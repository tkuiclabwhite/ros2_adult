#!/usr/bin/env python3
# coding=utf-8
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String, UInt8MultiArray, Int16, Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

from tku_msgs.msg import (
    Interface,
    SensorPackage,
    SensorSet,
    DrawImage,
    HeadPackage,
    SingleMotorData,
    Parameter,
    Dio,
)

# -------------------- Strategy Process Manager (Plan B-1) --------------------

@dataclass
class StrategyHandle:
    name: str
    popen: subprocess.Popen

class StrategyProcessManager:
    # Ensure only ONE strategy process runs at a time (ros2 run).
    def __init__(self):
        self._current: Optional[StrategyHandle] = None

    def is_running(self) -> bool:
        return self._current is not None and self._current.popen.poll() is None

    def current_name(self) -> Optional[str]:
        return self._current.name if self.is_running() else None

    def start(self, name: str, ros2_pkg: str, ros2_exec: str):
        if self.is_running():
            self.stop()

        cmd = ["ros2", "run", ros2_pkg, ros2_exec]

        # stdout/stderr -> terminal (方便你看策略 log)
        popen = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,  # new process group => killpg works
            stdout=None,
            stderr=None,
            text=True,
        )
        self._current = StrategyHandle(name=name, popen=popen)

    def stop(self, timeout_sec: float = 2.0):
        if not self.is_running():
            self._current = None
            return

        pgid = os.getpgid(self._current.popen.pid)

        # graceful stop first
        os.killpg(pgid, signal.SIGINT)
        try:
            self._current.popen.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self._current.popen.wait(timeout=timeout_sec)

        self._current = None


# ------------------------------ API Node ------------------------------------

class API(Node):
    ORANGE, YELLOW, BLUE, GREEN, BLACK, RED, WHITE, OTHERS = range(8)
    COLORS = ['orange', 'yellow', 'blue', 'green', 'black', 'red', 'white', 'others']

    def __init__(self, node_name: str = 'API'):
        super().__init__(node_name)

        # stamp / fps EMA
        self.label_matrix_stamp = (0, 0)
        self._lm_prev_ns = None
        self.lm_dt_ms_ema = None
        self.lm_fps_ema = None

        # callback groups
        self.imu_cbg = ReentrantCallbackGroup()
        self.image_cbg = ReentrantCallbackGroup()

        # QoS
        self.qos_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        # for high-rate feature topics (detections/label_matrix)
        self.qos_fast = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # -------------------- Publishers --------------------
        self.imu_reset_pub = self.create_publisher(SensorSet, '/sensorset', 10)
        self.singlemotor_pub = self.create_publisher(SingleMotorData, '/package/SingleMotorData', 10)
        self.SingleAbsolutePosition_pub = self.create_publisher(SingleMotorData, '/package/SingleAbsolutePosition', 10)

        self.generate_pub = self.create_publisher(Int16, '/ContinousMode_Topic', 10)
        self.continous_pub = self.create_publisher(Interface, '/ChangeContinuousValue_Topic', 10)
        self.body_auto_pub = self.create_publisher(Interface, '/SendBodyAuto_Topic', 10)

        self.head_motor_pub = self.create_publisher(HeadPackage, '/Head_Topic', 10)
        self.sector_pub = self.create_publisher(Int16, '/package/Sector', 10)

        self.draw_image_pub = self.create_publisher(DrawImage, '/drawimage', 10)
        self.walkparameter_pub = self.create_publisher(Parameter, '/strategy/walkparameter', 10)

        self.walking_json_pub = self.create_publisher(String, '/walking_params_update', 10)

        # Head scale control
        self.HEAD_PAN_ID = 1
        self.HEAD_TILT_ID = 2
        self.HEAD_PAN_CENTER = 2048
        self.HEAD_TILT_CENTER = 2048
        self.HEAD_PAN_RANGE = 600
        self.HEAD_TILT_RANGE = 500
        self.HEAD_DEFAULT_SPEED = 100

        # -------------------- Parameters (optional) --------------------
        self.declare_parameter("start", False)
        self.declare_parameter("mode", 0)

        # 實例變數 (在 __init__ 裡面)
        self.is_start: bool = False
        """
        硬體開關狀態：

        * **True** 代表啟動
        * **False** 代表停止
        """

        self._last_param_start: Optional[bool] = None
        self._last_param_mode: Optional[int] = None
        self.create_timer(0.05, self._sync_start_from_param)  # 20Hz

        # -------------------- IMU / YOLO subscriptions --------------------
        self.imu_sub = self.create_subscription(
            SensorPackage, '/package/sensorpackage', self.imu, 10, callback_group=self.imu_cbg
        )
        self.yolo_zed = self.create_subscription(
            Point, '/zed_yolo_ball', self.Yolo_Zed, 10, callback_group=self.imu_cbg
        )
        self.ContinuousValue_sub = self.create_subscription(
            Interface, '/ChangeContinuousValue_Topic', self.ContinuousValueFunction,
            self.qos_latest, callback_group=self.imu_cbg
        )

        # -------------------- Image / detection subscriptions --------------------
        self.label_matrix: Optional[np.ndarray] = None
        self.label_matrix_flatten: Optional[np.ndarray] = None
        self._bridge = CvBridge()

        self.label_img_sub = self.create_subscription(
            RosImage, '/label_matrix', self._label_image_cb, self.qos_fast, callback_group=self.image_cbg
        )

        self.latest_masks: Dict[str, Optional[np.ndarray]] = {c: None for c in self.COLORS}
        for c in self.COLORS:
            topic = f'/{c}_mask'
            self.create_subscription(
                UInt8MultiArray,
                topic,
                lambda msg, label=c: self._mask_callback(msg, label),
                10,
                callback_group=self.image_cbg,
            )

        self.latest_objects: Dict[str, List[dict]] = {c: [] for c in self.COLORS}
        self.latest_stamps: Dict[str, Tuple[int, int]] = {c: (0, 0) for c in self.COLORS}
        for c in self.COLORS:
            topic = f'/detections/{c}'
            self.create_subscription(
                String, topic, lambda msg, label=c: self._det_callback(msg, label),
                self.qos_fast, callback_group=self.image_cbg
            )

        # -------------------- state / stats --------------------
        self.roll = self.pitch = self.yaw = 0.0
        self.imu_rpy : List[float] = [self.roll, self.pitch, self.yaw]
        """
        IMU數值：
        
            * [0] roll：翻滾角
            * [1] pitch：俯仰角
            * [2] yaw：偏航角
        
        Example:
            >>> # 存取特定的姿態數值（例如：Roll 翻滾角）
            >>> # 索引 [0] 為 Roll, [1] 為 Pitch, [2] 為 Yaw
            >>> current_roll = api.imu_rpy[0]
        """

        self.pose_x = self.pose_y = self.pose_z = 0.0
        self.pose = [self.pose_x, self.pose_y, self.pose_z]
        self.xx, self.yy, self.tt = 0, 0, 0

        self.color_counts: List[int] = [0] * len(self.COLORS)
        """
        各顏色類別偵測到的物體總數清單。

        紀錄當前影像中，每一種特定顏色類別被偵測到的物體數量。

        * **顏色索引對照表**:
            * **[0] Orange**: 橘色物體數量
            * **[1] Yellow**: 黃色物體數量
            * **[2] Blue**: 藍色物體數量
            * **[3] Green**: 綠色物體數量
            * **[4] Black**: 黑色物體數量
            * **[5] Red**: 紅色物體數量
            * **[6] White**: 白色物體數量
            * **[7] Others**: 其他類別數量

        Example:
            >>> # 檢查畫面上是否有偵測到橘色物體
            >>> if api.color_counts[0] > 0:
            >>>     print(f"偵測到 {api.color_counts[0]} 個橘色目標！")
        """

        self.object_sizes: List[List[float]] = [[] for _ in self.COLORS]
        """
        各顏色類別偵測到的物體面積 (Area) 清單。

        內層清單則包含該顏色類別下，所有偵測到物體的面積資訊。

        Example:
            >>> api.object_sizes[0]    #會回傳一個包含所有橘色物體面積的清單。
            >>> api.object_sizes[0][0] #則是第一個偵測到的橘色物體面積。
        """

        self.object_x_min: List[List[int]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的最小 X 座標 (左邊界) 清單。

        這是一個二維清單，用於紀錄每個偵測物體在影像中的最左側像素位置。
        數值範圍介於(0 ~ 320)之間。

        Example:
            >>> # 取得第一個偵測到的橘色物體左邊界 (解析度寬度 0~320)
            >>> if api.object_x_min[0]:
            >>>     left_edge = api.object_x_min[0][0]
        """

        self.object_x_max: List[List[int]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的最大 X 座標 (右邊界) 清單。

        這是一個二維清單，用於紀錄每個偵測物體在影像中的最右側像素位置。
        數值範圍介於(0 ~ 320)之間。

        Example:
            >>> # 取得第一個偵測到的黃色物體右邊界
            >>> if api.object_x_max[1]:
            >>>     right_edge = api.object_x_max[1][0]
        """

        self.object_y_min: List[List[int]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的最小 Y 座標 (上邊界) 清單。

        這是一個二維清單，用於紀錄每個偵測物體在影像中的最上側像素位置。
        數值範圍介於(0 ~ 240)之間。

        Example:
            >>> # 取得第三個偵測到的藍色物體上邊界 (解析度高度 0~240)
            >>> if api.object_y_min[2]:
            >>>     top_edge = api.object_y_min[2][2]
        """

        self.object_y_max: List[List[int]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的最大 Y 座標 (下邊界) 清單。

        這是一個二維清單，用於紀錄每個偵測物體在影像中的最下側像素位置。
        數值範圍介於(0 ~ 240)之間。

        Example:
            >>> # 取得第二個偵測到的白色物體下邊界 (解析度高度 0~240)
            >>> if api.object_y_min[6]:
            >>>     top_edge = api.object_y_min[6][1]
        """

        self.new_object_info: bool = False

        # -------------------- Plan B Control Plane --------------------
        self._mgr = StrategyProcessManager()
        self._selected_strategy = "ar"

        self.strategy_name_sub = self.create_subscription(
            String, "/strategy/name", self._on_strategy_name, 10
        )
        self.strategy_start_sub = self.create_subscription(
            Bool, "/strategy/start", self._on_strategy_start, 10
        )
        self.strategy_status_pub = self.create_publisher(
            String, "/strategy/status", 10
        )

        # -------------------- 硬體 DIO 狀態 --------------------
        self.is_start = False
        self.dio_data = 0
        
        # 訂閱硬體節點發出的訊息
        self.dio_sub = self.create_subscription(
            Dio,
            '/package/dioarray',
            self._dio_callback,
            10
        )

        # 你要啟動的對應表：UI 只要送 name=ar/bb/sp
        # 注意：這裡的 exe 必須存在於 setup.py 的 console_scripts
        self._strategy_map = {
            "ar": ("strategy", "ar"),
            "bb": ("strategy", "bb"),
            "sp": ("strategy", "sp"),
            "obs": ("strategy", "obs"),
            "sr": ("strategy", "sr"),
            "wl": ("strategy", "wl"),
            "mar": ("strategy", "mar"),
            "pk": ("strategy", "pk"),
        }

        self._publish_strategy_status("idle")

    def _dio_callback(self, msg: Dio):
        """
        同步硬體實體開關狀態的回呼函式。

        監聽來自 `/package/dioarray` 主題的訊息，並在物理開關狀態發生變化時更新
        內部的 `is_start` 標記。 此變數通常作為所有策略程式
        執行迴圈的入口條件。

        Args:
            msg (Dio): 來自硬體底層的數位訊號包。 
                       其中 `msg.strategy` 欄位對應機器人背部的物理撥桿開關。

        Note:
            * **邊緣觸發 (Edge Detection)**：僅在狀態與上次不同時才執行記錄與邏輯更新，避免重複處理。
            * **視覺反饋**：使用 ANSI 轉義序列在終端機輸出彩色日誌（綠色為 START，紅色為 STOP）。
            * 目前版本為了安全性，註解掉了自動觸發進程管理器的功能。
        同步狀態 (把硬體的 True/False 給 API)
        """
        if msg.strategy != self.is_start:
            self.is_start = msg.strategy
            
            if self.is_start:
                self.get_logger().info("\033[92m[DIO] 物理開關開啟：START\033[0m")
                # 如果你需要它同時觸發進程管理器的話才保留下面這行
                # self._start_selected_strategy() 
            else:
                self.get_logger().info("\033[91m[DIO] 物理開關關閉：STOP\033[0m")
                # self._stop_strategy()
                
    # -------------------- Param Sync (optional) --------------------
    def _sync_start_from_param(self):
        """
        同步內部狀態與外部 ROS 2 參數的回呼函式。

        定期讀取節點參數 `start` 與 `mode`，並在偵測到參數異動時，
        自動觸發對應策略的啟動或停止流程。

        此函式實現了「參數驅動控制」，允許使用者透過 `ros2 param set` 
        或 GUI 工具遠端切換機器人任務（如從自動導引切換至平衡木）。

        Attributes Referenced:
            _last_param_start (bool): 紀錄上一次的啟動參數狀態。
            _last_param_mode (int): 紀錄上一次的模式參數狀態。
            _selected_strategy (str): 根據模式編號映射後的策略代號。

        Note:
            * **防洗版機制 (Anti-spam)**：僅在參數值確實發生改變時才執行後續邏輯，避免重複觸發進程管理器。
            * **模式映射**：目前預設編號為 0 (ar), 1 (bb), 2 (sp)。
        """
        start = bool(self.get_parameter("start").value)
        mode = int(self.get_parameter("mode").value)

        # avoid spamming
        changed = (start != self._last_param_start) or (mode != self._last_param_mode)
        if not changed:
            return
        self._last_param_start = start
        self._last_param_mode = mode

        # If you want param mode to map to strategy name, do it here:
        # e.g. mode=0->ar,1->bb,2->sp
        mode_map = {0: "ar", 1: "bb", 2: "sp"}
        if mode in mode_map:
            self._selected_strategy = mode_map[mode]

        # mimic start/stop
        if start:
            self._start_selected_strategy()
        else:
            self._stop_strategy()

    # -------------------- Plan B callbacks --------------------
    def _publish_strategy_status(self, state: str): 
        """       
        發送當前策略狀態的格式化字串至 ROS 2 主題。

        將內部的策略選擇狀態與進程管理器的運行狀態整合為一條字串訊息，
        並發布至 `/strategy/status`。 這對於外部 UI 
        監控機器人當前的任務切換情形至關重要。

        Args:
            state (str): 自定義的狀態描述詞（例如 "idle", "running", "error"）。

        Note:
            發布的字串格式為：`"{state}; selected={_selected_strategy}; running={current_name}"`。
            其中 `running` 欄位來自 `StrategyProcessManager` 的即時查詢結果。
        """
        msg = String()
        msg.data = f"{state}; selected={self._selected_strategy}; running={self._mgr.current_name()}"
        self.strategy_status_pub.publish(msg)

    def _on_strategy_name(self, msg: String):
        """
        處理策略名稱切換的回呼函式。

        監聽來自 `/strategy/name` 主題的字串訊息，並更新內部的預選策略代號 
        (`_selected_strategy`)。此動作僅更新標記，並不會立即啟動進程。

        Args:
            msg (String): 包含策略代號（如 "ar", "bb", "sp"）的 ROS 2 字串訊息。

        Note:
            * **輸入清洗**：使用 `.strip()` 去除字串前後的空白或換行符號，防止比對失敗。
            * **空值保護**：若收到的訊息內容為空，則直接忽略，維持目前的預選狀態。
            * **狀態廣播**：更新成功後會呼叫 `_publish_strategy_status` 同步外部狀態。
        """
        name = msg.data.strip()
        if not name:
            return
        self._selected_strategy = name
        self.get_logger().info(f"[strategy] selected = {name}")
        self._publish_strategy_status("selected")

    def _on_strategy_start(self, msg: Bool):
        """
        處理策略啟動與停止指令的回呼函式。

        監聽來自 `/strategy/start` 主題的布林訊息。當收到 `True` 時，
        會嘗試啟動目前預選的策略進程；收到 `False` 時則終止當前運行的策略。

        此函式是外部指令進入系統執行層的關鍵入口，實現了策略的遠端啟動功能。

        Args:
            msg (Bool): ROS 2 標準布林訊息包。
                - `True`: 呼叫 `_start_selected_strategy()` 執行任務。
                - `False`: 呼叫 `_stop_strategy()` 停止任務。

        Note:
            啟動前必須先透過 `_on_strategy_name` 選定策略名稱，否則啟動可能會因找不到對應功能包而失敗。
        """
        if msg.data:
            self._start_selected_strategy()
        else:
            self._stop_strategy()

    def _start_selected_strategy(self):
        """
        執行目前預選的機器人策略進程。

        根據 `_selected_strategy` 的內容，從 `_strategy_map` 中檢索對應的 
        ROS 2 功能包與執行檔名稱，並透過 `StrategyProcessManager` 啟動該進程。

        此函式包含完整的校驗與錯誤處理機制，確保在啟動失敗時能及時回報狀態。

        Raises:
            Exception: 若 `StrategyProcessManager` 在啟動進程時發生系統錯誤，會被內部捕捉並記錄。

        Note:
            * **白名單校驗**：若預選名稱不在 `_strategy_map` 中，將視為未知任務並終止啟動。
            * **狀態反饋**：啟動成功會發布 "running" 狀態；失敗或無效則發布 "error"。
            * **自動切換**：`StrategyProcessManager` 內部會處理「先停後開」的邏輯，此處僅負責發起啟動請求。
        """
        name = self._selected_strategy
        if name not in self._strategy_map:
            self.get_logger().error(f"[strategy] unknown: {name}")
            self._publish_strategy_status("error")
            return

        pkg, exe = self._strategy_map[name]
        try:
            self._mgr.start(name=name, ros2_pkg=pkg, ros2_exec=exe)
            self.get_logger().info(f"[strategy] started: {name}")
            self._publish_strategy_status("running")
        except Exception as e:
            self.get_logger().error(f"[strategy] start failed: {e}")
            self._publish_strategy_status("error")

    def _stop_strategy(self):
        """
        停止當前正在執行的機器人策略進程。

        透過 `StrategyProcessManager` 向背景進程發送終止訊號。
        此函式確保系統在切換任務或遭遇緊急狀況時，能回收系統資源並停止馬達控制指令。

        此函式包含例外處理機制，防止因進程無法終止而導致 API 主節點崩潰。

        Raises:
            Exception: 若在執行 `_mgr.stop()` 過程中發生系統級錯誤，會被內部捕捉並記錄。

        Note:
            * **優雅終止**：底層管理器通常先發送 `SIGINT` (Ctrl+C)，若超時則發送 `SIGKILL` 強制結束。
            * **狀態回報**：執行成功後會發布 "stopped" 狀態；若發生異常則回報 "error"。
        """
        try:
            self._mgr.stop()
            self.get_logger().info("[strategy] stopped")
            self._publish_strategy_status("stopped")
        except Exception as e:
            self.get_logger().error(f"[strategy] stop failed: {e}")
            self._publish_strategy_status("error")

    # -------------------- existing helpers / callbacks --------------------
    def sendBodyAutoCmd(self, x: float=0, y: float=0, theta: float=0, walking_mode: int = 0) -> None:
        """
        發送步態步長數值，同時啟動步態。

        將 X、Y 與Theta 發送至Walking，並同時發送啟動步態訊號
        (generate) 令步態開始執行動作。

        Args:
            x (float, optional): 前後移動步長。正值代表前進，負值代表後退。預設為 0。
            y (float, optional): 左右平移步長。正值代表向左，負值代表向右。預設為 0。
            theta (float, optional): 旋轉角度。正值代表左轉，負值代表右轉。預設為 0。
            walking_mode (int, optional): 步態模式編號。用於切換不同的行走模式。預設為 0。
            
                * **0 (連續步態)**
                * **1 (上板步態)**
                * **2 (下板步態)**

        Returns:
            None

        Example:
            >>> # 讓機器人以前進步長 300 穩定行走
            >>> api.sendBodyAutoCmd(x=300, y=0, theta=0, walking_mode=0)
            >>>
            >>> # 讓機器人向左轉彎同時前進
            >>> api.sendBodyAutoCmd(x=200, theta=5)
            >>>
            >>> # 執行上板步態
            >>> api.sendBodyAutoCmd(x=20000, walking_mode=1)

        """
        m = Interface()
        m.x, m.y, m.theta = int(x), int(y), int(theta)
        m.walking_mode = walking_mode
        self.body_auto_pub.publish(m)
        n = Int16()
        n.data = 1
        self.generate_pub.publish(n)


    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        """
        將數值限制在指定的區間內（數值鉗位）。

        此函式用於確保輸入參數（如馬達角度、移動速度等）不會超出物理硬體 
        或演算法所能承受的範圍，是預防硬體損毀的重要保護機制。

        Args:
            v (float): 待處理的原始數值。
            lo (float): 允許的最小值（下限）。
            hi (float): 允許的最大值（上限）。

        Returns:
            float: 處理後的數值。若 `v` 小於 `lo` 則回傳 `lo`；若大於 `hi` 則回傳 `hi`；
                   若在區間內則回傳 `v` 本身。

        Example:
            >>> API._clamp(1.5, -1.0, 1.0)
            1.0
            >>> API._clamp(-2.0, -1.0, 1.0)
            -1.0
        """
        return lo if v < lo else hi if v > hi else v

    def set_head(self, pan: float, tilt: float, speed: Optional[int] = None) -> None:
        """
        控制機器人頭部的左右 (Pan) 與上下 (Tilt) 角度。

        將輸入的比例值 (-1.0 ~ 1.0) 映射為硬體馬達的實際位置值，並發布至頭部控制主題。
        此函式會自動限制輸入範圍以保護馬達硬體。

        Args:
            pan (float): 左右轉動比例。範圍為 -1.0 (最右) 到 1.0 (最左)。
            tilt (float): 上下仰俯比例。範圍為 -1.0 (最下) 到 1.0 (最上)。
            speed (int, optional): 馬達轉動速度。若未指定，則使用 `HEAD_DEFAULT_SPEED`。

        Returns:
            None

        Note:
            * **線性映射**：計算公式為 `CENTER + (RATIO * RANGE)`。
            * **硬體保護**：內部呼叫 `_clamp` 確保輸出位置不會超出 `PAN_RANGE` 或 `TILT_RANGE`。
            * **分次發布**：函式會依序發布 Pan 與 Tilt 兩個 `HeadPackage` 訊息。
        """
        pan = self._clamp(pan, -1.0, 1.0)
        tilt = self._clamp(tilt, -1.0, 1.0)
        spd = self.HEAD_DEFAULT_SPEED if speed is None else int(speed)

        pan_pos = int(self.HEAD_PAN_CENTER + pan * self.HEAD_PAN_RANGE)
        tilt_pos = int(self.HEAD_TILT_CENTER + tilt * self.HEAD_TILT_RANGE)

        m1 = HeadPackage()
        m1.id, m1.position, m1.speed = self.HEAD_PAN_ID, pan_pos, spd
        self.head_motor_pub.publish(m1)

        m2 = HeadPackage()
        m2.id, m2.position, m2.speed = self.HEAD_TILT_ID, tilt_pos, spd
        self.head_motor_pub.publish(m2)

    def _is_newer_stamp(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """
        比較兩個 ROS 2 時間戳記，判斷 A 是否比 B 更晚（更新）。

        此函式遵循 ROS 2 的時間格式，將時間拆解為秒 (Seconds) 與奈秒 (Nanoseconds) 分別進行比較。
        常用於過濾掉因網路延遲或異步通訊導致的舊資料（如舊的 YOLO 偵測結果）。

        Args:
            a (Tuple[int, int]): 待檢查的時間戳記 (sec, nanosec)。
            b (Tuple[int, int]): 作為基準的舊時間戳記 (sec, nanosec)。

        Returns:
            bool: 若 A 的時間晚於 B 則回傳 `True`，否則回傳 `False`。

        Example:
            >>> api._is_newer_stamp((100, 500), (100, 200))  # 秒相同，比較奈秒
            True
            >>> api._is_newer_stamp((101, 0), (100, 9999))   # 秒較大，即為更新
            True
        """
        return (a[0] > b[0]) or (a[0] == b[0] and a[1] > b[1])

    def _recompute_stats(self) -> None:
        """
        重新計算並更新所有偵測物體的統計數據。

        此函式會遍歷 `latest_objects` 中儲存的所有顏色類別，提取每個物體的邊界框 (Bounding Box) 
        與面積資訊，並將其分類存入對應的成員變數清單中。

        更新的屬性包括：
            - color_counts (list): 各顏色類別偵測到的物體總數。
            - object_sizes (list[list]): 各物體的面積 (Area)。
            - object_x_min / x_max (list[list]): 物體邊界框的水平座標極值。
            - object_y_min / y_max (list[list]): 物體邊界框的垂直座標極值。

        Note:
            * **資料重置**：每次呼叫時會先清空舊數據，確保統計資訊與當前影像同步。
            * **容錯機制**：若物體資料格式錯誤（如缺少 bbox），會跳過該物體並發出警告日誌，避免程式崩潰。
            * **座標轉換**：會自動將寬度 (w) 與高度 (h) 轉換為極值座標（如 x + w = x_max）。
        """
        n = len(self.COLORS)
        self.color_counts = [0] * n
        self.object_sizes = [[] for _ in self.COLORS]
        self.object_x_min = [[] for _ in self.COLORS]
        self.object_x_max = [[] for _ in self.COLORS]
        self.object_y_min = [[] for _ in self.COLORS]
        self.object_y_max = [[] for _ in self.COLORS]

        for idx, color in enumerate(self.COLORS):
            lst = self.latest_objects.get(color, [])
            self.color_counts[idx] = len(lst)
            for o in lst:
                try:
                    x, y, w, h = o['bbox']
                    area = float(o.get('area', float(w * h)))
                except Exception as ex:
                    self.get_logger().warn(f"[stats] Malformed object for color {color}: {ex}")
                    continue
                self.object_sizes[idx].append(area)
                self.object_x_min[idx].append(int(x))
                self.object_x_max[idx].append(int(x + w))
                self.object_y_min[idx].append(int(y))
                self.object_y_max[idx].append(int(y + h))

    def _det_callback(self, msg: String, label: str) -> None:
        """
        處理特定顏色類別偵測結果的回呼函式。

        接收來自 `/detections/{label}` 的 JSON 字串訊息，解析後更新對應顏色的 
        物體清單與時間戳記。 隨後觸發統計數據重算，並標記新資訊標誌。

        Args:
            msg (String): 包含 YOLO 偵測結果（objects, stamp）的 JSON 字串。
            label (str): 物體的顏色類別名稱（例如 'orange', 'yellow'）。

        Note:
            * **時序校驗**：呼叫 `_is_newer_stamp` 確保不處理因網路延遲導致的舊封包。
            * **統計更新**：解析成功後會自動執行 `_recompute_stats` 更新座標清單。
            * **狀態標記**：將 `new_object_info` 設為 True，通知策略主迴圈有新的視覺資料。
            * **容錯處理**：包含完整的 JSON 解析例外捕捉，防止格式異常導致節點崩潰。
        """
        try:
            data = json.loads(msg.data)
            st = data.get('stamp', {})
            cur = (int(st.get('sec', 0)), int(st.get('nanosec', 0)))
            prev = self.latest_stamps.get(label, (0, 0))
            if not self._is_newer_stamp(cur, prev):
                return

            objs = data.get('objects', [])
            if not isinstance(objs, list):
                objs = []

            self.latest_objects[label] = objs
            self.latest_stamps[label] = cur

            self._recompute_stats()
            self.new_object_info = True

        except Exception as e:
            self.get_logger().error(f'[det] detections/{label} parse error: {e}')

    def _mask_callback(self, msg: UInt8MultiArray, label: str) -> None:
        """
        處理二值化遮罩影像的回呼函式。

        接收來自 `/{label}_mask` 主題的扁平化陣列資料，並根據訊息中的佈局 (Layout) 
        資訊將其重塑 (Reshape) 為二維矩陣格式。

        此函式處理的是各顏色類別的分割結果，儲存於 `latest_masks` 字典中。

        Args:
            msg (UInt8MultiArray): 包含影像原始資料與佈局資訊的 ROS 2 多維陣列訊息。
            label (str): 遮罩對應的顏色標籤（如 'orange', 'black'）。

        Note:
            * **結構校驗**：函式會檢查佈局維度 (dim) 是否至少包含兩維（列與欄），否則將視為格式錯誤。
            * **矩陣還原**：利用 `numpy.reshape` 將一維資料轉回影像矩陣空間。
            * **容錯處理**：若發生陣列大小不匹配或重塑失敗，會輸出錯誤日誌以供除錯。
        """
        dims = msg.layout.dim
        if len(dims) < 2:
            self.get_logger().error("[mask] layout 格式錯誤")
            return
        rows = dims[0].size
        cols = dims[1].size
        try:
            arr = np.array(msg.data, dtype=np.uint8).reshape((rows, cols))
            self.latest_masks[label] = arr
        except Exception as e:
            self.get_logger().error(f"[mask] 還原 {label} mask 失敗：{e}")

    def _label_image_cb(self, msg: RosImage) -> None:
        """
        處理標籤矩陣影像的回呼函式，並計算處理效能。

        接收來自視覺節點的標籤影像 (Label Image)，將其轉換為 NumPy 矩陣並儲存。
        同時利用時間戳記計算影格間距 (dt) 與指數移動平均 FPS (lm_fps_ema)。

        Args:
            msg (RosImage): 包含標籤資訊的 ROS 影像訊息，編碼預計為 'mono8'。

        Note:
            * **影像轉換**：使用 `CvBridge` 將 ROS 影像格式轉換為 OpenCV/NumPy 的二維矩陣。
            * **資料扁平化**：除了存儲二維矩陣 `label_matrix`，也會同步更新一維的 `label_matrix_flatten` 供快速索引。
            * **效能統計**：採用指數移動平均 (EMA, Alpha=0.2) 平滑化 FPS 數值，避免因單次網路延遲造成數值大幅跳動。
            * **單位轉換**：將秒與奈秒轉換為純奈秒 (ns) 進行計算，隨後換算為毫秒 (ms)。
        """
        try:
            arr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.label_matrix = arr
            self.label_matrix_flatten = arr.flatten()

            self.label_matrix_stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
            cur_ns = self.label_matrix_stamp[0] * 1_000_000_000 + self.label_matrix_stamp[1]
            if self._lm_prev_ns is not None and cur_ns > self._lm_prev_ns:
                dt_ms = (cur_ns - self._lm_prev_ns) / 1e6
                alpha = 0.2
                self.lm_dt_ms_ema = dt_ms if self.lm_dt_ms_ema is None else (1 - alpha) * self.lm_dt_ms_ema + alpha * dt_ms
                if self.lm_dt_ms_ema and self.lm_dt_ms_ema > 1e-6:
                    self.lm_fps_ema = 1000.0 / self.lm_dt_ms_ema
            self._lm_prev_ns = cur_ns

        except Exception as e:
            self.get_logger().error(f"[label_matrix(Image)] 轉換失敗：{e}")

    def imu(self, msg: SensorPackage) -> None:
        """
        處理慣性測量單元 (IMU) 數據的回呼函式。

        接收來自底層感測器節點的姿態訊息，並將歐拉角 (Euler Angles) 同步至內部的
        狀態變數中。這些數據對於機器人的跌倒偵測與動態平衡控制至關重要。

        Args:
            msg (SensorPackage): 包含 `roll` (翻滾角)、`pitch` (俯仰角) 與 
                                 `yaw` (偏航角) 的感測器數據包。

        Note:
            * **座標系統**：數據通常以弧度或角度表示，具體取決於底層驅動的定義。
            * **資料同步**：函式同時更新獨立的 `roll/pitch/yaw` 變數與整合後的 `imu_rpy` 清單。
        """
        self.roll = msg.roll
        self.pitch = msg.pitch
        self.yaw = msg.yaw
        self.imu_rpy = [self.roll, self.pitch, self.yaw]

    def Yolo_Zed(self, msg: Point) -> None:
        """
        處理 YOLO 與 ZED 相機整合後的空間定位數據回呼函式。

        接收來自視覺偵測節點的 3D 座標訊息，並將其同步至內部的姿態變數中。
        這些數據代表目標物相對於機器人相機座標系的實際物理位置 (x, y, z)。

        Args:
            msg (Point): 包含 `x` (左右偏移)、`y` (前後距離/深度) 與 
                         `z` (上下高度) 的空間點位數據。

        Note:
            * **座標映射**：根據 ZED 的標準，`x` 通常代表水平偏移，`y` 代表物體距離，`z` 代表物體高度。
            * **資料同步**：函式會同步更新獨立的 `pose_x/y/z` 與整合後的 `pose` 清單。
            * **應用場景**：常用於自動導引 (AR) 任務中精確追蹤目標物或避障。
        """
        self.pose_x = msg.x
        self.pose_y = msg.y
        self.pose_z = msg.z
        self.pose = [self.pose_x, self.pose_y, self.pose_z]

    def sendSensorReset(self, status: bool) -> None:
        """
        發送感測器重設訊號。

        此函式於校正 IMU 姿態，將Yaw/Roll/Pitch歸零。

        Args:
            status (bool): 重設狀態開關。
                * True: 觸發重設/校正動作。
                * False: 結束重設狀態或保持正常運作。

        Returns:
            None

        Note:
            * 只需要發送一次True，無須發送False指令

        Example:
            >>> # 重設imu
            >>> api.sendSensorReset(True)
        """
        rst = SensorSet()
        rst.reset = status
        self.imu_reset_pub.publish(rst)

    def sendbodyAuto(self, generate: int) -> None:
        """
        發送步態觸發訊號，切換機器人步態的開關。

        (使用現在Walking內的步長數值開啟/關閉步態)

        Args:
            generate (int): 執行訊號。
                * **1 (啟動或持續執行步態)**
                * **0 (停止執行步態)**

        Returns:
            None

        Note:
            * **與 [sendBodyAutoCmd] 的差異**：**[sendBodyAutoCmd]** 會同時更新步長數值並開啟步態；
              而 **[sendbodyAuto]** 僅單獨發送觸發訊號，不會改變目前的步長數值，並且無法且換行走模式，固定為連續步態。

        Example:
            >>> # 停止行走
            >>> api.sendbodyAuto(0)
            >>>
            >>> # 開啟/持續執行步態
            >>> api.sendbodyAuto(1)
        """
        m = Int16()
        m.data = generate
        self.generate_pub.publish(m)

    def sendContinuousValue(self, x: int, y: int, theta: int, walking_mode: int = 0) -> None:
        """
        發送連續移動步長數值。

        將 X、Y 與Theta發布至Walking。此函式通常配合已啟動的步態使用，

        與 **[sendBodyAutoCmd]** 不同，此函式不會發送啟動步態訊號，僅更新底層緩存的步長數值。

        Args:
            x (int): 前後移動步長。
            y (int): 左右移動步長。
            theta (int): 旋轉角度。
            walking_mode (int, optional): 步態模式編號。預設為 0。

        Returns:
            None

        Note:
            * **依賴性**：呼叫此函式前，通常需要先透過 **[sendbodyAuto(1)]** 啟動步態，機器人才會根據這些參數開始移動。

        Example:
            >>> # 1.先啟動步態引擎 (原地踏步)
            >>> api.sendbodyAuto(1)
            >>> # 2. 給予移動步長
            >>> api.sendContinuousValue(x=300, y=-100, theta=3)
        """
        m = Interface()
        m.x, m.y, m.theta = x, y, theta
        m.walking_mode = walking_mode # 寫入
        self.continous_pub.publish(m)

    def sendBodySector(self, sector: int) -> None:
        """
        發送執行動作磁區 (Sector) 指令。

        Args:
            sector (int): 動作區段編號。
                - 29固定為基礎站立姿勢

        Returns:
            None

        Note:
            * **不可中斷性**：啟動一個 Sector 後，通常需要等待該動作序列執行完畢，請在此指令後給足夠的延遲時間，否則可能導致關節衝突。
            * **與步態解耦**：此函式於執行非步態類型的固定動作，請確保執行 Sector 前，有中止連續步態。

        Example:
            >>> #回到站姿
            >>> api.sendBodySector(29)

        """
        m = Int16()
        m.data = sector
        self.sector_pub.publish(m)

    def sendSingleMotor(self, ID: int, Position: int, Speed: int) -> None:
        """
        發送單一馬達控制指令。

        直接指定馬達 ID 並發布 **[相對刻度]** 與速度指令。

        Args:
            ID (int): 目標馬達的硬體編號(1~22）。
            Position (int): 相對刻度值。
            Speed (int): 速度設定。

        Returns:
            None

        Note:
            * **警告**：此函式不包含軟體保護限制，使用不當可能導致關節超出物理極限而毀損硬體。

            * **與SingleAbsolutePosition的差異**： **sendSingleMotor** 是轉動相對刻度，而 **SingleAbsolutePosition** 是轉動絕對刻度

        Example:
            >>> # 轉動腰部馬達
            >>> api.sendSingleMotor(9,50,15)
        """
        m = SingleMotorData()
        m.id, m.position, m.speed = ID, Position, Speed
        self.singlemotor_pub.publish(m)

        

    def SingleAbsolutePosition(self, ID: int, Position: int, Speed: int) -> None:
        """
        發送單一馬達控制指令

        直接指定馬達 ID 並發布 **[絕對刻度]** 與速度指令

        Args:
            ID (int): 目標馬達的硬體編號。
            Position (int): 絕對刻度值。
            Speed (int): 速度設定。

        Returns:
            None
            
        Note:
            * **警告**：此函式不包含軟體保護限制，使用不當可能導致關節超出物理極限而毀損硬體。

            * **與sendSingleMotor的差異**： **sendSingleMotor** 是轉動相對刻度，而 **SingleAbsolutePosition** 是轉動絕對刻度

        Example:
            >>> # 腰部馬達回正
            >>> api.sendSingleMotor(9,2048,15)
        """
        m = SingleMotorData()
        m.id, m.position, m.speed = ID, Position, Speed
        self.SingleAbsolutePosition_pub.publish(m)        

    def sendHeadMotor(self, ID: int, Position: int, Speed: int) -> None:
        """
        發送指令至指定的頭部馬達。

        直接指定頭部馬達 ID、目標位置值與旋轉速度。

        Args:
            ID (int): 頭部馬達的硬體 ID。
                * 1為水平馬達(Horizontal)
                * 2為垂直馬達(Vertical)
            Position (int): 目標絕對刻度。範圍為0~4095。
            Speed (int): 馬達移動速度。

        Returns:
            None

        Note:
            * **無保護機制**：此函式不會自動過濾超出範圍的數值，請在呼叫此函式前在Image網頁界面確認Position是否能到達，以免硬體卡死。

        Example:
            >>> # 水平馬達回正
            >>> api.sendHeadMotor(1,2048,100)
        """
        m = HeadPackage()
        m.id, m.position, m.speed = ID, Position, Speed
        self.head_motor_pub.publish(m)

    def drawImageFunction(self, cnt: int, mode: int,
                          xmin: int, xmax: int, ymin: int, ymax: int,
                          r: int, g: int, b: int,thickness: int=1) -> None:
        """
        在Image網頁介面上繪製幾何圖形或標記。

        Args:
            cnt (int): 圖形編號，用於區分多個標記。
            mode (int): 繪製模式。
                * 1直線
                * 2矩形
                * 3圓形
            xmin (int): 圖形區域的左邊界座標。在圓形模式時為圓心x座標。
            xmax (int): 圖形區域的右邊界座標。在圓形模式時為圓形半徑。
            ymin (int): 圖形區域的上邊界座標。在圓形模式時為圓心y座標。
            ymax (int): 圖形區域的下邊界座標。在圓形模式時無作用。
            r (int): RGB 顏色空間中的紅色值 (0-255)。
            g (int): RGB 顏色空間中的綠色值 (0-255)。
            b (int): RGB 顏色空間中的藍色值 (0-255)。
            thickness (int, optional): 線條粗細。預設為 1。

        Returns:
            None

        Note:
            **圓形模式**：注意圓形模式參數代表含意不同。

        Example:
            >>> # 畫直線
            >>> api.drawImageFunction(1, 1, 0, 320, 120, 120, 255, 0, 0, 2)
            >>> # 畫矩形
            >>> api.drawImageFunction(2, 2, 150, 170 , 130, 110, 0, 255, 0, 1)
            >>> # 畫圓形
            >>> api.drawImageFunction(3, 3, 160, 100, 240, 0, 0, 0, 255, 1)
        """
        img = DrawImage()
        img.cnt = int(cnt)
        img.mode = int(mode)
        img.xmin, img.xmax = int(xmin), int(xmax)
        img.ymin, img.ymax = int(ymin), int(ymax)
        img.rvalue, img.gvalue, img.bvalue = int(r), int(g), int(b)
        img.thickness = int(thickness)
        self.draw_image_pub.publish(img)

    def sendWalkParameter(self,
            mode: int,
            com_y_swing: float,
            width_size: float,
            period_t: int,
            t_dsp: float,
            lift_height: float,
            stand_height: float,
            com_height: float,
        ):
        """
        發送標準步態參數。

        將基礎的行走參數封裝為 `Parameter` 訊息並發布。此函式主要用於
        一般的平地行走與基本的運動控制。

        Args:
            mode (int): 步態模式。
            com_y_swing (float): 重心 (CoM) 的左右擺幅。
            width_size (float): 雙腳間距（步寬）。
            period_t (int): 步態週期時間（單位通常為 ms）。
            t_dsp (float): 雙支撐相 (Double Support Phase) 時間比例。
            lift_height (float): 抬腳高度。
            stand_height (float): 站立高度（從地面到髖部）。
            com_height (float): 質心高度。
        """
        msg = Parameter()
        msg.mode = mode
        msg.com_y_swing = com_y_swing
        msg.width_size = width_size
        msg.period_t = period_t
        msg.t_dsp = t_dsp
        msg.lift_height = lift_height
        msg.stand_height = stand_height
        msg.com_height = com_height
        self.walkparameter_pub.publish(msg)

    def sendLCWalkParameter(self,
            # mode: int,
            com_y_swing: float, width_size: float,
            period_t: int, t_dsp: float, 
            stand_height: float = 23.5, com_height: float = 29.5,lift_height: float = 0,
            board_high: float = 0.0, clearance: float = 3.0,
            hip_roll: float = 0.0, ankle_roll: float = 0.0
        ):
        """
        發送步態參數給Walking，此函式包含上下板步態參數。

        Args:
            連續步態
                * com_y_swing (float): 起步重心(CoM)的左右補償，正值代表向左，負值代表向右。
                * width_size (float): 雙腳間距（步寬）。
                * period_t (int): 步態週期時間，越小踏步越快，以每20為一單位。
                * t_dsp (float): 雙支時間。
                * lift_height (float, optional): 抬腳高度。預設為 0。
                * stand_height (float, optional): 站立高度。預設為 23.5。
                * com_height (float, optional): 質心高度。預設為 29.5。

            上下板步態(包含除lift_height以外的連續步態參數)
                * board_high (float, optional): 木板的高度。預設為 0。
                * clearance (float, optional): 抬腳時的地面淨空高度。預設為 3。
                * hip_roll (float, optional): 髖部側傾補償角度。預設為 0。
                * ankle_roll (float, optional): 踝部側傾補償角度。預設為 0。

        Returns:
            None

        Note:
            * 如在啟動步態前無發送過步態參數，則會使用預設值。請確保在 **開啟步態** 前有使用此函數，或者在Walking網頁界面中 **Send/Load** 步態參數。
            
            * 在上下板步態時clearance會取代lift_height參數的位置

        Example:
            >>> api.sendLCWalkParameter(                
            ... com_y_swing  = float(-1.5),
            ... width_size   = float(4.5),
            ... period_t     = int(320),
            ... t_dsp        = float(0.1),
            ... lift_height  = float(2),
            ... stand_height = float(23.5),
            ... com_height   = float(29.5),
            ... )   
        """
        params = {
            # "walking_mode": mode,
            "com_y_swing": com_y_swing, "width_size": width_size,
            "period_t": period_t, "Tdsp": t_dsp,
            "lift_height": lift_height, "STAND_HEIGHT": stand_height,
            "COM_HEIGHT": com_height,
            "Board_High": board_high, "Clearance": clearance,
            "Hip_roll": hip_roll, "Ankle_roll": ankle_roll
        }
        msg = String()
        msg.data = json.dumps(params)
        self.walking_json_pub.publish(msg)

    def get_objects(self, color: Optional[str] = None) -> List[dict]:
        """
        獲取目前偵測到的物體列表。

        根據指定的顏色標籤檢索。若未指定顏色，則回傳所有已知 
        顏色類別的物體總表。

        Args:
            color (str, optional): 顏色類別名稱。預設為 None，回傳所有類別。
                
                * 顏色類別名稱：'orange', 'yellow', 'blue', 'green', 'black', 'red', 'white'
        
        Returns:
            List[dict]: 包含物體資訊
                * bbox：[x, y, w, h] (邊界框)

                    * x, y：矩形左上角的像素座標。
                    * w：矩形的寬度
                    * h：矩形的高度
                * centroid：[x, y] (質心 / 中心點)

                    * x, y：該物件幾何中心的像素座標。

                * area (該顏色區塊所佔據的總像素數量)

                * aspect_ratio (長寬比 (寬度 / 高度))

                * label (顏色標籤)

        Example:
            >>> # 1. 獲取當前畫面上所有已知顏色的偵測物體總表
            >>> all_detected = api.get_objects()
            >>>
            >>> # 2. 僅獲取特定標籤（例如：'orange'）的物體清單
            >>> orange_objs = api.get_objects(color='orange')
            >>>
            >>> # 3. 存取範例：獲取第一個橘色物體的中心點 X 座標與面積
            >>> if orange_objs:
            >>>     target = orange_objs[0]
            >>>     x_center = target['centroid'][0]
            >>>     size = target['area']
            >>>     print(f"Found orange target at X: {x_center}, Area: {size}")
            >>>
            >>> # 4. 進階篩選：找出畫面上面積最大（通常是最近）的藍色目標
            >>> blue_objs = api.get_objects('blue')
            >>> if blue_objs:
            >>>     closest_target = max(blue_objs, key=lambda x: x['area'])
        """
        if color is None:
            all_objs: List[dict] = []
            for c in self.COLORS:
                all_objs.extend(self.latest_objects.get(c, []))
            return all_objs
        return self.latest_objects.get(color, [])

    def ContinuousValueFunction(self, msg: Interface) -> None:
        """
        連續移動參數的狀態同步回呼函式。

        監聽來自 `/ChangeContinuousValue_Topic` 的訊息，並將其數值同步至內部變數 
        `xx`, `yy`, `tt` 中。

        此函式用於紀錄系統目前的「預計移動狀態」，方便在不直接存取發布者的情況下讀取運動參數。

        Args:
            msg (Interface): 包含 x, y, theta 的運動指令訊息。
        """
        self.xx = msg.x
        self.yy = msg.y
        self.tt = msg.theta


def main():
    rclpy.init()
    node = API()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()