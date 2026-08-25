#!/usr/bin/env python3
# coding=utf-8
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String, UInt8MultiArray, Int16, Bool
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

from tku_msgs.msg import (
    ZedImu,
    Zoom,
    Interface,
    SensorPackage,
    SensorSet,
    DrawImage,
    HeadPackage,
    SingleMotorData,
    Parameter,
    Dio,
)

# ------------------------------ API Node ------------------------------------

class API(Node):
    ORANGE, YELLOW, BLUE, GREEN, BLACK, RED, WHITE, OTHERS = range(8)
    COLORS = ['orange', 'yellow', 'blue', 'green', 'black', 'red', 'white', 'others']

    # 距離 class：與顏色各自獨立的分類系統，只是數量同為 8
    D1, D2, D3, D4, D5, D6, D7, D8 = range(8)
    DISTANCES = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

    # 疊合組合：每組綁定一個顏色 class 與一個距離 class
    SET1, SET2, SET3, SET4, SET5, SET6, SET7, SET8 = range(8)
    SETS = ['Set1', 'Set2', 'Set3', 'Set4', 'Set5', 'Set6', 'Set7', 'Set8']

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

        self.zoomin_pub = self.create_publisher(Zoom, '/Zoom_In_Topic', 10)

        self.draw_image_pub = self.create_publisher(DrawImage, '/drawimage', 10)
        self.draw_clear_pub = self.create_publisher(Bool, '/drawimage/clear', 10)
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

        # 實例變數 (在 __init__ 裡面)
        self.is_start: bool = False
        """
        硬體開關狀態：

        * **True** 代表啟動
        * **False** 代表停止
        """

        # -------------------- IMU / YOLO subscriptions --------------------
        self.imu_sub = self.create_subscription(
            SensorPackage, '/package/sensorpackage', self.imu, 10, callback_group=self.imu_cbg
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

        # -------------------- 深度 (depth_process_node) --------------------
        self.latest_depth_objects: Dict[str, List[dict]] = {d: [] for d in self.DISTANCES}
        self.latest_depth_stamps: Dict[str, Tuple[int, int]] = {d: (0, 0) for d in self.DISTANCES}
        for d in self.DISTANCES:
            self.create_subscription(
                String, f'/depth_detections/{d}',
                lambda msg, label=d: self._depth_det_callback(msg, label),
                self.qos_fast, callback_group=self.image_cbg
            )

        # 裁切對齊後的深度值（公釐 uint16），供 depth_at() 查任意座標
        self._depth_mm: Optional[np.ndarray] = None
        self.create_subscription(
            RosImage, '/depth_mm', self._depth_mm_cb,
            self.qos_fast, callback_group=self.image_cbg
        )

        # -------------------- 疊合 (overlap_node) --------------------
        self.latest_overlap_objects: Dict[str, List[dict]] = {t: [] for t in self.SETS}
        self.latest_overlap_stamps: Dict[str, Tuple[int, int]] = {t: (0, 0) for t in self.SETS}
        for t in self.SETS:
            self.create_subscription(
                String, f'/overlap_detections/{t}',
                lambda msg, label=t: self._overlap_det_callback(msg, label),
                self.qos_fast, callback_group=self.image_cbg
            )

        # -------------------- ZED 內建 IMU (zed_imu_node) --------------------
        self.zed_imu_sub = self.create_subscription(
            ZedImu, '/zed_imu/data', self._zed_imu_cb, 10, callback_group=self.imu_cbg
        )
        self.zed_imu_reset_pub = self.create_publisher(SensorSet, '/zed_sensorset', 10)

        # -------------------- ZED 里程計（最小驗證版，尚未做頭部補償） -----------
        self.odom_sub = self.create_subscription(
            Odometry, '/zed/zed_node/odom', self._odom_cb, 10, callback_group=self.imu_cbg
        )
        self.odom_reset_cli = self.create_client(Trigger, '/zed/zed_node/reset_odometry')

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

        self.xx, self.yy, self.tt = 0.0, 0.0, 0.0

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

        # 色模新增欄位：中心座標與寬高比。image.py 本來就有發，先前沒接。
        self.object_cx: List[List[int]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的中心 X 座標清單。

        這是一個二維清單，第一層為顏色索引，第二層為該顏色的第幾個物體。
        數值範圍介於 (0 ~ 960) 之間。

        Example:
            >>> # 取得第一個橘色物體的中心，用來對準頭部
            >>> if api.color_counts[api.ORANGE] > 0:
            >>>     cx = api.object_cx[api.ORANGE][0]
            >>>     offset = cx - 480      # 480 為畫面中心
        """

        self.object_cy: List[List[int]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的中心 Y 座標清單。

        這是一個二維清單，第一層為顏色索引，第二層為該顏色的第幾個物體。
        數值範圍介於 (0 ~ 600) 之間。

        Example:
            >>> if api.color_counts[api.RED] > 0:
            >>>     cy = api.object_cy[api.RED][0]
        """

        self.object_ratio: List[List[float]] = [[] for _ in self.COLORS]
        """
        各顏色類別物體邊界框的寬高比 (寬 / 高) 清單。

        用於形狀判斷，可過濾掉輪廓明顯不符的雜訊：

        * **約 1.0**：接近正方形或圓形（例如球）
        * **遠小於 1**：窄而高（例如直立的桿子、門柱）
        * **遠大於 1**：寬而扁（例如橫躺的橫桿、地面線條）

        Example:
            >>> # 只保留形狀接近圓形、且面積夠大的橘色目標
            >>> balls = [i for i in range(api.color_counts[api.ORANGE])
            ...          if api.object_sizes[api.ORANGE][i] > 3000
            ...          and 0.7 < api.object_ratio[api.ORANGE][i] < 1.4]
        """

        # -------------------- 深度統計（形狀比照色模） --------------------
        self.distance_counts: List[int] = [0] * len(self.DISTANCES)
        """
        各距離類別偵測到的物體總數清單。

        距離類別 D1~D8 的範圍在網頁的深度模式中設定，儲存於各 strategy 的
        ``depth.ini``。這 8 個區間與色模的 8 種顏色是**各自獨立**的分類系統，
        只是數量恰好相同。

        * **距離索引對照表**\ ：\ ``api.D1`` ~ ``api.D8``\ （即 0 ~ 7）

        Example:
            >>> # 檢查最近的距離區間內有沒有東西
            >>> if api.distance_counts[api.D1] > 0:
            >>>     print(f"D1 區間偵測到 {api.distance_counts[api.D1]} 個物體")
        """

        self.distance_sizes: List[List[float]] = [[] for _ in self.DISTANCES]
        """各距離類別物體的面積（像素數）清單。用法同 :attr:`object_sizes`。"""

        self.distance_x_min: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體邊界框的最小 X 座標 (左邊界)，範圍 (0 ~ 960)。"""

        self.distance_x_max: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體邊界框的最大 X 座標 (右邊界)，範圍 (0 ~ 960)。"""

        self.distance_y_min: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體邊界框的最小 Y 座標 (上邊界)，範圍 (0 ~ 600)。"""

        self.distance_y_max: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體邊界框的最大 Y 座標 (下邊界)，範圍 (0 ~ 600)。"""

        self.distance_cx: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體邊界框的中心 X 座標，範圍 (0 ~ 960)。"""

        self.distance_cy: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體邊界框的中心 Y 座標，範圍 (0 ~ 600)。"""

        self.distance_ratio: List[List[float]] = [[] for _ in self.DISTANCES]
        """各距離類別物體的寬高比 (寬 / 高)，判讀方式同 :attr:`object_ratio`。"""

        self.distance_cm: List[List[int]] = [[] for _ in self.DISTANCES]
        """
        各距離類別物體的平均距離清單，單位為**公分**。

        取該物體所有有效深度像素的平均值。因為物體本來就被距離區間框住，
        平均值必定落在該區間內。

        Example:
            >>> # 取得 D1 區間中最近的那個物體
            >>> n = api.distance_counts[api.D1]
            >>> if n > 0:
            >>>     j = min(range(n), key=lambda i: api.distance_cm[api.D1][i])
            >>>     print(f"最近的物體在 {api.distance_cm[api.D1][j]} 公分處")
        """

        self.distance_min_cm: List[List[int]] = [[] for _ in self.DISTANCES]
        """
        各距離類別物體**最近端**的距離，單位為公分。

        與 :attr:`distance_max_cm` 一起看可判斷物體的深度延伸範圍。
        正對鏡頭的球兩者會很接近；斜向的牆面則會相差很大。
        """

        self.distance_max_cm: List[List[int]] = [[] for _ in self.DISTANCES]
        """各距離類別物體**最遠端**的距離，單位為公分。"""

        self.new_distance_info: bool = False
        """深度偵測是否有新資料的旗標，用法同 :attr:`new_object_info`。"""

        # -------------------- 疊合統計（形狀比照色模） --------------------
        self.overlap_counts: List[int] = [0] * len(self.SETS)
        """
        各疊合組合偵測到的交集區塊總數清單。

        每一組 Set 綁定一個顏色類別與一個距離類別，交集即「**顏色符合且距離也符合**」
        的區域。組合內容在網頁的疊合模式中設定，儲存於各 strategy 的 ``overlap.ini``。

        * **組合索引對照表**\ ：\ ``api.SET1`` ~ ``api.SET8``\ （即 0 ~ 7）

        .. note::
           僅在色模為 ``All_color``、深度為 ``All_distance``、疊合為 ``All_overlap``
           時更新，且只計算已勾選 Enable 的組合。

        Example:
            >>> # Set1 設定為「紅色 x D1」時，代表 1.5 公尺內的紅色目標
            >>> if api.overlap_counts[api.SET1] > 0:
            >>>     print("找到近距離的紅色目標")
        """

        self.overlap_sizes: List[List[float]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的面積（像素數）清單。"""

        self.overlap_x_min: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的最小 X 座標 (左邊界)，範圍 (0 ~ 960)。"""

        self.overlap_x_max: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的最大 X 座標 (右邊界)，範圍 (0 ~ 960)。"""

        self.overlap_y_min: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的最小 Y 座標 (上邊界)，範圍 (0 ~ 600)。"""

        self.overlap_y_max: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的最大 Y 座標 (下邊界)，範圍 (0 ~ 600)。"""

        self.overlap_cx: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的中心 X 座標，範圍 (0 ~ 960)。"""

        self.overlap_cy: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的中心 Y 座標，範圍 (0 ~ 600)。"""

        self.overlap_ratio: List[List[float]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊的寬高比 (寬 / 高)。"""

        self.overlap_cm: List[List[int]] = [[] for _ in self.SETS]
        """
        各疊合組合交集區塊的平均距離，單位為公分。

        Example:
            >>> # 取得 Set1 中面積最大的目標，並取得其中心與距離
            >>> n = api.overlap_counts[api.SET1]
            >>> if n > 0:
            >>>     j = max(range(n), key=lambda i: api.overlap_sizes[api.SET1][i])
            >>>     cx = api.overlap_cx[api.SET1][j]
            >>>     distance = api.overlap_cm[api.SET1][j]
        """

        self.overlap_min_cm: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊**最近端**的距離，單位為公分。"""

        self.overlap_max_cm: List[List[int]] = [[] for _ in self.SETS]
        """各疊合組合交集區塊**最遠端**的距離，單位為公分。"""

        self.new_overlap_info: bool = False
        """疊合偵測是否有新資料的旗標，用法同 :attr:`new_object_info`。"""

        # -------------------- ZED IMU --------------------
        self.zed_imu_rpy: List[float] = [0.0, 0.0, 0.0]
        """
        ZED 相機內建 IMU 的姿態數值（單位：度），已套用歸零。

            * [0] roll：翻滾角
            * [1] pitch：俯仰角
            * [2] yaw：偏航角

        歸零請呼叫 :meth:`sendZedSensorReset`\ 。

        .. warning::
           這與 :attr:`imu_rpy`\ （機身上的 Arduino IMU）是**兩顆不同的感測器**，
           且 ZED 安裝在頭部雲台上 —— 只要頭部轉動，即使機身沒動 yaw 也會改變。
           兩者的數值不能直接互相比較。

        Example:
            >>> current_roll = api.zed_imu_rpy[0]
        """

        self.zed_imu_abs_rpy: List[float] = [0.0, 0.0, 0.0]
        """
        ZED 內建 IMU \ **未歸零**\ 的絕對姿態（單位：度），索引順序同 :attr:`zed_imu_rpy`\ 。

        roll 與 pitch 以重力為基準，屬於絕對值，可用來判斷相機真實的傾斜程度；
        yaw 沒有絕對基準（無磁力計校正），會隨時間漂移。
        """

        self.zed_gyro: List[float] = [0.0, 0.0, 0.0]
        """
        ZED 內建 IMU 的角速度 [x, y, z]，單位為 \ **deg/s**\ （原生 rad/s 已換算）。

        靜止時三軸都應接近 0，可用於偵測劇烈晃動或跌倒。
        """

        self.zed_accel: List[float] = [0.0, 0.0, 0.0]
        """
        ZED 內建 IMU 的線加速度 [x, y, z]，單位為 **m/s²**，數值含重力。

        靜止且水平放置時 z 軸約為 9.81，可作為感測器是否正常運作的快速檢查。
        """

        self.zed_imu_zeroed: bool = False
        """ZED IMU 是否已設定過零點。開機後尚未呼叫 :meth:`sendZedSensorReset` 前為 False。"""

        # -------------------- ZED 里程計（驗證用，尚未做頭部補償） -----------
        self.odom_xyz: List[float] = [0.0, 0.0, 0.0]
        """
        ZED 視覺里程計的位移 [x, y, z]，單位為**公分**。

        相對於節點啟動時的位置，或最近一次 :meth:`sendOdomReset` 的位置。

        .. warning::
           這是**相機**的位移，而相機安裝在頭部雲台上 —— 只轉頭而機身不動時
           數值同樣會變化。此外人形機器人走路時的晃動會影響視覺里程計的精度，
           場地特徵稀少時也可能漂移。**目前僅供觀察驗證，請勿讓策略依賴此數值。**

        Example:
            >>> api.sendOdomReset()
            >>> # ... 讓機器人前進一段距離 ...
            >>> print(f"前進了約 {api.odom_xyz[0]:.1f} 公分")
        """

        self.odom_yaw: float = 0.0
        """
        ZED 視覺里程計的偏航角，單位為度。

        與 :attr:`odom_xyz` 相同，這是**相機**而非機身的朝向。
        """

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
        同步狀態 (把硬體的 True/False 給 API)
        """
        if msg.strategy != self.is_start:
            self.is_start = msg.strategy
            
            if self.is_start:
                self.get_logger().info("\033[92m[DIO] 物理開關開啟：START\033[0m")
            else:
                self.get_logger().info("\033[91m[DIO] 物理開關關閉：STOP\033[0m")

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

        Note:
            * **可以給小數**：三個步長皆為浮點數，例如 ``theta=2.5``。
              整數傳入也沒問題，函式內部會自行轉型。

        Example:
            >>> # 讓機器人以前進步長 300 穩定行走
            >>> api.sendBodyAutoCmd(x=300, y=0, theta=0, walking_mode=0)
            >>>
            >>> # 讓機器人向左轉彎同時前進（可用小數微調轉向）
            >>> api.sendBodyAutoCmd(x=200, theta=2.5)
            >>>
            >>> # 執行上板步態
            >>> api.sendBodyAutoCmd(x=20000, walking_mode=1)

        """
        m = Interface()
        # float()：訊息欄位是 float64，rclpy 不接受 int，而既有策略大多傳整數
        m.x, m.y, m.theta = float(x), float(y), float(theta)
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
        self.object_cx = [[] for _ in self.COLORS]
        self.object_cy = [[] for _ in self.COLORS]
        self.object_ratio = [[] for _ in self.COLORS]

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
                cx, cy = o.get('centroid', (x + w // 2, y + h // 2))
                self.object_cx[idx].append(int(cx))
                self.object_cy[idx].append(int(cy))
                self.object_ratio[idx].append(
                    float(o.get('aspect_ratio', (w / h) if h else 0.0)))

    # -------------------- 深度 / 疊合 --------------------
    def _fill_stats(self, labels, latest, counts, sizes, x_min, x_max, y_min, y_max,
                    cx_l, cy_l, ratio_l, cm_l, cm_min_l, cm_max_l, tag):
        """把 detections JSON 攤平成與色模相同形狀的平行清單。

        深度與疊合的 JSON 欄位與色模完全相同，只是多了 distance_* 三個欄位，
        所以這裡用同一套邏輯處理，避免三份幾乎一樣的程式碼。
        """
        for lst in (sizes, x_min, x_max, y_min, y_max, cx_l, cy_l, ratio_l,
                    cm_l, cm_min_l, cm_max_l):
            for i in range(len(labels)):
                lst[i] = []

        for idx, name in enumerate(labels):
            objs = latest.get(name, [])
            counts[idx] = len(objs)
            for o in objs:
                try:
                    x, y, w, h = o['bbox']
                    area = float(o.get('area', float(w * h)))
                except Exception as ex:
                    self.get_logger().warn(f"[{tag}] Malformed object for {name}: {ex}")
                    continue
                sizes[idx].append(area)
                x_min[idx].append(int(x))
                x_max[idx].append(int(x + w))
                y_min[idx].append(int(y))
                y_max[idx].append(int(y + h))
                cx, cy = o.get('centroid', (x + w // 2, y + h // 2))
                cx_l[idx].append(int(cx))
                cy_l[idx].append(int(cy))
                ratio_l[idx].append(float(o.get('aspect_ratio', (w / h) if h else 0.0)))
                # 色模沒有距離欄位，缺的時候補 -1 以維持與其他清單等長
                cm_l[idx].append(int(o.get('distance_cm', -1)))
                cm_min_l[idx].append(int(o.get('distance_min_cm', -1)))
                cm_max_l[idx].append(int(o.get('distance_max_cm', -1)))

    def _recompute_distance_stats(self) -> None:
        """重算 D1~D8 的統計，欄位形狀與色模一致。"""
        self._fill_stats(
            self.DISTANCES, self.latest_depth_objects, self.distance_counts,
            self.distance_sizes, self.distance_x_min, self.distance_x_max,
            self.distance_y_min, self.distance_y_max, self.distance_cx,
            self.distance_cy, self.distance_ratio, self.distance_cm,
            self.distance_min_cm, self.distance_max_cm, "depth")

    def _recompute_overlap_stats(self) -> None:
        """重算 Set1~Set8 的統計，欄位形狀與色模一致。"""
        self._fill_stats(
            self.SETS, self.latest_overlap_objects, self.overlap_counts,
            self.overlap_sizes, self.overlap_x_min, self.overlap_x_max,
            self.overlap_y_min, self.overlap_y_max, self.overlap_cx,
            self.overlap_cy, self.overlap_ratio, self.overlap_cm,
            self.overlap_min_cm, self.overlap_max_cm, "overlap")

    def _json_det_callback(self, msg, label, latest, stamps, recompute):
        """detections JSON 的共用解析（色模以外的兩套共用）。"""
        try:
            data = json.loads(msg.data)
            st = data.get('stamp', {})
            cur = (int(st.get('sec', 0)), int(st.get('nanosec', 0)))
            if not self._is_newer_stamp(cur, stamps.get(label, (0, 0))):
                return False
            objs = data.get('objects', [])
            latest[label] = objs if isinstance(objs, list) else []
            stamps[label] = cur
            recompute()
            return True
        except Exception as e:
            self.get_logger().error(f'[det] {label} parse error: {e}')
            return False

    def _depth_det_callback(self, msg: String, label: str) -> None:
        """/depth_detections/{D1~D8} 的回呼。"""
        if self._json_det_callback(msg, label, self.latest_depth_objects,
                                   self.latest_depth_stamps,
                                   self._recompute_distance_stats):
            self.new_distance_info = True

    def _overlap_det_callback(self, msg: String, label: str) -> None:
        """/overlap_detections/{Set1~Set8} 的回呼。"""
        if self._json_det_callback(msg, label, self.latest_overlap_objects,
                                   self.latest_overlap_stamps,
                                   self._recompute_overlap_stats):
            self.new_overlap_info = True

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

    def setZoomIn(self, value: float = 1.0) -> None:
        """
        設定影像的裁切放大倍率。

        以畫面中心為基準裁切出 1/value 大小的區域，再放大回原本的解析度，
        效果等同於數位變焦。色模、深度、疊合三套系統會同時跟著改，
        所以放大後三邊的座標仍然彼此對齊。

        Args:
            value (float, optional): 放大倍率，範圍 1.0 ~ 5.0。預設為 1.0。
                1.0 表示不裁切（完整視野），數值越大視野越窄、目標越大。
                超出範圍會自動夾到邊界。

        Returns:
            None

        Note:
            * **與網頁共用**：這裡送出的值等同於在影像處理介面拉 ZoomIn 滑桿，
              網頁上的數字不會自動跟著跳，但機器人實際吃到的是最後送出的那個。
            * **不會寫進 ini**：只改當下的執行狀態，重開節點後會回到 ini 裡的設定值。
            * **視野會變窄**：放大後畫面邊緣的物體會直接消失在視野外，
              追蹤目標時建議搭配頭部轉動使用。

        Example:
            >>> # 遠處的球太小，放大兩倍看清楚
            >>> api.setZoomIn(2.0)
            >>>
            >>> # 找不到目標時退回完整視野
            >>> api.setZoomIn(1.0)
        """
        n = Zoom()
        # float()：欄位是 float32，rclpy 不收 int；夾範圍是因為 image.py 直接
        # 拿 width / zoom，傳 0 會讓它每一幀都噴 ZeroDivisionError
        n.zoomin = self._clamp(float(value), 1.0, 5.0)
        self.zoomin_pub.publish(n)

    # -------------------- 深度值查詢 --------------------
    def _depth_mm_cb(self, msg: RosImage) -> None:
        """快取 /depth_mm（裁切對齊後的公釐深度圖），供 depth_at() 使用。"""
        try:
            self._depth_mm = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            self.get_logger().error(f"[depth_mm] convert failed: {e}")

    def depth_at(self, x: int, y: int, radius: int = 5) -> Optional[float]:
        """查詢影像上任一點的深度，單位公分。

        預設取該點周圍 (2*radius+1) 見方鄰域的**中位數**而非單一像素值。
        ZED 深度圖有不少單像素的洞與雜訊，直接取單點很容易剛好打在無效值上，
        中位數對這兩者都免疫。

        Args:
            x (int): 影像 X 座標，範圍 0~960（與 object_x_min 等同一座標系）。
            y (int): 影像 Y 座標，範圍 0~600。
            radius (int): 取樣半徑。0 表示只取該點；預設 5 即 11x11 鄰域。

        Returns:
            float | None: 距離（公分）。座標超出範圍、尚未收到深度圖、
            或鄰域內全是無效值時回傳 None。

        Example:
            >>> # 查一個色模物體中心的距離
            >>> cx = api.object_cx[api.RED][0]
            >>> cy = api.object_cy[api.RED][0]
            >>> d = api.depth_at(cx, cy)
            >>> if d is not None and d < 80:
            >>>     print("紅色目標在 80 公分內")
        """
        img = self._depth_mm
        if img is None:
            return None
        h, w = img.shape[:2]
        x, y = int(x), int(y)
        if not (0 <= x < w and 0 <= y < h):
            return None

        r = max(0, int(radius))
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        roi = img[y0:y1, x0:x1]

        valid = roi[roi > 0]          # 0 代表無效（NaN/超出範圍已在上游歸零）
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 10.0     # 公釐 -> 公分

    # -------------------- ZED IMU --------------------
    def _zed_imu_cb(self, msg: ZedImu) -> None:
        """/zed_imu/data 的回呼，來源是 zed_imu_node。"""
        self.zed_imu_rpy = [msg.roll, msg.pitch, msg.yaw]
        self.zed_imu_abs_rpy = [msg.abs_roll, msg.abs_pitch, msg.abs_yaw]
        self.zed_gyro = list(msg.angular_velocity)
        self.zed_accel = list(msg.linear_acceleration)
        self.zed_imu_zeroed = bool(msg.zeroed)

    def sendZedSensorReset(self) -> None:
        """將 ZED IMU 目前的姿態設為零點。

        與 sendSensorReset()（Arduino IMU）互不影響，兩顆感測器各自歸零。
        歸零後 zed_imu_rpy 會變成 0，zed_imu_abs_rpy 則維持絕對值不變。
        """
        msg = SensorSet()
        msg.reset = True
        self.zed_imu_reset_pub.publish(msg)

    # -------------------- ZED 里程計 --------------------
    def _odom_cb(self, msg: Odometry) -> None:
        """/zed/zed_node/odom 的回呼，把位移換算成公分、朝向換算成度。"""
        p = msg.pose.pose.position
        self.odom_xyz = [p.x * 100.0, p.y * 100.0, p.z * 100.0]

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.degrees(math.atan2(siny, cosy))

    def sendOdomReset(self) -> None:
        """把 ZED 視覺里程計歸零（呼叫 /zed/zed_node/reset_odometry）。

        非阻塞：服務尚未就緒時只記錄警告，不會卡住策略主迴圈。
        """
        if not self.odom_reset_cli.service_is_ready():
            self.get_logger().warn("[odom] reset_odometry 服務尚未就緒")
            return
        self.odom_reset_cli.call_async(Trigger.Request())

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
        rst.reset = bool(status)
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
        m.data = int(generate)
        self.generate_pub.publish(m)

    def sendContinuousValue(self, x: float, y: float, theta: float, walking_mode: int = 0) -> None:
        """
        發送連續移動步長數值。

        將 X、Y 與Theta發布至Walking。此函式通常配合已啟動的步態使用，

        與 **[sendBodyAutoCmd]** 不同，此函式不會發送啟動步態訊號，僅更新底層緩存的步長數值。

        Args:
            x (float): 前後移動步長。
            y (float): 左右移動步長。
            theta (float): 旋轉角度。
            walking_mode (int, optional): 步態模式編號。預設為 0。

        Returns:
            None

        Note:
            * **依賴性**：呼叫此函式前，通常需要先透過 **[sendbodyAuto(1)]** 啟動步態，機器人才會根據這些參數開始移動。
            * **可以給小數**：三個步長皆為浮點數，整數傳入也沒問題，函式內部會自行轉型。

        Example:
            >>> # 1.先啟動步態引擎 (原地踏步)
            >>> api.sendbodyAuto(1)
            >>> # 2. 給予移動步長（可用小數微調）
            >>> api.sendContinuousValue(x=300, y=-100, theta=2.5)
        """
        m = Interface()
        # float()：訊息欄位是 float64，rclpy 不接受 int
        m.x, m.y, m.theta = float(x), float(y), float(theta)
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
        m.data = int(sector)
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
        m.id, m.position, m.speed = int(ID), int(Position), int(Speed)
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
        m.id, m.position, m.speed = int(ID), int(Position), int(Speed)
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
        m.id, m.position, m.speed = int(ID), int(Position), int(Speed)
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

    def clearDrawImage(self) -> None:
        """清除畫面上所有由 :meth:`drawImageFunction` 產生的圖形。

        影像節點是以 ``cnt`` 為鍵累積繪圖資料的，策略結束後圖形會一直留在畫面上。
        建議在策略停止時呼叫一次，以免殘留的圖形干擾下一個任務的判讀 ——
        切換 strategy **不會**自動清除，圖形會一直保留到有人明確清掉為止。

        網頁的 Clear Draw 按鈕走的是同一個介面。

        Example:
            >>> # 在策略結束前清乾淨
            >>> api.clearDrawImage()
        """
        msg = Bool()
        msg.data = True
        self.draw_clear_pub.publish(msg)

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
        # 這幾個是 float64 欄位，rclpy 不收 int。策略很自然會寫 com_y_swing=0，
        # 沒有這層轉型會直接 AssertionError
        msg.mode = int(mode)
        msg.com_y_swing = float(com_y_swing)
        msg.width_size = float(width_size)
        msg.period_t = int(period_t)
        msg.t_dsp = float(t_dsp)
        msg.lift_height = float(lift_height)
        msg.stand_height = float(stand_height)
        msg.com_height = float(com_height)
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