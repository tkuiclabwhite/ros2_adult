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
    """Ensure only ONE strategy process runs at a time (ros2 run)."""
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

        self.generate_pub = self.create_publisher(Int16, '/ContinousMode_Topic', 10)
        self.continous_pub = self.create_publisher(Interface, '/ChangeContinuousValue_Topic', 10)
        self.body_auto_pub = self.create_publisher(Interface, '/SendBodyAuto_Topic', 10)

        self.head_motor_pub = self.create_publisher(HeadPackage, '/Head_Topic', 10)
        self.sector_pub = self.create_publisher(Int16, '/package/Sector', 10)

        self.draw_image_pub = self.create_publisher(DrawImage, '/drawimage', 10)
        self.walkparameter_pub = self.create_publisher(Parameter, '/strategy/walkparameter', 10)

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

        self.is_start: bool = False
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
        self.imu_rpy = [self.roll, self.pitch, self.yaw]
        self.pose_x = self.pose_y = self.pose_z = 0.0
        self.pose = [self.pose_x, self.pose_y, self.pose_z]
        self.xx, self.yy, self.tt = 0, 0, 0

        self.color_counts: List[int] = [0] * len(self.COLORS)
        self.object_sizes: List[List[float]] = [[] for _ in self.COLORS]
        self.object_x_min: List[List[int]] = [[] for _ in self.COLORS]
        self.object_x_max: List[List[int]] = [[] for _ in self.COLORS]
        self.object_y_min: List[List[int]] = [[] for _ in self.COLORS]
        self.object_y_max: List[List[int]] = [[] for _ in self.COLORS]
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
        }

        self._publish_strategy_status("idle")

    def _dio_callback(self, msg: Dio):
        """
        純粹同步硬體開關狀態。
        當 strategy 為 True 時，解開所有策略的迴圈限制。
        """
        # 同步狀態 (把硬體的 True/False 給 API)
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
        msg = String()
        msg.data = f"{state}; selected={self._selected_strategy}; running={self._mgr.current_name()}"
        self.strategy_status_pub.publish(msg)

    def _on_strategy_name(self, msg: String):
        name = msg.data.strip()
        if not name:
            return
        self._selected_strategy = name
        self.get_logger().info(f"[strategy] selected = {name}")
        self._publish_strategy_status("selected")

    def _on_strategy_start(self, msg: Bool):
        if msg.data:
            self._start_selected_strategy()
        else:
            self._stop_strategy()

    def _start_selected_strategy(self):
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
        try:
            self._mgr.stop()
            self.get_logger().info("[strategy] stopped")
            self._publish_strategy_status("stopped")
        except Exception as e:
            self.get_logger().error(f"[strategy] stop failed: {e}")
            self._publish_strategy_status("error")

    # -------------------- existing helpers / callbacks --------------------
    def sendBodyAutoCmd(self, x: float, y: float, theta: float) -> None:
        m = Interface()
        m.x, m.y, m.theta = x, y, theta
        self.body_auto_pub.publish(m)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    def set_head(self, pan: float, tilt: float, speed: Optional[int] = None) -> None:
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
        return (a[0] > b[0]) or (a[0] == b[0] and a[1] > b[1])

    def _recompute_stats(self) -> None:
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
        self.roll = msg.roll
        self.pitch = msg.pitch
        self.yaw = msg.yaw
        self.imu_rpy = [self.roll, self.pitch, self.yaw]

    def Yolo_Zed(self, msg: Point) -> None:
        self.pose_x = msg.x
        self.pose_y = msg.y
        self.pose_z = msg.z
        self.pose = [self.pose_x, self.pose_y, self.pose_z]

    def sendSensorReset(self, status: bool) -> None:
        rst = SensorSet()
        rst.reset = status
        self.imu_reset_pub.publish(rst)

    def sendbodyAuto(self, generate: int) -> None:
        m = Int16()
        m.data = generate
        self.generate_pub.publish(m)

    def sendContinuousValue(self, x: int, y: int, theta: int) -> None:
        m = Interface()
        m.x, m.y, m.theta = x, y, theta
        self.continous_pub.publish(m)

    def sendBodySector(self, sector: int) -> None:
        m = Int16()
        m.data = sector
        self.sector_pub.publish(m)

    def sendSingleMotor(self, ID: int, Position: int, Speed: int) -> None:
        m = SingleMotorData()
        m.id, m.position, m.speed = ID, Position, Speed
        self.singlemotor_pub.publish(m)

    def sendHeadMotor(self, ID: int, Position: int, Speed: int) -> None:
        m = HeadPackage()
        m.id, m.position, m.speed = ID, Position, Speed
        self.head_motor_pub.publish(m)

    def drawImageFunction(self, cnt: int, mode: int,
                          xmin: int, xmax: int, ymin: int, ymax: int,
                          r: int, g: int, b: int,thickness: int=1) -> None:
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

    def get_objects(self, color: Optional[str] = None) -> List[dict]:
        if color is None:
            all_objs: List[dict] = []
            for c in self.COLORS:
                all_objs.extend(self.latest_objects.get(c, []))
            return all_objs
        return self.latest_objects.get(color, [])

    def ContinuousValueFunction(self, msg: Interface) -> None:
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