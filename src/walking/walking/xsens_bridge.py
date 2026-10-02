#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xsens MTi-620 橋接：/filter/* 與 /imu/* -> tku_msgs/SensorPackage + tku_msgs/XsensImu。

角色與 imu_node.py（Arduino IMU）完全相同 —— 都是 /package/sensorpackage 的來源，
所以 API.py、walking_node.py 與所有子策略**一行都不用改**就能換成新 IMU。
差別只在資料從哪裡來：

  - Arduino IMU 自己開 /dev/ttyTHS1 讀序列埠並解析
  - Xsens 由官方的 xsens_mti_ros2_driver 負責讀取，本節點只做座標慣例轉換、
    歸零與訊息轉發

同時另外發一份 tku_msgs/XsensImu 到 /xsens_imu/data，帶著 SensorPackage 塞不下的
資料（絕對角、角速度、free_acceleration、頻率、watchdog），給平衡控制與網頁
ImuMonitor 用。SensorPackage 那條維持最小介面，不動既有欄位語意。

歸零（/sensorset，與 API.sendSensorReset() 同一條）在**感測器原始座標系**用四元數做：
q_rel = q0⁻¹ ⊗ q_now。/filter/euler 只給歐拉角，但歐拉角重建回四元數是無損的，
所以不必為了歸零多開 /filter/quaternion 那條佔頻寬的輸出。
座標慣例（正負號 / 軸對調 / yaw 安裝偏移）在算完之後才套用到歐拉角上，
不會污染四元數運算。

注意：本節點與 imu_node 都發 /package/sensorpackage，**兩個不能同時跑**。
"""
import math

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from tku_msgs.msg import SensorPackage, SensorSet, XsensImu

from .imu_math import (
    euler_deg_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_to_euler_deg,
    wrap_deg,
)


class XsensBridgeNode(Node):
    def __init__(self):
        super().__init__('xsens_bridge')

        # ---- 來源 topic：driver 若改了 namespace 從這裡調，不用改程式 ----
        self.declare_parameter('euler_topic', '/filter/euler')
        self.declare_parameter('free_accel_topic', '/filter/free_acceleration')
        self.declare_parameter('gyro_topic', '/imu/angular_velocity')
        self.declare_parameter('reset_topic', '/sensorset')

        # ---- 發布 ----
        # 50Hz 對齊 walking_node 的步態迴圈；馬達端只有 20Hz，再高沒有意義
        self.declare_parameter('pub_hz', 50.0)
        self.declare_parameter('watchdog_sec', 1.0)

        # ---- 座標對位：IMU 貼上軀幹不可能完美水平，這裡做慣例轉換 ----
        # 優先用裝置端的 Alignment Reset 修正，這幾個參數是補剩下的軸向差異
        self.declare_parameter('swap_roll_pitch', False)
        self.declare_parameter('sign_roll', 1.0)
        self.declare_parameter('sign_pitch', 1.0)
        self.declare_parameter('sign_yaw', 1.0)
        self.declare_parameter('yaw_mount_offset', 0.0)

        euler_topic = str(self.get_parameter('euler_topic').value)
        free_accel_topic = str(self.get_parameter('free_accel_topic').value)
        gyro_topic = str(self.get_parameter('gyro_topic').value)
        reset_topic = str(self.get_parameter('reset_topic').value)

        pub_hz = float(self.get_parameter('pub_hz').value or 50.0)
        self.watchdog_sec = float(self.get_parameter('watchdog_sec').value or 1.0)

        self.swap_roll_pitch = bool(self.get_parameter('swap_roll_pitch').value)
        self.sign_roll = float(self.get_parameter('sign_roll').value or 1.0)
        self.sign_pitch = float(self.get_parameter('sign_pitch').value or 1.0)
        self.sign_yaw = float(self.get_parameter('sign_yaw').value or 1.0)
        self.yaw_mount_offset = float(self.get_parameter('yaw_mount_offset').value or 0.0)

        # ---- 狀態 ----
        self._euler = None        # 最新一筆原始歐拉角 (roll, pitch, yaw)，單位度
        self._gyro = [0.0, 0.0, 0.0]        # deg/s
        self._free_accel = [0.0, 0.0, 0.0]  # m/s^2
        self._q_zero = None       # 零點四元數，None 表示尚未歸零

        self._last_euler_ns = None   # 上一筆 euler 的到達時刻，watchdog 與頻率都靠它
        self._dt_ema = 0.0           # 平滑後的到達週期，秒
        self._rate_hz = 0.0
        self._warned_no_data = False
        self._euler_topic = euler_topic
        # 從沒收到過資料時 watchdog 從這裡起算，driver 根本沒起來才看得出來
        self._start_ns = self.get_clock().now().nanoseconds

        # 感測器資料只取最新一筆，堆積沒有意義。訂閱端用 BEST_EFFORT：
        # driver 不論發 RELIABLE 或 BEST_EFFORT 都收得到
        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 對外維持與 imu_node 相同的 QoS，walking_node / API 的預設訂閱才配得上
        qos_pkg = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Vector3Stamped, euler_topic, self.on_euler, qos_sensor)
        self.create_subscription(Vector3Stamped, free_accel_topic, self.on_free_accel, qos_sensor)
        self.create_subscription(Vector3Stamped, gyro_topic, self.on_gyro, qos_sensor)
        self.create_subscription(SensorSet, reset_topic, self.on_sensor_set, qos_pkg)

        self.pub_pkg = self.create_publisher(SensorPackage, '/package/sensorpackage', qos_pkg)
        self.pub_imu = self.create_publisher(XsensImu, '/xsens_imu/data', qos_pkg)

        # 預建訊息，避免每個週期重新配置
        self.msg_pkg = SensorPackage()
        self.msg_imu = XsensImu()

        self.timer = self.create_timer(1.0 / max(pub_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f'[Xsens] 訂閱 {euler_topic}，以 {pub_hz:.0f}Hz 發布 '
            f'/package/sensorpackage 與 /xsens_imu/data')

    # ------------------------------------------------------------------
    # 訂閱回呼
    # ------------------------------------------------------------------
    def on_euler(self, msg: Vector3Stamped):
        """/filter/euler：Xsens 給的就是「度」，不需要換算。"""
        self._euler = (msg.vector.x, msg.vector.y, msg.vector.z)

        now_ns = self.get_clock().now().nanoseconds
        if self._last_euler_ns is not None:
            dt = (now_ns - self._last_euler_ns) * 1e-9
            # 平滑「週期」再取倒數，不能平滑「瞬時頻率」——
            # 資料成批抵達時（USB 轉接的 FTDI latency timer、或 driver 排程延遲），批內 dt 接近 0，
            # 1/dt 會衝到天上去，平均起來報出遠高於實際的頻率
            if 0.0 < dt < 10.0:
                self._dt_ema = dt if self._dt_ema <= 0.0 else \
                    self._dt_ema * 0.9 + dt * 0.1
                self._rate_hz = 1.0 / self._dt_ema
        self._last_euler_ns = now_ns

        if self._warned_no_data:
            self._warned_no_data = False
            self.get_logger().info('[Xsens] 資料恢復')

    def on_gyro(self, msg: Vector3Stamped):
        """/imu/angular_velocity：原生 rad/s，轉成 deg/s 與角度單位一致。"""
        self._gyro = [
            math.degrees(msg.vector.x),
            math.degrees(msg.vector.y),
            math.degrees(msg.vector.z),
        ]

    def on_free_accel(self, msg: Vector3Stamped):
        """/filter/free_acceleration：已扣除重力，單位 m/s^2。"""
        self._free_accel = [msg.vector.x, msg.vector.y, msg.vector.z]

    def on_sensor_set(self, msg: SensorSet):
        """把目前姿態設為零點；與 API.sendSensorReset(True) 是同一條路。"""
        if not bool(getattr(msg, 'reset', False)):
            return
        if self._euler is None:
            self.get_logger().warn('[Xsens] 尚未收到 IMU 資料，無法歸零')
            return
        self._q_zero = euler_deg_to_quat(*self._euler)
        r, p, y = self._euler
        self.get_logger().info(
            f'[Xsens] 已歸零，零點原始姿態 roll={r:.2f} pitch={p:.2f} yaw={y:.2f}')

    # ------------------------------------------------------------------
    # 座標慣例
    # ------------------------------------------------------------------
    def _apply_mount(self, roll, pitch, yaw):
        """把感測器座標系的歐拉角轉成機器人慣例：先軸對調，再套正負號。

        正負號設錯會讓平衡補償往「加深傾倒」的方向作用，第一次測試務必有人扶著。
        驗證方式：機器人立正 -> 手動往前傾（pitch 應為正）、往右傾（roll 應為正）。
        """
        if self.swap_roll_pitch:
            roll, pitch = pitch, roll
        return (
            self.sign_roll * roll,
            self.sign_pitch * pitch,
            self.sign_yaw * yaw,
        )

    # ------------------------------------------------------------------
    def on_timer(self):
        if self._euler is None:
            self._check_alive()
            return

        alive = self._check_alive()

        mount_roll, mount_pitch, mount_yaw = self._apply_mount(*self._euler)

        # 絕對姿態。yaw 的安裝偏移只加在絕對值上 —— 相對 yaw 已經以零點為基準，
        # 再加一次偏移沒有意義
        abs_roll, abs_pitch = mount_roll, mount_pitch
        abs_yaw = wrap_deg(mount_yaw + self.yaw_mount_offset)

        # 相對姿態：在原始座標系用四元數扣掉零點，算完才套座標慣例
        if self._q_zero is None:
            roll, pitch, yaw = mount_roll, mount_pitch, mount_yaw
        else:
            q_now = euler_deg_to_quat(*self._euler)
            rel = quat_to_euler_deg(quat_multiply(quat_conjugate(self._q_zero), q_now))
            roll, pitch, yaw = self._apply_mount(*rel)

        # --- 相容介面：欄位與 imu_node 發的完全一樣 ---
        # SensorPackage 的 roll/pitch/yaw 是 float64，直接給浮點數即可
        pkg = self.msg_pkg
        pkg.roll = float(roll)
        pkg.pitch = float(pitch)
        pkg.yaw = float(yaw)
        self.pub_pkg.publish(pkg)

        # --- 完整介面 ---
        m = self.msg_imu
        m.roll, m.pitch, m.yaw = float(roll), float(pitch), float(yaw)
        m.abs_roll, m.abs_pitch, m.abs_yaw = float(abs_roll), float(abs_pitch), float(abs_yaw)
        m.angular_velocity = [float(v) for v in self._gyro]
        m.free_acceleration = [float(v) for v in self._free_accel]
        m.zeroed = self._q_zero is not None
        m.rate_hz = float(self._rate_hz)
        m.alive = alive
        self.pub_imu.publish(m)

    def _check_alive(self) -> bool:
        """走路時的震動最容易把細排線震鬆，斷線要能立刻看出來而不是靜靜卡住舊值。"""
        never_received = self._last_euler_ns is None
        last_ns = self._start_ns if never_received else self._last_euler_ns
        age = (self.get_clock().now().nanoseconds - last_ns) * 1e-9
        if age > self.watchdog_sec:
            if not self._warned_no_data:
                self._warned_no_data = True
                if never_received:
                    # 最常見的原因是 launch 找不到 xsens_mti_ros2_driver（沒 source ~/ros2_ws），
                    # driver 被略過，這裡是唯一會出聲的地方
                    self.get_logger().warn(
                        f'[Xsens] 啟動 {age:.1f}s 仍未收到 {self._euler_topic}，'
                        f'用 ros2 node list 確認 xsens_mti_node 有沒有起來')
                else:
                    self.get_logger().warn(
                        f'[Xsens] 已 {age:.1f}s 沒有收到 IMU 資料，'
                        f'檢查 driver 是否還活著、排線是否鬆脫')
            return False
        return not never_received


def main():
    rclpy.init()
    node = XsensBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
