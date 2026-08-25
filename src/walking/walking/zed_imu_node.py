#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZED 內建 IMU 橋接：sensor_msgs/Imu -> tku_msgs/ZedImu。

與 imu_node.py（Arduino IMU）的角色相同，差別在資料來源與歸零方式：

  - ZED 發布的是四元數姿態，需轉成歐拉角才能與現有介面對齊
  - 歸零用四元數做：q_rel = q0⁻¹ ⊗ q_now，而非歐拉角相減。
    歐拉角相減在大角度時會出錯（旋轉不可交換、萬向鎖），
    imu.py 的作法只是在小角度下剛好夠用
  - ZED SDK 沒有提供 IMU 歸零的呼叫（reset_odometry / set_pose 動的是位置追蹤的
    pose，不影響 /imu/data 的 orientation），所以零點只能在這裡自己維護

ZED 以 100Hz 發布，本節點用 timer 降到 pub_hz（預設 20，對齊 Arduino IMU）再發出。
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

from tku_msgs.msg import SensorSet, ZedImu


def quat_conjugate(q):
    """(x, y, z, w) 的共軛；單位四元數的共軛即為其逆。"""
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_multiply(a, b):
    """四元數乘法 a ⊗ b，格式皆為 (x, y, z, w)。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_to_euler_deg(q):
    """(x, y, z, w) -> (roll, pitch, yaw) 角度，ZYX 內旋順序。"""
    x, y, z, w = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 接近 ±90° 時 asin 的定義域會超出，夾住避免例外（萬向鎖）
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class ZedImuNode(Node):
    def __init__(self):
        super().__init__('zed_imu_node')

        self.declare_parameter('imu_topic', '/zed/zed_node/imu/data')
        self.declare_parameter('pub_hz', 20.0)
        imu_topic = str(self.get_parameter('imu_topic').value)
        pub_hz = float(self.get_parameter('pub_hz').value or 20.0)

        self._latest = None      # 最新一筆 sensor_msgs/Imu
        self._q_zero = None      # 零點四元數，None 表示尚未歸零

        # 感測器資料只取最新一筆，堆積沒有意義
        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_pkg = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_imu = self.create_subscription(
            Imu, imu_topic, self.on_imu, qos_sensor)
        # 沿用 Arduino IMU 那顆按鈕的訊息型別，不另外新增介面
        self.sub_reset = self.create_subscription(
            SensorSet, '/zed_sensorset', self.on_sensor_set, qos_pkg)
        self.pub = self.create_publisher(ZedImu, '/zed_imu/data', qos_pkg)

        self.msg = ZedImu()
        self.timer = self.create_timer(1.0 / max(pub_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f'[ZedImu] 訂閱 {imu_topic}，以 {pub_hz:.0f}Hz 發布 /zed_imu/data')

    # ------------------------------------------------------------------
    def on_imu(self, msg: Imu):
        self._latest = msg

    def on_sensor_set(self, msg: SensorSet):
        if not bool(getattr(msg, 'reset', False)):
            return
        if self._latest is None:
            self.get_logger().warn('[ZedImu] 尚未收到 IMU 資料，無法歸零')
            return
        o = self._latest.orientation
        self._q_zero = (o.x, o.y, o.z, o.w)
        r, p, y = quat_to_euler_deg(self._q_zero)
        self.get_logger().info(
            f'[ZedImu] 已歸零，零點絕對姿態 roll={r:.2f} pitch={p:.2f} yaw={y:.2f}')

    # ------------------------------------------------------------------
    def on_timer(self):
        src = self._latest
        if src is None:
            return

        o = src.orientation
        q_now = (o.x, o.y, o.z, o.w)

        abs_roll, abs_pitch, abs_yaw = quat_to_euler_deg(q_now)
        if self._q_zero is None:
            roll, pitch, yaw = abs_roll, abs_pitch, abs_yaw
        else:
            roll, pitch, yaw = quat_to_euler_deg(
                quat_multiply(quat_conjugate(self._q_zero), q_now))

        m = self.msg
        m.roll, m.pitch, m.yaw = roll, pitch, yaw
        m.abs_roll, m.abs_pitch, m.abs_yaw = abs_roll, abs_pitch, abs_yaw

        av = src.angular_velocity
        m.angular_velocity = [
            math.degrees(av.x), math.degrees(av.y), math.degrees(av.z)]

        la = src.linear_acceleration
        m.linear_acceleration = [la.x, la.y, la.z]

        m.zeroed = self._q_zero is not None
        self.pub.publish(m)


def main():
    rclpy.init()
    node = ZedImuNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
