#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from .imu import IMUService

# 依你專案的訊息型別匯入
from tku_msgs.msg import SensorPackage, SensorSet


def clamp_float(x, ndigits=2):
    try:
        return float(round(float(x), ndigits))
    except Exception:
        return 0.0


class IMUBridgeNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        # ---- 參數 ----
        self.declare_parameter('port', '/dev/ttyTHS1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('pub_hz', 20.0)          # 建議先跟 Arduino DMP rate 一致（你那邊是 20）
        self.declare_parameter('debug_raw', False)      # true 時會印出解析不到的原始行
        self.declare_parameter('open_wait_sec', 2.0)    # 打開串口後等 Arduino reset 的秒數

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = int(self.get_parameter('baud').get_parameter_value().integer_value or 115200)
        pub_hz = float(self.get_parameter('pub_hz').get_parameter_value().double_value or 20.0)
        debug_raw = bool(self.get_parameter('debug_raw').get_parameter_value().bool_value)
        open_wait_sec = float(self.get_parameter('open_wait_sec').get_parameter_value().double_value or 2.0)

        # ---- 啟動 IMU 執行緒（相對模式）----
        self.imu = IMUService(
            port=port,
            baud=baud,
            rel_mode=True,
            debug_raw=debug_raw,
            open_wait_sec=open_wait_sec
        )
        self.imu.start()
        self.get_logger().info(f'[IMUBridge] IMU thread started on {port} @ {baud}')

        # ---- QoS ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Pub/Sub ----
        self.pub_pkg = self.create_publisher(SensorPackage, '/package/sensorpackage', qos)
        self.sub_reset = self.create_subscription(SensorSet, '/sensorset', self.on_sensor_set, qos)

        # 預建 message，避免多次配置
        self.msg_pkg = SensorPackage()

        # 若訊息含力感測陣列，先填 0（依 tku_msgs 定義調整命名）
        try:
            if hasattr(self.msg_pkg, 'force_sensor_data') and isinstance(self.msg_pkg.force_sensor_data, list):
                self.msg_pkg.force_sensor_data = [0] * 8
            elif hasattr(self.msg_pkg, 'ForceSensorData') and isinstance(self.msg_pkg.ForceSensorData, list):
                self.msg_pkg.ForceSensorData = [0] * 8
        except Exception:
            pass

        period = 1.0 / max(pub_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self._sent_once = False

    # ---- Callbacks ----
    def on_sensor_set(self, msg: SensorSet):
        try:
            flag = False
            if hasattr(msg, 'reset'):
                flag = bool(msg.reset)
            elif hasattr(msg, 'Reset'):
                flag = bool(msg.Reset)
            if flag:
                self.imu.zero_here()
                self.get_logger().info('[IMUBridge] Zeroed by /sensorset')
        except Exception as e:
            self.get_logger().warn(f'[IMUBridge] on_sensor_set error: {e}')

    def on_timer(self):
        ypr = self.imu.latest()
        if ypr is None:
            return

        y, p, r = ypr  # deg

        # 填值（兼容欄位命名 yaw/Yaw 等）
        try:
            if hasattr(self.msg_pkg, 'yaw'):
                self.msg_pkg.yaw = clamp_float(y)
            elif hasattr(self.msg_pkg, 'Yaw'):
                self.msg_pkg.Yaw = clamp_float(y)

            if hasattr(self.msg_pkg, 'pitch'):
                self.msg_pkg.pitch = clamp_float(p)
            elif hasattr(self.msg_pkg, 'Pitch'):
                self.msg_pkg.Pitch = clamp_float(p)

            if hasattr(self.msg_pkg, 'roll'):
                self.msg_pkg.roll = clamp_float(r)
            elif hasattr(self.msg_pkg, 'Roll'):
                self.msg_pkg.Roll = clamp_float(r)
        except Exception:
            if hasattr(self.msg_pkg, 'yaw'):
                self.msg_pkg.yaw = float(y)
            elif hasattr(self.msg_pkg, 'Yaw'):
                self.msg_pkg.Yaw = float(y)

            if hasattr(self.msg_pkg, 'pitch'):
                self.msg_pkg.pitch = float(p)
            elif hasattr(self.msg_pkg, 'Pitch'):
                self.msg_pkg.Pitch = float(p)

            if hasattr(self.msg_pkg, 'roll'):
                self.msg_pkg.roll = float(r)
            elif hasattr(self.msg_pkg, 'Roll'):
                self.msg_pkg.Roll = float(r)

        self.pub_pkg.publish(self.msg_pkg)

        if not self._sent_once:
            self._sent_once = True
            # self.get_logger().info(f'[IMUBridge] First publish: yaw={y:.2f}, pitch={p:.2f}, roll={r:.2f}')

    # ---- teardown ----
    def destroy_node(self):
        try:
            self.imu.stop()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = IMUBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down.')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

