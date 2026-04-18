#!/usr/bin/env python3
#coding=utf-8
import rclpy
from rclpy.node import Node
from tku_msgs.msg import Dio
import serial
import serial.tools.list_ports
import time

class USBDIONode(Node):
    def __init__(self):
        super().__init__('usb_dionode')
        # 發布到原本的 dio topic
        self.publisher = self.create_publisher(Dio, '/package/dioarray', 10)
        self.ser = None
        # 每 0.05 秒檢查一次 (20Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("USB DIO Node started. Monitoring physical switch...")

    def find_port(self):
        """用 udev 命名的固定裝置：/dev/esp32"""
        target = '/dev/esp32'
        import os
        if os.path.exists(target):
            try:
                ser = serial.Serial(target, 115200, timeout=0.1)
                self.get_logger().info(f"✅ DIO 連線至: {target}")
                return ser
            except Exception as e:
                self.get_logger().warn(f"⚠️ 開啟 {target} 失敗: {e}")
                return None
        else:
            self.get_logger().warn("⚠️ 找不到 /dev/esp32（udev 規則可能沒生效）")
            return None

    def timer_callback(self):
        # 斷線重連機制
        if self.ser is None or not self.ser.is_open:
            self.ser = self.find_port()
            return

        try:
            if self.ser.in_waiting > 0:
                # 讀取 ESP32 傳來的 START 或 STOP
                # 加入 errors='ignore' 防止雜訊導致解碼失敗崩潰
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                
                if not line: return # 空行忽略

                # --- 關鍵修正：加入 Log 顯示，這樣你才看得到反應 ---
                self.get_logger().info(f"Serial Read: {line}") 
                # ----------------------------------------------

                msg = Dio()
                # 判斷字串內容，設定 bool 數值
                if "START" in line:
                    msg.strategy = True
                elif "STOP" in line:
                    msg.strategy = False
                else:
                    # 如果讀到的不是關鍵字，就不發布，避免干擾
                    return 

                msg.data = 0 
                self.publisher.publish(msg)
                
        except Exception as e:
            self.get_logger().error(f"Serial Error: {e}")
            self.ser = None

def main(args=None):
    rclpy.init(args=args)
    node = USBDIONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 關閉時若有連線則關閉，避免占用 Port
        if node.ser and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

# --- 關鍵修正：補上前後雙底線 ---
if __name__ == '__main__':
    main()