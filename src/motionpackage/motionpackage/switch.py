#!/usr/bin/env python3
#coding=utf-8
import rclpy
from rclpy.node import Node
from tku_msgs.msg import Dio
import Jetson.GPIO as GPIO

PIN = 32

class USBDIONode(Node):
    def __init__(self):
        super().__init__('usb_dionode')
        
        # 1. GPIO 初始化移到這裡，只需執行一次
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # 發布到原本的 dio topic
        self.publisher = self.create_publisher(Dio, '/package/dioarray', 10)
        
        # 每 0.05 秒檢查一次 (20Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("USB DIO Node started. Monitoring physical switch...")

        self.strategy_check = False

    def timer_callback(self):
        # 2. 移除 while True，讓 Timer 自然地每 0.05 秒觸發一次這段邏輯
        try:
            msg = Dio()
            state = GPIO.input(PIN)
            
            if state == GPIO.LOW:
                # self.get_logger().error("Strategy: True") # 若覺得洗頻可以註解掉這行
                msg.strategy = False
            else:  # state == GPIO.HIGH
                # self.get_logger().error("Strategy: False") # 若覺得洗頻可以註解掉這行
                msg.strategy = True
            
            if msg.strategy != self.strategy_check:
                self.get_logger().error(f"{msg.strategy}")
                self.strategy_check = msg.strategy
                
            # 3. 將訊息發布出去
            self.publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"GPIO Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = USBDIONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 4. 關閉節點時才清理 GPIO 資源
        node.get_logger().info("Shutting down... cleaning up GPIO.")
        GPIO.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()