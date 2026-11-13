import rclpy
import time
import sys
from std_msgs.msg import Float32MultiArray
# sys.path.append('/home/iclab/Desktop/kid_hurocup/src/strategy')
from strategy.API import API

class test(API):
    def __init__(self):
        super().__init__()  # 1. 不傳參數，沿用 API 預設設定
        
        # 2. 建立 Timer，每 0.033 秒 (約 30Hz) 自動執行一次 self.run
        self.timer = self.create_timer(1.0 / 30.0, self.run)

    def run(self):
        if self.is_start:
            self.get_logger().info("start")
            self.handTouch(True)
            self.get_logger().info(self.hand_mass['left'])
            time.sleep(10)
        else:
            self.get_logger().info("stop")


def main(args=None):
    rclpy.init(args=args)
    node = test()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # 3. 加上 rclpy.ok() 檢查，避免 Ctrl+C 時重複 shutdown 報錯
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

