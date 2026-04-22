import rclpy
import time
from collections import deque
from std_msgs.msg import String
from geometry_msgs.msg import Point
from strategy.API import API

ORIGIN_THETA = 0


class Mar(API):
    def __init__(self):
        super().__init__('mar_strategy')

        # YOLO 即時資料
        self.action = None
        self.x = 0.0
        self.y = 0.0
        self.area = 0.0
        self.angle = 0.0
        self.last_update = 0.0

        # 穩定箭頭判斷
        self.arrow_temp = deque(['None', 'None', 'None', 'None', 'None'], maxlen=5)
        self.stable_arrow = None
        self.can_turn_flag = False

        # 箭頭位置
        self.arrow_center_x = 0.0
        self.arrow_center_y = 0.0

        # 狀態機
        self.status = 'INIT'
        self.turn_now_flag = False
        self.is_start = False
        self.arrow_cnt_times = 0
        self.turn_start_yaw = 0.0

        # 訂閱 YOLO topic
        self.create_subscription(String, 'class_id_topic', self.yolo_callback, 10)
        self.create_subscription(Point, 'sign_coordinates', self.coord_callback, 10)

        self.create_timer(0.05, self.main_loop)

    def yolo_callback(self, msg: String):
        parts = msg.data.split(',')
        if len(parts) != 5:
            self.get_logger().warn(f'Bad class_id_topic format: {msg.data}')
            return

        try:
            self.action = parts[0].strip().lower()
            self.x = float(parts[1])
            self.y = float(parts[2])
            self.area = float(parts[3])
            self.angle = float(parts[4])
            self.last_update = time.time()
        except Exception as e:
            self.get_logger().error(f'YOLO parse error: {e}')

    def coord_callback(self, msg: Point):
        # 即時點位，可視需求保留
        pass

    def yolo_valid(self):
        return (time.time() - self.last_update) < 1.0

    def arrow_yolo(self):
        if self.yolo_valid():
            self.arrow_temp.append(self.action)

        self.get_logger().info(f'arrow_temp = {list(self.arrow_temp)}')

        uniq = set(self.arrow_temp)
        if len(uniq) == 1 and self.arrow_temp[0] != 'None':
            stable_label = self.arrow_temp[0]
            self.stable_arrow = stable_label
            self.can_turn_flag = stable_label in ('left', 'right','Right','Left')
            self.arrow_center_x = self.x
            self.arrow_center_y = self.y
            return True

        self.stable_arrow = None
        self.can_turn_flag = False
        return False

    def imu_go(self):
        
        self.theta = 0 + ORIGIN_THETA
        self.speed_x = 3500

        self.get_logger().debug(f'直走 yaw={self.yaw:.2f}')

        if 0 < self.arrow_center_x <= 140:
            self.theta = 5 + ORIGIN_THETA

        elif self.arrow_center_x >= 180:
            self.theta = -5 + ORIGIN_THETA

        else:
            if self.yaw > 5:
                self.theta = -3 + ORIGIN_THETA
                self.get_logger().debug('修正：右轉')
            elif self.yaw < -5:
                self.theta = 3 + ORIGIN_THETA
                self.get_logger().debug('修正：左轉')
            else:
                self.theta = 0 + ORIGIN_THETA

        self.sendContinuousValue(self.speed_x, 0, self.theta)

    def arrow_turn(self):
        print(1)
        if self.arrow_temp[0] == 'right':
            self.sendContinuousValue(1700, 0, -6 + ORIGIN_THETA)
        elif self.arrow_temp[0] == 'left':
            self.sendContinuousValue(1700, 0, 6 + ORIGIN_THETA)
        else:
            self.sendContinuousValue(-300, 0, 0)
            return

        if abs(self.yaw - self.turn_start_yaw) > 85:
            self.sendSensorReset(True)
            self.turn_now_flag = False
            self.can_turn_flag = False
            self.arrow_cnt_times = 0
            self.get_logger().info('箭頭轉彎結束')

    def main_loop(self):
        if self.is_start:
            # self.status = 'INIT'
            
            # print(self.status)
            # self.get_logger().info(self.can_turn_flag)
            if self.status == 'INIT':
                self.sendHeadMotor(2, 2800, 50)
                self.sendHeadMotor(1, 2048, 50)
                self.sendSensorReset(True)
                self.sendbodyAuto(1)
                self.status = 'ARROW_PART'
                self.get_logger().info('進入 ARROW_PART')

            elif self.status == 'ARROW_PART':
                
                if self.turn_now_flag:
                    self.arrow_turn()

                else:
                    # Change this line:
                    # self.arrow.yolo()
                    
                    # To this:
                    self.arrow_yolo()
                    
                    if self.can_turn_flag:
                        self.get_logger().info('穩定看到可轉向箭頭')

                        # 接近箭頭才開始轉
                        if self.arrow_center_y >= 185:
                            self.arrow_cnt_times += 1

                        if self.arrow_cnt_times >= 13:
                            self.turn_start_yaw = self.yaw
                            self.turn_now_flag = True
                            self.arrow_cnt_times = 0
                            self.get_logger().info('開始轉彎')
                            self.status = 'ARROW_PART'
                            
            
                    self.imu_go()

        else :
            if self.status != 'INIT':
                self.sendbodyAuto(0)
                self.status = 'INIT'
    

def main(args=None):
    rclpy.init(args=args)
    node = Mar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()