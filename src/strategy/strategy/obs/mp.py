import sys
import rclpy
from rclpy.node import Node
from strategy.API import API
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

# 設定初始值與馬達限制
HEAD_HORIZON = 2030
HEAD_VERTICAL = 2028

# 設定頭部旋轉範圍極限 (0 ~ 3000)
HEAD_HORIZON_MAXMIN = [1024, 3096]
HEAD_VERTICAL_MAXMIN = [1400, 2900]
WAIST_HORIZON_MAXMIN = [1250,3096]

WAIST_ID = 15

# 強制排除可能衝突的路徑
sys.path = [p for p in sys.path if 'matplotlib' not in p]

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    import mediapipe.self.python.solutions.pose as mp_pose

class PersonCoordinateNode(API):
    def __init__(self):
        # 修正：Node 與 API 初始化順序
        super().__init__('person_coordinate_node')
        self.subscription = self.create_subscription(
            Image, 
            'camera1/image_raw', 
            self.image_callback, 
            10
        )
        self.bridge = CvBridge()
        
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 初始化座標與變數
        self.head_horizon = HEAD_HORIZON
        self.head_vertical = HEAD_VERTICAL
        self.waist_horizon = 2048  # 假設腰部馬達初始在正中間
        self.px = 0
        self.py = 0
        self.search_num = 0
        self.new_waist_pos = self.waist_horizon
        self.reg = 1 # 搜尋方向控制
        self.get_logger().info("320x240 座標偵測與追蹤 Node 已啟動")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (320, 240))
            
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            self.px, self.py = 0, 0
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark[0] # 鼻子
                self.px, self.py = int(lm.x * 320), int(lm.y * 240)

                # 視覺回饋繪製
                cv2.circle(cv_image, (self.px, self.py), 5, (0, 0, 255), -1)
                cv2.line(cv_image, (self.px, 0), (self.px, 240), (0, 255, 0), 1)
                cv2.line(cv_image, (0, self.py), (320, self.py), (0, 255, 0), 1)
                cv2.putText(cv_image, f"X:{self.px}, Y:{self.py}", (self.px + 10, self.py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(cv_image, "No Person", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Target: ({self.px}, {self.py})")
            # cv2.imshow("Person Tracking", cv_image)
            self.drawImage()
            # 執行追蹤修正
            if self.is_start:
                if self.px != 0 and self.py != 0:
                    self.trace_revise(self.px, self.py, 100)
                    self.search_num = 0
                else:
                    self.get_logger().info('miss_target -> 需重新尋求')
                    # self.view_search(right_place=2800, left_place=1200, up_place=2800, down_place=1800, speed=100, delay=0.05)
            else:
                self.sendHeadMotor(1, HEAD_HORIZON, 50)
                self.sendHeadMotor(2, HEAD_VERTICAL, 50)
                self.SingleAbsolutePosition(WAIST_ID, 2048, 50)
                self.search_num = 0
                self.sendBodySector(29)
                # self.sendSingleMotor(WAIST_ID, 2048, 50)

            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"執行錯誤: {e}")

    def trace_revise(self, x_target, y_target, speed):
        # 計算影像中心偏差 (Center = 160, 120)
        x_difference = x_target - 160               
        y_difference = y_target - 120               
        
        # 像素偏差轉角度比例
        x_degree = x_difference * (65 / 320)         
        y_degree = y_difference * (38 / 240)         

        # --- 水平計算 ---
        new_head_h = self.head_horizon - round(x_degree * 4096 / 360 * 0.15)
        
        # --- 垂直計算 (已修正符號為相反) ---
        # 原本是 self.head_vertical - ... 現在改為 +
        new_head_v = self.head_vertical + round(y_degree * 4096 / 360 * 0.15)

        # 1. 水平追蹤與腰部補償
        if HEAD_HORIZON_MAXMIN[0] <= new_head_h <= HEAD_HORIZON_MAXMIN[1]:
            self.move_head(1, new_head_h, speed)
        else:
            self.get_logger().warn(f"頭部水平受限，改動腰部 ID:{WAIST_ID}")
            
            relative_step = -5 if x_difference > 0 else 5
            
            # 預判移動後的絕對位置，用來檢查範圍
            predicted_absolute_pos = self.waist_horizon + relative_step
            if WAIST_HORIZON_MAXMIN[0] <= predicted_absolute_pos <= WAIST_HORIZON_MAXMIN[1]:
                # waist_step = -10 if x_difference > 0 else 10
                # self.new_waist_pos = self.waist_horizon - waist_step
                # self.get_logger().info(f"self.new_waist_pos:{self.new_waist_pos}")
                # self.sendSingleMotor(WAIST_ID, waist_step, speed)
                # self.waist_horizon = self.new_waist_pos
                self.SingleAbsolutePosition(WAIST_ID, predicted_absolute_pos, speed)
                # 更新目前存放在程式裡的絕對位置紀錄
                self.waist_horizon = predicted_absolute_pos
            clamped_h = max(HEAD_HORIZON_MAXMIN[0], min(new_head_h, HEAD_HORIZON_MAXMIN[1]))
            self.move_head(1, clamped_h, speed)

        # 2. 垂直追蹤
        if HEAD_VERTICAL_MAXMIN[0] <= new_head_v <= HEAD_VERTICAL_MAXMIN[1]:
            self.move_head(2, new_head_v, speed)
        else:
            # 強制限制在範圍內，避免數值跑掉
            clamped_v = max(HEAD_VERTICAL_MAXMIN[0], min(new_head_v, HEAD_VERTICAL_MAXMIN[1]))
            self.move_head(2, clamped_v, speed)
    def move_head(self, ID, Position, Speed):
        self.sendHeadMotor(ID, Position, Speed)
        if ID == 1:
            self.head_horizon = Position
        elif ID == 2:
            self.head_vertical = Position

    def view_search(self, right_place, left_place, up_place, down_place, speed, delay):   
        # 決定搜尋順序
        turn_order = [3, 4, 1, 2] if self.reg > 0 else [1, 4, 3, 2]
        if self.search_num >= len(turn_order):
            self.search_num = 0

        search_flag = turn_order[self.search_num]

        if search_flag == 1: # 左尋
            if self.head_horizon >= left_place:
                self.move_head(1, self.head_horizon - speed, 100)
            else:
                self.search_num += 1
        
        elif search_flag == 3: # 右尋
            if self.head_horizon <= right_place:
                self.move_head(1, self.head_horizon + speed, 100)
            else:
                self.search_num += 1

        elif search_flag == 4: # 上尋
            if self.head_vertical <= up_place:
                self.move_head(2, self.head_vertical + speed, 100)
            else:
                self.search_num += 1  
        
        elif search_flag == 2: # 下尋
            if self.head_vertical >= down_place:
                self.move_head(2, self.head_vertical - speed, 100)
            else:
                self.search_num = 0 # 繞完一圈回到第一個動作
        
        time.sleep(delay)            

    def drawImage(self):
        self.drawImageFunction(1, 1, 160, 160, 0, 240, 255, 255, 255) 
        self.drawImageFunction(2, 1, 0, 320, 120, 120, 255, 255, 255) 

        self.drawImageFunction(3, 1, self.px, self.px, 0, 240, 255, 255, 255) 
        self.drawImageFunction(4, 1, 0, 320, self.py, self.py, 255, 255, 255) 
        

def main(args=None):
    rclpy.init(args=args)
    node = PersonCoordinateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
