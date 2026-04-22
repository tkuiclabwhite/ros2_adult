import sys
import rclpy
from rclpy.node import Node
from strategy.API import API
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import math

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
        self.prev_x_diff = 0
        self.prev_y_diff = 0
        self.search_angle = 0.0
        self.search_radius = 0.0
        self.new_waist_pos = self.waist_horizon
        self.reg = 1 # 搜尋方向控制        
        
        # 新增：目標丟失計數器
        self.miss_count = 0
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
# 執行追蹤修正
            if self.is_start:
                if self.px != 0 and self.py != 0:
                    self.trace_revise(self.px, self.py, 100)
                    self.search_angle = 0.0
                    self.search_radius = 0.0
                    self.miss_count = 0  # 找到人，重置丟失計數
                else:
                    self.miss_count += 1
                    if self.miss_count > 10:  # 容忍 10 幀沒看到人，才開始螺旋搜尋
                        self.get_logger().info('miss_target -> 開始螺旋尋求')
                        # 拿掉 delay 參數
                        self.view_search(
                            right_place=HEAD_HORIZON_MAXMIN[1], 
                            left_place=HEAD_HORIZON_MAXMIN[0], 
                            up_place=HEAD_VERTICAL_MAXMIN[1], 
                            down_place=HEAD_VERTICAL_MAXMIN[0], 
                            speed=80
                        )
                    else:
                        self.get_logger().info(f'目標閃爍... 等待確認 ({self.miss_count}/10)')
            else:
                self.sendHeadMotor(1, HEAD_HORIZON, 50)
                self.sendHeadMotor(2, HEAD_VERTICAL, 50)
                self.SingleAbsolutePosition(WAIST_ID, 2048, 50)
                self.sendBodySector(29)
                
                # 必須同步重置內部絕對座標變數，防止下次 true 時暴衝
                self.head_horizon = HEAD_HORIZON
                self.head_vertical = HEAD_VERTICAL
                self.waist_horizon = 2048
                
                # 同時重置搜尋與 PID 相關變數
                self.search_num = 0
                self.search_angle = 0.0
                self.search_radius = 0.0
                self.prev_x_diff = 0
                self.prev_y_diff = 0
                self.miss_count = 0

            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"執行錯誤: {e}")

    def trace_revise(self, x_target, y_target, speed):
        x_difference = x_target - 160               
        y_difference = y_target - 120               
        
        # --- 3. PD 控制防過衝 ---
        Kp = 0.1 # 比例：原始的 0.15
        Kd = 0.2 # 微分：預測趨勢，煞車用 (可依機器人反應微調)
        
        x_pd = (x_difference * Kp) + ((x_difference - self.prev_x_diff) * Kd)
        y_pd = (y_difference * Kp) + ((y_difference - self.prev_y_diff) * Kd)
        
        # 紀錄本次誤差給下一幀使用
        self.prev_x_diff = x_difference
        self.prev_y_diff = y_difference
        
        x_degree = x_pd * (65 / 320)         
        y_degree = y_pd * (38 / 240)         

        # --- 頭部計算 ---
        new_head_h = self.head_horizon - round(x_degree * 4096 / 360)
        new_head_v = self.head_vertical + round(y_degree * 4096 / 360)

        # 1. 水平追蹤與 4. 腰部回正
        if HEAD_HORIZON_MAXMIN[0] <= new_head_h <= HEAD_HORIZON_MAXMIN[1]:
            # 頭部在安全範圍 -> 檢查腰部是否需要回正 2048
            if self.waist_horizon != 2048:
                # 決定腰部回歸中心的步長
                if abs(self.waist_horizon - 2048) < 15:
                    waist_shift = 2048 - self.waist_horizon
                else:
                    waist_shift = 15 if self.waist_horizon < 2048 else -15
                
                self.waist_horizon += waist_shift
                self.SingleAbsolutePosition(WAIST_ID, self.waist_horizon, speed)
                
                # 腰部往中心轉了，頭部必須「反向轉動」同等刻度，鏡頭才能鎖定在原目標上
                new_head_h -= waist_shift 

            clamped_h = max(HEAD_HORIZON_MAXMIN[0], min(new_head_h, HEAD_HORIZON_MAXMIN[1]))
            self.move_head(1, int(clamped_h), speed)
            
        else:
            self.get_logger().warn(f"頭部水平受限，計算腰部 ID:{WAIST_ID}")
            
            # --- 腰部專用 PD 計算 ---
            # 腰部負載較重，Kp 與 Kd 建議比頭部小，確保重心穩定
            Kp_waist = 0.05
            Kd_waist = 0.1  
            
            # 計算腰部的 PD 補償量
            x_pd_waist = (x_difference * Kp_waist) + ((x_difference - self.prev_x_diff) * Kd_waist)
            
            # 轉換為馬達刻度
            waist_diff = round((x_pd_waist * 65 / 320) * (4096 / 360))
            
            # 決定移動方向 (依照你之前的邏輯：人在右側 x_diff > 0 時，馬達數值要減少)
            predicted_absolute_pos = self.waist_horizon - waist_diff
            
            # 檢查是否超出腰部物理極限
            if WAIST_HORIZON_MAXMIN[0] <= predicted_absolute_pos <= WAIST_HORIZON_MAXMIN[1]:
                self.SingleAbsolutePosition(WAIST_ID, predicted_absolute_pos, speed)
                self.waist_horizon = predicted_absolute_pos
            else:
                self.get_logger().warn("腰部已達物理範圍邊限")

            # 頭部依然固定在邊界
            clamped_h = max(HEAD_HORIZON_MAXMIN[0], min(new_head_h, HEAD_HORIZON_MAXMIN[1]))
            self.move_head(1, int(clamped_h), speed)

        # 2. 垂直追蹤
        clamped_v = max(HEAD_VERTICAL_MAXMIN[0], min(new_head_v, HEAD_VERTICAL_MAXMIN[1]))
        self.move_head(2, int(clamped_v), speed)
    def move_head(self, ID, Position, Speed):
        self.sendHeadMotor(ID, Position, Speed)
        if ID == 1:
            self.head_horizon = Position
        elif ID == 2:
            self.head_vertical = Position

    def view_search(self, right_place, left_place, up_place, down_place, speed):   
        # 螺旋擴展參數 (配合高頻率 callback 需調得很小)
        angle_step = 0.1   # 每次轉動的角度增量，越小畫圈越圓
        radius_step = 2.5  # 半徑增量，越小圈圈擴大得越慢
        
        self.search_angle += angle_step
        self.search_radius += radius_step
        
        target_h = int(2048 + self.search_radius * math.cos(self.search_angle))
        target_v = int(2028 + self.search_radius * math.sin(self.search_angle))
        
        if target_h > right_place or target_h < left_place or target_v > up_place or target_v < down_place:
            self.get_logger().info('搜尋達邊界，重置回中心點')
            self.search_angle = 0.0
            self.search_radius = 0.0
            target_h = 2048
            target_v = 2028

        self.move_head(1, target_h, speed)
        self.move_head(2, target_v, speed)
        # 絕對不能在這裡放 time.sleep()！

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
