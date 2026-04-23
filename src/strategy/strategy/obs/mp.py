import sys
import rclpy
from rclpy.node import Node
from strategy.API import API
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import math
import threading
import os

# 設定初始值與馬達限制
HEAD_HORIZON = 2030
HEAD_VERTICAL = 2028

# 設定頭部旋轉範圍極限 (0 ~ 3000)
HEAD_HORIZON_MAXMIN = [1024, 3096]
HEAD_VERTICAL_MAXMIN = [1400, 2900]
WAIST_HORIZON_MAXMIN = [1250,3096]

DIR_HEAD_H = -1
DIR_HEAD_V = 1

WAIST_ID = 15

# 強制排除可能衝突的路徑
sys.path = [p for p in sys.path if 'matplotlib' not in p]

try:
    import mediapipe as mp
    # 全面使用相容性最高的 Solutions API
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
except Exception as e:
    print(f"MediaPipe 匯入失敗: {e}")

class PersonCoordinateNode(API):
    def __init__(self):
        super().__init__('person_coordinate_node')
        self.subscription = self.create_subscription(Image, 'camera1/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        
        # --- 1. 手勢辨識初始化 (Solutions 版本) ---
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- 2. Pose 模型初始化 (Solutions 版本) ---
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sys_msg = "穩定版 (Solutions API) 已啟動"
        self.gesture_msg = ""

        self.head_horizon = HEAD_HORIZON
        self.head_vertical = HEAD_VERTICAL
        self.waist_horizon = 2048
        self.px = 0
        self.py = 0
        self.search_num = 0
        self.reg = 1
        
        # PID 與搜尋變數
        self.prev_x_diff = 0
        self.prev_y_diff = 0
        self.search_angle = 0.0
        self.search_radius = 0.0
        self.miss_count = 0
        self.target_lost_frames = 0
        self.people_count = 0
        self.untracked_people = []

        self.photo_cooldown = 0
        self.switch_cooldown = 0

        # self.is_start = True
        self.was_started = True

        # 照片儲存設定
        self.save_dir = "snap"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.get_logger().info("全 Solutions API 模式已啟動")

        self.printer_thread = StatusPrinterThread(self)
        self.printer_thread.start()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (320, 240))
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 冷卻時間扣減
            if self.photo_cooldown > 0: self.photo_cooldown -= 1
            if self.switch_cooldown > 0: self.switch_cooldown -= 1

            # ==========================================
            # A. 手勢辨識邏輯 (取代原本的 GestureRecognizer)
            # ==========================================
            gesture_direction = None
            hand_results = self.hands.process(image_rgb)

            # ==========================================
            # A. 手勢辨識邏輯 (Solutions 版本)
            # ==========================================
            gesture_direction = None
            hand_results = self.hands.process(image_rgb)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    
                    # 建立一個計算兩點之間距離的匿名函式
                    get_dist = lambda p1, p2: math.hypot(lm[p1].x - lm[p2].x, lm[p1].y - lm[p2].y)
                    
                    # 判定手指是否伸直：指尖(Tip)到手腕(0)的距離 > 第二關節(PIP)到手腕的距離
                    index_open  = get_dist(8, 0)  > get_dist(6, 0)  # 食指
                    middle_open = get_dist(12, 0) > get_dist(10, 0) # 中指
                    ring_open   = get_dist(16, 0) > get_dist(14, 0) # 無名指
                    pinky_open  = get_dist(20, 0) > get_dist(18, 0) # 小指

                    # --- 只有在 self.is_start == True 時才執行手勢動作 ---
                    if self.is_start:
                        # 1. 功能一：比 YA (嚴格要求：食指中指伸出，且無名指小指必須收起)
                        if index_open and middle_open and not ring_open and not pinky_open and self.photo_cooldown == 0:
                            filename = f"robot_photo_{int(time.time())}.jpg"
                            save_path = os.path.join(self.save_dir, filename)
                            cv2.imwrite(save_path, cv_image) 
                            self.gesture_msg = f"已拍照存檔: {filename}"
                            self.photo_cooldown = 30 

                        # 2. 功能二：食指指向 (嚴格要求：只有食指伸出，其他三指必須收起)
                        elif index_open and not middle_open and not ring_open and not pinky_open and self.switch_cooldown == 0:
                            # 改用食指尖(8) 與 食指掌根(5) 的 X 座標來判斷方向，比對手腕更精準
                            dx = lm[8].x - lm[5].x 
                            
                            # 加大判定閾值，手要確實往側邊指才觸發
                            if abs(dx) > 0.05:
                                shift_val = 400 if dx > 0 else -400
                                new_h = self.head_horizon + (DIR_HEAD_H * shift_val)
                                
                                clamped_h = int(max(HEAD_HORIZON_MAXMIN[0], min(new_h, HEAD_HORIZON_MAXMIN[1])))
                                
                                self.move_head(1, clamped_h, 80)
                                self.gesture_msg = f"向{ '右' if dx>0 else '左' }查看"
                                self.switch_cooldown = 45 
                    else:
                        # 待機狀態無視指令
                        if (index_open and middle_open and not ring_open and not pinky_open) or \
                           (index_open and not middle_open and not ring_open and not pinky_open):
                            self.gesture_msg = "系統待機中，無視手勢指令"

            # ==========================================
            # B. 人體追蹤邏輯 (使用解決方案模式)
            # ==========================================
            results = self.pose.process(image_rgb)
            self.px, self.py = 0, 0
            self.untracked_people = [] # Solutions Pose 僅偵測單人，故此處清空

            if results.pose_landmarks:
                self.people_count = 1
                # 取得鼻子 (0) 作為追蹤目標
                nose = results.pose_landmarks.landmark[0]
                self.px = int(nose.x * 320)
                self.py = int(nose.y * 240)
                self.target_lost_frames = 0
                
                # 若有切換手勢，在單人模式下顯示提示 (因為沒有第二個人可以切換)
                if gesture_direction:
                    self.gesture_msg = f"已指向{gesture_direction}，但單人模式無備選目標"
                    self.switch_cooldown = 30

            #     # 在 OpenCV 畫面上畫出鎖定標誌
            #     cv2.circle(cv_image, (self.px, self.py), 5, (0, 0, 255), -1)
            #     cv2.line(cv_image, (self.px, 0), (self.px, 240), (0, 255, 0), 1)
            #     cv2.line(cv_image, (0, self.py), (320, self.py), (0, 255, 0), 1)
            else:
                self.people_count = 0
            
            # --- 共用的馬達控制與搜尋邏輯 ---
            if self.is_start:
                self.was_started = True
                if self.px != 0 and self.py != 0:
                    self.trace_revise(self.px, self.py, 100)
                    self.search_angle = 0.0
                    self.search_radius = 0.0
                    self.miss_count = 0 
                else:
                    self.miss_count += 1
                    if self.miss_count > 10: 
                        self.view_search(
                            right_place=HEAD_HORIZON_MAXMIN[1], 
                            left_place=HEAD_HORIZON_MAXMIN[0], 
                            up_place=HEAD_VERTICAL_MAXMIN[1], 
                            down_place=HEAD_VERTICAL_MAXMIN[0], 
                            speed=80
                        )
                        self.prev_x_diff = 0
                        self.prev_y_diff = 0
            else:
                if self.was_started:
                    self.reset_system() # 抽取成函式保持整潔
                    self.was_started = False
            
            self.drawImage()

            # try:
            #     cv2.imshow("Person Tracking Dashboard", cv_image)
            #     cv2.waitKey(1)
            # except Exception:
            #     pass

        except Exception as e:
            self.get_logger().error(f"執行錯誤: {e}")

    def reset_system(self):
        self.sys_msg = "正在復位馬達..."
        self.sendHeadMotor(1, HEAD_HORIZON, 50)
        self.sendHeadMotor(2, HEAD_VERTICAL, 50)
        self.SingleAbsolutePosition(WAIST_ID, 2048, 50)
        self.sendBodySector(29)
        self.head_horizon, self.head_vertical, self.waist_horizon = HEAD_HORIZON, HEAD_VERTICAL, 2048
        self.search_angle = self.search_radius = self.prev_x_diff = self.prev_y_diff = self.miss_count = 0
        self.px = self.py = 0
        self.sys_msg = "停止運作 (Standby)"

    def trace_revise(self, x_target, y_target, speed):
        x_difference, y_difference = x_target - 160, y_target - 120
        Kp, Kd = 0.1, 0.2 
        
        x_pd = (x_difference * Kp) + ((x_difference - self.prev_x_diff) * Kd)
        y_pd = (y_difference * Kp) + ((y_difference - self.prev_y_diff) * Kd)
        
        self.prev_x_diff, self.prev_y_diff = x_difference, y_difference
        
        x_degree, y_degree = x_pd * (65 / 320), y_pd * (38 / 240)
        
        # 乘上方向常數，只要在最上方改 1 或 -1 就能一鍵反轉
        new_head_h = self.head_horizon + (DIR_HEAD_H * round(x_degree * 4096 / 360))
        new_head_v = self.head_vertical + (DIR_HEAD_V * round(y_degree * 4096 / 360))

        if HEAD_HORIZON_MAXMIN[0] <= new_head_h <= HEAD_HORIZON_MAXMIN[1]:
            self.sys_msg = "正常鎖定追蹤中"
            if self.waist_horizon != 2048:
                waist_shift = (2048 - self.waist_horizon) if abs(self.waist_horizon - 2048) < 15 else (15 if self.waist_horizon < 2048 else -15)
                self.waist_horizon += waist_shift
                self.SingleAbsolutePosition(WAIST_ID, self.waist_horizon, speed)
                new_head_h -= waist_shift 
            self.move_head(1, int(max(HEAD_HORIZON_MAXMIN[0], min(new_head_h, HEAD_HORIZON_MAXMIN[1]))), speed)
        else:
            self.sys_msg = f"啟動腰部補償 ID:{WAIST_ID}"
            
            # --- 補回：腰部專用 PD 參數，方便隨時微調 ---
            Kp_waist = 0.05
            Kd_waist = 0.1  
            
            x_pd_waist = (x_difference * Kp_waist) + ((x_difference - self.prev_x_diff) * Kd_waist)
            waist_diff = round((x_pd_waist * 65 / 320) * (4096 / 360))
            predicted_pos = self.waist_horizon - waist_diff
            # ------------------------------------------

            if WAIST_HORIZON_MAXMIN[0] <= predicted_pos <= WAIST_HORIZON_MAXMIN[1]:
                self.SingleAbsolutePosition(WAIST_ID, predicted_pos, speed)
                self.waist_horizon = predicted_pos
            else:
                self.sys_msg = "腰部達邊限!"
            self.move_head(1, int(max(HEAD_HORIZON_MAXMIN[0], min(new_head_h, HEAD_HORIZON_MAXMIN[1]))), speed)

        self.move_head(2, int(max(HEAD_VERTICAL_MAXMIN[0], min(new_head_v, HEAD_VERTICAL_MAXMIN[1]))), speed)

    def move_head(self, ID, Position, Speed):
        self.sendHeadMotor(ID, Position, Speed)
        if ID == 1: self.head_horizon = Position
        elif ID == 2: self.head_vertical = Position

    def view_search(self, right_place, left_place, up_place, down_place, speed):   
        self.search_angle += 0.1
        self.search_radius += 2.5
        target_h = int(2048 + self.search_radius * math.cos(self.search_angle))
        target_v = int(2028 + self.search_radius * math.sin(self.search_angle))
        
        if target_h > right_place or target_h < left_place or target_v > up_place or target_v < down_place:
            self.sys_msg = "搜尋達邊界重置"
            self.search_angle = self.search_radius = 0.0
            target_h, target_v = 2048, 2028
        self.move_head(1, target_h, speed)
        self.move_head(2, target_v, speed)

    def drawImage(self):
        self.drawImageFunction(1, 1, 160, 160, 0, 240, 255, 255, 255) 
        self.drawImageFunction(2, 1, 0, 320, 120, 120, 255, 255, 255) 
        self.drawImageFunction(3, 1, self.px, self.px, 0, 240, 255, 0, 0) 
        self.drawImageFunction(4, 1, 0, 320, self.py, self.py, 255, 0, 0) 
        # 覆蓋清理矩形圖層防止殘影
        for i in range(5, 7): self.drawImageFunction(i, 2, 0, 0, 0, 0, 0, 0, 0)
        
class StatusPrinterThread(threading.Thread):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.daemon = True
        self.running = True

    def run(self):
        while self.running and rclpy.ok():
            self.val_print()
            time.sleep(0.1)
    
    def val_print(self):
        if getattr(self.node, 'head_horizon', None) is None: return
        try:
            sys.stdout.write("\033[H\033[J")
            if not self.node.is_start:
                status_str = "停止運作 (Standby)"
            elif self.node.px != 0:
                status_str = "目標鎖定中"
            elif self.node.miss_count > 10:
                status_str = "螺旋搜尋中"
            else:
                status_str = "目標閃爍等待確認"

            sys.stdout.write(f"\
#==============系統狀態==============#\n\
is_start         : {self.node.is_start}\n\
System Status    : {status_str}\n\
System Message   : {self.node.sys_msg}\n\
Gesture_msg      : {getattr(self.node, 'gesture_msg', '')}\n\
Miss Count       : {self.node.miss_count} / 10\n\
People Detected  : {self.node.people_count} 人\n\
#===============視覺目標=============#\n\
Target (px, py)  : ({self.node.px}, {self.node.py})\n\
Error  (x, y)    : ({self.node.prev_x_diff:.1f}, {self.node.prev_y_diff:.1f})\n\
#===============馬達狀態=============#\n\
Head Horizon (1) : {self.node.head_horizon}\n\
Head Vertical(2) : {self.node.head_vertical}\n\
Waist Horizon({WAIST_ID}): {self.node.waist_horizon}\n\
#===============搜尋參數=============#\n\
Search Angle     : {self.node.search_angle:.2f}\n\
Search Radius    : {self.node.search_radius:.2f}\n\
#====================================#\n")
            sys.stdout.flush()
        except Exception: pass

def main(args=None):
    rclpy.init(args=args)
    node = PersonCoordinateNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        # cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()