# # # import rclpy
# # # from rclpy.node import Node
# # # from geometry_msgs.msg import Point, Twist

# # # class SignControlPFilter(Node):
# # #     def __init__(self):
# # #         super().__init__('sign_control_p_filter')

# # #         # --- 訂閱與發布 ---
# # #         self.sub_coord = self.create_subscription(
# # #             Point, 
# # #             'sign_coordinates', 
# # #             self.coord_callback, 
# # #             10
# # #         )
# # #         self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)

# # #         # --- 參數設定 (調整這裡) ---
# # #         self.img_width = 640
# # #         self.target_center_x = self.img_width / 2  # 目標中心 320
# # #         self.dead_zone = 20    # 死區：誤差小於 20 就不動

# # #         # 1. P 控制參數
# # #         self.kp = 0.005        # 比例增益 (視車速需求調整)

# # #         # 2. 濾波參數 (Alpha)
# # #         # 範圍 0.0 ~ 1.0
# # #         # 值越小 = 越平滑，但反應越慢 (Lag)
# # #         # 值越大 = 越接近原始數據，反應快但較抖
# # #         self.filter_alpha = 0.3 
        
# # #         self.max_angular_vel = 0.5  # 最大轉速限制

# # #         # --- 狀態變數 ---
# # #         self.filtered_x = None  # 儲存濾波後的 X 數值

# # #         self.get_logger().info("P控制 + 濾波節點已啟動 (Alpha=0.3)")

# # #     def coord_callback(self, msg):
# # #         raw_x = msg.x
        
# # #         # --- 步驟 1: 低通濾波 (Low-Pass Filter) ---
# # #         # 演算法： 新值 = (權重 * 當前測量值) + ((1-權重) * 上一次濾波值)
# # #         if self.filtered_x is None:
# # #             self.filtered_x = raw_x  # 第一次接收，直接設為當前值
# # #         else:
# # #             self.filtered_x = (self.filter_alpha * raw_x) + \
# # #                               ((1 - self.filter_alpha) * self.filtered_x)

# # #         # --- 步驟 2: 計算誤差 (使用濾波後的 X) ---
# # #         error = self.target_center_x - self.filtered_x

# # #         # --- 步驟 3: P 控制計算 ---
# # #         if abs(error) < self.dead_zone:
# # #             cmd_angular_z = 0.0
# # #             status = "已對準"
# # #         else:
# # #             cmd_angular_z = self.kp * error
# # #             status = "修正中"

# # #         # --- 步驟 4: 安全限制 ---
# # #         cmd_angular_z = max(min(cmd_angular_z, self.max_angular_vel), -self.max_angular_vel)

# # #         # --- 步驟 5: 發布指令 ---
# # #         twist = Twist()
# # #         twist.linear.x = 0.0
# # #         twist.angular.z = cmd_angular_z
# # #         self.pub_vel.publish(twist)

# # #         # Log 顯示：可以看出原始值(Raw)跟濾波值(Filt)的差異
# # #         self.get_logger().info(
# # #             f"Raw: {raw_x:.0f} -> Filt: {self.filtered_x:.0f} | "
# # #             f"Err: {error:.0f} | Cmd: {cmd_angular_z:.3f}"
# # #         )

# # # def main(args=None):
# # #     rclpy.init(args=args)
# # #     node = SignControlPFilter()
# # #     try:
# # #         rclpy.spin(node)
# # #     except KeyboardInterrupt:
# # #         pass
# # #     finally:
# # #         stop_twist = Twist()
# # #         node.pub_vel.publish(stop_twist)
# # #         node.destroy_node()
# # #         rclpy.shutdown()

# # # if __name__ == '__main__':
# # #     main()

# # import rclpy
# # from rclpy.node import Node
# # from geometry_msgs.msg import Point, Twist
# # import math

# # class HumanoidSignControl(Node):
# #     def __init__(self):
# #         super().__init__('humanoid_sign_control')

# #         self.sub_coord = self.create_subscription(
# #             Point, 'sign_coordinates', self.coord_callback, 10)
# #         self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)

# #         # --- 參數設定 (人形機器人特化) ---
# #         self.img_width = 320
# #         self.target_center_x = self.img_width / 2
        
# #         # 1. 死區加大：人形機器人站著也會微晃，死區太小會導致它一直抖動
# #         self.dead_zone = 40  
        
# #         # 2. P 值調小：人形機器人轉向很慢，給太大沒意義，反而會超調(Overshoot)
# #         self.kp = 0.002      

# #         # 3. 濾波係數 (Alpha)：越小越平滑。
# #         # 0.1 代表新數據只佔 10%，舊數據佔 90%，抗抖動能力極強
# #         self.filter_alpha = 0.15 

# #         # 4. 速度限制
# #         self.max_angular_vel = 0.3  # 絕對不要設太快，人形機器人轉太快會離心力摔倒

# #         # 5. 加速度限制 (Slew Rate Limit) - 保護機制核心
# #         # 每次迴圈 (假設 30Hz) 速度最多只能改變這麼多
# #         self.accel_limit = 0.02     

# #         # --- 狀態變數 ---
# #         self.filtered_x = None
# #         self.current_vel_z = 0.0  # 紀錄目前實際發出的速度

# #         self.get_logger().info("人形機器人專用控制節點已啟動 (含防摔倒機制)")

# #     def coord_callback(self, msg):
# #         raw_x = msg.x
        
# #         # --- 1. 低通濾波 (抗走路晃動) ---
# #         if self.filtered_x is None:
# #             self.filtered_x = raw_x
# #         else:
# #             self.filtered_x = (self.filter_alpha * raw_x) + \
# #                               ((1 - self.filter_alpha) * self.filtered_x)

# #         # --- 2. 計算誤差 ---
# #         error = self.target_center_x - self.filtered_x

# #         # --- 3. P 控制計算目標速度 ---
# #         if abs(error) < self.dead_zone:
# #             target_vel_z = 0.0
# #             status = "穩定區"
# #         else:
# #             target_vel_z = self.kp * error
# #             status = "轉向修正"

# #         # --- 4. 絕對速度限制 (Hard Limit) ---
# #         target_vel_z = max(min(target_vel_z, self.max_angular_vel), -self.max_angular_vel)

# #         # --- 5. 平滑加減速 (Slew Rate Limiter) - 防止摔倒關鍵 ---
# #         # 計算「目標速度」與「當前速度」的差
# #         diff = target_vel_z - self.current_vel_z

# #         # 如果變化量超過限制，就只增加限制量的速度
# #         if diff > self.accel_limit:
# #             self.current_vel_z += self.accel_limit
# #         elif diff < -self.accel_limit:
# #             self.current_vel_z -= self.accel_limit
# #         else:
# #             self.current_vel_z = target_vel_z  # 變化量很小，直接到位

# #         # 強制歸零優化：如果目標是停下來，且速度已經很小，直接歸零 (避免滑行)
# #         if target_vel_z == 0.0 and abs(self.current_vel_z) < 0.05:
# #             self.current_vel_z = 0.0

# #         # --- 6. 發布指令 ---
# #         twist = Twist()
# #         twist.linear.x = 0.0  # 原地踏步旋轉
# #         twist.angular.z = self.current_vel_z
# #         self.pub_vel.publish(twist)

# #         self.get_logger().info(
# #             f"[{status}] Err:{error:.0f} | "
# #             f"Target:{target_vel_z:.3f} | "
# #             f"Actual:{self.current_vel_z:.3f}" # 注意看這個，它會慢慢追上 Target
# #         )

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = HumanoidSignControl()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         # 結束時發送停止指令
# #         stop_twist = Twist()
# #         node.pub_vel.publish(stop_twist)
# #         node.destroy_node()
# #         rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()
# #!/usr/bin/env python3
# #coding=utf-8

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile

# from std_msgs.msg import String
# from geometry_msgs.msg import Point

# from collections import Counter, deque
# import numpy as np
# import sys 
# import time
# import math

# # 假設 Python_API 是純 Python 函式庫，不依賴 rospy
# # 如果它依賴 rospy，這裡會報錯，需要修改 Python_API
# try:
#     from Python_API import Sendmessage
# except ImportError:
#     print("警告: 找不到 Python_API，請確認路徑或環境")
#     # 這裡做一個假的 class 避免程式直接掛掉，方便測試結構
#     class Sendmessage:
#         def __init__(self):
#             self.is_start = True
#             self.DIOValue = 48
#             self.imu_value_Yaw = 0.0
#             self.yolo_Label = 'None'
#             self.color_mask_subject_size = [[0]]
#             self.color_mask_subject_XMin = [[0]]
#             self.color_mask_subject_XMax = [[0]]
#             self.color_mask_subject_YMin = [[0]]
#             self.color_mask_subject_YMax = [[0]]
#             self.Label_Model = b'\x00'*320*240
#         def sendHeadMotor(self, id, pos, speed): pass
#         def sendSensorReset(self, a, b, c): pass
#         def drawImageFunction(self, *args): pass
#         def sendContinuousValue(self, *args): pass
#         def sendBodyAuto(self, *args): pass

# ORIGIN_THETA = 0
# ORIGIN_SPEED = 4000
# send = Sendmessage()

# class Coordinate: 
#     def __init__(self, x, y):
#         self.x, self.y = x, y
#     def __add__(self, other):
#         return Coordinate((self.x + other.x), (self.y + other.y))
#     def __sub__(self, other):
#         return Coordinate((self.x - other.x), (self.y - other.y))
#     def __floordiv__(self, other):
#         return Coordinate((self.x // other), (self.y // other))

# class MarNode(Node):
#     def __init__(self):
#         super().__init__('mar_strategy_node')
#         self.get_logger().info("MAR Strategy Node (ROS 2) Started")

#         # --- 訂閱 YOLO 節點 (QoS 設為 10) ---
#         qos = QoSProfile(depth=10)
        
#         # 1. 訂閱類別名稱與面積 (String)
#         self.sub_yolo_str = self.create_subscription(
#             String, 
#             'class_id_topic', 
#             self.cb_yolo_string, 
#             qos
#         )
        
#         # 2. 訂閱座標點 (Point)
#         self.sub_yolo_point = self.create_subscription(
#             Point, 
#             'sign_coordinates', 
#             self.cb_yolo_point, 
#             qos
#         )

#         # --- 變數初始化 ---
#         self.yolo_received_label = 'None'
#         self.yolo_received_area = 0.0
#         self.yolo_received_x = 0.0
#         self.yolo_received_y = 0.0
        
#         # 策略變數
#         self.initial_vars()
#         self.seek_line = Seek_line(self) # 傳入 self 以便 Seek_line 使用 logger

#         # --- 建立主迴圈 Timer (20Hz = 0.05s) ---
#         self.timer = self.create_timer(0.05, self.strategy_loop)

#     def initial_vars(self):
#         self.stright_head_flag = False
#         self.avoid_mode = False
#         self.avoid_timer = 0
#         self.speed_x = 0
#         self.speed_y = 0
#         self.theta = 0
#         self.arrow_center = Coordinate(0, 0)
#         self.arrow_temp = deque(['None', 'None', 'None', 'None', 'None'], maxlen=5) 
#         self.search_done = False   
#         self.search_num = 0              
#         self.search_flag = 0             
#         self.head_horizon = 2048         
#         self.head_vertical = 2048        
#         self.reg = 1
        
#         # 狀態機
#         self.status = 'First'
#         self.can_turn_flag = False                                         
#         self.turn_now_flag = False
#         self.arrow_cnt_times = 0
#         self.yaw_temp = 0
        
#         # 硬體初始化指令
#         send.sendHeadMotor(2, 2048, 50)
#         send.sendHeadMotor(1, 2048, 50)
#         send.sendSensorReset(1, 1, 1)

#     # --- Callbacks ---
#     def cb_yolo_string(self, msg):
#         # 格式: "name,cx,bbox_bottom,area"
#         try:
#             data = msg.data.split(',')
#             if len(data) >= 4:
#                 self.yolo_received_label = data[0]
#                 self.yolo_received_area = float(data[3])
#             else:
#                 self.yolo_received_label = 'None'
#         except Exception as e:
#             self.get_logger().warn(f"YOLO String Parse Error: {e}")

#     def cb_yolo_point(self, msg):
#         self.yolo_received_x = msg.x
#         self.yolo_received_y = msg.y

#     # --- 策略邏輯函式 ---

#     def view_search(self, right_place, left_place, up_place, down_place, speed, delay):   
#         # 注意：在 ROS2 Timer 回調中使用 sleep 會阻塞接收 YOLO 訊息
#         # 建議將 delay 設極小，或改寫邏輯移除 sleep
        
#         if self.reg > 0:
#             turn_order = [3, 4, 1, 2]   # 右→上→左→下
#         else:
#             turn_order = [1, 4, 3, 2]   # 左→上→右→下

#         if self.search_num >= len(turn_order):
#             self.search_num = 0

#         self.search_flag = turn_order[self.search_num]
#         # self.get_logger().info(f"搜尋箭頭: Mode {self.search_flag}")

#         # === 檢查是否已經找到箭頭 (來自 YOLO Topic) ===
#         if self.yolo_received_label != 'None' and self.yolo_received_area > 0:
#             self.arrow_center.x = self.yolo_received_x
#             self.arrow_center.y = self.yolo_received_y
            
#             self.get_logger().info(f"找到箭頭 {self.yolo_received_label}，切換到追蹤模式")
#             self.search_done = True
#             self.lock_arrow()
#             return  

#         # === 沒找到 → 繼續掃描 ===
#         if self.search_flag == 1:   # 左尋
#             if self.head_horizon >= left_place:
#                 self.move_head(1, self.head_horizon, 880, 880, speed)
#                 self.head_horizon -= speed
#                 # time.sleep(delay) # 警告：這會卡住接收
#             else:
#                 self.search_num += 1

#         elif self.search_flag == 4: # 上尋
#             if self.head_vertical <= up_place:
#                 self.move_head(2, self.head_vertical, 880, 880, speed)
#                 self.head_vertical += speed
#                 # time.sleep(delay)
#             else:
#                 self.search_num += 1  

#         elif self.search_flag == 3: # 右尋
#             if self.head_horizon <= right_place:
#                 self.move_head(1, self.head_horizon, 880, 880, speed)
#                 self.head_horizon += speed
#                 # time.sleep(delay) 
#             else:
#                 self.search_num += 1      

#         elif self.search_flag == 2: # 下尋
#             if self.head_vertical >= down_place:
#                 self.move_head(2, self.head_vertical, 880, 880, speed)
#                 self.head_vertical -= speed
#                 # time.sleep(delay)   
#             else:
#                 self.search_num = 0
        
#         # 檢查更新
#         self.arrow_yolo()
#         if self.yolo_received_label != 'None':
#              self.search_done = True

#     def move_head(self, ID, Position, head_max_x, head_max_y, Speed):
#         send.sendHeadMotor(ID, Position, Speed)
#         if ID == 1:
#             self.head_horizon = Position
#             if abs(self.head_horizon - 2048) > head_max_x:
#                 self.head_horizon = 2048 + head_max_x if (self.head_horizon - 2048) > 0 else 2048 - head_max_x
#         else:
#             self.head_vertical = Position
#             if abs(self.head_vertical - 2048) > head_max_y:
#                 self.head_vertical = 2048 + head_max_y if (self.head_vertical - 2048) > 0 else 2048 - head_max_y
        
#     def lock_arrow(self, speed=10):
#         """ 讓頭部追蹤箭頭中心 """
#         # 更新資料
#         self.arrow_temp.append(self.yolo_received_label)
#         self.arrow_center.x = self.yolo_received_x
#         self.arrow_center.y = self.yolo_received_y

#         if self.yolo_received_label != 'None':
#             # === 基本參數 ===
#             # 假設 YOLO 解析度為 640x480，中心為 320, 240
#             # 如果是 320x240，請改為 160, 120
#             img_center_x, img_center_y = 160, 120 
            
#             x_diff = self.arrow_center.x - img_center_x
#             y_diff = self.arrow_center.y - img_center_y

#             # === 誤差轉角度 (係數需依實際鏡頭調整) ===
#             x_degree = x_diff * (38.0 / 320.0)
#             y_degree = y_diff * (55.0 / 240.0)

#             horizon_target = 2048 - round(x_degree * 4096 / 360 )
#             vertical_target = 2048 - round(y_degree * 4096 / 360 )

#             send.sendHeadMotor(1, horizon_target, speed)
#             send.sendHeadMotor(2, vertical_target, speed)
            
#             if vertical_target <= 2000:
#                 self.stright_head_flag = True
        
#             # self.get_logger().info(f"[鎖定] X:{self.arrow_center.x} Y:{self.arrow_center.y}")
#         else:
#             pass
#             # self.get_logger().debug("miss_target")

#     def arrow_yolo(self):
#         # 更新座標
#         self.arrow_center.x = self.yolo_received_x
#         self.arrow_center.y = self.yolo_received_y
#         self.arrow_temp.append(self.yolo_received_label)
        
#         arrow_cnt_temp = len(set(self.arrow_temp))
#         if arrow_cnt_temp == 1 and self.arrow_temp[0] != 'None':
#             self.can_turn_flag = False
#             if self.arrow_temp[0] == 'left' or self.arrow_temp[0] == 'right':
#                 self.can_turn_flag = True
            
#             # 可以在這裡呼叫 send.drawImageFunction，但因為是訂閱接收，
#             # 座標轉換要確保正確
#             return True
#         return False

#     def arrow_turn(self):
#         self.yaw = send.imu_value_Yaw
#         # self.get_logger().info(f"Turn Yaw: {self.yaw}")

#         if self.arrow_temp[0] == 'right':
#             # self.get_logger().info('箭頭：右轉')
#             send.sendContinuousValue(2000, 0, 0, -8 + ORIGIN_THETA, 0)
#         elif self.arrow_temp[0] == 'left':
#             # self.get_logger().info('箭頭：左轉')
#             send.sendContinuousValue(2000, 0, 0, 8 + ORIGIN_THETA, 0)
            
#         if abs(self.yaw) > 85:
#             send.sendSensorReset(0, 0, 1)    # imu Reset
#             self.get_logger().info('箭頭轉彎結束')
#             self.turn_now_flag = False
#             self.can_turn_flag = False

#     def stright_avoid(self):
#         self.yaw = send.imu_value_Yaw
#         # self.get_logger().info("開始避開箭頭")
#         self.speed_x = 2000 
        
#         if self.yaw > 0 : 
#             if abs(self.yaw) < 50:
#                 self.theta = -8
#                 send.sendContinuousValue(self.speed_x, -500, 0, ORIGIN_THETA + self.theta, 0)
#             else:
#                 send.sendSensorReset(0, 0, 1) 
#                 self.get_logger().info("✅ 避開完成")
                
#         elif self.yaw < 0: 
#             if abs(self.yaw) < 50:
#                 self.theta = 8
#                 send.sendContinuousValue(self.speed_x, 500, 0, ORIGIN_THETA + self.theta, 0)
#             else:
#                 send.sendSensorReset(0, 0, 1) 
#                 self.get_logger().info("✅ 避開完成")
                
#         self.stright_head_flag = False
#         self.avoid_mode = False

#     def imu_go(self):
#         self.yaw = send.imu_value_Yaw
#         self.speed_x = 2000
#         self.arrow_temp.append(self.yolo_received_label)

#         # 修正模式
#         if 0 < self.arrow_center.x <= 100:
#             self.theta = 5
#             send.sendContinuousValue(self.speed_x, 100, 0, self.theta + ORIGIN_THETA, 0)
#         elif self.arrow_center.x >= 220:
#             self.theta = -5
#             send.sendContinuousValue(self.speed_x, 100, 0, self.theta + ORIGIN_THETA, 0)
#         else:
#             if self.yaw > 5:
#                 self.theta = -3 + ORIGIN_THETA
#             elif self.yaw < -5:
#                 self.theta = 3 + ORIGIN_THETA
    
#         send.sendContinuousValue(self.speed_x, 100, 0, self.theta, 0)

#     # --- 主策略迴圈 (替代原本的 main + while) ---
#     def strategy_loop(self):
#         # 檢查機器人是否啟動
#         if send.is_start:
#             # 狀態機
#             if self.status == 'First':
#                 self.initial_vars()
#                 # 這裡不能用 time.sleep(1) 會卡住，建議改用計數器或一次性旗標
#                 # 暫時為了相容邏輯保留，但這在 ROS2 會有風險
#                 # time.sleep(1) 
                
#                 send.sendBodyAuto(0, 0, 0, 0, 1, 0)
                
#                 # 根據 DIO 判斷進入哪一關
#                 if send.DIOValue == 48:
#                     self.status = 'Arrow_Part'
#                 else:
#                     self.status = 'Arrow_Part'
                
#                 send.sendHeadMotor(2, 2048, 50)
#                 self.search_done = False
#                 self.get_logger().info(f"進入狀態: {self.status}")

#             elif self.status == 'Arrow_Part':
#                 if not self.search_done:
#                     # 搜尋 (注意這裡去掉了 time.sleep，動作會變快，需要調整 speed)
#                     self.view_search(2700, 1400, 2500, 1500, 20, 0.0)
#                 else :
#                     self.lock_arrow()
                    
#                     if self.arrow_temp[0] == 'stright': 
#                         self.avoid_mode = True  

#                     if self.turn_now_flag:
#                         self.arrow_turn()
#                     elif self.avoid_mode and self.stright_head_flag :
#                         self.stright_avoid()
#                     else: 
#                         self.arrow_yolo()
                        
#                         if self.can_turn_flag:
#                             # 這裡要確認 Y 軸閾值 (140) 是否符合 YOLO 座標系
#                             if self.arrow_center.y >= 140:
#                                 self.arrow_cnt_times += 1
#                             if self.arrow_cnt_times >= 10:
#                                 self.turn_now_flag = True
#                                 self.arrow_cnt_times = 0
#                                 self.get_logger().info("準備轉彎!")
                    
#                         self.imu_go()    
#         else:
#             # 停止狀態
#             if self.status != 'First':
#                 self.status = 'First'
#                 send.sendBodyAuto(0, 0, 0, 0, 1, 0) # 關閉

# class Seek_line:
#     def __init__(self, node):
#         self.node = node # 為了使用 logger
#         self.upper_center = Coordinate(0, 0)
#         self.lower_center = Coordinate(0, 0)

#     def cvt_list2d2numpy(self, list2d):
#         max_len = max([len(sub_lst) for sub_lst in list2d])
#         np_array = np.vstack([np.pad(np.array(lst), (0, (max_len - len(lst)))) for lst in list2d])
#         return np_array

#     def update(self):
#         # 這裡的邏輯保持不變，但請確認 send.color_mask 相關變數在 ROS2 環境下是否能正常更新
#         try:
#             img_size = self.cvt_list2d2numpy(send.color_mask_subject_size)
#             img_xmin = self.cvt_list2d2numpy(send.color_mask_subject_XMin)
#             img_xmax = self.cvt_list2d2numpy(send.color_mask_subject_XMax)
#             img_ymin = self.cvt_list2d2numpy(send.color_mask_subject_YMin)
#             img_ymax = self.cvt_list2d2numpy(send.color_mask_subject_YMax)

#             filter_img_size = img_size > 380
#             has_object = filter_img_size.any()
#             send.data_check = False
            
#             if not has_object:
#                 self.upper_center.x, self.upper_center.y = 0, 0
#                 self.lower_center.x, self.lower_center.y = 0, 0
#                 return
            
#             img_xmin_new = int(img_xmin[filter_img_size].min()) 
#             img_xmax_new = int(img_xmax[filter_img_size].max()) 
#             img_ymin_new = int(img_ymin[filter_img_size].min())
#             img_ymax_new = int(img_ymax[filter_img_size].max())
            
#             # send.drawImageFunction(7, 1, img_xmin_new, img_xmax_new, img_ymin_new, img_ymax_new, 255, 0, 255)
            
#             img_data = np.frombuffer(send.Label_Model, dtype = np.uint8)
#             img_data = img_data.reshape(240, 320)
#             img_data = img_data[img_ymin_new : img_ymax_new, img_xmin_new : img_xmax_new]
            
#             y_coord, x_coord = np.where(img_data != 0)

#             if len(x_coord) == 0:
#                 self.upper_center.x, self.upper_center.y = 0, 0
#                 self.lower_center.x, self.lower_center.y = 0, 0
#                 return
            
#             middle_y = (np.max(y_coord) + np.min(y_coord)) // 2 
#             upper_filter = y_coord <= middle_y
#             upper_x, upper_y = x_coord[upper_filter], y_coord[upper_filter]
#             if len(upper_x) > 0:
#                 self.upper_center.x = np.mean(upper_x) + img_xmin_new 
#                 self.upper_center.y = np.mean(upper_y) + img_ymin_new

#             lower_filter = y_coord > middle_y
#             lower_x, lower_y = x_coord[lower_filter], y_coord[lower_filter]
#             if len(lower_x) > 0:
#                 self.lower_center.x = np.mean(lower_x) + img_xmin_new
#                 self.lower_center.y = np.mean(lower_y) + img_ymin_new
            
#             # send.drawImageFunction(2, 0, int(self.upper_center.x), int(self.lower_center.x), int(self.upper_center.y), int(self.lower_center.y), 0, 0, 0)
#         except Exception as e:
#             self.node.get_logger().error(f"Seek_line error: {e}")

#     def calculate_slope(self):
#         delta = self.upper_center - self.lower_center
#         if delta.x == 0:
#             return float("inf")
#         return delta.y / delta.x

# def main(args=None):
#     rclpy.init(args=args)
#     node = MarNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
# coding=utf-8

import rclpy
from collections import deque
import time
from API import API 
from tku_msgs.msg import Dio

# === 全域常數 ===
ORIGIN_THETA = 0
HEAD_CENTER_X = 2048
HEAD_CENTER_Y = 2048

class Coordinate:
    def __init__(self, x, y):
        self.x, self.y = x, y

class Mar(API):
    def __init__(self):
        # 初始化父類別 API ('mar_strategy' 為節點名稱)
        super().__init__('mar_strategy')
        
        # === 變數初始化 ===
        self.DIOValue = 0            # 儲存 DIO 數值
        self.status = 'First'
        
        # 狀態旗標
        self.straight_head_flag = False
        self.avoid_mode = False
        self.search_done = False
        self.can_turn_flag = False
        self.turn_now_flag = False
        
        # 運動參數
        self.speed_x = 0
        self.theta = 0
        self.yaw = 0.0               # 當前機器人角度
        self.arrow_cnt_times = 0
        
        # 視覺參數
        self.arrow_center = Coordinate(0, 0)
        self.arrow_temp = deque(['None']*5, maxlen=5)
        
        # 搜尋參數
        self.search_num = 0             
        self.search_flag = 0            
        self.head_horizon = HEAD_CENTER_X         
        self.head_vertical = HEAD_CENTER_Y        
        self.reg = 1

        # === 建立計時器 (20Hz) ===
        # 使用 Timer 取代 while 迴圈，避免阻塞 ROS 通訊
        self.create_timer(0.05, self.main_loop) 
        
        self.get_logger().info("Mar Strategy Node Initialized")

    # === 硬體回調：抓取 DIO 數值 ===
    def _dio_callback(self, msg: Dio):
        super()._dio_callback(msg) # 保持 API 父類別的功能 (更新 is_start)
        
        # 嘗試從 msg 取得數值 (相容 data 或 value 欄位)
        if hasattr(msg, 'data'):
            self.DIOValue = msg.data
        elif hasattr(msg, 'value'):
            self.DIOValue = msg.value
        else:
            pass # 如果格式不符，暫不更新

    # === 初始化參數 ===
    def initial(self):
        self.can_turn_flag = False                             
        self.turn_now_flag = False
        self.arrow_cnt_times = 0
        self.sendSensorReset(True) # Reset IMU (API 方法)

    # === 視覺邏輯：取得箭頭資訊 ===
    def get_arrow_info(self):
        # 使用 API 的 get_objects 功能
        # 請確認 'orange' 是否正確對應你的模型設定，或是改為 'person', 'sign' 等類別名
        objects = self.get_objects('orange') 
        
        if not objects: 
            return 'None', 0, 0
            
        # 找出面積最大的物件 (bbox 格式: x, y, w, h)
        target = max(objects, key=lambda x: x['bbox'][2] * x['bbox'][3])
        
        raw_label = target.get('label', 'unknown')
        x, y, w, h = target['bbox']
        
        # 回傳: 標籤, 中心X, 中心Y
        return raw_label, int(x + w / 2), int(y + h / 2)

    # === 頭部控制安全包裝 ===
    def move_head_safe(self, ID, pos, speed):
        self.sendHeadMotor(ID, int(pos), speed)

    # === 搜尋策略 ===
    def view_search(self, right_place, left_place, up_place, down_place, speed, delay):
        turn_order = [3, 4, 1, 2] if self.reg > 0 else [1, 4, 3, 2]
        
        if self.search_num >= len(turn_order): 
            self.search_num = 0
            
        self.search_flag = turn_order[self.search_num]
        
        # 每次搜尋移動前，先看一眼有沒有找到
        label, cx, cy = self.get_arrow_info()
        if label != 'None':
            self.search_done = True
            self.arrow_center.x = cx
            self.arrow_center.y = cy
            self.lock_arrow() # 立即鎖定
            return

        # 執行搜尋動作
        if self.search_flag == 1: # 左
            if self.head_horizon >= left_place: self.head_horizon -= speed
            else: self.search_num += 1
        elif self.search_flag == 4: # 上
            if self.head_vertical <= up_place: self.head_vertical += speed
            else: self.search_num += 1  
        elif self.search_flag == 3: # 右
            if self.head_horizon <= right_place: self.head_horizon += speed
            else: self.search_num += 1      
        elif self.search_flag == 2: # 下
            if self.head_vertical >= down_place: self.head_vertical -= speed
            else: self.search_num = 0
        
        self.move_head_safe(1, self.head_horizon, speed)
        self.move_head_safe(2, self.head_vertical, speed)

    # === 鎖定箭頭 (追蹤) ===
    def lock_arrow(self, speed=20):
        if self.arrow_center.x != 0:
            # 計算誤差 (假設影像中心為 160, 120)
            x_diff = self.arrow_center.x - 160
            y_diff = self.arrow_center.y - 120
            
            # 轉換為馬達刻度
            horizon_target = HEAD_CENTER_X - int(x_diff * (38.0/320.0) * 4096 / 360)
            vertical_target = HEAD_CENTER_Y - int(y_diff * (55.0/240.0) * 4096 / 360)
            
            self.sendHeadMotor(1, horizon_target, speed)
            self.sendHeadMotor(2, vertical_target, speed)
            
            # 更新當前頭部位置紀錄
            self.head_horizon = horizon_target
            self.head_vertical = vertical_target
            
            # 如果頭低到一定程度，判定為接近目標
            self.straight_head_flag = True if vertical_target <= 2000 else False

    # === 處理 YOLO 決策邏輯 ===
    def process_yolo_logic(self):
        label, cx, cy = self.get_arrow_info()
        
        # 更新全域變數
        self.arrow_center.x = cx
        self.arrow_center.y = cy
        self.arrow_temp.append(label)
        
        # 檢查連續 5 次是否一致
        unique = set(self.arrow_temp)
        if len(unique) == 1 and label != 'None':
            # 判斷是否為轉彎類別
            self.can_turn_flag = True if label in ['left', 'right'] else False
            return True
        return False

    # === 動作：轉彎 ===
    def arrow_turn(self):
        target = self.arrow_temp[0]
        self.get_logger().info(f"執行轉彎: {target}")
        
        theta = -8 if target == 'right' else 8
        self.sendContinuousValue(2000, 0, theta + ORIGIN_THETA)
        
        # 檢查轉彎角度 (yaw)
        if abs(self.yaw) > 85:
            self.sendSensorReset(True)
            self.turn_now_flag = False
            self.can_turn_flag = False
            self.arrow_temp.clear() # 清除歷史紀錄，避免重複觸發

    # === 動作：避障 ===
    def straight_avoid(self):
        self.speed_x = 2000
        # 根據 Yaw 進行 S 型避障
        if self.yaw > 0:
            if abs(self.yaw) < 50: 
                self.sendContinuousValue(self.speed_x, -500, -8)
            else: 
                self.sendSensorReset(True)
                self.avoid_mode = False
                self.get_logger().info("避障完成 (右)")
        elif self.yaw < 0:
            if abs(self.yaw) < 50: 
                self.sendContinuousValue(self.speed_x, 500, 8)
            else: 
                self.sendSensorReset(True)
                self.avoid_mode = False
                self.get_logger().info("避障完成 (左)")
        
        self.straight_head_flag = False

    # === 動作：IMU 直走修正 ===
    def imu_go(self):
        self.speed_x = 2000
        # 1. 視覺修正 (如果箭頭偏左或偏右)
        if 0 < self.arrow_center.x <= 100:
            self.sendContinuousValue(self.speed_x, 100, 5 + ORIGIN_THETA)
            return
        elif self.arrow_center.x >= 220:
            self.sendContinuousValue(self.speed_x, 100, -5 + ORIGIN_THETA)
            return
        
        # 2. IMU 修正 (保持直線)
        theta_fix = -3 if self.yaw > 5 else 3 if self.yaw < -5 else 0
        self.sendContinuousValue(self.speed_x, 100, theta_fix + ORIGIN_THETA)

    # ==========================================
    #                主 邏 輯 區
    # ==========================================
    def main_loop(self):
        # [重要] 確保 self.yaw 隨時與 API 接收到的 IMU 數值同步
        # 假設父類別 API 有 imu_value_Yaw 這個屬性
        if hasattr(self, 'imu_value_Yaw'):
            self.yaw = self.imu_value_Yaw
        
        # 1. 最外層：檢查 API 的總開關 (is_start)
        if self.is_start:
            
            # 2. 狀態：First (初始化與等待訊號)
            if self.status == 'First':
                self.initial()
                
                # 發送 0 代表原地踏步或準備 (參數 x, y, theta)
                self.sendBodyAutoCmd(0, 0, 0) 
                
                # 初始化頭部位置
                self.sendHeadMotor(2, 2048, 50)
                self.sendHeadMotor(1, 2048, 50)
                self.search_done = False
                
                # 等待 DIO 訊號 48
                if self.DIOValue == 48:
                    self.get_logger().info("收到訊號 48，切換至 Arrow_Part")
                    self.status = 'Arrow_Part'
                else:
                    self.get_logger().info(f"等待啟動訊號 (DIO: {self.DIOValue})", throttle_duration_sec=1)

            # 3. 狀態：Arrow_Part (邏輯核心)
            elif self.status == 'Arrow_Part':
                
                if not self.search_done:
                    self.view_search(2700, 1400, 2500, 1500, 20, 0.1)
                else:
                    self.lock_arrow()
                    has_arrow = self.process_yolo_logic()
                    label = self.arrow_temp[0]

                    # 判斷是否進入避障模式 (straight)
                    if has_arrow and label in ['stright', 'straight']:
                        self.avoid_mode = True

                    # 狀態機執行優先順序
                    if self.turn_now_flag:
                        self.arrow_turn()
                    elif self.avoid_mode and self.straight_head_flag:
                        self.straight_avoid()
                    else:
                        # 轉彎判斷 (接近時)
                        if self.can_turn_flag:
                            if self.arrow_center.y >= 140: # 接近判斷閾值
                                self.arrow_cnt_times += 1
                            if self.arrow_cnt_times >= 10:
                                self.turn_now_flag = True
                                self.arrow_cnt_times = 0
                        
                        # 平常直走
                        self.imu_go()
        
        else:
            # 總開關被關閉時的行為
            if self.status != 'First':
                self.get_logger().info("機器人停止")
                self.status = 'First'
                self.sendBodyAutoCmd(0, 0, 0) # 停止動作

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