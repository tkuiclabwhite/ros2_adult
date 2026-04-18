#!/usr/bin/env python3
# coding=utf-8

import rclpy
from collections import deque
import time
from strategy.API import API
from tku_msgs.msg import Dio
from std_msgs.msg import String  
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tku_msgs.msg import SensorPackage

# === 全域設定 ===
HEAD_CENTER_X = 2048
HEAD_CENTER_Y = 2048
ORIGIN_THETA = 0

# === 參數設定區 ===
AREA_THRESHOLD = 2000    # 面積閥值
TURN_TARGET_ANGLE = 85    # 轉彎目標角度
SHIFT_RIGHT_TIME = 2.0    # 直走模式：向右平移時間
GO_STRAIGHT_TIME = 3.0    # 直走模式：直走穿越時間
LEAVE_WALK_TIME = 2.0     # 離開路口時間
LOST_TARGET_TIMEOUT = 5.0 # [新增] 沒看到標誌多久後才開始擺頭 (秒)

class Coordinate:
    def __init__(self, x, y):
        self.x, self.y = x, y

class Mar(API):
    def __init__(self):

        # 1. 先執行父類別初始化 (這時候會建立舊的、連不上的訂閱)
        super().__init__('mar_strategy')
        
        # === [關鍵修正] 在這裡直接覆寫 IMU 訂閱設定 ===
        # 說明：因為 API_node.py 預設用 Reliable，但底層可能是 BestEffort
        # 所以我們把舊的砍掉，自己建一個 BestEffort 的來連線。
        
        # (1) 銷毀舊訂閱
        if self.imu_sub:
            self.destroy_subscription(self.imu_sub)
            
        # (2) 設定正確的 QoS (Best Effort)
        qos_policy = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        # (3) 重新建立訂閱 (綁定原本的 self.imu 函式)
        self.imu_sub = self.create_subscription(
            SensorPackage, 
            '/package/sensorpackage', 
            self.imu, 
            qos_policy,
            callback_group=self.imu_cbg
        )
        self.get_logger().info("已在 mar1.py 中強制修正 IMU 訂閱為 Best Effort")
        # ============================================
        
        
        self.is_start = True
        self.status = 'First'
        self.sub_state = 'SEARCH'
        
        self.arrow_center = Coordinate(0, 0)
        self.current_area = 0
        self.target_label = 'None'
        
        self.search_num = 0             
        self.search_flag = 0            
        self.head_horizon = HEAD_CENTER_X         
        self.head_vertical = HEAD_CENTER_Y        
        self.reg = 1

        self.action_start_time = 0
        self.last_yolo_time = 0 # 記錄最後一次看到目標的時間
        self.latest_yolo_data = None 

        self.yolo_sub = self.create_subscription(String, 'class_id_topic', self.yolo_callback, 10)
        self.create_timer(0.05, self.main_loop) 
        
        self.get_logger().info("Mar Strategy Initialized (Smart Search Mode)")

    def _sync_start_from_param(self): pass

    def _dio_callback(self, msg: Dio):
        if hasattr(msg, 'data'): self.DIOValue = msg.data
        elif hasattr(msg, 'value'): self.DIOValue = msg.value

    def yolo_callback(self, msg):
        try:
            parts = msg.data.split(',')
            if len(parts) == 4:
                self.latest_yolo_data = {
                    'label': parts[0],      
                    'cx': int(parts[1]),    
                    'cy': int(parts[2]),    
                    'area': float(parts[3]) 
                }
                # 只要有收到數據，就更新「最後看到的時間」
                self.last_yolo_time = time.time() 
        except Exception as e:
            self.get_logger().error(f"YOLO Parse Error: {e}")

    def get_arrow_info(self):
        # 這裡的判斷是為了 filter 掉「瞬間」的丟失 (0.5s)
        # 但在 view_search 我們會用更長的時間 (5s) 來判斷是否要擺頭
        if time.time() - self.last_yolo_time > 0.5 or self.latest_yolo_data is None:
            return 'None', 0, 0, 0
        d = self.latest_yolo_data
        return d['label'], d['cx'], d['cy'], d['area']

    def initial(self):
        self.sendSensorReset(True) 
        # 重置時頭部回正
        self.head_horizon = HEAD_CENTER_X
        self.head_vertical = HEAD_CENTER_Y
        self.sendHeadMotor(1, self.head_horizon, 50)
        self.sendHeadMotor(2, self.head_vertical, 50)
        self.get_logger().info("SET initial")
        self.sub_state = 'SEARCH'

    # === [修改] 智慧搜尋模式 ===
    def view_search(self, right_place, left_place, speed):
        # 1. 身體停止
        self.sendContinuousValue(0, 0, 0)
        
        # 2. 檢查有沒有看到東西
        label, cx, cy, area = self.get_arrow_info()
        
        if label != 'None':
            self.target_label = label
            self.arrow_center.x = cx; self.arrow_center.y = cy
            self.current_area = area
            self.get_logger().info(f"[Search] 找到目標: {label} -> 開始接近")
            
            # 找到後頭部回正 (為了讓身體對準邏輯接手)
            self.sendHeadMotor(1, HEAD_CENTER_X, 50)
            self.sendHeadMotor(2, HEAD_CENTER_Y, 50)
            
            self.sub_state = 'APPROACH'
            return

        # 3. [新增] 判斷是否「剛跟丟」或「還在等待範圍內」
        time_since_lost = time.time() - self.last_yolo_time
        
        if time_since_lost < LOST_TARGET_TIMEOUT:
            # 尚未超過 5 秒 -> 保持頭部置中，不做擺頭，耐心等待
            self.get_logger().info(f"等待目標出現... ({time_since_lost:.1f}s / {LOST_TARGET_TIMEOUT}s)", throttle_duration_sec=1.0)
            self.head_horizon = HEAD_CENTER_X
            self.head_vertical = HEAD_CENTER_Y
            self.sendHeadMotor(1, self.head_horizon, 50)
            self.sendHeadMotor(2, self.head_vertical, 50)
            return

        # 4. 超過 5 秒都沒看到 -> 開始左右擺頭
        self.get_logger().info("超時未發現目標 -> 啟動左右掃描", throttle_duration_sec=1.0)
        
        turn_order = [3, 1] if self.reg > 0 else [1, 3]
        if self.search_num >= len(turn_order): self.search_num = 0
        self.search_flag = turn_order[self.search_num]
        
        if self.search_flag == 1: 
            if self.head_horizon >= left_place: self.head_horizon -= speed
            else: self.search_num += 1
        elif self.search_flag == 3: 
            if self.head_horizon <= right_place: self.head_horizon += speed
            else: self.search_num += 1

        self.head_vertical = HEAD_CENTER_Y
        self.sendHeadMotor(1, self.head_horizon, speed)
        self.sendHeadMotor(2, self.head_vertical, speed)

    # === 身體對準邏輯 ===
    def body_align_and_approach(self):
        self.sendHeadMotor(1, HEAD_CENTER_X, 50)
        self.sendHeadMotor(2, HEAD_CENTER_Y, 50)
        
        center_x = 160
        deadband = 20
        speed_x = 2000 
        theta = 0      
        
        if self.arrow_center.x < (center_x - deadband):
            theta = 3
        elif self.arrow_center.x > (center_x + deadband):
            theta = -3
        else:
            theta = 0
            if self.yaw > 5: theta = -2
            elif self.yaw < -5: theta = 2

        self.sendContinuousValue(speed_x, 0, theta + ORIGIN_THETA)

    # === 執行動作邏輯 ===
    def execute_action_logic(self):
        label = self.target_label
        
        if label in ['left', 'right']:
            target_yaw = TURN_TARGET_ANGLE if label == 'left' else -TURN_TARGET_ANGLE
            current_yaw = self.imu_rpy[2]
            
            is_finish = False
            if label == 'left' and current_yaw >= target_yaw: is_finish = True
            if label == 'right' and current_yaw <= target_yaw: is_finish = True
            
            if is_finish:
                self.get_logger().info(f"[Action] {label} 完成")
                self.sendSensorReset(True) 
                self.action_start_time = time.time()
                self.sub_state = 'LEAVE'
            else:
                turn_speed = 8 if label == 'left' else -8
                self.sendContinuousValue(0, 0, turn_speed)  #轉彎
                self.get_logger().info(f"轉彎中 Yaw:{current_yaw:.1f}", throttle_duration_sec=0.5)

        elif label in ['straight', 'stright']:
            elapsed = time.time() - self.action_start_time
            if elapsed < SHIFT_RIGHT_TIME:
                self.sendContinuousValue(0, -1500, 0) 
                self.get_logger().info("向右平移中", throttle_duration_sec=0.5)
            elif elapsed < (SHIFT_RIGHT_TIME + GO_STRAIGHT_TIME):
                self.sendContinuousValue(2500, 0, 0)
                self.get_logger().info("穿越中", throttle_duration_sec=0.5)
            else:
                self.get_logger().info("直走策略完成")
                self.sendSensorReset(True)
                
                # [修改] 直走完成後，也要把頭回正，方便 SEARCH 狀態一開始先看前面
                self.head_horizon = HEAD_CENTER_X
                self.head_vertical = HEAD_CENTER_Y
                self.sendHeadMotor(1, self.head_horizon, 50)
                self.sendHeadMotor(2, self.head_vertical, 50)
                
                self.sub_state = 'SEARCH'

    # === 主邏輯 ===
    def main_loop(self):
        if not self.is_start:
            return

        if self.status == 'First':
            self.initial()
            self.sendbodyAuto(1)
            # self.sendContinuousValue(0, 0, 0)
            self.status = 'Arrow_Part'
            self.sub_state = 'SEARCH'
            self.get_logger().info("=== 系統啟動 ===")
            # 初始化一下時間，避免剛開機就直接擺頭
            self.last_yolo_time = time.time()

        elif self.status == 'Arrow_Part':
            
            label, cx, cy, area = self.get_arrow_info()
            if label != 'None':
                self.arrow_center.x = cx
                self.arrow_center.y = cy
                self.current_area = area
                if self.sub_state == 'SEARCH' or label == self.target_label:
                    self.target_label = label

            if self.sub_state == 'SEARCH':
                self.view_search(2700, 1400, 20)
                
            elif self.sub_state == 'APPROACH':
                if self.current_area > AREA_THRESHOLD:
                    self.get_logger().info(f"到達目標 Area:{self.current_area:.0f}")
                    self.sendContinuousValue(0, 0, 0)
                    self.sendSensorReset(True) 
                    self.action_start_time = time.time()
                    self.sub_state = 'ACTION'
                else:
                    self.body_align_and_approach()
                    self.get_logger().info(f"接近中 | Area:{self.current_area:.0f}", throttle_duration_sec=0.5)

            elif self.sub_state == 'ACTION':
                self.execute_action_logic()

            elif self.sub_state == 'LEAVE':
                elapsed = time.time() - self.action_start_time
                if elapsed < LEAVE_WALK_TIME:
                    self.sendContinuousValue(2500, 0, 0)
                    self.get_logger().info("離開路口中", throttle_duration_sec=0.5)
                else:
                    self.get_logger().info("離開完成 -> 重新搜尋")
                    self.initial()

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