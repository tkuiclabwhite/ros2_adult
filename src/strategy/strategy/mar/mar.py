import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from collections import deque, Counter
import tkinter as tk
from tkinter import ttk
import threading
import sys
import os
import yaml
from pathlib import Path

# 導入機器人控制 API
try:
    from strategy.API import API
    from tku_msgs.msg import Dio
    from tku_msgs.msg import SensorPackage
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("[WARNING] strategy.API or tku_msgs not found. Robot control will be disabled.")


# ============================================================================
# P Controller Class (純比例控制，避免積分飽和)
# ============================================================================
class PController:
    """純比例控制器（避免 I/D 項的問題）"""
    def __init__(self, Kp=1.0, output_min=-100, output_max=100, deadband=0):
        self.Kp = Kp
        self.output_min = output_min
        self.output_max = output_max
        self.deadband = deadband  # 死區
        self.is_angle_mode = False
    
    def set_angle_mode(self, enabled=True):
        """啟用角度模式（處理 -180~180 環繞）"""
        self.is_angle_mode = enabled
    
    def normalize_angle(self, angle):
        """標準化角度到 [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def compute(self, target, current):
        """計算 P 控制輸出"""
        # 計算誤差
        error = target - current
        if self.is_angle_mode:
            error = self.normalize_angle(error)
        
        # 死區處理
        if abs(error) < self.deadband:
            return 0.0
        
        # P 控制
        output = self.Kp * error
        
        # 限幅
        output = max(min(output, self.output_max), self.output_min)
        
        return output
    
    def reset(self):
        """重置（P 控制不需要，但保留介面一致性）"""
        pass


# ============================================================================
# YOLO Test Node (ROS2 節點)
# ============================================================================
class YOLOTestNode(Node):
    def __init__(self, gui):
        super().__init__('yolo_test_node')
        
        self.gui = gui
        
        # === 初始化機器人控制 API ===
        self.api = None
        if API_AVAILABLE:
            try:
                self.api = API('yolo_strategy_api')
                self.api_mode = True
                self.get_logger().info('[API] Using strategy.API control')
                try:
                    self.api.sendSensorReset()
                    self.get_logger().info('[IMU] Sensor calibrated ✓')
                except:
                    pass  # 忽略校準失敗
            except Exception as e:
                self.api_mode = False
                self.get_logger().error(f'[API] Failed to initialize API: {e}')
        else:
            self.api_mode = False
            self.get_logger().warn('[API] Robot control API not available (Test mode)')
        
        # === 機器人運動控制參數（策略只改這些數值）===
        self.target_speed_x = 0
        self.target_speed_y = 0
        self.target_theta = 0
        self.target_head_horizontal = 2048
        self.target_head_vertical = 2048
        self.target_head_speed = 50
        
        # === 機器人行走狀態 ===
        self.robot_walking = False
        self.start_warning_shown = False
        
        # === IMU 資料 ===
        self.current_yaw = 0.0
        self.target_yaw = None
        self.yaw_tolerance = 5.0  # 角度容忍範圍 (度)
        self.imu_available = False
        self.last_imu_time = None
        self.imu_watchdog_timeout = 0.5  # 0.5秒沒更新就視為斷線
        self.imu_data_source = "API"  # "API" 或 "TOPIC"
        
        # === 影像與偵測資訊 ===
        self.sign_name = None
        self.sign_area = 0
        self.center_x = 0
        self.center_y = 0
        
        # === 偵測超時機制 (針對 3-4Hz 優化) ===
        self.last_detection_time = None
        self.detection_timeout = 0.8  # 0.8秒內沒收到新偵測就視為遺失目標（給2-3幀緩衝）
        
        self.img_center_x = 160
        self.img_center_y = 120
        self.align_threshold = 20
        
        # === P-Control 對齊控制器 ===
        self.align_controller = PController(
            Kp=3.0,  # 初始值，會從 GUI 動態讀取
            output_min=-800,
            output_max=800,
            deadband=5
        )
        
        # === 狀態機變數 ===
        self.current_state = '[SEARCH] Searching for signs...'
        self.head_searching = True
        self.alignment_done = False
        self.head_search_step = 0
        
        # === 彈性穩定性檢測參數 ===
        self.STABILITY_WINDOW = 5      # 觀察窗口大小(最近5次)
        self.STABILITY_THRESHOLD = 3   # 需要至少3次相同才算穩定
        self.arrow_temp = deque(maxlen=self.STABILITY_WINDOW)
        self.arrow_stable = False
        self.stable_sign = None        # 當前穩定的號誌類型
        
        # === 動作執行計時器 ===
        self.action_executing = False
        self.action_start_time = None
        self.action_duration = 0.0
        self.action_timeout = 5.0  # IMU 控制超時時間
        self.current_action_name = ""
        self.use_imu_control = True  # 是否使用 IMU 控制
        
        # === 三段式閃避狀態 ===
        self.avoid_step = 0  # 0=未開始, 1=右移, 2=直走, 3=左移
        self.avoid_step_start_time = None
        
        # === 基礎運動參數 ===
        self.ORIGIN_THETA = 0
        self.BASE_SPEED = 2000
        self.STRAFE_SPEED = 400
        self.TURN_SPEED = 8  # 轉彎角速度
        
        # === 建立訂閱 ===
        self.class_id_sub = self.create_subscription(
            String, 
            'class_id_topic', 
            self.class_id_callback, 
            10
        )
        
        self.coord_sub = self.create_subscription(
            Point,
            'sign_coordinates',
            self.coord_callback,
            10
        )
        
        # === 嘗試訂閱 IMU Topic（如果存在）===
        try:
            # TODO: 下週確認實際的 msg 型別和 topic 名稱
            self.imu_sub = self.create_subscription(
                SensorPackage,  # 暫時用 String，下週改成實際型別
                '/package/sensorpackage',  # 確認實際 topic 名稱
                self.imu_topic_callback,
                10
            )
            self.imu_data_source = "TOPIC"
            self.get_logger().info('[IMU] Subscribed to IMU topic')
        except:
            self.get_logger().warn('[IMU] No IMU topic, will use API polling')
        
        # === 主控制迴圈 (100Hz) - 持續發送運動指令 ===
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        # === 策略更新迴圈 (10Hz) - 更新策略狀態 ===
        self.strategy_timer = self.create_timer(0.1, self.strategy_update)
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('   YOLO Sign Action Node Started (P-Control)')
        self.get_logger().info(f'   Stability: {self.STABILITY_THRESHOLD}/{self.STABILITY_WINDOW} detections')
        if API_AVAILABLE:
            self.get_logger().info('   Robot Control: ENABLED (strategy.API)')
        else:
            self.get_logger().info('   Robot Control: DISABLED (Test Mode)')
        self.get_logger().info('=' * 50 + '\n')

    def normalize_angle(self, angle):
        """將角度標準化到 [-180, 180] 範圍"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def angle_difference(self, target, current):
        """計算兩個角度之間的最短差異"""
        diff = self.normalize_angle(target - current)
        return diff

    def imu_topic_callback(self, msg):
        """從 Topic 接收 IMU（下週實作）"""
        # TODO: 下週確認 msg 的格式，例如：
        # yaw = msg.orientation.z  # 或其他欄位
        # self.update_imu(yaw)
        self.update_imu(msg.yaw)
        
        pass

    def update_imu(self, yaw):
        """更新 IMU 數據"""
        self.current_yaw = yaw
        self.last_imu_time = self.get_clock().now().nanoseconds / 1e9
        
        # 第一次收到資料時設為可用
        if not self.imu_available:
            self.imu_available = True
            self.get_logger().info(f'[IMU] Connected! Current yaw: {yaw:.1f}°')

    def is_imu_healthy(self):
        """動態檢查 IMU 健康狀態（Watchdog）"""
        if self.last_imu_time is None:
            return False
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.last_imu_time
        
        # 超時判定
        if elapsed > self.imu_watchdog_timeout:
            if self.imu_available:  # 狀態改變時記錄
                self.get_logger().warn(f'[IMU] Watchdog timeout! ({elapsed:.2f}s)')
                self.imu_available = False
            return False
        
        return True

    def start_robot_walking(self):
        """啟動機器人行走"""
        if not self.api_mode:
            if not self.start_warning_shown:
                self.get_logger().warn('[ROBOT] API not available - Cannot start walking')
                self.start_warning_shown = True
            return
            
        if not self.robot_walking:
            try:
                self.send_body_auto(1)
                self.robot_walking = True
                self.get_logger().info('[ROBOT] Walking started (sendbodyAuto(1))')
            except Exception as e:
                self.get_logger().error(f'[ROBOT] Failed to start walking: {e}')
    
    def stop_robot_walking(self):
        """停止機器人行走"""
        if not self.api_mode:
            return
            
        if self.robot_walking:
            try:
                self.send_body_auto(0)
                self.robot_walking = False
                self.get_logger().info('[ROBOT] Walking stopped (sendbodyAuto(0))')
            except Exception as e:
                self.get_logger().error(f'[ROBOT] Failed to stop walking: {e}')

    def class_id_callback(self, msg):
        """接收偵測結果"""
        try:
            parts = msg.data.split(',')
            if len(parts) == 4:
                self.sign_name = parts[0]
                self.center_x = int(parts[1])
                bbox_y_max = int(parts[2])
                self.sign_area = int(parts[3])
                
                # 更新最後偵測時間
                self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
                
                if self.sign_name != 'None' and self.head_searching:
                    self.head_searching = False
                    self.get_logger().info('[HEAD] Sign detected, stop head searching')
                
                # 加入歷史記錄
                self.arrow_temp.append(self.sign_name)
                
                # 檢查穩定性
                self.check_stability()
                
        except Exception as e:
            self.get_logger().error(f'Parse message failed: {e}')

    def coord_callback(self, msg):
        """接收座標資訊"""
        self.center_x = int(msg.x)
        self.center_y = int(msg.y)

    def check_stability(self):
        """彈性穩定性檢測: 5次內有3-4次相同即可"""
        
        # 過濾掉 'None' 後統計
        valid_detections = [sign for sign in self.arrow_temp if sign != 'None']
        
        if len(valid_detections) == 0:
            # 沒有有效偵測
            if self.arrow_stable:
                self.get_logger().info('[WARN] Lost all detections, resetting stability')
                self.arrow_stable = False
                self.stable_sign = None
            return
        
        # 統計每個號誌出現的次數
        counter = Counter(valid_detections)
        most_common_sign, count = counter.most_common(1)[0]
        
        # 計算佔比
        window_size = len(self.arrow_temp)
        ratio = count / window_size if window_size > 0 else 0
        
        # 判斷是否穩定: 至少出現 STABILITY_THRESHOLD 次
        is_stable = count >= self.STABILITY_THRESHOLD
        
        if is_stable:
            # 檢查是否是新的穩定狀態
            if not self.arrow_stable or self.stable_sign != most_common_sign:
                self.arrow_stable = True
                self.stable_sign = most_common_sign
                self.get_logger().info(
                    f'[OK] Sign STABLE: {most_common_sign.upper()} '
                    f'({count}/{window_size} = {ratio*100:.0f}%)'
                )
        else:
            # 不穩定
            if self.arrow_stable:
                self.get_logger().info(
                    f'[WARN] Stability lost: {most_common_sign} only {count}/{window_size}'
                )
                self.arrow_stable = False
                self.stable_sign = None

    def check_detection_timeout(self):
        """檢查偵測是否超時（沒收到新資料）"""
        if self.last_detection_time is None:
            return False
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.last_detection_time
        
        if elapsed > self.detection_timeout:
            # 偵測超時，清空目標
            if self.sign_name and self.sign_name != 'None':
                self.get_logger().warn(f'[TIMEOUT] No detection for {elapsed:.1f}s, clearing target')
                self.sign_name = 'None'
                self.sign_area = 0
                self.arrow_stable = False
                self.stable_sign = None
                self.alignment_done = False
                self.head_searching = True
                return True
        return False

    def control_loop(self):
        """主控制迴圈 - 持續發送運動指令 (100Hz)"""
        # 從 API 獲取 IMU（如果沒有 Topic）
        if self.imu_data_source == "API" and self.api_mode and self.api:
            try:
                if hasattr(self.api, 'imu_rpy'):
                    self.update_imu(self.api.imu_rpy[2])
            except Exception as e:
                self.get_logger().error(f'[IMU] API read failed: {e}')
        
        # === 檢查 Robot Control 是否啟用 ===
        if not self.gui.get_robot_control_enabled():
            # 如果關閉控制，確保機器人停止
            if self.robot_walking:
                self.stop_robot_walking()
            return
        
        # === 如果啟用控制但機器人未行走，啟動它 ===
        if not self.robot_walking:
            self.start_robot_walking()
            
        if self.api_mode and self.api:
            # 持續發送頭部指令
            self.send_head_motor(1, self.target_head_horizontal, self.target_head_speed)
            self.send_head_motor(2, self.target_head_vertical, self.target_head_speed)
            
            # 持續發送移動指令
            self.send_continuous_value(self.target_speed_x, self.target_speed_y, self.target_theta)

    def strategy_update(self):
        """策略更新迴圈 - 根據狀態更新目標值 (10Hz)"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # === 檢查偵測超時 ===
        if not self.action_executing:
            self.check_detection_timeout()
        
        # 檢查動作執行時間
        if self.action_executing and self.action_start_time is not None:
            elapsed = current_time - self.action_start_time
            
            # === IMU 控制模式 ===
            if self.use_imu_control and self.target_yaw is not None:
                angle_error = abs(self.angle_difference(self.target_yaw, self.current_yaw))
                
                if angle_error < self.yaw_tolerance:
                    # 角度達到目標
                    self.get_logger().info(f'[IMU] Target angle reached! Error: {angle_error:.2f}°')
                    self.action_executing = False
                    self.target_yaw = None
                    self.reset_to_search()
                    
                elif elapsed > self.action_timeout:
                    # 超時保護
                    self.get_logger().warn(f'[IMU] Timeout! Angle error: {angle_error:.2f}°')
                    self.action_executing = False
                    self.target_yaw = None
                    self.reset_to_search()
                else:
                    # 持續轉向
                    remaining = self.action_timeout - elapsed
                    state = f'[TURNING] {self.current_action_name} | Error: {angle_error:.1f}° | Time: {remaining:.1f}s'
                    if state != self.current_state:
                        self.current_state = state
            
            # === 三段式閃避控制 ===
            elif self.current_action_name == 'AVOID STRAIGHT':
                self.update_3step_avoidance(current_time)
            
            # === 一般時間控制模式 ===
            else:
                if elapsed < self.action_duration:
                    remaining = self.action_duration - elapsed
                    state = f'[EXECUTING] {self.current_action_name} ... {remaining:.1f}s'
                    if state != self.current_state:
                        self.current_state = state
                else:
                    self.action_executing = False
                    self.reset_to_search()
        
        # 如果沒有在執行動作，更新策略
        if not self.action_executing:
            self.update_strategy()

    def update_3step_avoidance(self, current_time):
        """更新三段式閃避動作"""
        if self.avoid_step_start_time is None:
            self.avoid_step_start_time = current_time
            self.avoid_step = 1
        
        step_elapsed = current_time - self.avoid_step_start_time
        
        # 獲取各階段時間
        step1_duration = self.gui.get_avoid_step1_duration()
        step2_duration = self.gui.get_avoid_step2_duration()
        step3_duration = self.gui.get_avoid_step3_duration()
        strafe_speed = self.gui.get_avoid_strafe_speed()
        
        # 階段 1: 右移
        if self.avoid_step == 1:
            if step_elapsed < step1_duration:
                self.target_speed_x = self.BASE_SPEED * 0
                self.target_speed_y = strafe_speed  # 向右
                self.target_theta = self.ORIGIN_THETA * 0
                
                remaining = step1_duration - step_elapsed
                state = f'[AVOID-1/3] Moving RIGHT ... {remaining:.1f}s'
                if state != self.current_state:
                    self.current_state = state
            else:
                # 進入階段 2
                self.avoid_step = 2
                self.avoid_step_start_time = current_time
                self.get_logger().info('[AVOID] Step 1 complete → Step 2')
        
        # 階段 2: 直走
        elif self.avoid_step == 2:
            if step_elapsed < step2_duration:
                self.target_speed_x = self.BASE_SPEED
                self.target_speed_y = 0  # 純直走
                self.target_theta = self.ORIGIN_THETA
                
                remaining = step2_duration - step_elapsed
                state = f'[AVOID-2/3] Moving FORWARD ... {remaining:.1f}s'
                if state != self.current_state:
                    self.current_state = state
            else:
                # 進入階段 3
                self.avoid_step = 3
                self.avoid_step_start_time = current_time
                self.get_logger().info('[AVOID] Step 2 complete → Step 3')
        
        # 階段 3: 左移
        elif self.avoid_step == 3:
            if step_elapsed < step3_duration:
                self.target_speed_x = self.BASE_SPEED * 0
                self.target_speed_y = -strafe_speed  # 向左
                self.target_theta = self.ORIGIN_THETA * 0
                
                remaining = step3_duration - step_elapsed
                state = f'[AVOID-3/3] Moving LEFT ... {remaining:.1f}s'
                if state != self.current_state:
                    self.current_state = state
            else:
                # 三段式完成
                self.get_logger().info('[AVOID] 3-Step avoidance complete!')
                self.action_executing = False
                self.avoid_step = 0
                self.avoid_step_start_time = None
                self.reset_to_search()

    def reset_to_search(self):
        """重置到搜尋狀態"""
        self.head_searching = True
        self.alignment_done = False
        self.head_search_step = 0
        self.arrow_stable = False
        self.stable_sign = None
        self.arrow_temp.clear()  # 清空歷史
        self.target_yaw = None
        self.avoid_step = 0
        self.avoid_step_start_time = None
        
        # 重置運動參數為前進
        self.target_speed_x = self.BASE_SPEED
        self.target_speed_y = 0
        self.target_theta = self.ORIGIN_THETA
        
        self.current_state = '[SEARCH] Searching for signs...'
        self.get_logger().info('[RESET] Back to search mode')

    def update_strategy(self):
        """更新策略 - 使用 stable_sign 而非 sign_name"""
        area_threshold = self.gui.get_area_threshold()
        
        if self.head_searching:
            new_state = '[HEAD_SEARCH] Moving head to search for signs...'
            self.perform_head_search()
        
        elif not self.sign_name or self.sign_name == 'None':
            new_state = '[SEARCH] Searching for signs...'
            self.head_searching = True
            self.target_speed_x = self.BASE_SPEED
            self.target_speed_y = 0
            self.target_theta = self.ORIGIN_THETA
            
        elif not self.arrow_stable:
            # 顯示當前偵測到的內容和確認進度
            valid_count = len([s for s in self.arrow_temp if s != 'None'])
            new_state = f'[DETECT] Found {self.sign_name.upper()} (confirming {valid_count}/{self.STABILITY_WINDOW}...)'
            self.target_speed_x = self.BASE_SPEED
            self.target_speed_y = 0
            self.target_theta = self.ORIGIN_THETA
            
        elif not self.alignment_done:
            x_diff = abs(self.center_x - self.img_center_x)
            
            if x_diff > self.align_threshold:
                new_state = f'[ALIGN] Aligning to center (X offset: {self.center_x - self.img_center_x}px)'
                self.perform_alignment()
            else:
                self.alignment_done = True
                new_state = f'[ALIGNED] Sign centered, checking distance...'
                self.get_logger().info('[ALIGN] Alignment complete')
                
        elif self.alignment_done:
            if self.sign_area < area_threshold:
                new_state = f'[LOCK] Target {self.stable_sign.upper()} | Area: {self.sign_area} | Status: Approaching'
                self.perform_approach()
            else:
                # 使用穩定的號誌類型來執行動作
                if self.stable_sign in ['left', 'right']:
                    new_state = f'[ACTION] Execute {self.stable_sign.upper()} turn'
                    self.execute_action(self.stable_sign)
                elif self.stable_sign == 'straight':
                    new_state = f'[ACTION] Execute 3-step avoidance'
                    self.execute_action('straight')
        else:
            new_state = '[SEARCH] Searching for signs...'
        
        if new_state != self.current_state:
            self.current_state = new_state
            self.display_state()
    
    # === 機器人控制方法（兼容 strategy.API） ===
    def send_head_motor(self, motor_id, position, speed):
        """發送頭部馬達控制指令"""
        if not self.api_mode or not self.api:
            return
        self.api.sendHeadMotor(motor_id, position, speed)
    
    def send_continuous_value(self, speed_x, speed_y, theta):
        """發送連續移動指令"""
        if not self.api_mode or not self.api:
            return
        self.api.sendContinuousValue(speed_x, speed_y, theta)
    
    def send_body_auto(self, mode):
        """啟用/停止自動控制模式"""
        if not self.api_mode or not self.api:
            return
        self.api.sendbodyAuto(mode)
    
    def enable_robot_control(self):
        """啟用機器人控制"""
        if self.api_mode and self.api:
            self.get_logger().info('[CONTROL] Enabling robot body auto mode...')
    
    def disable_robot_control(self):
        """停止機器人控制"""
        if self.api_mode and self.api:
            self.get_logger().info('[CONTROL] Disabling robot body auto mode...')
            # 停止所有運動
            self.target_speed_x = 0
            self.target_speed_y = 0
            self.target_theta = 0
    
    def perform_head_search(self):
        """頭部搜尋 - 只改變目標位置"""
        search_positions = [2048, 2048, 2048, 2048, 2048]
        
        if self.head_search_step < len(search_positions):
            self.target_head_horizontal = search_positions[self.head_search_step]
            self.target_head_speed = 30
            
            self.head_search_step += 1
        else:
            self.head_search_step = 0
        
        # 搜尋時保持前進
        self.target_speed_x = self.BASE_SPEED
        self.target_speed_y = 0
        self.target_theta = self.ORIGIN_THETA
    
    def perform_alignment(self):
        """對齊 - 使用 P-Control 橫移"""
        x_error = self.center_x - self.img_center_x
        
        # 更新 P-Control 增益（從 GUI 讀取）
        self.align_controller.Kp = self.gui.get_align_kp()
        
        # 計算橫移速度（P-Control）
        strafe_speed = int(self.align_controller.compute(self.img_center_x, self.center_x))
        
        # 限制最大速度
        max_strafe = self.gui.get_avoid_strafe_speed()
        strafe_speed = max(min(strafe_speed, max_strafe), -max_strafe)
        
        # 根據誤差大小調整前進速度
        abs_error = abs(x_error)
        if abs_error > self.align_threshold * 2:  # 誤差很大時
            forward_speed = int(self.BASE_SPEED * 0.0)  # 幾乎停止，專注對齊
        elif abs_error > self.align_threshold:
            forward_speed = int(self.BASE_SPEED * 0.0)  # 減速前進
        else:
            forward_speed = self.BASE_SPEED  # 全速前進
        
        self.target_speed_x = forward_speed
        self.target_speed_y = strafe_speed
        self.target_theta = self.ORIGIN_THETA
        
        direction = "LEFT" if strafe_speed < 0 else "RIGHT"
        self.get_logger().debug(f'[P-ALIGN] err:{x_error} → spd:{strafe_speed} ({direction})')
    
    def perform_approach(self):
        """靠近 - 只改變目標速度"""
        self.target_speed_x = self.BASE_SPEED
        self.target_speed_y = 0
        self.target_theta = self.ORIGIN_THETA

    def execute_action(self, action):
        """執行動作 - 設定目標值並開始計時"""
        self.action_executing = True
        self.action_start_time = self.get_clock().now().nanoseconds / 1e9
        
        if action == 'left':
            self.current_action_name = 'LEFT TURN'
            msg = '<< Execute LEFT turn >>'
            
            # 使用 IMU 健康檢查（Watchdog）
            if self.is_imu_healthy():
                self.use_imu_control = True
                left_angle = self.gui.get_left_turn_angle()
                self.target_yaw = self.normalize_angle(self.current_yaw + left_angle)
                
                self.target_speed_x = self.BASE_SPEED
                self.target_speed_y = 0
                self.target_theta = self.TURN_SPEED
                
                self.get_logger().info(f'[IMU] Left turn: {self.current_yaw:.1f}° → {self.target_yaw:.1f}°')
            else:
                # 降級為時間控制
                self.use_imu_control = False
                self.action_duration = self.gui.get_left_turn_duration()
                self.target_speed_x = self.BASE_SPEED
                self.target_speed_y = 0
                self.target_theta = 8 + self.ORIGIN_THETA
                self.get_logger().warn('[FALLBACK] IMU unhealthy, using timed turn')
            
        elif action == 'right':
            self.current_action_name = 'RIGHT TURN'
            msg = '>> Execute RIGHT turn >>'
            
            # 使用 IMU 健康檢查（Watchdog）
            if self.is_imu_healthy():
                self.use_imu_control = True
                right_angle = self.gui.get_right_turn_angle()
                self.target_yaw = self.normalize_angle(self.current_yaw - right_angle)
                
                self.target_speed_x = self.BASE_SPEED
                self.target_speed_y = 0
                self.target_theta = -self.TURN_SPEED
                
                self.get_logger().info(f'[IMU] Right turn: {self.current_yaw:.1f}° → {self.target_yaw:.1f}°')
            else:
                # 降級為時間控制
                self.use_imu_control = False
                self.action_duration = self.gui.get_right_turn_duration()
                self.target_speed_x = self.BASE_SPEED
                self.target_speed_y = 0
                self.target_theta = -8 + self.ORIGIN_THETA
                self.get_logger().warn('[FALLBACK] IMU unhealthy, using timed turn')
            
        elif action == 'straight':
            # === 三段式閃避動作 ===
            self.use_imu_control = False
            self.current_action_name = 'AVOID STRAIGHT'
            msg = '|| Execute 3-Step AVOIDANCE ||'
            
            # 重置三段式狀態
            self.avoid_step = 0
            self.avoid_step_start_time = None
        
        self.get_logger().warn('=' * 45)
        self.get_logger().warn(f'   {msg}')
        self.get_logger().warn('=' * 45)

    def display_state(self):
        """顯示狀態"""
        self.get_logger().info(f'\n{"─"*45}')
        self.get_logger().info(f'  {self.current_state}')
        self.get_logger().info(f'{"─"*45}\n')


# ============================================================================
# YOLO GUI (Tkinter 界面 - 優化排版)
# ============================================================================
class YOLOGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Sign Control Panel (P-Control)")
        self.root.geometry("850x850")  # 降低高度
        self.root.configure(bg='#2c3e50')
        
        # === 參數變數 ===
        self.area_threshold = tk.IntVar(value=8000)
        self.left_turn_duration = tk.DoubleVar(value=2.0)
        self.right_turn_duration = tk.DoubleVar(value=2.0)
        
        # IMU 控制的轉向角度
        self.left_turn_angle = tk.IntVar(value=90)
        self.right_turn_angle = tk.IntVar(value=90)
        
        # 三段式閃避參數
        self.avoid_step1_duration = tk.DoubleVar(value=1.5)
        self.avoid_step2_duration = tk.DoubleVar(value=1.5)
        self.avoid_step3_duration = tk.DoubleVar(value=1.5)
        self.avoid_strafe_speed = tk.IntVar(value=500)
        
        # P-Control 對齊參數
        self.align_kp = tk.DoubleVar(value=3.0)
        
        self.robot_control_enabled = tk.BooleanVar(value=False)
        
        # === GUI 元件參考 ===
        self.control_toggle_btn = None
        self.control_status_label = None
        self.walking_status_label = None
        self.val_labels = {}
        
        # === YAML 配置檔路徑 ===
        # self.config_file = Path('/mar/yolo_config.yaml')
        self.config_file = Path('yolo_config.yaml')
        
        # === Thread-Safe 輪詢變數 ===
        self.latest_sign_name = None
        self.latest_center_x = 0
        self.latest_center_y = 0
        self.latest_sign_area = 0
        self.latest_current_yaw = 0.0
        self.latest_target_yaw = None
        self.latest_imu_healthy = False
        self.latest_robot_walking = False
        self.latest_head_h = 2048
        self.latest_head_v = 2048
        self.latest_speed_x = 0
        self.latest_speed_y = 0
        self.latest_theta = 0.0
        self.latest_state = '[SEARCH] Ready'
        self.latest_action = 'Idle'
        
        self.setup_ui()
        
        # 載入配置
        self.load_config()
        
        # 啟動自動儲存（每 5 秒）
        self.auto_save_config()
        
        # 啟動 GUI 輪詢（每 100ms）
        self.poll_node_data()
        
    def setup_ui(self):
        # ========== 1. Title (縮小 padding) ==========
        title_frame = tk.Frame(self.root, bg='#34495e', pady=5)
        title_frame.pack(fill='x')
        
        title = tk.Label(title_frame, text="YOLO Sign Control Panel", 
                        font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
        title.pack()
        
        subtitle = tk.Label(title_frame, text="P-Control Alignment + IMU Watchdog", 
                           font=('Arial', 9), bg='#34495e', fg='#ecf0f1')
        subtitle.pack()
        
        # ========== 2. Robot Control Toggle ==========
        control_frame = tk.Frame(self.root, bg='#34495e', pady=3)
        control_frame.pack(fill='x', padx=10)
        
        tk.Label(control_frame, text="ROBOT CONTROL", 
                font=('Arial', 10, 'bold'), bg='#34495e', fg='white').pack()
        
        self.control_toggle_btn = tk.Button(
            control_frame,
            text="[ OFF ] Simulation Mode",
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            command=self.toggle_robot_control,
            padx=15,
            pady=3,
            relief='raised',
            borderwidth=2,
            cursor='hand2'
        )
        self.control_toggle_btn.pack(pady=3)
        
        self.control_status_label = tk.Label(
            control_frame,
            text="Commands will NOT be sent to robot",
            font=('Arial', 9),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.control_status_label.pack()
        
        self.walking_status_label = tk.Label(
            control_frame,
            text="Robot: STOPPED",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#e74c3c'
        )
        self.walking_status_label.pack(pady=1)
        
        # ========== 3. 左右並排：Turn Angles + 3-Step Avoidance ==========
        params_row1 = tk.Frame(self.root, bg='#2c3e50')
        params_row1.pack(fill='x', padx=10, pady=3)
        
        # 左側：Turn Angles
        angle_frame = tk.LabelFrame(params_row1, text="Turn Angles (IMU)", 
                                     font=('Arial', 10, 'bold'),
                                     bg='#34495e', fg='#f1c40f', 
                                     padx=3, pady=3)
        angle_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        
        self.create_compact_slider(angle_frame, "Left (°)", 30, 180, 
                                 self.left_turn_angle, self.update_left_angle_label, 
                                 resolution=5, is_int=True, color='#3498db')
        self.create_compact_slider(angle_frame, "Right (°)", 30, 180, 
                                 self.right_turn_angle, self.update_right_angle_label, 
                                 resolution=5, is_int=True, color='#e74c3c')
        
        # 右側：3-Step Avoidance
        avoid_frame = tk.LabelFrame(params_row1, text="3-Step Avoidance", 
                                     font=('Arial', 10, 'bold'),
                                     bg='#9b59b6', fg='white', 
                                     padx=3, pady=3)
        avoid_frame.pack(side='left', fill='both', expand=True, padx=(5,0))
        
        self.create_compact_slider(avoid_frame, "Step1(s)", 0.5, 5.0, 
                                 self.avoid_step1_duration, self.update_avoid_step1_label, 
                                 resolution=0.1, color='#e74c3c')
        self.create_compact_slider(avoid_frame, "Step2(s)", 0.5, 5.0, 
                                 self.avoid_step2_duration, self.update_avoid_step2_label, 
                                 resolution=0.1, color='#2ecc71')
        self.create_compact_slider(avoid_frame, "Step3(s)", 0.5, 5.0, 
                                 self.avoid_step3_duration, self.update_avoid_step3_label, 
                                 resolution=0.1, color='#3498db')
        self.create_compact_slider(avoid_frame, "Strafe", 100, 1000, 
                                 self.avoid_strafe_speed, self.update_strafe_speed_label, 
                                 resolution=50, is_int=True, color='#f39c12')
        
        # ========== 4. Other Parameters (單行排列) ==========
        control_params = tk.LabelFrame(self.root, text="Other Parameters", 
                                     font=('Arial', 10, 'bold'),
                                     bg='#ecf0f1', fg='#2c3e50', 
                                     padx=3, pady=3)
        control_params.pack(fill='x', padx=10, pady=3)
        
        self.create_compact_slider(control_params, "Area Thres", 500, 10000, 
                                 self.area_threshold, self.update_area_label, 
                                 is_int=True, color='#3498db')
        self.create_compact_slider(control_params, "Align Kp", 0.5, 10.0, 
                                 self.align_kp, self.update_align_kp_label, 
                                 resolution=0.1, color='#9b59b6')
        self.create_compact_slider(control_params, "Left Time(s)", 3.0, 10.0, 
                                 self.left_turn_duration, self.update_left_label, 
                                 resolution=0.1, color='#95a5a6')
        self.create_compact_slider(control_params, "Right Time(s)", 3.0, 10.0, 
                                 self.right_turn_duration, self.update_right_label, 
                                 resolution=0.1, color='#95a5a6')

        # ========== 5. 左右並排：Detection Info + IMU Monitor ==========
        info_row = tk.Frame(self.root, bg='#2c3e50')
        info_row.pack(fill='x', padx=10, pady=3)
        
        # 左側：Detection Info
        info_frame = tk.LabelFrame(info_row, text="Detection", 
                                  font=('Arial', 10, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50',
                                  padx=5, pady=3)
        info_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        
        self.sign_label = tk.Label(info_frame, text="Sign: --", font=('Arial', 9), bg='#ecf0f1', anchor='w')
        self.sign_label.pack(fill='x', padx=2)
        
        self.area_label = tk.Label(info_frame, text="Area: 0", font=('Arial', 9), bg='#ecf0f1', anchor='w')
        self.area_label.pack(fill='x', padx=2)
        
        self.coord_label = tk.Label(info_frame, text="Coord: X=0, Y=0", font=('Arial', 9), bg='#ecf0f1', anchor='w')
        self.coord_label.pack(fill='x', padx=2)
        
        self.action_display_label = tk.Label(info_frame, text="Action: --", 
                                             font=('Arial', 10, 'bold'), 
                                             bg='#ecf0f1', fg='#e74c3c', anchor='w')
        self.action_display_label.pack(fill='x', pady=(3,0), padx=2)
        
        # 右側：IMU Monitor
        imu_frame = tk.LabelFrame(info_row, text="IMU Monitor", 
                                   font=('Arial', 10, 'bold'),
                                   bg='#2c3e50', fg='#f39c12',
                                   padx=5, pady=3)
        imu_frame.pack(side='left', fill='both', expand=True, padx=(5,0))
        
        lbl_style = {'bg': '#2c3e50', 'fg': '#ecf0f1', 'font': ('Consolas', 9)}
        val_style = {'bg': '#2c3e50', 'fg': '#3498db', 'font': ('Consolas', 10, 'bold')}
        
        yaw_row = tk.Frame(imu_frame, bg='#2c3e50')
        yaw_row.pack(fill='x')
        tk.Label(yaw_row, text="Yaw:", **lbl_style).pack(side='left')
        self.imu_yaw_val = tk.Label(yaw_row, text="0.0°", **val_style)
        self.imu_yaw_val.pack(side='left', padx=3)
        
        target_row = tk.Frame(imu_frame, bg='#2c3e50')
        target_row.pack(fill='x')
        tk.Label(target_row, text="Target:", **lbl_style).pack(side='left')
        self.imu_target_val = tk.Label(target_row, text="--", **val_style)
        self.imu_target_val.pack(side='left', padx=3)
        
        health_row = tk.Frame(imu_frame, bg='#2c3e50')
        health_row.pack(fill='x', pady=(3,0))
        tk.Label(health_row, text="Health:", **lbl_style).pack(side='left')
        self.imu_health_indicator = tk.Label(
            health_row, 
            text="● OFFLINE", 
            font=('Arial', 9, 'bold'),
            bg='#2c3e50', 
            fg='#e74c3c'
        )
        self.imu_health_indicator.pack(side='left', padx=3)

        # ========== 6. Motor & Motion Monitor (壓縮成 2 行) ==========
        motor_frame = tk.LabelFrame(self.root, text="Motor & Motion", 
                                   font=('Arial', 10, 'bold'),
                                   bg='#2c3e50', fg='#f1c40f',
                                   padx=5, pady=3)
        motor_frame.pack(fill='x', padx=10, pady=3)
        
        lbl_style = {'bg': '#2c3e50', 'fg': '#ecf0f1', 'font': ('Consolas', 9)}
        val_style = {'bg': '#2c3e50', 'fg': '#2ecc71', 'font': ('Consolas', 9, 'bold')}

        # 第一行：Head H, Head V, Speed X
        row1 = tk.Frame(motor_frame, bg='#2c3e50')
        row1.pack(fill='x')
        
        tk.Label(row1, text="Head H:", **lbl_style).pack(side='left', padx=2)
        self.motor_h_val = tk.Label(row1, text="2048", **val_style)
        self.motor_h_val.pack(side='left', padx=3)
        
        tk.Label(row1, text=" | ", bg='#2c3e50', fg='#7f8c8d').pack(side='left')
        
        tk.Label(row1, text="Head V:", **lbl_style).pack(side='left', padx=2)
        self.motor_v_val = tk.Label(row1, text="2048", **val_style)
        self.motor_v_val.pack(side='left', padx=3)
        
        tk.Label(row1, text=" | ", bg='#2c3e50', fg='#7f8c8d').pack(side='left')
        
        tk.Label(row1, text="Speed X:", **lbl_style).pack(side='left', padx=2)
        self.move_x_val = tk.Label(row1, text="0", **val_style)
        self.move_x_val.pack(side='left', padx=3)

        # 第二行：Speed Y, Theta
        row2 = tk.Frame(motor_frame, bg='#2c3e50')
        row2.pack(fill='x')
        
        tk.Label(row2, text="Speed Y:", **lbl_style).pack(side='left', padx=2)
        self.move_y_val = tk.Label(row2, text="0", **val_style)
        self.move_y_val.pack(side='left', padx=3)
        
        tk.Label(row2, text=" | ", bg='#2c3e50', fg='#7f8c8d').pack(side='left')
        
        tk.Label(row2, text="Theta:", **lbl_style).pack(side='left', padx=2)
        self.move_th_val = tk.Label(row2, text="0.00", **val_style)
        self.move_th_val.pack(side='left', padx=3)
        
        # ========== 7. Current State ==========
        state_frame = tk.LabelFrame(self.root, text="State", 
                                   font=('Arial', 10, 'bold'),
                                   bg='#ecf0f1', fg='#2c3e50',
                                   padx=3, pady=3)
        state_frame.pack(fill='x', padx=10, pady=3)
        
        self.state_label = tk.Label(state_frame, text="[SEARCH] Ready", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#f39c12', fg='white',
                                    padx=8, pady=3)
        self.state_label.pack(fill='x')
        
        # ========== 8. Log (縮小高度) ==========
        log_frame = tk.LabelFrame(self.root, text="Log", 
                                 font=('Arial', 10, 'bold'),
                                 bg='#ecf0f1', fg='#2c3e50',
                                 padx=3, pady=3)
        log_frame.pack(fill='both', expand=True, padx=10, pady=3)
        
        scroll = tk.Scrollbar(log_frame)
        scroll.pack(side='right', fill='y')
        
        self.log_text = tk.Text(log_frame, height=4,  # 降低高度
                               font=('Consolas', 8),
                               bg='#2c3e50', fg='#ecf0f1',
                               yscrollcommand=scroll.set,
                               state='disabled')
        self.log_text.pack(fill='both', expand=True)
        scroll.config(command=self.log_text.yview)

    def create_compact_slider(self, parent, label, from_, to, var, cmd, resolution=1, is_int=False, color='#3498db'):
        frame = tk.Frame(parent, bg=parent['bg'])
        frame.pack(fill='x', pady=1)
        
        label_bg = parent['bg']
        fg_color = 'white' if label_bg in ['#34495e', '#9b59b6', '#2c3e50'] else 'black'
        
        tk.Label(frame, text=label, font=('Arial', 9), bg=label_bg, 
                fg=fg_color, width=12, anchor='w').pack(side='left')
        
        val_text = str(var.get())
        val_lbl = tk.Label(frame, text=val_text, font=('Arial', 9, 'bold'), 
                           bg=color, fg='white', width=5)
        val_lbl.pack(side='right', padx=3)
        self.val_labels[str(var)] = val_lbl
        
        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
                         orient='horizontal', variable=var, command=cmd,
                         showvalue=0, bg=label_bg, troughcolor=color, highlightthickness=0,
                         width=8)
        scale.pack(side='left', fill='x', expand=True, padx=3)

    def update_val_label(self, var, value, is_int=False):
        lbl = self.val_labels.get(str(var))
        if lbl:
            if is_int:
                lbl.config(text=str(int(float(value))))
            else:
                lbl.config(text=f"{float(value):.1f}")

    def update_area_label(self, value): 
        self.update_val_label(self.area_threshold, value, is_int=True)
    
    def update_left_label(self, value): 
        self.update_val_label(self.left_turn_duration, value)
    
    def update_right_label(self, value): 
        self.update_val_label(self.right_turn_duration, value)
    
    def update_left_angle_label(self, value): 
        self.update_val_label(self.left_turn_angle, value, is_int=True)
    
    def update_right_angle_label(self, value): 
        self.update_val_label(self.right_turn_angle, value, is_int=True)
    
    def update_avoid_step1_label(self, value): 
        self.update_val_label(self.avoid_step1_duration, value)
    
    def update_avoid_step2_label(self, value): 
        self.update_val_label(self.avoid_step2_duration, value)
    
    def update_avoid_step3_label(self, value): 
        self.update_val_label(self.avoid_step3_duration, value)
    
    def update_strafe_speed_label(self, value): 
        self.update_val_label(self.avoid_strafe_speed, value, is_int=True)
    
    def update_align_kp_label(self, value): 
        self.update_val_label(self.align_kp, value)
    
    def toggle_robot_control(self):
        current = self.robot_control_enabled.get()
        new_state = not current
        self.robot_control_enabled.set(new_state)
        
        if new_state:
            self.control_toggle_btn.config(text="[ ON ] Active Mode", bg='#2ecc71')
            self.control_status_label.config(text="Commands WILL be sent to robot", fg='#2ecc71')
            self.log_message('[CONTROL] Robot control ENABLED')
            if hasattr(self, 'node'):
                self.node.enable_robot_control()
        else:
            self.control_toggle_btn.config(text="[ OFF ] Simulation Mode", bg='#e74c3c')
            self.control_status_label.config(text="Commands will NOT be sent to robot", fg='#ecf0f1')
            self.log_message('[CONTROL] Robot control DISABLED')
            if hasattr(self, 'node'):
                self.node.disable_robot_control()
    
    # ========== Thread-Safe 輪詢機制 ==========
    def poll_node_data(self):
        """定期從 Node 讀取資料並更新 GUI（Thread-Safe）"""
        if hasattr(self, 'node'):
            # 讀取 Node 的變數（不呼叫 GUI 更新）
            self.latest_sign_name = self.node.sign_name
            self.latest_center_x = self.node.center_x
            self.latest_center_y = self.node.center_y
            self.latest_sign_area = self.node.sign_area
            self.latest_current_yaw = self.node.current_yaw
            self.latest_target_yaw = self.node.target_yaw
            self.latest_imu_healthy = self.node.is_imu_healthy()
            self.latest_robot_walking = self.node.robot_walking
            self.latest_head_h = self.node.target_head_horizontal
            self.latest_head_v = self.node.target_head_vertical
            self.latest_speed_x = self.node.target_speed_x
            self.latest_speed_y = self.node.target_speed_y
            self.latest_theta = self.node.target_theta
            self.latest_state = self.node.current_state
            
            # 更新 GUI
            self.update_detection(self.latest_sign_name, self.latest_center_x, 
                                 self.latest_center_y, self.latest_sign_area)
            self.update_imu_status(self.latest_current_yaw, self.latest_target_yaw)
            self.update_imu_health(self.latest_imu_healthy)
            self.update_walking_status(self.latest_robot_walking)
            self.update_motor_status(self.latest_head_h, self.latest_head_v,
                                   self.latest_speed_x, self.latest_speed_y, self.latest_theta)
            self.update_state(self.latest_state)
        
        # 每 100ms 輪詢一次
        self.root.after(100, self.poll_node_data)
    
    # ========== Getter 方法 ==========
    def get_robot_control_enabled(self): 
        return self.robot_control_enabled.get()
    
    def get_area_threshold(self): 
        return self.area_threshold.get()
    
    def get_left_turn_duration(self): 
        return self.left_turn_duration.get()
    
    def get_right_turn_duration(self): 
        return self.right_turn_duration.get()
    
    def get_left_turn_angle(self): 
        return self.left_turn_angle.get()
    
    def get_right_turn_angle(self): 
        return self.right_turn_angle.get()
    
    def get_avoid_step1_duration(self): 
        return self.avoid_step1_duration.get()
    
    def get_avoid_step2_duration(self): 
        return self.avoid_step2_duration.get()
    
    def get_avoid_step3_duration(self): 
        return self.avoid_step3_duration.get()
    
    def get_avoid_strafe_speed(self): 
        return self.avoid_strafe_speed.get()
    
    def get_align_kp(self): 
        return self.align_kp.get()
    
    # ========== 更新方法（僅更新 GUI 元件）==========
    def update_detection(self, sign_name, x, y, area):
        self.sign_label.config(text=f"Sign: {sign_name.upper() if sign_name else '--'}")
        self.coord_label.config(text=f"Coord: X={x}, Y={y}")
        self.area_label.config(text=f"Area: {area}")
    
    def update_imu_status(self, current_yaw, target_yaw):
        try:
            self.imu_yaw_val.config(text=f"{current_yaw:.1f}°")
            if target_yaw is not None:
                self.imu_target_val.config(text=f"{target_yaw:.1f}°", fg='#e74c3c')
            else:
                self.imu_target_val.config(text="--", fg='#3498db')
        except:
            pass
    
    def update_imu_health(self, is_healthy):
        try:
            if is_healthy:
                self.imu_health_indicator.config(text="● HEALTHY", fg='#2ecc71')
            else:
                self.imu_health_indicator.config(text="● OFFLINE", fg='#e74c3c')
        except:
            pass
    
    def update_walking_status(self, is_walking):
        try:
            if is_walking:
                self.walking_status_label.config(text="Robot: WALKING", fg='#2ecc71')
            else:
                self.walking_status_label.config(text="Robot: STOPPED", fg='#e74c3c')
        except:
            pass
    
    def update_motor_status(self, head_h, head_v, spd_x, spd_y, theta):
        try:
            self.motor_h_val.config(text=f"{int(head_h)}")
            self.motor_v_val.config(text=f"{int(head_v)}")
            self.move_x_val.config(text=f"{int(spd_x)}")
            self.move_y_val.config(text=f"{int(spd_y)}")
            self.move_th_val.config(text=f"{theta:.2f}")
        except:
            pass

    def update_robot_action(self, action):
        """這個方法由 Node 直接呼叫，所以保留（但不是最佳實踐）"""
        if self.action_display_label is None: 
            return
        self.action_display_label.config(text=f"Action: {action}")
        if "LEFT" in action or "STRAFE LEFT" in action: 
            self.action_display_label.config(fg='#3498db')
        elif "RIGHT" in action or "STRAFE RIGHT" in action: 
            self.action_display_label.config(fg='#e74c3c')
        elif "FORWARD" in action: 
            self.action_display_label.config(fg='#2ecc71')
        elif "Head Search" in action: 
            self.action_display_label.config(fg='#9b59b6')
        elif "Step" in action:
            self.action_display_label.config(fg='#f39c12')
        else: 
            self.action_display_label.config(fg='#2c3e50')
        
    def update_state(self, state):
        self.state_label.config(text=state)
        if 'AVOID' in state:
            self.state_label.config(bg='#9b59b6')
        elif 'TURNING' in state or 'EXECUTING' in state: 
            self.state_label.config(bg='#e74c3c')
        elif 'ACTION' in state: 
            self.state_label.config(bg='#c0392b')
        elif 'LOCK' in state or 'ALIGNED' in state: 
            self.state_label.config(bg='#2ecc71')
        elif 'ALIGN' in state: 
            self.state_label.config(bg='#27ae60')
        elif 'HEAD_SEARCH' in state: 
            self.state_label.config(bg='#3498db')
        elif 'DETECT' in state: 
            self.state_label.config(bg='#f39c12')
        else: 
            self.state_label.config(bg='#95a5a6')
    
    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert('end', f'{message}\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')
    
    # ========== YAML 配置儲存/載入 ==========
    # def load_config(self):
    #     """載入 YAML 配置檔"""
    #     if not self.config_file.exists():
    #         self.log_message('[CONFIG] No config file found, using defaults')
    #         return
        
    #     try:
    #         with open(self.config_file, 'r') as f:
    #             config = yaml.safe_load(f)
            
    #         if config:
    #             self.area_threshold.set(config.get('area_threshold', 8000))
    #             self.left_turn_angle.set(config.get('left_turn_angle', 90))
    #             self.right_turn_angle.set(config.get('right_turn_angle', 90))
    #             self.left_turn_duration.set(config.get('left_turn_duration', 2.0))
    #             self.right_turn_duration.set(config.get('right_turn_duration', 2.0))
    #             self.avoid_step1_duration.set(config.get('avoid_step1_duration', 1.5))
    #             self.avoid_step2_duration.set(config.get('avoid_step2_duration', 1.5))
    #             self.avoid_step3_duration.set(config.get('avoid_step3_duration', 1.5))
    #             self.avoid_strafe_speed.set(config.get('avoid_strafe_speed', 500))
    #             self.align_kp.set(config.get('align_kp', 3.0))
            
    #         self.log_message(f'[CONFIG] Loaded from {self.config_file}')
    #     except Exception as e:
    #         self.log_message(f'[CONFIG] Load failed: {e}')
    
    def load_config(self):
        """載入 YAML 配置檔"""
        if not self.config_file.exists():
            self.log_message('[CONFIG] No config file found, using defaults')
           # 即使沒有檔案，也要更新標籤為預設值
            self.update_area_label(self.area_threshold.get())
            self.update_left_angle_label(self.left_turn_angle.get())
            self.update_right_angle_label(self.right_turn_angle.get())
            self.update_left_label(self.left_turn_duration.get())
            self.update_right_label(self.right_turn_duration.get())
            self.update_avoid_step1_label(self.avoid_step1_duration.get())
            self.update_avoid_step2_label(self.avoid_step2_duration.get())
            self.update_avoid_step3_label(self.avoid_step3_duration.get())
            self.update_strafe_speed_label(self.avoid_strafe_speed.get())
            self.update_align_kp_label(self.align_kp.get())
            return
    
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
        
            if config:
            # 設定滑桿位置
                self.area_threshold.set(config.get('area_threshold', 8000))
                self.left_turn_angle.set(config.get('left_turn_angle', 90))
                self.right_turn_angle.set(config.get('right_turn_angle', 90))
                self.left_turn_duration.set(config.get('left_turn_duration', 2.0))
                self.right_turn_duration.set(config.get('right_turn_duration', 2.0))
                self.avoid_step1_duration.set(config.get('avoid_step1_duration', 1.5))
                self.avoid_step2_duration.set(config.get('avoid_step2_duration', 1.5))
                self.avoid_step3_duration.set(config.get('avoid_step3_duration', 1.5))
                self.avoid_strafe_speed.set(config.get('avoid_strafe_speed', 500))
                self.align_kp.set(config.get('align_kp', 3.0))
            
                # 手動更新標籤顯示（因為 .set() 不會觸發 command 回調）
                self.update_area_label(self.area_threshold.get())
                self.update_left_angle_label(self.left_turn_angle.get())
                self.update_right_angle_label(self.right_turn_angle.get())
                self.update_left_label(self.left_turn_duration.get())
                self.update_right_label(self.right_turn_duration.get())
                self.update_avoid_step1_label(self.avoid_step1_duration.get())
                self.update_avoid_step2_label(self.avoid_step2_duration.get())
                self.update_avoid_step3_label(self.avoid_step3_duration.get())
                self.update_strafe_speed_label(self.avoid_strafe_speed.get())
                self.update_align_kp_label(self.align_kp.get())
        
            self.log_message(f'[CONFIG] Loaded from {self.config_file}')
        except Exception as e:
            self.log_message(f'[CONFIG] Load failed: {e}')
    
    def save_config(self):
        """儲存 YAML 配置檔"""
        config = {
            'area_threshold': self.area_threshold.get(),
            'left_turn_angle': self.left_turn_angle.get(),
            'right_turn_angle': self.right_turn_angle.get(),
            'left_turn_duration': self.left_turn_duration.get(),
            'right_turn_duration': self.right_turn_duration.get(),
            'avoid_step1_duration': self.avoid_step1_duration.get(),
            'avoid_step2_duration': self.avoid_step2_duration.get(),
            'avoid_step3_duration': self.avoid_step3_duration.get(),
            'avoid_strafe_speed': self.avoid_strafe_speed.get(),
            'align_kp': self.align_kp.get(),
        }
        
        try:
            # 確保目錄存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            # 不要每次都 log，太吵
        except Exception as e:
            self.log_message(f'[CONFIG] Save failed: {e}')
    
    def auto_save_config(self):
        """每 5 秒自動儲存配置"""
        self.save_config()
        self.root.after(5000, self.auto_save_config)  # 5000ms = 5秒
    
    def run(self):
        self.root.mainloop()


# ============================================================================
# ROS2 Spin Thread
# ============================================================================
def ros_spin(node):
    rclpy.spin(node)


# ============================================================================
# Main Function
# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    gui = YOLOGUI()
    node = YOLOTestNode(gui)
    
    # 讓 GUI 可以存取 node
    gui.node = node
    
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()
    
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        # 關閉時停止機器人
        if node.api_mode and node.api:
            node.send_body_auto(0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from geometry_msgs.msg import Point
# from collections import deque, Counter
# import tkinter as tk
# from tkinter import ttk
# import threading
# import sys
# import os

# # 導入機器人控制 API
# try:
#     from strategy.API import API
#     from tku_msgs.msg import Dio
#     API_AVAILABLE = True
# except ImportError:
#     API_AVAILABLE = False
#     print("[WARNING] strategy.API or tku_msgs not found. Robot control will be disabled.")


# class YOLOTestNode(Node):
#     def __init__(self, gui):
#         super().__init__('yolo_test_node')
        
#         self.gui = gui
        
#         # === 初始化機器人控制 API ===
#         self.api = None
#         if API_AVAILABLE:
#             try:
#                 self.api = API('yolo_strategy_api')
#                 self.api_mode = True
#                 self.get_logger().info('[API] Using strategy.API control')
#                 self.gui.log_message('[API] Robot API loaded successfully')
#             except Exception as e:
#                 self.api_mode = False
#                 self.get_logger().error(f'[API] Failed to initialize API: {e}')
#                 self.gui.log_message(f'[ERROR] API initialization failed: {e}')
#         else:
#             self.api_mode = False
#             self.get_logger().warn('[API] Robot control API not available (Test mode)')
#             self.gui.log_message('[WARNING] API not loaded - Simulation mode only')
        
#         # === 機器人運動控制參數（策略只改這些數值）===
#         self.target_speed_x = 0
#         self.target_speed_y = 0
#         self.target_theta = 0
#         self.target_head_horizontal = 2048
#         self.target_head_vertical = 2048
#         self.target_head_speed = 50
        
#         # === 機器人行走狀態 ===
#         self.robot_walking = False
#         self.start_warning_shown = False
        
#         # === IMU 資料 ===
#         self.current_yaw = 0.0
#         self.target_yaw = None
#         self.yaw_tolerance = 5.0  # 角度容忍範圍 (度)
#         self.imu_available = False
        
#         # === 影像與偵測資訊 ===
#         self.sign_name = None
#         self.sign_area = 0
#         self.center_x = 0
#         self.center_y = 0
        
#         # === 偵測超時機制 (針對 3-4Hz 優化) ===
#         self.last_detection_time = None
#         self.detection_timeout = 0.5  # 0.5秒內沒收到新偵測就視為遺失目標
        
#         self.img_center_x = 160
#         self.img_center_y = 120
#         self.align_threshold = 20
        
#         # === 狀態機變數 ===
#         self.current_state = '[SEARCH] Searching for signs...'
#         self.head_searching = True
#         self.alignment_done = False
#         self.head_search_step = 0
        
#         # === 彈性穩定性檢測參數 ===
#         self.STABILITY_WINDOW = 5      # 觀察窗口大小(最近5次)
#         self.STABILITY_THRESHOLD = 3   # 需要至少3次相同才算穩定
#         self.arrow_temp = deque(maxlen=self.STABILITY_WINDOW)
#         self.arrow_stable = False
#         self.stable_sign = None        # 當前穩定的號誌類型
        
#         # === 動作執行計時器 ===
#         self.action_executing = False
#         self.action_start_time = None
#         self.action_duration = 0.0
#         self.action_timeout = 5.0  # IMU 控制超時時間
#         self.current_action_name = ""
#         self.use_imu_control = True  # 是否使用 IMU 控制
        
#         # === 三段式閃避狀態 ===
#         self.avoid_step = 0  # 0=未開始, 1=右移, 2=直走, 3=左移
#         self.avoid_step_start_time = None
        
#         # === 基礎運動參數 ===
#         self.ORIGIN_THETA = 0
#         self.BASE_SPEED = 2500
#         self.STRAFE_SPEED = 400
#         self.TURN_SPEED = 8  # 轉彎角速度
        
#         # === 建立訂閱 ===
#         self.class_id_sub = self.create_subscription(
#             String, 
#             'class_id_topic', 
#             self.class_id_callback, 
#             10
#         )
        
#         self.coord_sub = self.create_subscription(
#             Point,
#             'sign_coordinates',
#             self.coord_callback,
#             10
#         )
        
#         # === 主控制迴圈 (100Hz) - 持續發送運動指令 ===
#         self.control_timer = self.create_timer(0.01, self.control_loop)
        
#         # === 策略更新迴圈 (10Hz) - 更新策略狀態 ===
#         self.strategy_timer = self.create_timer(0.1, self.strategy_update)
        
#         self.get_logger().info('=' * 50)
#         self.get_logger().info('   YOLO Sign Action Node Started (3-Step Avoidance)')
#         self.get_logger().info(f'   Stability: {self.STABILITY_THRESHOLD}/{self.STABILITY_WINDOW} detections')
#         if API_AVAILABLE:
#             self.get_logger().info('   Robot Control: ENABLED (strategy.API)')
#         else:
#             self.get_logger().info('   Robot Control: DISABLED (Test Mode)')
#         self.get_logger().info('=' * 50 + '\n')

#     def normalize_angle(self, angle):
#         """將角度標準化到 [-180, 180] 範圍"""
#         while angle > 180:
#             angle -= 360
#         while angle < -180:
#             angle += 360
#         return angle

#     def angle_difference(self, target, current):
#         """計算兩個角度之間的最短差異"""
#         diff = self.normalize_angle(target - current)
#         return diff

#     def update_imu(self, yaw):
#         """更新 IMU 數據 (從 API 獲取)"""
#         self.current_yaw = yaw
#         self.imu_available = True
#         self.gui.update_imu_status(yaw, self.target_yaw)

#     def start_robot_walking(self):
#         """啟動機器人行走"""
#         if not self.api_mode:
#             if not self.start_warning_shown:
#                 self.get_logger().warn('[ROBOT] API not available - Cannot start walking')
#                 self.gui.log_message('[ERROR] Cannot start - API not loaded')
#                 self.gui.update_walking_status(False)
#                 self.start_warning_shown = True
#             return
            
#         if not self.robot_walking:
#             try:
#                 self.send_body_auto(1)
#                 self.robot_walking = True
#                 self.get_logger().info('[ROBOT] Walking started (sendbodyAuto(1))')
#                 self.gui.log_message('[ROBOT] Walking started ✓')
#                 self.gui.update_walking_status(True)
#             except Exception as e:
#                 self.get_logger().error(f'[ROBOT] Failed to start walking: {e}')
#                 self.gui.log_message(f'[ERROR] Failed to start walking: {e}')
#                 self.gui.update_walking_status(False)
    
#     def stop_robot_walking(self):
#         """停止機器人行走"""
#         if not self.api_mode:
#             return
            
#         if self.robot_walking:
#             try:
#                 self.send_body_auto(0)
#                 self.robot_walking = False
#                 self.get_logger().info('[ROBOT] Walking stopped (sendbodyAuto(0))')
#                 self.gui.log_message('[ROBOT] Walking stopped ✓')
#                 self.gui.update_walking_status(False)
#             except Exception as e:
#                 self.get_logger().error(f'[ROBOT] Failed to stop walking: {e}')
#                 self.gui.log_message(f'[ERROR] Failed to stop walking: {e}')

#     def class_id_callback(self, msg):
#         """接收偵測結果"""
#         try:
#             parts = msg.data.split(',')
#             if len(parts) == 4:
#                 self.sign_name = parts[0]
#                 self.center_x = int(parts[1])
#                 bbox_y_max = int(parts[2])
#                 self.sign_area = int(parts[3])
                
#                 # 更新最後偵測時間
#                 self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
                
#                 self.gui.update_detection(self.sign_name, self.center_x, self.center_y, self.sign_area)
                
#                 if self.sign_name != 'None' and self.head_searching:
#                     self.head_searching = False
#                     self.get_logger().info('[HEAD] Sign detected, stop head searching')
#                     self.gui.log_message('[HEAD] Sign detected, stop head searching')
                
#                 # 加入歷史記錄
#                 self.arrow_temp.append(self.sign_name)
                
#                 # 檢查穩定性
#                 self.check_stability()
                
#         except Exception as e:
#             self.get_logger().error(f'Parse message failed: {e}')

#     def coord_callback(self, msg):
#         """接收座標資訊"""
#         self.center_x = int(msg.x)
#         self.center_y = int(msg.y)

#     def check_stability(self):
#         """彈性穩定性檢測: 5次內有3-4次相同即可"""
        
#         # 過濾掉 'None' 後統計
#         valid_detections = [sign for sign in self.arrow_temp if sign != 'None']
        
#         if len(valid_detections) == 0:
#             # 沒有有效偵測
#             if self.arrow_stable:
#                 self.get_logger().info('[WARN] Lost all detections, resetting stability')
#                 self.arrow_stable = False
#                 self.stable_sign = None
#             return
        
#         # 統計每個號誌出現的次數
#         counter = Counter(valid_detections)
#         most_common_sign, count = counter.most_common(1)[0]
        
#         # 計算佔比
#         window_size = len(self.arrow_temp)
#         ratio = count / window_size if window_size > 0 else 0
        
#         # 判斷是否穩定: 至少出現 STABILITY_THRESHOLD 次
#         is_stable = count >= self.STABILITY_THRESHOLD
        
#         if is_stable:
#             # 檢查是否是新的穩定狀態
#             if not self.arrow_stable or self.stable_sign != most_common_sign:
#                 self.arrow_stable = True
#                 self.stable_sign = most_common_sign
#                 self.get_logger().info(
#                     f'[OK] Sign STABLE: {most_common_sign.upper()} '
#                     f'({count}/{window_size} = {ratio*100:.0f}%)'
#                 )
#                 self.gui.log_message(
#                     f'[OK] {most_common_sign.upper()} stable '
#                     f'({count}/{window_size})'
#                 )
#         else:
#             # 不穩定
#             if self.arrow_stable:
#                 self.get_logger().info(
#                     f'[WARN] Stability lost: {most_common_sign} only {count}/{window_size}'
#                 )
#                 self.arrow_stable = False
#                 self.stable_sign = None

#     def check_detection_timeout(self):
#         """檢查偵測是否超時（沒收到新資料）"""
#         if self.last_detection_time is None:
#             return False
        
#         current_time = self.get_clock().now().nanoseconds / 1e9
#         elapsed = current_time - self.last_detection_time
        
#         if elapsed > self.detection_timeout:
#             # 偵測超時，清空目標
#             if self.sign_name and self.sign_name != 'None':
#                 self.get_logger().warn(f'[TIMEOUT] No detection for {elapsed:.1f}s, clearing target')
#                 self.gui.log_message(f'[TIMEOUT] Lost sign (no update for {elapsed:.1f}s)')
#                 self.sign_name = 'None'
#                 self.sign_area = 0
#                 self.arrow_stable = False
#                 self.stable_sign = None
#                 self.alignment_done = False
#                 self.head_searching = True
#                 return True
#         return False

#     def control_loop(self):
#         """主控制迴圈 - 持續發送運動指令 (100Hz)"""
#         # 從 API 獲取 IMU
#         if self.api_mode and self.api and hasattr(self.api, 'imu_rpy'):
#             self.update_imu(self.api.imu_rpy[2])
        
#         # 更新 GUI 數值顯示
#         self.gui.update_motor_status(
#             self.target_head_horizontal,
#             self.target_head_vertical,
#             self.target_speed_x,
#             self.target_speed_y,
#             self.target_theta
#         )
        
#         # === 檢查 Robot Control 是否啟用 ===
#         if not self.gui.get_robot_control_enabled():
#             # 如果關閉控制，確保機器人停止
#             if self.robot_walking:
#                 self.stop_robot_walking()
#             return
        
#         # === 如果啟用控制但機器人未行走，啟動它 ===
#         if not self.robot_walking:
#             self.start_robot_walking()
            
#         if self.api_mode and self.api:
#             # 持續發送頭部指令
#             self.send_head_motor(1, self.target_head_horizontal, self.target_head_speed)
#             self.send_head_motor(2, self.target_head_vertical, self.target_head_speed)
            
#             # 持續發送移動指令
#             self.send_continuous_value(self.target_speed_x, self.target_speed_y, self.target_theta)

#     def strategy_update(self):
#         """策略更新迴圈 - 根據狀態更新目標值 (10Hz)"""
#         current_time = self.get_clock().now().nanoseconds / 1e9
        
#         # === 檢查偵測超時 ===
#         if not self.action_executing:
#             self.check_detection_timeout()
        
#         # 檢查動作執行時間
#         if self.action_executing and self.action_start_time is not None:
#             elapsed = current_time - self.action_start_time
            
#             # === IMU 控制模式 ===
#             if self.use_imu_control and self.target_yaw is not None:
#                 angle_error = abs(self.angle_difference(self.target_yaw, self.current_yaw))
                
#                 if angle_error < self.yaw_tolerance:
#                     # 角度達到目標
#                     self.get_logger().info(f'[IMU] Target angle reached! Error: {angle_error:.2f}°')
#                     self.gui.log_message(f'[IMU] Turn complete (error: {angle_error:.2f}°)')
#                     self.action_executing = False
#                     self.target_yaw = None
#                     self.reset_to_search()
                    
#                 elif elapsed > self.action_timeout:
#                     # 超時保護
#                     self.get_logger().warn(f'[IMU] Timeout! Angle error: {angle_error:.2f}°')
#                     self.gui.log_message(f'[IMU] Timeout (error: {angle_error:.2f}°)')
#                     self.action_executing = False
#                     self.target_yaw = None
#                     self.reset_to_search()
#                 else:
#                     # 持續轉向
#                     remaining = self.action_timeout - elapsed
#                     state = f'[TURNING] {self.current_action_name} | Error: {angle_error:.1f}° | Time: {remaining:.1f}s'
#                     if state != self.current_state:
#                         self.current_state = state
#                         self.gui.update_state(state)
            
#             # === 三段式閃避控制 ===
#             elif self.current_action_name == 'AVOID STRAIGHT':
#                 self.update_3step_avoidance(current_time)
            
#             # === 一般時間控制模式 ===
#             else:
#                 if elapsed < self.action_duration:
#                     remaining = self.action_duration - elapsed
#                     state = f'[EXECUTING] {self.current_action_name} ... {remaining:.1f}s'
#                     if state != self.current_state:
#                         self.current_state = state
#                         self.gui.update_state(state)
#                 else:
#                     self.action_executing = False
#                     self.reset_to_search()
        
#         # 如果沒有在執行動作，更新策略
#         if not self.action_executing:
#             self.update_strategy()

#     def update_3step_avoidance(self, current_time):
#         """更新三段式閃避動作"""
#         if self.avoid_step_start_time is None:
#             self.avoid_step_start_time = current_time
#             self.avoid_step = 1
        
#         step_elapsed = current_time - self.avoid_step_start_time
        
#         # 獲取各階段時間
#         step1_duration = self.gui.get_avoid_step1_duration()
#         step2_duration = self.gui.get_avoid_step2_duration()
#         step3_duration = self.gui.get_avoid_step3_duration()
#         strafe_speed = self.gui.get_avoid_strafe_speed()
        
#         # 階段 1: 右移
#         if self.avoid_step == 1:
#             if step_elapsed < step1_duration:
#                 self.target_speed_x = self.BASE_SPEED * 0
#                 self.target_speed_y = strafe_speed  # 向右
#                 self.target_theta = self.ORIGIN_THETA * 0
                
#                 remaining = step1_duration - step_elapsed
#                 state = f'[AVOID-1/3] Moving RIGHT ... {remaining:.1f}s'
#                 if state != self.current_state:
#                     self.current_state = state
#                     self.gui.update_state(state)
#                     self.gui.update_robot_action("Step 1: Move RIGHT")
#             else:
#                 # 進入階段 2
#                 self.avoid_step = 2
#                 self.avoid_step_start_time = current_time
#                 self.get_logger().info('[AVOID] Step 1 complete → Step 2')
#                 self.gui.log_message('[AVOID] Step 1→2: RIGHT→FORWARD')
        
#         # 階段 2: 直走
#         elif self.avoid_step == 2:
#             if step_elapsed < step2_duration:
#                 self.target_speed_x = self.BASE_SPEED
#                 self.target_speed_y = 0  # 純直走
#                 self.target_theta = self.ORIGIN_THETA
                
#                 remaining = step2_duration - step_elapsed
#                 state = f'[AVOID-2/3] Moving FORWARD ... {remaining:.1f}s'
#                 if state != self.current_state:
#                     self.current_state = state
#                     self.gui.update_state(state)
#                     self.gui.update_robot_action("Step 2: Move FORWARD")
#             else:
#                 # 進入階段 3
#                 self.avoid_step = 3
#                 self.avoid_step_start_time = current_time
#                 self.get_logger().info('[AVOID] Step 2 complete → Step 3')
#                 self.gui.log_message('[AVOID] Step 2→3: FORWARD→LEFT')
        
#         # 階段 3: 左移
#         elif self.avoid_step == 3:
#             if step_elapsed < step3_duration:
#                 self.target_speed_x = self.BASE_SPEED * 0
#                 self.target_speed_y = -strafe_speed  # 向左
#                 self.target_theta = self.ORIGIN_THETA * 0
                
#                 remaining = step3_duration - step_elapsed
#                 state = f'[AVOID-3/3] Moving LEFT ... {remaining:.1f}s'
#                 if state != self.current_state:
#                     self.current_state = state
#                     self.gui.update_state(state)
#                     self.gui.update_robot_action("Step 3: Move LEFT")
#             else:
#                 # 三段式完成
#                 self.get_logger().info('[AVOID] 3-Step avoidance complete!')
#                 self.gui.log_message('[AVOID] 3-Step complete ✓')
#                 self.action_executing = False
#                 self.avoid_step = 0
#                 self.avoid_step_start_time = None
#                 self.reset_to_search()

#     def reset_to_search(self):
#         """重置到搜尋狀態"""
#         self.head_searching = True
#         self.alignment_done = False
#         self.head_search_step = 0
#         self.arrow_stable = False
#         self.stable_sign = None
#         self.arrow_temp.clear()  # 清空歷史
#         self.target_yaw = None
#         self.avoid_step = 0
#         self.avoid_step_start_time = None
        
#         # 重置運動參數為前進
#         self.target_speed_x = self.BASE_SPEED
#         self.target_speed_y = 0
#         self.target_theta = self.ORIGIN_THETA
        
#         self.current_state = '[SEARCH] Searching for signs...'
#         self.gui.update_state(self.current_state)
#         self.get_logger().info('[RESET] Back to search mode')
#         self.gui.log_message('[RESET] Back to search mode')

#     def update_strategy(self):
#         """更新策略 - 使用 stable_sign 而非 sign_name"""
#         area_threshold = self.gui.get_area_threshold()
        
#         if self.head_searching:
#             new_state = '[HEAD_SEARCH] Moving head to search for signs...'
#             self.perform_head_search()
        
#         elif not self.sign_name or self.sign_name == 'None':
#             new_state = '[SEARCH] Searching for signs...'
#             self.head_searching = True
#             self.target_speed_x = self.BASE_SPEED
#             self.target_speed_y = 0
#             self.target_theta = self.ORIGIN_THETA
            
#         elif not self.arrow_stable:
#             # 顯示當前偵測到的內容和確認進度
#             valid_count = len([s for s in self.arrow_temp if s != 'None'])
#             new_state = f'[DETECT] Found {self.sign_name.upper()} (confirming {valid_count}/{self.STABILITY_WINDOW}...)'
#             self.target_speed_x = self.BASE_SPEED
#             self.target_speed_y = 0
#             self.target_theta = self.ORIGIN_THETA
            
#         elif not self.alignment_done:
#             x_diff = abs(self.center_x - self.img_center_x)
            
#             if x_diff > self.align_threshold:
#                 new_state = f'[ALIGN] Aligning to center (X offset: {self.center_x - self.img_center_x}px)'
#                 self.perform_alignment()
#             else:
#                 self.alignment_done = True
#                 new_state = f'[ALIGNED] Sign centered, checking distance...'
#                 self.get_logger().info('[ALIGN] Alignment complete')
#                 self.gui.log_message('[ALIGN] Alignment complete')
                
#         elif self.alignment_done:
#             if self.sign_area < area_threshold:
#                 new_state = f'[LOCK] Target {self.stable_sign.upper()} | Area: {self.sign_area} | Status: Approaching'
#                 self.perform_approach()
#             else:
#                 # 使用穩定的號誌類型來執行動作
#                 if self.stable_sign in ['left', 'right']:
#                     new_state = f'[ACTION] Execute {self.stable_sign.upper()} turn'
#                     self.execute_action(self.stable_sign)
#                 elif self.stable_sign == 'straight':
#                     new_state = f'[ACTION] Execute 3-step avoidance'
#                     self.execute_action('straight')
#         else:
#             new_state = '[SEARCH] Searching for signs...'
        
#         if new_state != self.current_state:
#             self.current_state = new_state
#             self.gui.update_state(new_state)
#             self.display_state()
    
#     # === 機器人控制方法（兼容 strategy.API） ===
#     def send_head_motor(self, motor_id, position, speed):
#         """發送頭部馬達控制指令"""
#         if not self.api_mode or not self.api:
#             return
#         self.api.sendHeadMotor(motor_id, position, speed)
    
#     def send_continuous_value(self, speed_x, speed_y, theta):
#         """發送連續移動指令"""
#         if not self.api_mode or not self.api:
#             return
#         self.api.sendContinuousValue(speed_x, speed_y, theta)
    
#     def send_body_auto(self, mode):
#         """啟用/停止自動控制模式"""
#         if not self.api_mode or not self.api:
#             return
#         self.api.sendbodyAuto(mode)
    
#     def enable_robot_control(self):
#         """啟用機器人控制"""
#         if self.api_mode and self.api:
#             self.get_logger().info('[CONTROL] Enabling robot body auto mode...')
#             self.gui.log_message('[CONTROL] Robot body auto mode ENABLED')
    
#     def disable_robot_control(self):
#         """停止機器人控制"""
#         if self.api_mode and self.api:
#             self.get_logger().info('[CONTROL] Disabling robot body auto mode...')
#             # 停止所有運動
#             self.target_speed_x = 0
#             self.target_speed_y = 0
#             self.target_theta = 0
#             self.gui.log_message('[CONTROL] Robot body auto mode DISABLED')
    
#     def perform_head_search(self):
#         """頭部搜尋 - 只改變目標位置"""
#         search_positions = [2048, 2048, 2048, 2048, 2048]
        
#         if self.head_search_step < len(search_positions):
#             self.target_head_horizontal = search_positions[self.head_search_step]
#             self.target_head_speed = 30
            
#             self.gui.update_robot_action(f"Head Search (Position: {self.target_head_horizontal})")
#             self.head_search_step += 1
#         else:
#             self.head_search_step = 0
        
#         # 搜尋時保持前進
#         self.target_speed_x = self.BASE_SPEED
#         self.target_speed_y = 0
#         self.target_theta = self.ORIGIN_THETA
    
#     def perform_alignment(self):
#         """對齊 - 使用左右橫移同時保持前進"""
#         x_diff = self.center_x - self.img_center_x
        
#         if x_diff < -self.align_threshold:
#             self.gui.update_robot_action("STRAFE LEFT (Aligning)")
#             # 同時前進+橫移
#             self.target_speed_x = int(self.BASE_SPEED * 0.6)  # 降速前進
#             self.target_speed_y = -self.STRAFE_SPEED
#             self.target_theta = self.ORIGIN_THETA
#             self.get_logger().debug('[ALIGN] STRAFE LEFT + FORWARD')
            
#         elif x_diff > self.align_threshold:
#             self.gui.update_robot_action("STRAFE RIGHT (Aligning)")
#             # 同時前進+橫移
#             self.target_speed_x = int(self.BASE_SPEED * 0.6)
#             self.target_speed_y = self.STRAFE_SPEED
#             self.target_theta = self.ORIGIN_THETA
#             self.get_logger().debug('[ALIGN] STRAFE RIGHT + FORWARD')
    
#     def perform_approach(self):
#         """靠近 - 只改變目標速度"""
#         self.gui.update_robot_action("Move FORWARD (Approaching)")
#         self.target_speed_x = self.BASE_SPEED
#         self.target_speed_y = 0
#         self.target_theta = self.ORIGIN_THETA

#     def execute_action(self, action):
#         """執行動作 - 設定目標值並開始計時"""
#         self.action_executing = True
#         self.action_start_time = self.get_clock().now().nanoseconds / 1e9
        
#         if action == 'left':
#             self.current_action_name = 'LEFT TURN'
#             msg = '<< Execute LEFT turn >>'
            
#             # 使用 IMU 控制
#             if self.imu_available:
#                 self.use_imu_control = True
#                 left_angle = self.gui.get_left_turn_angle()
#                 self.target_yaw = self.normalize_angle(self.current_yaw + left_angle)
                
#                 self.target_speed_x = self.BASE_SPEED
#                 self.target_speed_y = 0
#                 self.target_theta = self.TURN_SPEED
                
#                 self.get_logger().info(f'[IMU] Left turn: {self.current_yaw:.1f}° → {self.target_yaw:.1f}°')
#                 self.gui.log_message(f'[IMU] Left turn target: {self.target_yaw:.1f}°')
#             else:
#                 # 降級為時間控制
#                 self.use_imu_control = False
#                 self.action_duration = self.gui.get_left_turn_duration()
#                 self.target_speed_x = self.BASE_SPEED
#                 self.target_speed_y = 0
#                 self.target_theta = 8 + self.ORIGIN_THETA
#                 self.gui.log_message('[WARN] IMU unavailable, using timed control')
            
#             self.gui.update_robot_action("TURN LEFT (90 degrees)")
            
#         elif action == 'right':
#             self.current_action_name = 'RIGHT TURN'
#             msg = '>> Execute RIGHT turn >>'
            
#             # 使用 IMU 控制
#             if self.imu_available:
#                 self.use_imu_control = True
#                 right_angle = self.gui.get_right_turn_angle()
#                 self.target_yaw = self.normalize_angle(self.current_yaw - right_angle)
                
#                 self.target_speed_x = self.BASE_SPEED
#                 self.target_speed_y = 0
#                 self.target_theta = -self.TURN_SPEED
                
#                 self.get_logger().info(f'[IMU] Right turn: {self.current_yaw:.1f}° → {self.target_yaw:.1f}°')
#                 self.gui.log_message(f'[IMU] Right turn target: {self.target_yaw:.1f}°')
#             else:
#                 # 降級為時間控制
#                 self.use_imu_control = False
#                 self.action_duration = self.gui.get_right_turn_duration()
#                 self.target_speed_x = self.BASE_SPEED
#                 self.target_speed_y = 0
#                 self.target_theta = -8 + self.ORIGIN_THETA
#                 self.gui.log_message('[WARN] IMU unavailable, using timed control')
            
#             self.gui.update_robot_action("TURN RIGHT (90 degrees)")
            
#         elif action == 'straight':
#             # === 三段式閃避動作 ===
#             self.use_imu_control = False
#             self.current_action_name = 'AVOID STRAIGHT'
#             msg = '|| Execute 3-Step AVOIDANCE ||'
            
#             # 重置三段式狀態
#             self.avoid_step = 0
#             self.avoid_step_start_time = None
            
#             # 初始速度會在 update_3step_avoidance 中設定
#             self.gui.update_robot_action("Starting 3-Step Avoidance")
        
#         self.get_logger().warn('=' * 45)
#         self.get_logger().warn(f'   {msg}')
#         self.get_logger().warn('=' * 45)
#         self.gui.log_message(f'[ACTION] {msg}')

#     def display_state(self):
#         """顯示狀態"""
#         self.get_logger().info(f'\n{"─"*45}')
#         self.get_logger().info(f'  {self.current_state}')
#         self.get_logger().info(f'{"─"*45}\n')


# class YOLOGUI:
#     def __init__(self):
#         self.root = tk.Tk()
#         self.root.title("YOLO Sign Control Panel (3-Step Avoidance)")
#         self.root.geometry("850x1150")
#         self.root.configure(bg='#2c3e50')
        
#         self.area_threshold = tk.IntVar(value=8000)
#         self.left_turn_duration = tk.DoubleVar(value=2.0)
#         self.right_turn_duration = tk.DoubleVar(value=2.0)
        
#         # IMU 控制的轉向角度
#         self.left_turn_angle = tk.IntVar(value=90)
#         self.right_turn_angle = tk.IntVar(value=90)
        
#         # 三段式閃避參數
#         self.avoid_step1_duration = tk.DoubleVar(value=1.5)  # 右移時間
#         self.avoid_step2_duration = tk.DoubleVar(value=1.5)  # 直走時間
#         self.avoid_step3_duration = tk.DoubleVar(value=1.5)  # 左移時間
#         self.avoid_strafe_speed = tk.IntVar(value=500)       # 橫移速度
        
#         self.robot_control_enabled = tk.BooleanVar(value=False)
        
#         self.action_display_label = None
#         self.control_toggle_btn = None
#         self.control_status_label = None
#         self.walking_status_label = None
        
#         self.val_labels = {}
        
#         self.setup_ui()
        
#     def setup_ui(self):
#         # 1. Title
#         title_frame = tk.Frame(self.root, bg='#34495e', pady=10)
#         title_frame.pack(fill='x')
        
#         title = tk.Label(title_frame, text="YOLO Sign Control Panel", 
#                         font=('Arial', 17, 'bold'), bg='#34495e', fg='white')
#         title.pack()
        
#         subtitle = tk.Label(title_frame, text="with 3-Step Avoidance", 
#                            font=('Arial', 11), bg='#34495e', fg='#ecf0f1')
#         subtitle.pack()
        
#         # 2. ROBOT CONTROL TOGGLE
#         control_frame = tk.Frame(self.root, bg='#34495e', pady=5)
#         control_frame.pack(fill='x', padx=10)
        
#         tk.Label(control_frame, text="ROBOT CONTROL", 
#                 font=('Arial', 11, 'bold'), bg='#34495e', fg='white').pack()
        
#         self.control_toggle_btn = tk.Button(
#             control_frame,
#             text="[ OFF ] Simulation Mode",
#             font=('Arial', 13, 'bold'),
#             bg='#e74c3c',
#             fg='white',
#             command=self.toggle_robot_control,
#             padx=20,
#             pady=5,
#             relief='raised',
#             borderwidth=3,
#             cursor='hand2'
#         )
#         self.control_toggle_btn.pack(pady=5)
        
#         self.control_status_label = tk.Label(
#             control_frame,
#             text="Commands will NOT be sent to robot",
#             font=('Arial', 10),
#             bg='#34495e',
#             fg='#ecf0f1'
#         )
#         self.control_status_label.pack()
        
#         self.walking_status_label = tk.Label(
#             control_frame,
#             text="Robot Status: STOPPED",
#             font=('Arial', 11, 'bold'),
#             bg='#34495e',
#             fg='#e74c3c'
#         )
#         self.walking_status_label.pack(pady=2)
        
#         # 3. IMU Turn Angles
#         angle_frame = tk.LabelFrame(self.root, text="Turn Angles (IMU Control)", 
#                                      font=('Arial', 11, 'bold'),
#                                      bg='#34495e', fg='#f1c40f', 
#                                      padx=5, pady=5)
#         angle_frame.pack(fill='x', padx=10, pady=5)
        
#         self.create_compact_slider(angle_frame, "Left Turn (°)", 30, 180, 
#                                  self.left_turn_angle, self.update_left_angle_label, 
#                                  resolution=5, is_int=True, color='#3498db')
#         self.create_compact_slider(angle_frame, "Right Turn (°)", 30, 180, 
#                                  self.right_turn_angle, self.update_right_angle_label, 
#                                  resolution=5, is_int=True, color='#e74c3c')
        
#         # 4. 3-Step Avoidance Parameters
#         avoid_frame = tk.LabelFrame(self.root, text="3-Step Avoidance (Straight Sign)", 
#                                      font=('Arial', 11, 'bold'),
#                                      bg='#9b59b6', fg='white', 
#                                      padx=5, pady=5)
#         avoid_frame.pack(fill='x', padx=10, pady=5)
        
#         self.create_compact_slider(avoid_frame, "Step1 Time(s)", 0.5, 10.0, 
#                                  self.avoid_step1_duration, self.update_avoid_step1_label, 
#                                  resolution=0.1, color='#e74c3c')
#         self.create_compact_slider(avoid_frame, "Step2 Time(s)", 0.5, 10.0, 
#                                  self.avoid_step2_duration, self.update_avoid_step2_label, 
#                                  resolution=0.1, color='#2ecc71')
#         self.create_compact_slider(avoid_frame, "Step3 Time(s)", 0.5, 10.0, 
#                                  self.avoid_step3_duration, self.update_avoid_step3_label, 
#                                  resolution=0.1, color='#3498db')
#         self.create_compact_slider(avoid_frame, "Strafe Speed", 100, 1000, 
#                                  self.avoid_strafe_speed, self.update_strafe_speed_label, 
#                                  resolution=50, is_int=True, color='#f39c12')
        
#         # 添加說明
#         hint = tk.Label(avoid_frame, 
#                        text="Step1: RIGHT | Step2: FORWARD | Step3: LEFT",
#                        font=('Arial', 9), bg='#9b59b6', fg='#ecf0f1')
#         hint.pack(pady=(5,0))
        
#         # 5. Other Parameters
#         control_params = tk.LabelFrame(self.root, text="Other Parameters", 
#                                      font=('Arial', 11, 'bold'),
#                                      bg='#ecf0f1', fg='#2c3e50', 
#                                      padx=5, pady=5)
#         control_params.pack(fill='x', padx=10, pady=5)
        
#         self.create_compact_slider(control_params, "Area Thres", 1000, 20000, 
#                                  self.area_threshold, self.update_area_label, 
#                                  is_int=True, color='#3498db')
#         self.create_compact_slider(control_params, "Left Time (s)", 0.5, 10.0, 
#                                  self.left_turn_duration, self.update_left_label, 
#                                  resolution=0.1, color='#95a5a6')
#         self.create_compact_slider(control_params, "Right Time (s)", 0.5, 10.0, 
#                                  self.right_turn_duration, self.update_right_label, 
#                                  resolution=0.1, color='#95a5a6')

#         # 6. Detection Info
#         info_frame = tk.LabelFrame(self.root, text="Detection Info", 
#                                   font=('Arial', 11, 'bold'),
#                                   bg='#ecf0f1', fg='#2c3e50',
#                                   padx=10, pady=5)
#         info_frame.pack(fill='x', padx=10, pady=5)
        
#         self.sign_label = tk.Label(info_frame, text="Sign: --", font=('Arial', 10), bg='#ecf0f1', width=15, anchor='w')
#         self.sign_label.grid(row=0, column=0, sticky='w', padx=2)
        
#         self.area_label = tk.Label(info_frame, text="Area: 0", font=('Arial', 10), bg='#ecf0f1', width=15, anchor='w')
#         self.area_label.grid(row=0, column=1, sticky='w', padx=2)
        
#         self.coord_label = tk.Label(info_frame, text="Coord: X=0, Y=0", font=('Arial', 10), bg='#ecf0f1', width=25, anchor='w')
#         self.coord_label.grid(row=1, column=0, columnspan=2, sticky='w', padx=2)
        
#         self.action_display_label = tk.Label(info_frame, text="Action: --", 
#                                              font=('Arial', 11, 'bold'), 
#                                              bg='#ecf0f1', fg='#e74c3c', anchor='w')
#         self.action_display_label.grid(row=2, column=0, columnspan=2, sticky='w', pady=(5,0), padx=2)

#         # 7. IMU Monitor
#         imu_frame = tk.LabelFrame(self.root, text="IMU Monitor", 
#                                    font=('Arial', 11, 'bold'),
#                                    bg='#2c3e50', fg='#f39c12',
#                                    padx=10, pady=5)
#         imu_frame.pack(fill='x', padx=10, pady=5)
        
#         lbl_style = {'bg': '#2c3e50', 'fg': '#ecf0f1', 'font': ('Consolas', 10)}
#         val_style = {'bg': '#2c3e50', 'fg': '#3498db', 'font': ('Consolas', 11, 'bold')}
        
#         tk.Label(imu_frame, text="Current Yaw:", **lbl_style).grid(row=0, column=0, sticky='e', padx=2)
#         self.imu_yaw_val = tk.Label(imu_frame, text="0.0°", **val_style)
#         self.imu_yaw_val.grid(row=0, column=1, sticky='w', padx=5)
        
#         tk.Label(imu_frame, text="Target Yaw:", **lbl_style).grid(row=0, column=2, sticky='e', padx=2)
#         self.imu_target_val = tk.Label(imu_frame, text="--", **val_style)
#         self.imu_target_val.grid(row=0, column=3, sticky='w', padx=5)

#         # 8. Motor & Motion Monitor
#         motor_frame = tk.LabelFrame(self.root, text="Motor & Motion Monitor", 
#                                    font=('Arial', 11, 'bold'),
#                                    bg='#2c3e50', fg='#f1c40f',
#                                    padx=10, pady=5)
#         motor_frame.pack(fill='x', padx=10, pady=5)
        
#         lbl_style = {'bg': '#2c3e50', 'fg': '#ecf0f1', 'font': ('Consolas', 10)}
#         val_style = {'bg': '#2c3e50', 'fg': '#2ecc71', 'font': ('Consolas', 11, 'bold')}

#         tk.Label(motor_frame, text="Head H:", **lbl_style).grid(row=0, column=0, sticky='e', padx=2)
#         self.motor_h_val = tk.Label(motor_frame, text="2048", **val_style)
#         self.motor_h_val.grid(row=0, column=1, sticky='w', padx=5)

#         tk.Label(motor_frame, text="Head V:", **lbl_style).grid(row=1, column=0, sticky='e', padx=2)
#         self.motor_v_val = tk.Label(motor_frame, text="2048", **val_style)
#         self.motor_v_val.grid(row=1, column=1, sticky='w', padx=5)

#         tk.Label(motor_frame, text=" | ", bg='#2c3e50', fg='#7f8c8d').grid(row=0, column=2, rowspan=2)

#         tk.Label(motor_frame, text="Speed X:", **lbl_style).grid(row=0, column=3, sticky='e', padx=2)
#         self.move_x_val = tk.Label(motor_frame, text="0", **val_style)
#         self.move_x_val.grid(row=0, column=4, sticky='w', padx=5)

#         tk.Label(motor_frame, text="Speed Y:", **lbl_style).grid(row=1, column=3, sticky='e', padx=2)
#         self.move_y_val = tk.Label(motor_frame, text="0", **val_style)
#         self.move_y_val.grid(row=1, column=4, sticky='w', padx=5)

#         tk.Label(motor_frame, text="Theta:", **lbl_style).grid(row=0, column=5, sticky='e', padx=2)
#         self.move_th_val = tk.Label(motor_frame, text="0.00", **val_style)
#         self.move_th_val.grid(row=0, column=6, sticky='w', padx=5)
        
#         # 9. Current State
#         state_frame = tk.LabelFrame(self.root, text="State", 
#                                    font=('Arial', 11, 'bold'),
#                                    bg='#ecf0f1', fg='#2c3e50',
#                                    padx=5, pady=5)
#         state_frame.pack(fill='x', padx=10, pady=5)
        
#         self.state_label = tk.Label(state_frame, text="[SEARCH] Ready", 
#                                     font=('Arial', 11, 'bold'),
#                                     bg='#f39c12', fg='white',
#                                     padx=10, pady=5)
#         self.state_label.pack(fill='x')
        
#         # 10. Log
#         log_frame = tk.LabelFrame(self.root, text="Log", 
#                                  font=('Arial', 11, 'bold'),
#                                  bg='#ecf0f1', fg='#2c3e50',
#                                  padx=5, pady=5)
#         log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
#         scroll = tk.Scrollbar(log_frame)
#         scroll.pack(side='right', fill='y')
        
#         self.log_text = tk.Text(log_frame, height=5, 
#                                font=('Consolas', 9),
#                                bg='#2c3e50', fg='#ecf0f1',
#                                yscrollcommand=scroll.set,
#                                state='disabled')
#         self.log_text.pack(fill='both', expand=True)
#         scroll.config(command=self.log_text.yview)

#     def create_compact_slider(self, parent, label, from_, to, var, cmd, resolution=1, is_int=False, color='#3498db'):
#         frame = tk.Frame(parent, bg=parent['bg'])
#         frame.pack(fill='x', pady=2)
        
#         label_bg = parent['bg']
#         fg_color = 'white' if label_bg in ['#34495e', '#9b59b6'] else 'black'
        
#         tk.Label(frame, text=label, font=('Arial', 10), bg=label_bg, 
#                 fg=fg_color, width=14, anchor='w').pack(side='left')
        
#         val_text = str(var.get())
#         val_lbl = tk.Label(frame, text=val_text, font=('Arial', 10, 'bold'), 
#                            bg=color, fg='white', width=6)
#         val_lbl.pack(side='right', padx=5)
#         self.val_labels[str(var)] = val_lbl
        
#         scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
#                          orient='horizontal', variable=var, command=cmd,
#                          showvalue=0, bg=label_bg, troughcolor=color, highlightthickness=0,
#                          width=10)
#         scale.pack(side='left', fill='x', expand=True, padx=5)

#     def update_val_label(self, var, value, is_int=False):
#         lbl = self.val_labels.get(str(var))
#         if lbl:
#             if is_int:
#                 lbl.config(text=str(int(float(value))))
#             else:
#                 lbl.config(text=f"{float(value):.1f}")

#     def update_area_label(self, value): 
#         self.update_val_label(self.area_threshold, value, is_int=True)
    
#     def update_left_label(self, value): 
#         self.update_val_label(self.left_turn_duration, value)
    
#     def update_right_label(self, value): 
#         self.update_val_label(self.right_turn_duration, value)
    
#     def update_left_angle_label(self, value): 
#         self.update_val_label(self.left_turn_angle, value, is_int=True)
    
#     def update_right_angle_label(self, value): 
#         self.update_val_label(self.right_turn_angle, value, is_int=True)
    
#     def update_avoid_step1_label(self, value): 
#         self.update_val_label(self.avoid_step1_duration, value)
    
#     def update_avoid_step2_label(self, value): 
#         self.update_val_label(self.avoid_step2_duration, value)
    
#     def update_avoid_step3_label(self, value): 
#         self.update_val_label(self.avoid_step3_duration, value)
    
#     def update_strafe_speed_label(self, value): 
#         self.update_val_label(self.avoid_strafe_speed, value, is_int=True)
    
#     def toggle_robot_control(self):
#         current = self.robot_control_enabled.get()
#         new_state = not current
#         self.robot_control_enabled.set(new_state)
        
#         if new_state:
#             self.control_toggle_btn.config(text="[ ON ] Active Mode", bg='#2ecc71')
#             self.control_status_label.config(text="Commands WILL be sent to robot", fg='#2ecc71')
#             self.log_message('[CONTROL] Robot control ENABLED')
#             if hasattr(self, 'node'):
#                 self.node.enable_robot_control()
#         else:
#             self.control_toggle_btn.config(text="[ OFF ] Simulation Mode", bg='#e74c3c')
#             self.control_status_label.config(text="Commands will NOT be sent to robot", fg='#ecf0f1')
#             self.log_message('[CONTROL] Robot control DISABLED')
#             if hasattr(self, 'node'):
#                 self.node.disable_robot_control()
    
#     def get_robot_control_enabled(self): 
#         return self.robot_control_enabled.get()
    
#     def get_area_threshold(self): 
#         return self.area_threshold.get()
    
#     def get_left_turn_duration(self): 
#         return self.left_turn_duration.get()
    
#     def get_right_turn_duration(self): 
#         return self.right_turn_duration.get()
    
#     def get_left_turn_angle(self): 
#         return self.left_turn_angle.get()
    
#     def get_right_turn_angle(self): 
#         return self.right_turn_angle.get()
    
#     def get_avoid_step1_duration(self): 
#         return self.avoid_step1_duration.get()
    
#     def get_avoid_step2_duration(self): 
#         return self.avoid_step2_duration.get()
    
#     def get_avoid_step3_duration(self): 
#         return self.avoid_step3_duration.get()
    
#     def get_avoid_strafe_speed(self): 
#         return self.avoid_strafe_speed.get()
    
#     def update_detection(self, sign_name, x, y, area):
#         self.sign_label.config(text=f"Sign: {sign_name.upper() if sign_name else '--'}")
#         self.coord_label.config(text=f"Coord: X={x}, Y={y}")
#         self.area_label.config(text=f"Area: {area}")
    
#     def update_imu_status(self, current_yaw, target_yaw):
#         try:
#             self.imu_yaw_val.config(text=f"{current_yaw:.1f}°")
#             if target_yaw is not None:
#                 self.imu_target_val.config(text=f"{target_yaw:.1f}°", fg='#e74c3c')
#             else:
#                 self.imu_target_val.config(text="--", fg='#3498db')
#         except:
#             pass
    
#     def update_walking_status(self, is_walking):
#         try:
#             if is_walking:
#                 self.walking_status_label.config(text="Robot Status: WALKING", fg='#2ecc71')
#             else:
#                 self.walking_status_label.config(text="Robot Status: STOPPED", fg='#e74c3c')
#         except:
#             pass
    
#     def update_motor_status(self, head_h, head_v, spd_x, spd_y, theta):
#         try:
#             self.motor_h_val.config(text=f"{int(head_h)}")
#             self.motor_v_val.config(text=f"{int(head_v)}")
#             self.move_x_val.config(text=f"{int(spd_x)}")
#             self.move_y_val.config(text=f"{int(spd_y)}")
#             self.move_th_val.config(text=f"{theta:.2f}")
#         except:
#             pass

#     def update_robot_action(self, action):
#         if self.action_display_label is None: 
#             return
#         self.action_display_label.config(text=f"Action: {action}")
#         if "LEFT" in action or "STRAFE LEFT" in action: 
#             self.action_display_label.config(fg='#3498db')
#         elif "RIGHT" in action or "STRAFE RIGHT" in action: 
#             self.action_display_label.config(fg='#e74c3c')
#         elif "FORWARD" in action: 
#             self.action_display_label.config(fg='#2ecc71')
#         elif "Head Search" in action: 
#             self.action_display_label.config(fg='#9b59b6')
#         elif "Step" in action:
#             self.action_display_label.config(fg='#f39c12')
#         else: 
#             self.action_display_label.config(fg='#2c3e50')
        
#     def update_state(self, state):
#         self.state_label.config(text=state)
#         if 'AVOID' in state:
#             self.state_label.config(bg='#9b59b6')
#         elif 'TURNING' in state or 'EXECUTING' in state: 
#             self.state_label.config(bg='#e74c3c')
#         elif 'ACTION' in state: 
#             self.state_label.config(bg='#c0392b')
#         elif 'LOCK' in state or 'ALIGNED' in state: 
#             self.state_label.config(bg='#2ecc71')
#         elif 'ALIGN' in state: 
#             self.state_label.config(bg='#27ae60')
#         elif 'HEAD_SEARCH' in state: 
#             self.state_label.config(bg='#3498db')
#         elif 'DETECT' in state: 
#             self.state_label.config(bg='#f39c12')
#         else: 
#             self.state_label.config(bg='#95a5a6')
    
#     def log_message(self, message):
#         self.log_text.config(state='normal')
#         self.log_text.insert('end', f'{message}\n')
#         self.log_text.see('end')
#         self.log_text.config(state='disabled')
    
#     def run(self):
#         self.root.mainloop()


# def ros_spin(node):
#     rclpy.spin(node)


# def main(args=None):
#     rclpy.init(args=args)
    
#     gui = YOLOGUI()
#     node = YOLOTestNode(gui)
    
#     # 讓 GUI 可以存取 node
#     gui.node = node
    
#     ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
#     ros_thread.start()
    
#     try:
#         gui.run()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         # 關閉時停止機器人
#         if node.api_mode and node.api:
#             node.send_body_auto(0)
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()