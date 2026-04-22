#!/usr/bin/env python3
import rclpy
import os
import sys
import time
import cv2
import numpy as np
import threading
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import MultiThreadedExecutor

# 假設這是你的 API 來源
from strategy.API import API

# ================= 參數定義 =================
FOCUS_MATRIX = [
     1, 1, 2, 3, 4, 5, 6,
     7, 7, 8, 8, 9,10,11,
    12,14,14,12,
    11,10, 9, 8, 8, 7, 7,
     6, 5, 4, 3, 2, 1, 1
]


HEAD_HORIZONTAL = 2030
HEAD_VERTICAL = 2800
#原地步態
STAY_X     = 1700
STAY_Y     = 0
STAY_THETA = 0
#大前進
MAX_FORWARD_X       = -6000
MAX_FORWARD_Y       = 0
MAX_FORWARD_THETA   = 0
#小前進
SMALL_FOEWARD_X     = 3000
SMALL_FOEWARD_Y     = 0
SMALL_FOEWARD_THETA = 0
#dx右轉(45)
TURN_RIGHT_X     = 1700
TURN_RIGHT_Y     = -1000
TURN_RIGHT_THETA = -5
#右轉(90)
TURN_RIGHT_90_X     = 1700
TURN_RIGHT_90_Y     = -1000
TURN_RIGHT_90_THETA = -5
#imu右轉
IMU_RIGHT_X = 1700
IMU_RIGHT_Y = -1000
#IMU__RIGHT_THETA max(5,-5)
#dx左轉(45)
TURN_LEFT_X = 2000
TURN_LEFT_Y = 600
TURN_LEFT_THETA = 5
#左轉(90)
TURN_LEFT_90_X     = 2000
TURN_LEFT_90_Y     = 600
TURN_LEFT_90_THETA = 5
#imu左轉
IMU_LEFT_X = 2000
IMU_LEFT_Y = 600
#IMU_LEFT_THETA max(5,-5)

#開局動作
PRE_ACT = 'start' # 'start' 'preturn_L' 'preturn_R'
PRE_TURN_ANGLE = 20
TURN_HEAD_RANGE = 12
TURN_90           = False #是否啟用轉90度判斷
WALK_FORWARD_ZONE = 6   #可從障礙物中間通過的可容忍範圍
# RECHECK_ZONE      = 7   #障礙物偏離中心重新判斷轉向
ACCEL_STEP        = 100 #每秒增加/減少的速度量 
IMU_FIX           = 3   #imu修正容許值


class NonBlockingDelay:
    def __init__(self):
        self.start_time = 0.0
        self.is_waiting = False

    def check(self, delay_seconds):
        """ 檢查是否已經經過指定的秒數 """
        if not self.is_waiting:
            # 第一次進來，開始計時
            self.start_time = time.time()
            self.is_waiting = True
            return False
        
        if time.time() - self.start_time >= delay_seconds:
            # 時間到，重置狀態以供下次使用，並回傳 True
            self.is_waiting = False
            return True
            
        # 時間還沒到
        return False
        
    def reset(self):
        """ 強制重置計時器 (當狀態被外部強制切換時可呼叫) """
        self.is_waiting = False
# ================= Calculate (負責影像與視覺參數計算) =================
class Calculate():
    def __init__(self, robot_core):
        self.robot = robot_core  
        self.bridge = CvBridge()
        
        self.depth = [24] * 32   
        self.Deep_Matrix = []
        self.red_width = 0
        self.deep_x = 0
        self.deep_y = 24
        self.center_deep = 24
        self.left_deep = 24
        self.right_deep = 24
        self.deep_sum = 0
        self.deep_sum_l = 0
        self.deep_sum_r = 0
        
        self.now_speed = 0
        self.speed_x = 0
        self.y = 0
        self.theta = 0
        self.last_action = "stop"
        
        # 除錯用變數
        self.img_received = False
        self.last_error = "None"

    def convert(self, msg: Image):
        try:
            self.img_received = True 
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.red_width = 0
            # 【修改 1】：不要使用 self.Deep_Matrix，改用區域變數 local_matrix
            local_matrix = []
            
            cv_image = cv2.resize(cv_image, (320, 240), interpolation=cv2.INTER_NEAREST)
            cv_image_2 = cv2.resize(cv_image, (32, 24), interpolation=cv2.INTER_NEAREST)
            
            for compress_width in range(0, 32):
                local_matrix.append(0) # 👈 修改這裡
                for compress_height in range(23, -1, -1):
                    b = cv_image_2.item(compress_height, compress_width, 0)
                    g = cv_image_2.item(compress_height, compress_width, 1)
                    r = cv_image_2.item(compress_height, compress_width, 2)
                    if (b == 128 and g == 0 and r == 128) or (b ==128 and g ==128 and r ==0):
                        local_matrix[compress_width] = 23 - compress_height # 👈 修改這裡
                        break
                    if b == 255 and g == 255 and r == 0:
                        local_matrix[compress_width] = 18 - compress_height # 👈 修改這裡
                        break
                    if compress_height == 0:
                        local_matrix[compress_width] = 24 # 👈 修改這裡
                        
            # 【修改 2】：全部算完且長度確定是 32 後，再原子性地覆蓋給 self.depth
            self.depth = local_matrix

            self.calculate()
            return self.depth
            
        except Exception as e:
            self.last_error = f"Convert Error: {e}"
            return self.depth

    def calculate(self):
        try:
            # ----------------Filter_matrix-------------------
            filter_matrix = [max(0, a - b) for a, b in zip(FOCUS_MATRIX, self.depth)]
            x_center_num = sum(
                i for i, num in enumerate(FOCUS_MATRIX - np.array(self.depth)) if num >= 0
            )
            x_center_cnt = np.sum(np.array(FOCUS_MATRIX) - np.array(self.depth) >= 0)
            x_center = (x_center_num / x_center_cnt) if x_center_cnt > 0 else 0
            
            left_weight_matrix = list(range(32))
            right_weight_matrix = list(range(31, -1, -1))
            right_weight = np.dot(filter_matrix, right_weight_matrix)
            left_weight = np.dot(filter_matrix, left_weight_matrix)
            
            # self.deep_y = min(np.array(self.depth))
            invaders = [self.depth[i] for i in range(32) if self.depth[i] < FOCUS_MATRIX[i]]
            self.deep_y = min(invaders) if invaders else 24

            self.deep_sum = sum(np.array(self.depth))
            self.deep_sum_l = sum(np.array(self.depth)[0:16])
            self.deep_sum_r = sum(np.array(self.depth)[17:32])
            self.left_deep = np.array(self.depth)[4]
            self.right_deep = np.array(self.depth)[28]
            self.center_deep = np.array(self.depth)[16]
            
            x_boundary = 31 if left_weight > right_weight else 0
            self.deep_x = x_center - x_boundary
            
            if self.deep_x > 16:
                self.deep_x = 16
            elif self.deep_x < -16:
                self.deep_x = -16        
        except Exception as e:
            self.last_error = f"Calc Error: {e}"

    def move(self, action_id):
        self.last_action = action_id
            
        actions = {
            "stay": {"x": STAY_X, "y": STAY_Y, "theta": STAY_THETA},
            "max_speed": {"x": MAX_FORWARD_X, "y": MAX_FORWARD_Y, "theta": MAX_FORWARD_THETA},
            "small_forward": {"x": SMALL_FOEWARD_X, "y": SMALL_FOEWARD_Y, "theta": SMALL_FOEWARD_THETA},
            "imu_fix": {"x": IMU_RIGHT_X if getattr(self.robot, 'yaw', 0) > 0 else IMU_LEFT_X, "y": 0, "theta": self.imu_angle()},
            "turn_right": {"x": TURN_RIGHT_X, "y": TURN_RIGHT_X, "theta": TURN_RIGHT_90_THETA},
            "turn_right_90": {"x": TURN_RIGHT_90_X, "y": TURN_RIGHT_90_Y, "theta": TURN_RIGHT_90_THETA},
            "turn_left": {"x": TURN_LEFT_X, "y": 0, "theta": TURN_LEFT_THETA},
            "turn_left_90": {"x": TURN_LEFT_90_X, "y": TURN_LEFT_90_Y, "theta": TURN_LEFT_90_THETA},
            "stay_wait": {"x": STAY_X, "y": STAY_Y, "theta": STAY_THETA},
            # "preturn_left": {"x": TURN_LEFT_X, "y": 0, "theta": TURN_LEFT_THETA},
            # "preturn_right": {"x": TURN_RIGHT_X, "y": 0, "theta": TURN_RIGHT_THETA},
        }
        
        action = actions.get(action_id, None)
        if action is not None:
            x = action["x"]
            self.y = action["y"]
            self.theta = action["theta"]
            
            if action_id in ['max_speed', 'stay_wait']:
                if self.speed_x < x:
                    self.speed_x = min(self.speed_x + ACCEL_STEP, x)
                elif self.speed_x > x:
                    self.speed_x = max(self.speed_x - ACCEL_STEP, x)

            self.robot.sendContinuousValue(self.speed_x, self.y, self.theta)

    def imu_angle(self):
        imu_ranges = [
            (180, -5), (90, -5), (60, -5), (45, -5), (20, -4), (10, -3), (5, -3),
            (2, -2), (0, 0), (-2, 2), (-5, 3), (-10, 3), (-20, 4), (-45, 5),
            (-60, 5), (-90, 5), (-180, 5),
        ]
        for imu_range in imu_ranges:
            if getattr(self.robot, 'yaw', 0) >= imu_range[0]:
                return imu_range[1]
        return 0


# ================= RobotStatus =================
class RobotStatus():
    def __init__(self, calc, robot):
        self.calc = calc
        self.robot = robot
        
        self.is_start           = False
        self.action_id          = "stop"
        self.ContinuousValue    = [0,0,0]
        self.obs_status         = "" 
        self.pre_status         = ""
        
        self.imu                = 0
        self.deep_x             = 0
        self.deep_y             = 0
        self.deep_sum_l         = 0
        self.deep_sum_r         = 0
        self.left_deep          = 0
        self.right_deep         = 0
        self.center_deep        = 0
        
        self.last_update_time   = time.time()
        self.running            = True
        self.grid_sent = False  # 新增旗標：確保只發送一次繪圖訊息
        self.last_update_time = time.time()

    def draw_focus_grid(self):
        """ 繪製 FOCUS_MATRIX 的階梯狀邊界 (修正連接邏輯與通訊塞車) """
        try:
            time.sleep(1) # 給系統一點初始化的緩衝時間
            
            for i in range(len(FOCUS_MATRIX)):
                v = FOCUS_MATRIX[i]
                
                # 計算 X 軸範圍 (每格 10 像素)
                x_start = i * 10
                x_end = (i + 1) * 10
                
                # 計算 Y 軸座標 (公式：y = 240 - v * 10)
                y_now = 240 - (v * 10)
                
                # 1. 繪製橫線 - 修改 ID 範圍為 100~131 (確保在 255 以內)
                self.robot.drawImageFunction(100 + i, 1, int(x_start), int(x_end), int(y_now), int(y_now), 255, 255, 255)
                time.sleep(0.03) # 【關鍵】：每畫一條線停頓 30 毫秒，讓硬體消化封包
                
                # 2. 繪製垂直連接線
                if i > 0:
                    y_prev = 240 - (FOCUS_MATRIX[i-1] * 10)
                    
                    if y_now != y_prev:
                        y_top = min(y_now, y_prev)
                        y_bottom = max(y_now, y_prev)
                        
                        # 修改 ID 範圍為 150~181 (確保在 255 以內，且不與橫線衝突)
                        self.robot.drawImageFunction(150 + i, 1, int(x_start), int(x_start), int(y_top), int(y_bottom), 255, 255, 255)
                        time.sleep(0.03) # 【關鍵】：同樣給予硬體消化時間
            
            # 標記已發送，防止重複執行
            self.grid_sent = True
            
        except Exception as e:
            self.calc.last_error = f"Grid Draw Error: {e}"
    def update(self):
        self.is_start           = getattr(self.robot, 'is_start', False)
        self.action_id          = self.calc.last_action
        self.ContinuousValue    = [self.calc.speed_x, self.calc.y, self.calc.theta] 
        self.obs_status         = self.robot.status
        self.pre_status         = self.robot.pre_status
        self.imu                = getattr(self.robot, 'yaw', 0)
        self.deep_x             = self.calc.deep_x
        self.deep_y             = self.calc.deep_y
        self.deep_sum_l         = self.calc.deep_sum_l
        self.deep_sum_r         = self.calc.deep_sum_r
        self.left_deep          = self.calc.left_deep
        self.right_deep         = self.calc.right_deep
        self.center_deep        = self.calc.center_deep

    def val_print(self):
        self.update()
        sys.stdout.write("\033[H\033[J")
        start_str = "True" if self.is_start else "False (PAUSED)"
        
        # 顯示影像是否正常接收
        img_str = "\033[92mOK\033[0m" if self.calc.img_received else "\033[91mNO IMAGE (/build_image)\033[0m"
        
        sys.stdout.write(f"\
#==============機器人狀態==============#\n\
is_start         : {start_str}\n\
action_id        : {self.action_id}\n\
ContinuousValue  : {self.ContinuousValue}\n\
#===============避障狀態===============#\n\
status           : {self.obs_status}\n\
pre_status       : {self.pre_status}\n\
#===============系統狀態===============#\n\
Image Topic      : {img_str}\n\
System Error     : {self.calc.last_error}\n\
#===============基礎數據===============#\n\
imu (yaw)        : {self.imu}\n\
running          : {self.running}\n\
deep_x           : {self.deep_x}\n\
deep_y           : {self.deep_y}\n\
center_deep      : {self.center_deep}\n\
deep_sum_lr      : {self.deep_sum_l} {self.deep_sum_r}\n\
deep_sum         : {self.calc.deep_sum}\n\
LCRdeep          : {self.left_deep} {self.center_deep} {self.right_deep}\n\
#=====================================#\n\
")
        sys.stdout.flush()

    def draw_function(self):        
        try:
            if getattr(self.robot, 'is_start', False) and not self.grid_sent:
                self.draw_focus_grid()
            else:
                draw_x = abs(self.deep_x*10) if self.deep_x < 0 else 320-(self.deep_x*10)
                self.robot.drawImageFunction(1,1,draw_x,draw_x,0,240,255,0,0)            
                self.robot.drawImageFunction(2,1,0,320,240-self.deep_y*10,240-self.deep_y*10,255,0,255)
        except Exception as e:
            self.calc.last_error = f"Draw Error: {e}"

    def runThread(self):
        self.running = True
        while rclpy.ok():
            try:
                if time.time() - self.last_update_time > 0.1: # 10Hz 刷新率
                    self.val_print()
                    self.draw_function()
                    self.last_update_time = time.time()
                time.sleep(0.05)
            except Exception as e:
                # 如果這條 Thread 真的崩潰，至少會印出錯誤而不會被吃掉
                print(f"Thread Error: {e}")


# ================= Obs =================
class Obs(API):
    def __init__(self):
        super().__init__('obs_node')
        
        self.calc = Calculate(self)
        self.status_mgr = RobotStatus(self.calc, self)

        self.cbg = ReentrantCallbackGroup()

        self.image_sub = self.create_subscription(
            Image, 'processed_image', self.calc.convert, 10, callback_group=self.cbg
        )
        self.delay = NonBlockingDelay()
        self.turn_head_step = 0

        self.timer = self.create_timer(0.05, self.main, callback_group=self.cbg)
        
        self.status_thread = threading.Thread(target=self.status_mgr.runThread, daemon=True)
        self.status_thread.start()

        self.status = PRE_ACT
        self.pre_status = ''
        self.imu_ok = False
        self.body_auto = False
        
        self.initial()

    def initial(self):
        self.pre_status = ''
        self.imu_ok = False
        self.body_auto = False
        self.sendHeadMotor(1, HEAD_HORIZONTAL, 50)
        self.sendHeadMotor(2, HEAD_VERTICAL, 50)
        time.sleep(2)
        self.sendSensorReset(True)
        self.left_deep_sum = 0
        self.right_deep_sum = 0
        self.deep_sum = 0
        self.turn_head_step = 0

    def walk_switch(self):
        if self.body_auto:
            self.sendbodyAuto(0)
            self.body_auto = False
        else:
            self.sendbodyAuto(1)
            self.body_auto = True

    def main(self):
        # self.calc.calculate()
        
        # 你的測試碼：強制啟動。若你想用硬體開關控制，請把這行註解掉
        # self.is_start = True 
        # time.sleep(10)
        
        if getattr(self, 'is_start', False):
            #1.預轉身=========================================
            if self.status == 'preturn_L':
                if not self.body_auto:
                    self.walk_switch()
                if self.imu_rpy[2] < PRE_TURN_ANGLE:
                    self.calc.move("turn_left")
                else:
                    self.pre_status = self.status
                    self.status = 'starting_walking_with_obs'
                    
            elif self.status == 'preturn_R':
                if not self.body_auto:
                    self.walk_switch()
                if self.imu_rpy[2] > -PRE_TURN_ANGLE:
                    self.calc.move("turn_right")
                else:
                    self.pre_status = self.status
                    self.status = 'starting_walking_with_obs'
            #2.判斷面前是否有障礙物==============================
            elif self.status == 'start':
                if not self.body_auto:
                    self.walk_switch()
                if self.calc.deep_y < 24:#小於24代表有東西
                    self.pre_status = self.status
                    self.status = 'starting_walking_with_obs'
                else:
                    self.pre_status = self.status
                    self.status = 'starting_walking_without_obs'
            #3.有障礙物時======================================
            elif self.status == 'starting_walking_with_obs':
                if abs(self.calc.deep_x) <= WALK_FORWARD_ZONE:#障礙物在兩側 可從中間過
                    self.pre_status = self.status
                    self.status = 'walk_forword'
                elif -14 < self.calc.deep_x < -WALK_FORWARD_ZONE:#障礙物在右邊
                    self.calc.move("turn_left")
                elif WALK_FORWARD_ZONE < self.calc.deep_x < 14:#障礙物在左邊
                    self.calc.move("turn_right")

                elif -16 <= self.calc.deep_x <= -14 or 14 <= self.calc.deep_x <= 16:#障礙物在正中間
                    self.pre_status = self.status
                    self.status = 'imu_fix'

            #4.執行轉頭策略(障礙物在正中間、整體偏左)===============
            elif self.status =='turn_right_90':
                if self.imu_rpy[2] > -50:
                    self.calc.move("turn_right_90")
                else:
                    self.pre_status = self.status
                    self.status = "walk_forword"
            #5.執行轉頭策略(障礙物在正中間、整體偏右)===============
            elif self.status =='turn_left_90':
                if self.imu_rpy[2] < 50:
                    self.calc.move("turn_left_90")
                else:
                    self.pre_status = self.status
                    self.status = "walk_forword"
            #6.直走(障礙物在兩側)================================
            elif self.status == 'walk_forword':
                if self.calc.deep_y < 24:
                    self.calc.move("small_forward")
                    if abs(self.calc.deep_x) > WALK_FORWARD_ZONE+1:
                        self.pre_status = self.status
                        self.status = 'starting_walking_with_obs'
                else:
                    self.pre_status = self.status
                    self.status = 'starting_walking_without_obs'
            #7.沒障礙物時========================================
            elif self.status == 'starting_walking_without_obs':
                if self.calc.deep_y < 24 :
                    self.pre_status = self.status
                    self.status = 'stay_wait'
                    # self.status = 'starting_walking_with_obs'
                else:
                    self.calc.move("max_speed")
            #8.高速看到障礙物時緩減速==============================
            elif self.status == 'stay_wait':
                if self.calc.speed_x <= STAY_X:#等待到減速完成
                    self.pre_status = self.status
                    self.status = 'imu_fix'
                else:
                    self.calc.move("stay_wait")
            #9.imu修正回0=======================================
            elif self.status == 'imu_fix':                
                if(abs(self.imu_rpy[2]) > IMU_FIX):
                    self.calc.move("imu_fix")                
                else:
                    if (self.calc.left_deep < TURN_HEAD_RANGE) and (self.calc.right_deep < TURN_HEAD_RANGE) and (self.calc.center_deep < TURN_HEAD_RANGE):
                        self.pre_status = self.status
                        self.status = 'turn_head'
                    elif self.calc.deep_y < 24:
                        self.pre_status = self.status
                        self.status = 'starting_walking_with_obs'
                    else:
                        self.pre_status = self.status
                        self.status = 'starting_walking_without_obs'
            elif self.status == 'turn_head':
                    self.calc.move("stay")
                    if self.turn_head_step == 0:
                        self.sendHeadMotor(1, HEAD_HORIZONTAL-500, 100)
                        self.sendHeadMotor(2, HEAD_VERTICAL, 50)
                        self.delay.reset() # 確保計時器重置
                        self.turn_head_step = 1 # 切換到下一步
                        
                    # 第 1 步：不卡死地等待 1 秒，讀取右邊數值，然後向左看
                    elif self.turn_head_step == 1:
                        if self.delay.check(1.0):
                            self.left_deep_sum = self.calc.deep_sum
                            self.sendHeadMotor(1, HEAD_HORIZONTAL+500, 100)
                            self.sendHeadMotor(2, HEAD_VERTICAL, 50)
                            self.turn_head_step = 2 # 切換到下一步
                            
                    # 第 2 步：不卡死地等待 1 秒，讀取左邊數值，頭部回正並退出狀態
                    elif self.turn_head_step == 2:
                        if self.delay.check(1.0):
                            self.right_deep_sum = self.calc.deep_sum
                            self.sendHeadMotor(1, HEAD_HORIZONTAL, 100)
                            self.sendHeadMotor(2, HEAD_VERTICAL, 50)
                            
                            # 決策接下來的主狀態
                            self.pre_status = self.status
                            if self.left_deep_sum > self.right_deep_sum:
                                self.status = 'turn_right_90'
                            else:
                                self.status = 'turn_left_90'
                                
                            # 【非常重要】離開前把計步器歸零，下次轉頭才能從第 0 步開始！
                            self.turn_head_step = 0

        else:
            if self.body_auto:
                self.walk_switch()
                self.initial()
            # time.sleep(1)
            
# ================= 執行進入點 =================
def main(args=None):
    rclpy.init(args=args)
    
    node = Obs()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()