#!/usr/bin/env python3
# coding=utf-8
import rclpy
import time
from strategy.API import API  # 確保此路徑與您的 API.py 一致
from tku_msgs.msg import Dio
import time
import math
import numpy as np#!/usr/bin/env python3
# coding=utf-8
import rclpy
import time
from strategy.API import API  # 確保此路徑與您的 API.py 一致
from tku_msgs.msg import Dio
import time
import math
import numpy as np
import cv2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tku_msgs.msg import SensorPackage

# --- 全域參數 (對齊原始邏輯) ---
WIGHT = 678
HEAD_MOTOR_START = 2800    
HEAD_MOTOR_FINISH = 2900    
FLAG1 = False  
THETA_FIX = -0.2
PRETURN = 1  # 1:右 2:左

# 原始權重邏輯判斷 (修正重複判斷與賦值錯誤)
if WIGHT == 678:
    THIRD_LINE = 203
    FOURTH_LINE = 190
    SPEED = 3500
    PICK_ONE = 70
    PICK_TWO = 71
    LIFT = 72
    LIFT2 = 73
else:
    THIRD_LINE = 215
    SPEED = 1800
    PICK_ONE = 401
    PICK_TWO = 402
    LIFT = 404

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ObjectInfo:
    color_dict = {'Orange': 0, 'Yellow': 1, 'Blue': 2, 'Green': 3, 'Black': 4, 'Red': 5, 'White': 6}

    def __init__(self, master_node, color, object_type,):
        # 正確寫法：直接接收外部傳入的節點實例
        self.node = master_node
        self.color = self.color_dict[color]
        self.edge_max = Coordinate(0, 0)
        self.edge_min = Coordinate(0, 0)
        self.center = Coordinate(0, 0)
        self.get_target = False
        self.target_size = 0

        update_strategy = {'rise_line': self.get_line_object, 
                           'weight_bar': self.get_bar_object}
        self.find_object = update_strategy[object_type]

    def get_line_object(self):
        # 透過實例存取主節點更新好的資料
        sizes = self.node.object_sizes[self.color]
        if sizes:
            max_size = max(sizes)
            return sizes.index(max_size) if max_size > 1000 else None
        

    def get_bar_object(self):
        sizes = self.node.object_sizes[self.color]
        if sizes:
            max_size = max(sizes)
            return sizes.index(max_size) if max_size > 50 else None
        
    
    def update(self, ID):
        object_idx = self.find_object()
        if object_idx is not None:
            self.get_target = True
            print("i see")
            # 取得邊界座標
            self.edge_max.x = self.node.object_x_max[self.color][object_idx]
            self.edge_min.x = self.node.object_x_min[self.color][object_idx]
            self.edge_max.y = self.node.object_y_max[self.color][object_idx]
            self.edge_min.y = self.node.object_y_min[self.color][object_idx]
            
            # --- 核心修改：直接使用 (Min + Max) // 2 計算中心點 ---
            # 使用 // 確保結果為 int，避免 ROS2 的型態斷言錯誤606
            self.center.x = (self.edge_min.x + self.edge_max.x) // 2
            self.center.y = (self.edge_min.y + self.edge_max.y) // 2
            
            self.target_size = self.node.object_sizes[self.color][object_idx]
            
            # 呼叫主節點畫圖功能
            self.node.drawImageFunction(int(ID), 1, self.edge_min.x, self.edge_max.x, self.edge_min.y, self.edge_max.y, 0, 0, 255)
        else:
            print("i cant see")
            self.edge_max.x = self.edge_min.x = self.edge_max.y = self.edge_min.y = 0
            self.center.x = self.center.y = self.target_size = 0
            self.get_target = False

class WeightLift(API):
    def __init__(self):
        super().__init__('WeightLift_Node')
        # 初始化子物件，傳入 self (API 節點)
        self.line = ObjectInfo(self, 'White', 'rise_line')
        self.bar = ObjectInfo(self, 'Red', 'weight_bar')       
        self.init_logic()
        self.create_timer(0.2, self.main_strategy)

    def init_logic(self):
        self.theta = 0
        self.ctrl_status = 'head_shake'
        self.body_auto = False
        self.third_line = False
        self.fourth_line = False
        self.stop = True
        self.real_bar_center = 160
        self.speed = 0
        self.crew = False

    def walk_switch(self):
        time.sleep(0.5)
        
        if self.body_auto:
            self.body_auto = False
            self.sendbodyAuto(0)
        else:
            self.body_auto = True
            self.sendbodyAuto(1)

    def imu_fix(self):
        theta = 0
        if self.imu_rpy[2] > 2.0:
            theta = -2
        elif self.imu_rpy[2] < -2.0:
            theta = 2
        return theta

    def walk_parameter(self, yaw, Y_COM):
        # self.sendSensorReset(True)
        self.sendLCWalkParameter(                        
                        com_y_swing  = float(Y_COM),   #起步步態補償
                        width_size   = float(0),  #雙腳距離
                        period_t     = int(420),  #步態頻率
                        t_dsp = float(0),
                        lift_height = float(5),
                        stand_height = float(50), #機器人初始站姿高度
                        com_height   = float(40)  #質心高度
                    )               

    def walking(self, yaw, Y_COM):
        if not self.body_auto:
            self.walk_parameter(yaw, Y_COM)
            self.walk_switch()
        self.theta = self.imu_fix()
        if self.ctrl_status == 'fourth_line':
            #if self.speed < 1800: self.speed += 200
            self.sendContinuousValue(3500, 0, self.theta) #60:self.theta #70:self.theta #80:self.theta+1
        elif self.ctrl_status == 'second_line':
            self.sendContinuousValue(4000, 500, self.theta) #60:self.theta #70:self.theta #80:self.theta+1
        else:
            self.sendContinuousValue(3500, 0, self.theta) #60:self.theta #70:self.theta #80:self.theta+1

    def main_strategy(self):
        if self.is_start:
            self.get_logger().info(f'ctrl_status : {self.ctrl_status}')
            # self.sendSensorReset(True)
            if self.ctrl_status == 'head_shake':
                print("head_shake")
                # self.sendBodySector(15)  #提右手拉左腳
                # time.sleep(1)
                # self.sendSensorReset(True)
    
                self.stop = False
                time.sleep(0.5)
                self.ctrl_status = 'preturn'

            # 更新視覺資訊 (會讀取 API 內部的 object_sizes 等資料)
                # self.bar.update(1)
                # self.line.update(2)

            if self.ctrl_status == 'preturn':
                print(PRETURN)
                self.get_logger().info(f'preturn')

                if not self.body_auto:
                    self.walk_parameter(0, 0)
                    self.walk_switch()
                    self.sendSensorReset(True)
                    self.theta = self.imu_fix()
                    if PRETURN == 1: # 右
                        self.bar.update(1)
                        self.line.update(2)
                        while True:
                            rclpy.spin_once(self, timeout_sec=0)
                            if 110 <= self.bar.center.x or self.bar.center.x > 260: break
                            self.bar.update(1)
                            self.line.update(2)
                            self.sendContinuousValue(2500, 0, self.theta)
                            self.get_logger().info(f"紅色preturn (R) = {self.bar.center.x}")               
                    elif PRETURN == 2: # 左
                        self.bar.update(1)
                        self.line.update(2)
                        while True:
                            rclpy.spin_once(self, timeout_sec=0)
                            if 30 <= self.bar.center.x or self.bar.center.x > 180: break
                            self.bar.update(1)
                            self.line.update(2)
                            self.sendContinuousValue(2500, -1000, self.theta)
                            self.get_logger().info(f"紅色preturn (L) = {self.bar.center.x}")
                            if self.bar.center.x <= 188:
                                self.sendContinuousValue(1000, -1000, self.theta)
            
                self.ctrl_status = 'start_line'
                time.sleep(0.5)
            elif self.ctrl_status == 'start_line':
                # 根據 IMU 角度補償平移或轉向
                self.get_logger().info(f"紅色preturn (L) = {self.bar.center.x}")
                self.get_logger().info(f"紅色preturn (Y) = {self.bar.center.y}")
                self.bar.update(1)
                self.theta = self.imu_fix()
                #if self.bar.center.x > 170:
                        #self.sendContinuousValue(800, -800, -1)
                        #print("右轉")
                    #elif self.bar.center.x < 145 and self.bar.center.x > 0:
                        #self.sendContinuousValue(800, 800, 1)
                        #print("左轉") 
                if self.bar.center.x > 165:
                    self.sendContinuousValue(2000, -800, self.theta) 
                    print("右平移")
                elif self.bar.center.x < 170 and self.bar.center.x > 0:
                    self.sendContinuousValue(2000, 1200, self.theta) 
                    print("左平移")  
                else:
                    self.walking(0, 0) # 直走
                if self.bar.center.y >= 123:
                    self.ctrl_status = 'turn_straight'

            elif self.ctrl_status == 'turn_straight':
                print("turn_straight")
                self.theta = self.imu_fix()
                self.sendContinuousValue(-1700, 200, self.theta) #1:self.theta-2 #2:self.theta+0.25
                if abs(self.theta) <= 0.2:
                    time.sleep(0.5)
                    self.ctrl_status = 'pick_up'
                    self.get_logger().info(f'ctrl_status : {self.ctrl_status}')
            elif self.ctrl_status == 'pick_up':
                print("pick_up")
                if self.body_auto: 
                    self.walk_switch()
                time.sleep(2.5)
                #self.sendBodySector(123)  #拉左腳
                time.sleep(1)
                if self.body_auto: 
                    self.walk_switch()
                self.crew = True
                time.sleep(2)
                self.sendBodySector(PICK_ONE)
                time.sleep(17)
                self.sendBodySector(PICK_TWO)
                time.sleep(11)
                self.bar.update(1)
                self.sendHeadMotor(2, 2700, 100)
                time.sleep(1)
                #self.sendBodySector(234)
                # time.sleep(1)
                
                self.real_bar_center = self.bar.center.x
                self.ctrl_status = 'second_line'

            elif self.ctrl_status == 'second_line':
                print("second_line")
                self.line.update(2)
                self.walking(-1, 0)                                    #一舉
                if self.line.edge_min.y < 200 and self.line.edge_min.y > 50:   #白線
                    self.third_line = True 
                print(self.third_line)
                self.get_logger().info(f"white_Y_min = {self.line.edge_min.y}")
                self.get_logger().info(f"white_Y_max = {self.line.edge_max.y}")
                if self.line.edge_max.y >= THIRD_LINE and self.third_line :
                    self.ctrl_status = 'rise_up'
                    time.sleep(5.5)

            elif self.ctrl_status == 'rise_up':
                print("rise_up")
                if self.body_auto: 
                    self.walk_switch()
                time.sleep(2)
                self.sendBodySector(int(LIFT))
                # 根據重量決定舉起後的等待時間
                # wait_time = 23.5 if WIGHT == 90 else 21 if WIGHT == 80 else 21 if WIGHT == 70 else 17
                # time.sleep(wait_time)
                # # 根據舉起時的中心偏移進行位移修正
                # if 165 < self.real_bar_center < 210:
                #     for _ in range(min(int((self.real_bar_center - 165) // 7), 4)):
                #         self.sendBodySector(606)
                #     time.sleep(1.5) 
                # elif 120 < self.real_bar_center < 155:
                #     for _ in range(min(int((165 - self.real_bar_center) // 7), 4)):
                #         self.sendBodySector(607)
                time.sleep(4)
                #self.sendBodySector(234)   #60迴圈外 ＃70迴圈內
                #time.sleep(0.5)
                if FLAG1:
                    sector_fix = 3336 if WIGHT == 90 else 3335 if WIGHT == 80 else 3334 if WIGHT == 70 else 3333 if WIGHT == 60 else None
                    if sector_fix: self.sendBodySector(int(sector_fix))
                self.ctrl_status = 'fourth_line'

            elif self.ctrl_status == 'fourth_line':
                print("fourth_line")
                self.line.update(2)
                self.walking(0, -2)
                if self.line.edge_min.y < 95 and self.line.edge_min.y > 75:   #白線
                    self.fourth_line = True 
                print(self.fourth_line)
                self.get_logger().info(f"white_Y_min = {self.line.edge_min.y}")
                self.get_logger().info(f"white_Y_max = {self.line.edge_max.y}")
                self.sendHeadMotor(2,2700, 100)
                if self.line.edge_max.y >= FOURTH_LINE and self.fourth_line :
                    self.ctrl_status = 'final'
                    time.sleep(3)
            elif self.ctrl_status == 'final':
                print("final")
                if self.body_auto: 
                    self.walk_switch()
                time.sleep(1)
                self.sendBodySector(LIFT2)
                time.sleep(9.5)
                self.ctrl_status = 'end'

        else: # 停止狀態
            self.bar.update(1)
            self.line.update(2)
            if self.body_auto:
                self.walk_switch()
            if not self.stop and self.crew:
                self.sendBodySector(14)
                time.sleep(10)
                self.sendBodySector(29)
                self.sendHeadMotor(2, HEAD_MOTOR_FINISH, 100)
                self.init_logic()
        self.get_logger().info(f'ctrl_status : {self.ctrl_status}')


def main(args=None):
    rclpy.init(args=args)
    node = WeightLift()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

    #motor = MotorMove()
    #target = TargetLocation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
