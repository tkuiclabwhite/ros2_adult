#!/usr/bin/env python3
#coding=utf-8
import sys
from strategy.API import API
from tku_msgs.msg import Dio
import rclpy
from rclpy.duration import Duration
from std_msgs.msg import String
import time
import math
import numpy as np
import cv2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


# 2025.4.17
#======================================================================================

CORRECT       = [-300, 0, 0]        # 原地踏步修正
LEFT_CORRECT  = [-350, -100, 3]        # 左旋修正
RIGHT_CORRECT = [-350, 0, -3]       # 右旋修正
#                 x , y , theta 

#====================================================================================

BASKET_SIZE_60_90 = [3070, 1250]      #sector 5301                   # 投籃時測量的籃框距離方法 #五分投籃時站姿高度看籃框size測距離
FIVEPOINT_HEAD_Y_DEGREE = [2010]      #投出去偏向左邊＝>頭 往左轉（大）-朝1960 ;  投出去偏向右邊＝>頭往右轉（小）-朝1940    #投籃前頭會固定一個角度，並扭腰

#=====================================================================================
CATCH_BALL_CORRECT = 1550        #1500

#CATCH_BALL_LINE  = [1680, 1, 1580]            # slow_degree, stop_degree, backward_degree
CATCH_BALL_LINE  = [2600, 2500, 2400]
TWO_POINT_LINE   = [1800, 1700, 1660]            # slow_degree, stop_degree, backward_degree
THREE_POINT_LINE = [86, 65, 62, 60]             # forward_slow_distance > forward_stop_distance > backward_stop_distance > backward_slow_distance
FIVE_POINT_LINE  = [160, 156, 150, 145]           # srward_slow_distance > forward_stop_distance > backward_stop_distance > backward_slow_distance

# 計算焦距判斷距離
BASTET_LENGTH =  20  #增加以下全域變數
FOCAL_LENGTH  = 352  # 333 
TEST_DISTANCE = 150

VALUEE = 55
#球測試
#VALUEE = 2  #框測試.  比賽時輸入的狀態決定投的策略  取代Diovalue
#VALUEE = 22 #2分球
#VALUEE = 33 #3分球
#VALUEE = 55 #5分球
#api.send = api.sendmessage()


class TargetLocation():
    def __init__(self,api_instance):
        self.initial()
        self.api = api_instance
        self.ball_x = 0
        self.ball_y = 0
        self.basket_x = 0
        self.basket_y = 0
        self.ball_size = 0
        self.basket_size = 0
        self.color_mask_subject_red = 0
        self.color_mask_subject_orange = 0
        self.ball_x_min = 0
        self.ball_y_min = 0
        self.ball_x_max = 0
        self.ball_y_max = 0
        self.basket_x_max = 0
        self.basket_x_min = 0
        self.basket_y_min = 0
        self.basket_y_max = 0
        self.basket_length = 0 #增加 

        # self.color_counts = [] 
        # self.get_object_cx = [[]] 
        # self.get_object_cy = [[]] 
        # self.object_sizes = [[]]
    
    def initial(self):
        self.ball_x = 0
        self.ball_y = 0
        self.basket_x = 0
        self.basket_y = 0
        self.ball_size = 0
        self.basket_size = 0
        self.color_mask_subject_red = 0
        self.color_mask_subject_orange = 0
        self.ball_x_min = 0
        self.ball_y_min = 0
        self.ball_x_max = 0
        self.ball_y_max = 0
        self.basket_x_min = 0
        self.basket_y_min = 0
        self.basket_x_max = 0
        self.basket_y_max = 0
        self.basket_length = 0 #增加
               
    def ball_parameter(self):   #利用色模建籃球
    
        self.color_mask_subject_orange = self.api.color_counts[0]

        self.ball_x = 0
        self.ball_y = 0
        self.ball_size = 0
        for j in range (self.color_mask_subject_orange):   #將所有看到的橘色物件編號
            cx = (self.api.object_x_max[0][j] + self.api.object_x_min[0][j]) // 2
            cy = (self.api.object_y_max[0][j] + self.api.object_y_min[0][j]) // 2
            if 310 > cx > 10 and 230 > cy   > 10 and self.api.object_sizes [0][j] > 200:
    
                if  self.api.object_sizes [0][j] > self.ball_size: #用大小過濾物件 #?????900待測試
                    self.ball_x =  cx
                    self.ball_y =  cy 
                    self.ball_size = self.api.object_sizes [0][j]
                    self.ball_x_min = self.api.object_x_min[0][j] 
                    self.ball_y_min = self.api.object_y_min[0][j] 
                    self.ball_x_max = self.api.object_x_max[0][j] 
                    self.ball_y_max = self.api.object_y_max[0][j]
    
    def basket_parameter(self): #利用色模建籃框
        
        self.color_mask_subject_red = self.api.color_counts[5]
        # print(f"[DEBUG] color_counts = {self.api.color_counts}")  # 確認各顏色數量
        # print(f"[DEBUG] 紅色物件數量 = {self.color_mask_subject_red}")
        # print(f"[DEBUG] object_sizes[5] = {self.api.object_sizes[5]}")  # 看實際大小

        self.basket_x = 0
        self.basket_y = 0
        self.basket_size = 0
        self.basket_length = 1 #增加
        # print(self.api.object_sizes [5])
        for j in range (self.color_mask_subject_red):     #將所有看到的紅色物件編號
            if self.api.object_sizes [5][j] > 200:
    
                if  40000 > self.api.object_sizes [5][j] > self.basket_size:  #用大小過濾物件(濾雜訊)
                    self.basket_x = (self.api.object_x_max[5][j] + self.api.object_x_min[5][j]) // 2
                    self.basket_y = (self.api.object_y_max[5][j] + self.api.object_y_min[5][j]) // 2
                    self.basket_size = self.api.object_sizes [5][j]
                    self.basket_x_min = self.api.object_x_min[5][j] 
                    self.basket_y_min = self.api.object_y_min[5][j] 
                    self.basket_x_max = self.api.object_x_max[5][j] 
                    self.basket_y_max = self.api.object_y_max[5][j]
                    self.basket_length = self.api.object_y_max[5][j] - self.api.object_y_min[5][j]  #增加
                    # print(f"[DEBUG] basket y_min={self.api.object_y_min[5][j]}, y_max={self.api.object_y_max[5][j]}, basket_length={self.basket_length}")
            
    
class MotorMove():
    
        def __init__(self,api_instance):
            self.api = api_instance
            self.initial()
            self.target = TargetLocation(self.api)           #腰當下的刻度
            self.head_horizon = 2048                #頭部水平刻度
            self.head_vertical = 2048               #頭部垂直刻度
            self.waist_position = 2048              #腰當下的刻度
            self.search_num = 0
            self.now_x = 0                          #現在要移動的x量
            self.now_y = 0                          #現在要移動的y量
            self.now_theta = 0                      #現在要旋轉的theta量
            self.throw_strength = 0                 #不知道
            self.distance_new = 0
            self.now_state = 0
            self.directly = False
            self.catch_correct = False
            self.reg = 2048
            self.desire_waist_degree = 2048
            self.size_correct = True
            self.corrected_size = 0
            self.turn_flag = True
            self.basket_distance_x = 0 #增加
            self.line_flag = 0


            #self.api.get_logger().info(f"[MotorMove init] head_horizon={self.head_horizon}")

    
        def initial(self):
            self.head_horizon = 2048                #頭部水平刻度
            self.head_vertical = 2048               #頭部垂直刻度
            self.waist_position = 2048              #腰當下的刻度
            self.search_num = 0
            self.now_x = 0                          #現在要移動的x量
            self.now_y = 0                          #現在要移動的y量
            self.now_theta = 0                      #現在要旋轉的theta量
            self.throw_strength = 0                 #不知道
            self.distance_new = 0
            self.directly = False
            self.catch_correct = False
            self.reg = 2048
            self.desire_waist_degree = 2048
            self.size_correct = True
            self.corrected_size = 0
            self.turn_flag = True
            self.basket_distance_x = 0 #增加
            self.line_flag = 0
    
    
        def draw(self):
            self.target.ball_parameter()
            self.target.basket_parameter()
            self.api.drawImageFunction(1, 1, 160, 160, 0, 240, 255, 255, 255) 
            self.api.drawImageFunction(2, 1, 0, 320, 120, 120, 255, 255, 255)
            self.api.drawImageFunction(3, 2, self.target.ball_x_min , self.target.ball_x_max , self.target.ball_y_max , self.target.ball_y_min, 255, 0, 255)
            self.api.drawImageFunction(4, 2, self.target.basket_x_min , self.target.basket_x_max , self.target.basket_y_max , self.target.basket_y_min, 255, 120, 255)
            # api.send.drawImageFunction(5, 1, api.send.yolo_XMin, api.send.yolo_XMax, api.send.yolo_YMin, api.send.yolo_YMax, 0, 255, 0)

    
        def move_head(self, ID, Position,head_max_x, head_max_y, Speed):  #把相對頭部變化變絕對(call 2048就變2048)
            self.api.sendHeadMotor(ID,Position,Speed)
            self.target.ball_parameter()
            self.target.basket_parameter()
            if ID == 1:
                self.head_horizon =  Position
                if abs(self.head_horizon - 2030) > head_max_x:
                    if (self.head_horizon - 2030 ) > 0:
                        self.head_horizon = 2030 + head_max_x
                    elif (self.head_horizon - 2030 ) < 0:
                        self.head_horizon = 2030 - head_max_x
    
            else :
                self.head_vertical = Position
                if abs(self.head_vertical - 2048) > head_max_y :
                    if (self.head_vertical - 2048 ) > 0 :
                        self.head_vertical = 2048 - head_max_y    
                    elif (self.head_vertical - 2048) < 0 :    
                        self.head_vertical = 2048 + head_max_y    
    
        def waist_rotate(self, waist_x, Speed):
            self.api.sendSingleMotor(15, waist_x-self.waist_position, Speed)#####
            self.waist_position =  waist_x 
    
        ####################################### view search #######################################
        def view_search(self, right_place, left_place, up_place, down_place, speed, delay):   
            self.api.get_logger().info(f'motor.reg =  {self.reg}')
            if self.reg > 0:
                turn_order = [3, 4, 1, 2]
                print("1")
            else:
                turn_order = [1, 4, 3, 2]
                print("2")
    
            if self.search_num >= len(turn_order):
                self.search_num = 0
    
            self.search_flag = turn_order[self.search_num]
    
            if self.search_flag == 1:
                if self.head_horizon >= left_place:
                    self.api.get_logger().debug(f'左尋')
                    self.api.get_logger().info(f'左尋')
                    self.move_head(1, self.head_horizon, 880, 880, speed)
                    self.head_horizon = self.head_horizon - speed
                    time.sleep(delay)
                    print("1-1")
                else:
                    self.search_num += 1
                    time.sleep(delay)
    
            elif self.search_flag == 4:
                if self.head_vertical <= up_place:
                    self.api.get_logger().debug(f'上尋')
                    self.api.get_logger().info(f'上尋')
                    self.move_head(2, self.head_vertical, 880, 880, speed)
                    self.head_vertical = self.head_vertical + speed
                    time.sleep(delay)
                    print("1-2")
                else:
                    self.search_num += 1  
                    time.sleep(delay*5)
                        
            elif  self.search_flag == 3:
                self.api.get_logger().debug(f'右尋')
                self.api.get_logger().info(f'右尋')
                if  self.head_horizon <= right_place:
                    self.move_head(1, self.head_horizon, 880, 880, speed)
                    self.head_horizon = self.head_horizon + speed
                    time.sleep(delay)
                    print("1-3") 
                else:
                    self.search_num += 1
                    time.sleep(delay*5)      
            
            elif self.search_flag ==  2:
                self.api.get_logger().debug(f'下尋')
                self.api.get_logger().info(f'下尋')
                if self.head_vertical >= down_place:
                    self.move_head(2, self.head_vertical, 880, 880, speed)      #頭向下的極限
                    self.head_vertical = self.head_vertical - speed
                    time.sleep(delay) 
                    print("1-4")  
                else:
                    self.search_num = 0
                    time.sleep(delay*5)
    
        ####################################### view search #######################################
          
        def trace_revise(self, x_target, y_target, speed):    #看誤差調整頭的角度(讓頭看向籃框或球)
            if x_target != 0 and y_target != 0:
                x_difference =  x_target - 160               #目標與中心x差距         
                y_difference =  y_target - 120               #目標與中心y差距
                x_degree = x_difference * (65/ 320)         #目標與中心x角度 65
                y_degree = y_difference * (38 / 240)         #目標與中心y角度
                self.move_head(1, self.head_horizon - round(x_degree * 4096 / 360 *0.15), 1200, 1200, speed)
                self.move_head(2, self.head_vertical + round(y_degree * 4096 / 360 *0.15), 1000, 1000, speed)
                time.sleep(0.1)
            else :
                self.api.get_logger().info(f'miss_target -> 需重新尋求')
                self.move_head(2, 2400, 880, 880, 50)
        def body_trace_rotate(self, degree): #步態旋轉到可拿球的角度
            x_body_rotate = self.head_horizon - 2030 #身體需要旋轉多少
            if x_body_rotate > degree:
                # self.MoveContinuous(LEFT_CORRECT[0], LEFT_CORRECT[1], LEFT_CORRECT[2], 100, 100, 8)
                # self.get_logger().info(f'右轉修正 = {x_body_rotate}')
                time.sleep(0.05)
            elif x_body_rotate < -degree :
                # self.MoveContinuous(RIGHT_CORRECT[0], RIGHT_CORRECT[1], RIGHT_CORRECT[2], 100, 100, 8)
                # self.get_logger().info(f'左轉修正 = {x_body_rotate}') # G863
                time.sleep(0.05)
    
        def ball_trace_straight(self, slow_degree, stop_degree, backward_degree):   #前進後退至可找拿球的距離
        ######################################## ball_trace_straight 副函式 ########################################
            if self.head_vertical > slow_degree:  #大前進
                self.MoveContinuous(1000+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 75, 75, 2) 
                self.api.get_logger().info(f'大前進, self.head_vertical= {self.head_vertical}')
    
            # elif stop_degree < self.head_vertical < slow_degree:  #進入減速範圍
            #     self.MoveContinuous(700+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 100, 100, 2)
            #     self.api.get_logger().info(f'進入減速範圍, self.head_vertical = {self.head_vertical}')
    
            # elif self.head_vertical < backward_degree: # 大後退
            #     self.MoveContinuous(-1000+CORRECT[0],0+CORRECT[1],0+CORRECT[2],100,100,2)
            #     self.api.get_logger().info(f'大後退, self.head_vertical = {self.head_vertical}')                
            
            elif stop_degree < self.head_vertical < slow_degree:  #進入減速範圍
                # self.MoveContinuous(800+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 100, 100, 2)
                self.api.get_logger().info(f'進入減速範圍, self.head_vertical = {self.head_vertical}')
    
            elif self.head_vertical < backward_degree: 
                # self.MoveContinuous(-650+CORRECT[0],0+CORRECT[1],0+CORRECT[2],50,50,2)
                self.api.get_logger().info(f'大後退, self.head_vertical = {self.head_vertical}')                
    
        def Owl_Rotate(self, turn_degree):
    
            self.api.get_logger().info(f"head_horizon = {self.head_horizon}")
            self.MoveW = self.head_horizon - turn_degree
            if abs(self.MoveW) > 2:
                pass
            else:
                self.MoveW = -2 if self.MoveW < 0 else 2 
                # elif self.MoveW < -3:
                #     self.MoveW = -3
            self.waist_rotate((self.waist_position + self.MoveW), 15)
            self.api.get_logger().info(f"貓頭鷹修腰 head_horizon = {self.head_horizon}")
            self.api.get_logger().info(f'self.MoveW = {self.MoveW}')
            time.sleep(0.1)
    
        def Null_WaistFix(self, turn_final): # 轉腰調整Basket.X與Baskethead_verticalBaseLine的誤差
     
            if (self.waist_position - 10) <= turn_final:
                self.desire_waist_degree -= 10
                self.waist_rotate(self.desire_waist_degree, 30)
                self.api.get_logger().info(f'waist_position = {self.waist_position}')
                time.sleep(0.15)
            else:
                self.api.get_logger().info(f'self.waist_position =  {self.waist_position}') 
                self.api.get_logger().error(f'fail')
    
        def degree_straight(self, slow_degree, stop_degree, backward_degree):   # 前進後退至可投球球的距離 2分
        ######################################## degree_straight 副函式 ######################################## 
            self.api.get_logger().debug(f'walk_to_basket')
            self.target.basket_parameter()
            self.api.get_logger().info(f'target.basket_x = {self.target.basket_x}, target.basket_y = {self.target.basket_y}, target.basket_size = {self.target.basket_size}')
            self.trace_revise(self.target.basket_x, self.target.basket_y, 35)
                
            if self.head_vertical > slow_degree:  #大前進
                self.MoveContinuous(1200+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 75, 75, 2) 
                self.api.get_logger().info(f'大前進, self.head_vertical =  {self.head_vertical}')
                time.sleep(0.05)
    
            elif stop_degree < self.head_vertical < slow_degree :  #進入減速範圍
                self.MoveContinuous(1000+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 50, 50, 2)
                self.api.get_logger().info(f'進入減速範圍, self.head_vertical =  {self.head_vertical}')
                time.sleep(0.05)
        
            elif self.head_vertical < backward_degree: 
                self.MoveContinuous(-500+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 80, 80, 2)
                self.api.get_logger().info(f'大後退, self.head_vertical =  {self.head_vertical}')               
                time.sleep(0.05)
    
        def distance_straight(self,forward_slow_distance, forward_stop_distance, backward_stop_distance, backward_slow_distance):   # 前進後退至可投球球的距離 3分 5分
        ########################################  distance_straight副函式 ########################################
            self.api.get_logger().debug(f'walk to line')
            self.target.basket_parameter()
            # self.get_logger().info(f'target.basket_yolo_x = {target.basket_yolo_x}, target.basket_yolo_y = {target.basket_yolo_y}, target.basket_size = {target.basket_size}')
            self.trace_revise(self.target.basket_x, self.target.basket_y,100)
            self.basket_distance()
    
            if self.basket_distance_x < backward_slow_distance:                         #大後退
                # self.MoveContinuous(-600+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 50, 50, 2)
                # self.get_logger().info(f'大後退, target.basket_size = {target.basket_size}')
                self.api.get_logger().info(f'離籃球框很近, 大後退, 籃球框距離 = {self.basket_distance_x}')
                
                time.sleep(0.05)
    
            elif backward_stop_distance > self.basket_distance_x > backward_slow_distance:  #進入後退減速範圍
                # self.MoveContinuous(-700+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 100, 100, 2)
                # self.get_logger().info(f'進入後退減速範圍, target.basket_size = {target.basket_size}')
                self.api.get_logger().info(f'接近籃球框, 進入後退減速範圍, 籃球框距離 = {self.basket_distance_x}')
                time.sleep(0.05)
    
            elif forward_slow_distance > self.basket_distance_x > forward_stop_distance:    #進入前進減速範圍
                # self.MoveContinuous(600+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 100, 100, 2)
                # self.get_logger().info(f'進入前進減速範圍, target.basket_size = {target.basket_size}')      
                self.api.get_logger().info(f'接近籃球框, 進入前進減速範圍, 籃球框距離 = {self.basket_distance_x}')          
                time.sleep(0.05)
    
            elif self.basket_distance_x > forward_slow_distance:                        #大前進
                # self.MoveContinuous(1000+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 100, 100, 2)    
                # self.get_logger().info(f'大前進, target.basket_size = {target.basket_size}')        
                self.api.get_logger().info(f'離籃球框很遠, 大前進, 籃球框距離 = {self.basket_distance_x}')          
                time.sleep(0.05)
        
        def MoveContinuous(self ,expect_x ,expect_y ,expect_theta ,add_x ,add_y ,add_theta):  #步態移動的馬達緩衝(調整距離:Now_X與Now_Y為現在要移動的x量與現在要移動的y量)
            if abs(self.now_x - expect_x) < add_x:
                self.now_x = expect_x
            else:
                if self.now_x < expect_x:
                    self.now_x += add_xball_parameter
                elif self.now_x > expect_x:
                    self.now_x -= add_x
                else:
                    pass
    
            if abs(self.now_y - expect_y) < add_y:
                self.now_y = expect_y
            else:
                if self.now_y < expect_y:
                    self.now_y += add_y
                elif self.now_y > expect_y:
                    self.now_y -= add_y
                else:
                    pass
    
            if abs(self.now_theta - expect_theta) < add_theta:
                self.now_theta = expect_theta
            else:
                if self.now_theta < expect_theta :
                    self.now_theta += add_theta
                elif self.now_theta > expect_theta :
                    self.now_theta -= add_theta
                else:
                    pass
    
            # self.get_logger().info(f'now_x = {self.now_x}, now_y = {self.now_y}, now_theta = {self.now_theta}') 
            self.api.sendContinuousValue(int(self.now_x), int(self.now_y), int(self.now_theta)) #####
    
    
        def bodyauto_close(self,next_state):    #步態移動的開關控制(原地踏步)
            if self.now_state == next_state :    
                pass
            elif self.now_state != next_state :
                self.api.sendbodyAuto(next_state)
                self.now_state = next_state
    
        def basket_distance(self):      #增加整段
            focal_x = TEST_DISTANCE * self.target.basket_length / BASTET_LENGTH
            self.api.get_logger().info(f'focal = {focal_x}')
            self.basket_distance_x = FOCAL_LENGTH * BASTET_LENGTH / self.target.basket_length
            #self.get_logger().info(f'ball_distance = {self.basket_distance_x}')
    
    
class Coordinate:
        def __init__(self, x, y):
            self.x, self.y = x, y
        def __add__(self, other):
            return Coordinate((self.x + other.x), (self.y + other.y))
        def __sub__(self, other):
            return Coordinate((self.x - other.x), (self.y - other.y))
        def __floordiv__(self, other):
            return Coordinate((self.x // other), (self.y // other))


class BasketBall(API):
    def __init__(self):
        super().__init__('bb')  
        self.head_y_down_adjust = False
        self.head_y_up_adjust = False
        self.ready_dunk = False
        self.ready_shoot = False
        self.aiming_finish = False
        self.step = 'begin'
        self.sw = 0
        self.target = TargetLocation(self) #新加上的
        self.motor = MotorMove(self)#新加上的
        self.create_timer(0.2, self.main)
        

    def initial(self):
        self.head_y_down_adjust = False
        self.head_y_up_adjust = False
        self.ready_dunk = False
        self.ready_shoot = False
        self.aiming_finish = False
        self.step = 'begin'
        self.sw = 0
        self.is_start = True

    def main(self):
        #self.get_logger().info("A: before ball_parameter")
        self.target.ball_parameter()
        #self.get_logger().info("B: after ball_parameter")
        #self.get_logger().info("C: before basket_parameter")
        self.target.basket_parameter()
        #self.get_logger().info("D: after basket_parameter")
        #self.get_logger().info(f"orange_count={self.target.color_mask_subject_orange}, ball_size={self.target.ball_size}")
   
  ##############???????
        self.is_start = True
        if  self.is_start: #api.send.Web
            self.get_logger().info(f'step = {self.step}')
            self.get_logger().info(f'ball_size = {self.target.ball_size}')
            self.motor.draw()
            
            if self.step == 'begin':
                self.begin()

            elif self.step == 'find_ball':
                self.find_ball()   

            elif self.step == 'start_gait':
                self.start_gait()

            elif self.step == 'walk_to_ball':
                self.walk_to_ball()

            elif self.step == 'waist_fix':
                self.waist_fix()

            elif self.step == 'catch_ball' :
                self.catch_ball() 
            
            elif self.step == 'find_basket':
                self.find_basket()

            elif self.step == 'stratagy_2':
                self.stratagy_2()

            elif self.step == 'stratagy_3':
                self.stratagy_3()

            elif self.step == 'stratagy_5':
                self.stratagy_5()   
        
        elif VALUEE == 1 :  # ball size   上上下下
            self.target.ball_parameter()
            if self.target.ball_size <= 350:   # 球在視野中太小
                self.get_logger().debug(f'球在視野中太小 -> 大範圍尋球')
                # motor.view_search_left(2428, 1668, 1800, 1200, 40, 0.05) 2551
                # self.motor.view_search(2700, 1678, 1660, 1200, 100, 0.05)
                self.motor.view_search(2700, 1678, 2600, 2300, 120, 0.05)
                self.target.ball_parameter() 

            elif self.target.ball_size > 350:   # 球在視野中夠大

                if abs(self.target.ball_x - 160) > 10  or abs(self.target.ball_y - 120) > 10:  # 讓球在畫面中心
                    self.get_logger().debug(f'球在視野中夠大 -> 鎖定球')
                    self.target.ball_parameter()
                    self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 65) 
                    time.sleep(0.05)
                else: 

                    self.motor.reg = 2030 - self.motor.head_horizon
                    self.motor.search_num = 0

            self.get_logger().info(f'Head_vertical = {self.motor.head_vertical}')
            self.get_logger().info(f'Ball_size = {self.target.ball_size}')
            time.sleep(0.05)
            self.step == 'test' 
           

        elif VALUEE == 2:       # basket size   上下上下
            self.motor.draw()
            self.motor.basket_distance() #增加

            # print("valuee=2")
            # time.sleep(2)
            # print(self.target.ball_x)
            self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 65)
            self.get_logger().info(f'Head_vertical = {self.motor.head_vertical}') #增加
            self.get_logger().info(f'籃球框距離 = {self.motor.basket_distance_x}')
            self.get_logger().info(f'籃球框大小 = {self.target.basket_size}')
            time.sleep(0.2)
            self.step == 'test' 
        

        elif self.step != 'begin' :
            self.sendHeadMotor(1, 2048, 30)
            self.sendHeadMotor(2, 2048, 30)
            self.target.initial()
            self.motor.initial()
            self.initial()
            time.sleep(0.05)
            self.motor.bodyauto_close(0)
            time.sleep(1)
            self.sendBodySector(29)
            time.sleep(0.05)
            self.step = 'begin'
            self.get_logger().debug(f'-------------------reset and stoping-------------------------')
            self.get_logger().info(f'主策略指撥關閉 -> 機器人回復初始狀態')


    def begin(self):
        ####################################### switch #######################################

        #self.api.sendSensorReset(1, 1, 1) 
        self.sendSensorReset(True)       
        if VALUEE == 22 and self.is_start:   # 開啟二分策略  下下下下
            self.sw = 2
            self.get_logger().info(f'SW = {self.sw}')

        elif VALUEE == 33: # 開啟三分策略  上下下下                                 
            self.sw = 3
            self.get_logger().info(f'SW = {self.sw}')

        elif VALUEE == 55: # 開啟五分策略  上上上下                                 
            self.sw = 5
            self.get_logger().info(f'SW = {self.sw}')

        else :
            self.sw = 3
            self.get_logger().info(f'SW = {self.sw}')

        ######################################## switch #######################################
        self.get_logger().debug(f'開始執行初始化')

        #self.api.sendBodySector(6)      ############步態調整############
        #time.sleep(0.05)
        #self.api.sendBodySector(8) 
        #time.sleep(0.05)
        # self.sendBodySector(1) 
        time.sleep(0.05)   


        self.step = 'find_ball'
        time.sleep(0.5)

        
    def find_ball(self):
        self.target.ball_parameter()

        if self.head_y_down_adjust:
            time.sleep(1)
            self.get_logger().debug(f'頭部抬起尋框')
            self.motor.move_head(2,1700,1250,880,50)                
            self.head_y_down_adjust = True
            time.sleep(1)
        
        else:
            if self.target.ball_size <= 350:   # 球在視野中太小
                self.get_logger().info(f'球在視野中太小 -> 大範圍尋球')
                # motor.view_search_left(2428, 1668, 1800, 1200, 40, 0.05)
                # self.motor.move_head(2, 1700, 880, 880, 50)   # 先抬頭到 1700
                time.sleep(0.5)
                self.motor.view_search(2700, 1678, 2600, 2300, 120, 0.05)
                self.target.ball_parameter() 

            elif self.target.ball_size > 350:   # 球在視野中夠大

                if abs(self.target.ball_x - 160) > 25  or abs(self.target.ball_y - 120) > 25:  # 讓球在畫面中心
                    self.get_logger().info(f'球在視野中夠大 -> 鎖定球')
                    self.target.ball_parameter()
                    self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 65) 
                    time.sleep(0.1)

                elif (CATCH_BALL_LINE[2] <= self.motor.head_vertical <= CATCH_BALL_LINE[1]) and (abs(self.motor.head_horizon-2030) <= 540):
                    self.get_logger().info(f'到達夾球範圍 STOP!!, self.head_vertical = {self.motor.head_vertical}')                
                    time.sleep(0.05)
                    self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 50) 
                    self.get_logger().debug(f'到達可move_head夾球位置')
                    self.get_logger().info(f'蹲下準備夾球')
                    time.sleep(1)
                    
                    self.motor.reg = 2030 - self.motor.head_horizon
                    self.motor.search_num = 0
                    self.motor.directly = True
                    self.step = 'waist_fix'

                else: 

                    self.motor.reg = 2030 - self.motor.head_horizon
                    self.motor.search_num = 0
                    self.step = 'start_gait'   


    def start_gait(self):
        self.target.ball_parameter()
        self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 30) 
        self.motor.bodyauto_close(0)
        time.sleep(0.05)

        if (self.motor.head_vertical <= CATCH_BALL_LINE[2]-50): # 球太近，先後退一段距離
            self.get_logger().info(f'球太大 -> 大倒退')
            self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 30) 
            # self.motor.MoveContinuous(-1200+CORRECT[0], 0+CORRECT[1], 0+CORRECT[2], 100, 100, 1) # 超大後退

        else:
            self.get_logger().debug(f'可進行微小修正')
            self.step = 'walk_to_ball'

    
    def walk_to_ball(self):
        self.target.ball_parameter()   
        self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 60)
        self.get_logger().info(f"head_vertical = {self.motor.head_vertical}")

        if (CATCH_BALL_LINE[2] <= self.motor.head_vertical <= CATCH_BALL_LINE[1]) and (abs(self.motor.head_horizon-2030) <= 110):  # 到達夾球位置
            self.motor.bodyauto_close(0) # 步態停止
            self.get_logger().info(f'到達夾球範圍 STOP!!, self.head_vertical = {self.motor.head_vertical}')                
            time.sleep(0.05)
            self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 40) 
            self.get_logger().debug(f'到達可夾球位置')
            self.get_logger().info(f'蹲下準備夾球')
            time.sleep(1)
            
            
            self.get_logger().info(f'頭往右轉')
            self.motor.move_head(1, 1820, 880, 880, 50)
            time.sleep(2) 
            self.step = 'waist_fix'

        else:
            self.target.ball_parameter()
            self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 35) 

            if abs(self.motor.head_horizon-2030) > 100:
                self.get_logger().debug(f'頭部馬達水平刻度偏差 -> 步態影響')
                self.get_logger().info(f'rotate調整')
                # self.motor.body_trace_rotate(30)

            else:
                self.get_logger().debug(f'頭部馬達垂直刻度與抓球角度差太多')
                self.get_logger().info(f'straight調整')
                # self.motor.ball_trace_straight(CATCH_BALL_LINE[0], CATCH_BALL_LINE[1], CATCH_BALL_LINE[2])        


    def waist_fix(self):
        self.target.ball_parameter()
        self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 60)
        if abs(self.target.ball_x - 160) > 1  or abs(self.target.ball_y - 120) > 1:  # 讓球在畫面中心
            self.get_logger().info(f'球在視野中夠大 -> 鎖定球')
            self.motor.trace_revise(self.target.ball_x, self.target.ball_y, 100) 
            self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
            #time.sleep(0.05)
            
        else:
            if (self.motor.head_horizon - 1940) > 5: 
                self.get_logger().info(f'球不在視野中間 -> 貓頭鷹修腰')
                # self.get_logger().info(f"motor.head_horizon = {motor.head_horizon}")
                self.motor.Owl_Rotate(1940)
            else :
                self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
                self.get_logger().info(f'球水平位置在中間')
                self.step = 'catch_ball'

    
    def catch_ball(self):
        self.get_logger().info(f"target.ball_size = {self.target.ball_size}")

        if self.target.ball_size < CATCH_BALL_CORRECT:
            self.get_logger().info(f'夾球修正')
            # self.sendBodySector(333)
            # time.sleep(1)
            # self.get_logger().info(f'正常夾球動作')
            # self.sendBodySector(687)
            self.sendBodySector(11) 
            time.sleep(5) 
            self.sendBodySector(12) 
            time.sleep(3)
            self.sendBodySector(14) 
            time.sleep(3)
            self.sendBodySector(15) 
            time.sleep(3)
            self.sendBodySector(18) 
            time.sleep(3)
            self.motor.catch_correct = True

        else:
            self.get_logger().info(f'正常夾球動作')
            self.sendBodySector(11) 
            time.sleep(5) 
            self.sendBodySector(12) 
            time.sleep(3)
            self.sendBodySector(14) 
            time.sleep(3)
            self.sendBodySector(15) 
            time.sleep(3)
            self.sendBodySector(18) 
            time.sleep(3)
            
            
        self.get_logger().info(f'腰部回正')
        self.motor.waist_rotate(2048,70)
        time.sleep(0.5) 

        if self.motor.catch_correct:
            self.get_logger().info(f'根據各自夾球動作回復站姿')
            # self.sendBodySector(444)
            # time.sleep(1)
            self.get_logger().info(f'回復站姿')
            self.sendBodySector(29) 
            time.sleep(0.5)
            

        else:
            self.get_logger().info(f'回復站姿')
            self.sendBodySector(29) 
            time.sleep(0.5)
           

        self.step = 'find_basket'  


    def find_basket(self):
        self.target.basket_parameter()
        if self.target.basket_size < 25:
            self.get_logger().debug(f'籃框在視野裡太小 -> 尋框')
            self.get_logger().info(f'basket_size =  {self.target.basket_size}')

            if not self.head_y_up_adjust:
                time.sleep(0.5)
                self.get_logger().debug(f'頭部抬起尋框')
                self.motor.move_head(2,1900,880,880,50)                
                self.head_y_up_adjust = True
                time.sleep(0.5)

            else:                                   
                self.get_logger().debug(f'開始尋框')
                self.get_logger().info(f'target.basket_x = {self.target.basket_x}, target.basket_y = {self.target.basket_y}, target.basket_size = {self.target.basket_size}')
                ####################################### view search #######################################
                self.motor.view_search(2700, 1678, 2600, 2300, 120, 0.04)
                

        else:                                
            self.get_logger().debug(f'籃框在視野裡夠大 -> 判斷策略所需前往的位置')

            if self.sw == 3:
                if abs(self.target.basket_x - 160) > 6  or abs(self.target.basket_y - 120) > 8:  #讓basket在畫面中心
                    self.get_logger().debug(f'匡在視野中夠大 -> 鎖定匡')
                    self.target.ball_parameter()
                    self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 40) 
                    time.sleep(0.05)
                else:
                    self.get_logger().info(f'3分球')
                    self.motor.bodyauto_close(0)
                    time.sleep(0.5)
                    self.step = 'stratagy_3'
                
            elif self.sw == 5:
                if abs(self.target.basket_x - 160) > 6  or abs(self.target.basket_y - 120) > 8:  #讓basket在畫面中心
                    self.get_logger().debug(f'匡在視野中夠大 -> 鎖定匡')
                    self.target.ball_parameter()
                    self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 40) 
                    time.sleep(0.05)
                else:
                    if self.motor.directly:
                        self.ready_shoot = True
                        self.step = 'stratagy_5'
                        self.target.basket_parameter()
                        time.sleep(2)
                        self.get_logger().info(f'五分球動作預備')
                        # self.sendBodySector(5301)
                        time.sleep(4)   
                        self.get_logger().debug(f'頭部調整') 
                        self.get_logger().debug(f'頭部水平旋轉調整')                                              
                        self.motor.move_head(1, FIVEPOINT_HEAD_Y_DEGREE[0], 880, 880, 50)
                        time.sleep(1)
                        self.get_logger().debug(f'頭部垂直旋轉調整')
                        self.motor.move_head(2, 2048, 880, 880, 50)

                    else:
                        self.get_logger().info(f'5分球')
                        self.motor.bodyauto_close(0)
                        time.sleep(0.5)
                        self.step = 'stratagy_5'
                
            elif self.sw == 2:
                if abs(self.target.basket_x - 160) > 12  or abs(self.target.basket_y - 120) > 16:  #讓basket在畫面中心
                    self.get_logger().debug(f'球在視野中夠大 -> 鎖定球')
                    self.target.ball_parameter()
                    self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 25) 
                    time.sleep(0.05)
                else:
                    self.get_logger().info(f'2分球')
                    self.motor.bodyauto_close(0)
                    time.sleep(1)
                    self.step = 'stratagy_2'


    ######################################## 二分球仿造catch_ball ######################################## 
    
    # def stratagy_2(self):
    #     self.target.basketauto_parameter()

    #     if not self.ready_dunk:  
    #         self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 50)
    #         self.get_logger().info(f'垂直刻度 = {self.motor.head_vertical}')
    #         self.get_logger().info(f'水平刻度 = {self.motor.head_horizon}')
    #         if ((TWO_POINT_LINE[2]) <= self.motor.head_vertical <= TWO_POINT_LINE[1]) and (abs(self.motor.head_horizon - 2048) <= 150): 
    #             self.get_logger().info(f'到達可投籃角度 STOP!!, self.head_vertical =  {self.motor.head_vertical}')
    #             self.ready_dunk = True
    #             self.get_logger().info(f'到達可投籃大小 STOP!!, target.basket_size = {self.target.basket_size} ,corrected_size = {self.motor.corrected_size}')
    #             self.motor.bodyauto_close(0)
    #             time.sleep(1)
    #             self.target.basket_parameter()
    #             time.sleep(1)
    #             self.get_logger().info(f'伸手準備投籃')
    #             self.sendBodySector(887)
    #             time.sleep(1.5)
    #             self.get_logger().info(f'頭往右轉')
    #             self.motor.move_head(1, 1820, 880, 1100, 50)
    #             time.sleep(1)
    #         else:

    #             if abs(self.motor.head_horizon - 2048) > 200:
    #                 self.get_logger().info(f'頭部馬達水平刻度偏差 -> 步態影響')
    #                 self.get_logger().info(f'rotate調整')
    #                 time.sleep(0.05)
    #                 self.motor.body_trace_rotate(40)

    #             else :
    #                 self.get_logger().info(f'頭部馬達垂直刻度與抓球角度差太多')
    #                 self.get_logger().info(f'straight調整')
    #                 time.sleep(0.05)
    #                 self.motor.degree_straight(TWO_POINT_LINE[0], TWO_POINT_LINE[1], TWO_POINT_LINE[2]) 

    #     else: 
    #         if self.target.basket_x != 0 :
    #             if abs(self.target.basket_x- 160) > 1  or abs(self.target.basket_y - 120) > 2:  #讓匡在畫面中心
    #                 self.get_logger().info(f'匡在視野中夠大 -> 鎖定匡')
    #                 self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 55) 
    #                 self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
    #                 time.sleep(0.05)
    #             else:
    #                 if abs(self.motor.head_horizon-1935) > 85: 
    #                     self.get_logger().info(f'匡不在視野中間 -> 貓頭鷹修腰')
    #                     # self.get_logger().info(f"motor.head_horizon = {motor.head_horizon}")
    #                     self.motor.Owl_Rotate(1935)    #2030
    
    #                 # if abs(self.target.basket_x-160) > 3:
    #                 #     self.target.basket_parameter()
    #                 #     self.get_logger().debug(f'腰部修正')
    #                 #     self.motor.WaistFix(self.target.basket_x, 160)
    #                 #     self.get_logger().info(f'abs(target.basket_x - 160) = {abs(self.target.basket_x - 160)}')

    #                 else:
    #                     time.sleep(1)
    #                     self.get_logger().info(f'執行2分球投籃')
    #                     self.sendBodySector(987)
    #                     self.step = "finish"
    #         else:
    #             self.get_logger().info(f'框不在視野中 -> 往左邊轉腰')
    #             self.motor.Null_WaistFix(1990)     #2348


    # ######################################## 三分球用size判斷 ########################################

    # def stratagy_3(self):
    #     self.target.basket_parameter()
    #     if not self.ready_shoot:  
    #         self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 65)
    #         self.get_logger().info(f'籃球框距離 = {self.motor.basket_distance_x}')
    #         self.get_logger().info(f'水平刻度 = {self.motor.head_horizon}')
    #         self.get_logger().info(f'target.basket_x = {self.target.basket_x}')

    #         if (THREE_POINT_LINE[1] >=  self.motor.basket_distance_x >= THREE_POINT_LINE[2]) and (abs(self.motor.head_horizon - 2048) <= 100) and not self.motor.turn_flag: 
    #             self.motor.line_flag += 1
    #             self.get_logger().info(f'line_flag = {self.motor.line_flag}')
                
    #             time.sleep(0.25)
                
    #             if (self.motor.line_flag >= 5):
    #                 self.ready_shoot = True
    #                 self.get_logger().info(f'到達可投籃大小 STOP!!, 籃球框距離 = {self.motor.basket_distance_x}')
    #                 self.motor.bodyauto_close(0)
    #                 self.target.basket_parameter()
    #                 time.sleep(1)
    #                 self.get_logger().info(f'3分球預備動作')
    #                 self.sendBodySector(887)
    #                 time.sleep(2)
    #                 self.get_logger().debug(f'頭部水平旋轉調整')
    #                 self.motor.move_head(1,1840, 880, 880, 30)
    #                 time.sleep(1)
    #         else:
    #             if abs(self.motor.head_horizon - 2048) > 100:
    #                 self.get_logger().info(f'頭部馬達水平刻度偏差 > 步態影響的')
    #                 self.get_logger().info(f'rotate調整')
    #                 time.sleep(0.05)
    #                 self.motor.body_trace_rotate(30)
    #                 self.motor.turn_flag = True

    #             else :
    #                 self.get_logger().info(f'頭部馬達垂直刻度與抓球角度差太多')
    #                 self.get_logger().info(f'straight調整')
    #                 time.sleep(0.05)
    #                 self.motor.distance_straight(THREE_POINT_LINE[0], THREE_POINT_LINE[1], THREE_POINT_LINE[2], THREE_POINT_LINE[3])
    #                 self.motor.turn_flag = False
            
    #     else:
    #         if self.target.basket_x != 0 :
    #             self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
    #             if abs(self.target.basket_x- 160) > 7 or abs(self.target.basket_y - 120) > 7:  #讓匡在畫面中心
    #                 self.get_logger().info(f'匡在視野中夠大 -> 鎖定匡')
    #                 self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 65) 
    #                 # self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
    #                 time.sleep(0.1)
    #             else: 
    #                 if abs(self.motor.head_horizon-1985) > 45: #10
    #                     self.get_logger().info(f'匡不在視野中間->貓頭鷹修腰')
    #                     # self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
    #                     self.motor.Owl_Rotate(1985)
                        
    #                 else:
    #                     time.sleep(0.5)
    #                     self.get_logger().info(f'手臂旋轉調整')
    #                     self.sendBodySector(5) 
    #                     time.sleep(0.5) 
    #                     self.get_logger().info(f'開爪')
    #                     self.sendBodySector(886)
    #                     self.get_logger().info(f'投籃')
    #                     self.sendBodySector(988) 
    #                     self.step = "finish"    
    #         # else:
    #         #     self.get_logger().info(f'框不在視野中 -> 往左邊轉腰')

    ######################################## 五分球用size判斷 ########################################

    def stratagy_5(self):
            self.target.basket_parameter()
            self.motor.basket_distance()
            if not self.ready_shoot:
                self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 65)
                self.get_logger().info(f'籃球框距離 = {self.motor.basket_distance_x}')
                self.get_logger().info(f'水平刻度 = {self.motor.head_horizon}')
                if (FIVE_POINT_LINE[1] >= self.motor.basket_distance_x >= FIVE_POINT_LINE[2]) and (abs(self.motor.head_horizon - 2030) <= 200) and not self.motor.turn_flag:
    
                    self.motor.line_flag += 1
                    self.get_logger().info(f'line_flag = {self.motor.line_flag}')
                    self.get_logger().info(f'籃球框距離 = {self.motor.basket_distance_x}')
                    time.sleep(0.1)
                    
                    if (self.motor.line_flag >= 5):
                        self.ready_shoot = True
                        self.get_logger().info(f'到達可投籃大小 STOP!!, target.basket_distance = {self.motor.basket_distance_x}')
                        self.motor.bodyauto_close(0)
                        self.target.basket_parameter()
                        time.sleep(2)
                        self.get_logger().info(f'5分球預備動作')
                        # self.sendBodySector(50)
                        # time.sleep(2)  
                        # self.sendBodySector(51)
                        # time.sleep(4)  
                        self.get_logger().info(f'頭部調整') 
                        self.get_logger().info(f'頭部水平旋轉調整')                                            
                        self.motor.move_head(1, FIVEPOINT_HEAD_Y_DEGREE[0], 880, 880, 50)
                        time.sleep(1)
                        self.get_logger().debug(f'頭部垂直旋轉調整')
                        self.motor.move_head(2, 2048, 880, 880, 50)
                        time.sleep(1)
    
                else:
    
                    if abs(self.motor.head_horizon-2030) < 80:
                        self.get_logger().info(f'頭部馬達水平刻度偏差 -> 步態影響的')
                        self.get_logger().info(f'rotate調整')
                        time.sleep(0.05)
                        # self.motor.body_trace_rotate(30)
                        self.motor.turn_flag = True
    
                    else:
                        self.get_logger().info(f'頭部馬達垂直刻度與抓球角度差太多')
                        self.get_logger().info(f'straight調整')
                        time.sleep(0.05)
                        # self.motor.distance_straight(FIVE_POINT_LINE[0], FIVE_POINT_LINE[1], FIVE_POINT_LINE[2], FIVE_POINT_LINE[3])
                        self.motor.turn_flag = False
            else:
    
                if self.target.basket_x != 0 :
                    if abs(self.target.basket_x- 160) > 1  or abs(self.target.basket_y - 120) > 1:  #讓匡在畫面中心
                        self.get_logger().info(f'匡在視野中夠大 -> 鎖定匡')
                        self.motor.trace_revise(self.target.basket_x, self.target.basket_y, 45) 
                        self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
                        time.sleep(0.05)
                    else:
                        if abs(self.motor.head_horizon-2010) > 4: 
                            self.get_logger().info(f'匡不在視野中間->貓頭鷹修腰')
                            self.get_logger().info(f"motor.head_horizon = {self.motor.head_horizon}")
                            self.motor.Owl_Rotate(2010)
    
                        else:
                            time.sleep(0.5)
                            # self.get_logger().info(f'手臂旋轉調整')
                            # self.api.sendBodySector(5) 
                            # time.sleep(0.05) 
                            self.get_logger().info(f'開爪')
                            # self.sendBodySector(23)
                            # time.sleep(3)
                            self.get_logger().info(f'投籃')
                            # self.sendBodySector(24)
                            # self.api.sendBodySector(5503)
                            time.sleep(2)
                            self.get_logger().info(f'motor.throw_strength  = {self.motor.throw_strength}')
                            self.step = "finish"
                    
                else:
                    self.get_logger().debug(f'框不在視野中 -> 五分球不會發生拉')

    



def main(args=None):
    rclpy.init(args=args)
    node = BasketBall()

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


   ###orange_count=0, ball_size=0