#!/usr/bin/env python3
#coding=utf-8
import sys
from strategy.API import API
from tku_msgs.msg import Dio
import rclpy
from rclpy.duration import Duration
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import math
import numpy as np
import threading
import cv2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from strategy.sr.calculate_edge import deep_calculate

#--校正量--#
#原地量校正
FORWARD_CORRECTION         = -1700
#平移校正
TRANSLATION_CORRECTION     = 0
#旋轉校正
THETA_CORRECTION           = 0

#基礎變化量(前進&平移)
BASE_CHANGE                = 1000         

#上下板後路徑規劃
ROUTE_PLAN_FLAG            = True

'''
平移最大量2000
平移旋轉最大量-700,1500,8
自轉90度-300,-1000,8,5-6秒
180度 10秒
'''

#[FORWARD,TRANSLATION,THETA,TIME......] 0,0,0,0
ROUTE_PLAN_LAYER_ONE       = [0,0,0,0]
ROUTE_PLAN_LAYER_TWO       = [0,0,0,0]
ROUTE_PLAN_LAYER_TREE      = [0,0,0,0]
ROUTE_PLAN_LAYER_FORE      = [0,0,0,0]
ROUTE_PLAN_LAYER_FIVE      = [0,0,0,0]
ROUTE_PLAN_LAYER_SIX       = [0,0,0,0]
ROUTE_PLAN_LAYER_SEVEN     = [0,0,0,0]
ROUTE_PLAN = [
                ROUTE_PLAN_LAYER_ONE,
                ROUTE_PLAN_LAYER_TWO,
                ROUTE_PLAN_LAYER_TREE,
                ROUTE_PLAN_LAYER_FORE,
                ROUTE_PLAN_LAYER_FIVE,
                ROUTE_PLAN_LAYER_SIX,
                ROUTE_PLAN_LAYER_SEVEN
             ]

BOARD_LAYER_CONFIG = {
    "UP" :{
        "LCUP": 40000,               #上板前進量
        "STAND_CORRECT": True,      #上板看板子站姿
        "STAND_CORRECT_SECTOR": 1, #上板看板子站姿sector        
        "WALK_PARAM": dict(com_y_swing=5, width_size=0, period_t=360, t_dsp=0,
                            clearance=14, board_high=5, stand_height=50,
                            com_height=40), #上板步態參數
        "NORMAL_OFFSET" : [1,3,1],#[中心值,邊緣值,曲率] [2,2,1](2,2,2,2,2,2) / [1,4,1](4,2,1,1,2,4) / [1,4,0.5](4,2,1,1,2,4)
        "U_OFFSET"      : [6,4,2] #[中心值,邊緣值,曲率] [6,4,2](2,2,3,3,2,2) /
    },

    1: {
        "LCUP": None, #上板前進量 (數值 or None)
        "WALK_PARAM_YN": False, #是否使用自訂步態參數
        "WALK_PARAM": dict(com_y_swing=5, width_size=0, period_t=360, t_dsp=0,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40),#上板步態參數
        "LC_CORRECT": True,      #上板站姿微調
        "LC_CORRECT_SECTOR": 2, #上板站姿微調sector
        "LC_U": False, #上板U形板
    },

    2: {
        "LCUP": None, #上板前進量 (數值 or None)
        "WALK_PARAM_YN": False, #是否使用自訂步態參數
        "WALK_PARAM": dict(com_y_swing=5, width_size=0, period_t=360, t_dsp=0,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40),#上板步態參數
        "LC_CORRECT": True,      #上板站姿微調
        "LC_CORRECT_SECTOR": 2, #上板站姿微調sector
        "LC_U": False, #上板U形板
    },

    3: {
        "LCUP": None, #上板前進量 (數值 or None)
        "WALK_PARAM_YN": False, #是否使用自訂步態參數
        "WALK_PARAM": dict(com_y_swing=5, width_size=0, period_t=360, t_dsp=0,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40),#上板步態參數
        "LC_CORRECT": True,      #上板站姿微調
        "LC_CORRECT_SECTOR": 2, #上板站姿微調sector
        "LC_U": False, #上板U形板
    },

    "DOWN" :{
        "LCDOWN": 35000,               #下板前進量
        "STAND_CORRECT": True,      #下板看板子站姿
        "STAND_CORRECT_SECTOR": 1, #下板看板子站姿sector
        "WALK_PARAM": dict(com_y_swing=-1, width_size=0, period_t=360, t_dsp=0,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40), #下板步態參數
        "NORMAL_OFFSET" : [1,3,1], #[中心值,邊緣值,曲率] [2,2,1](2,2,2,2,2,2) / [1,4,1](4,2,1,1,2,4)
        "U_OFFSET"      : [2,6,2]  #[中心值,邊緣值,曲率] [2,6,2](3,2,1,1,2,3)
    },

    4: {
        "LCDOWN": None, #下板前進量 (數值 or None)
        "WALK_PARAM_YN": False, #是否使用自訂步態參數
        "WALK_PARAM": dict(com_y_swing=-6, width_size=0, period_t=360, t_dsp=0.4,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40), #下板步態參數
        "LC_CORRECT": False,      #下板站姿微調
        "LC_CORRECT_SECTOR": 210, #下板站姿微調sector
        "LC_U": False, #下板U形板
    },

    5: {
        "LCDOWN": None, #下板前進量 (數值 or None)
        "WALK_PARAM_YN": False, #是否使用自訂步態參數
        "WALK_PARAM": dict(com_y_swing=-6, width_size=0, period_t=360, t_dsp=0.4,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40), #下板步態參數
        "LC_CORRECT": False,      #下板站姿微調
        "LC_CORRECT_SECTOR": 210, #下板站姿微調sector
        "LC_U": False, #下板U形板
    },

    6: {
        "LCDOWN": None, #下板前進量 (數值 or None)
        "WALK_PARAM_YN": False, #是否使用自訂步態參數
        "WALK_PARAM": dict(com_y_swing=-6, width_size=0, period_t=360, t_dsp=0.4,
                            clearance=8, board_high=3, stand_height=50,
                            com_height=40), #下板步態參數
        "LC_CORRECT": False,      #下板站姿微調
        "LC_CORRECT_SECTOR": 210, #下板站姿微調sector
        "LC_U": False, #下板U形板
    },
}


DRAW_FUNCTION_FLAG         = True                 #影像繪圖開關
START_LAYER                = 1
BOARD_COLOR                = ["Green"  ,           #板子顏色(根據比賽現場調整)
                              "Blue"   ,           #Blue Red Yellow Green
                              "Red", 
                              "Yellow" , 
                              "Red"    , 
                              "Blue"   , 
                              "Green"]              
#----------#                       右腳           左腳
#                              左 ,  中,  右|  左,  中,   右
FOOT                       = [120, 137, 155, 180, 195, 210]
HEAD_HORIZONTAL            = 2048                  #頭水平
HEAD_VERTICAL              = 2700                  #頭垂直 #down 2750

##判斷值
FOOTBOARD_LINE             = 225                   #基準線
UP_WARNING_DISTANCE        = 6                     #上板危險距離
DOWN_WARNING_DISTANCE      = 0                     #下板危險距離
GO_UP_DISTANCE             = 20                    #上板距離
GO_DOWN_DISTANCE           = 3                     #下板距離

FORWORD_CHANGE_LINE = {"MIN_NORMAL": 50, "NORMAL_BIG": 100, "BIG_SUPER": 150} #前進判斷線 {小 ~ 一般, 一般 ~ 大, 大 ~ 超大}

BACK                = {"MIN": -4000, "NORMAL": -6000} #後退{小後退,後退}

FORWARD             = {"MIN": 2000,  "NORMAL": 3000, "BIG": 4000  , "SUPER": 5000}    #前進{小前進,前進,大前進,超大前進}

TRANSLATION         = {"MIN": 3000,  "NORMAL": 5000, "BIG": 6000}      #平移{小平移,平移,大平移}

THETA               = {"MIN": 5,    "NORMAL": 8,    "BIG": 10}              #旋轉{小旋轉,旋轉,大旋轉}

SLOPE               = {"MIN": 2,    "NORMAL": 4,    "BIG": 6}  #斜{小斜,斜,大斜}

#左基礎參數
LEFT_THETA                 = 1
#右基礎參數
RIGHT_THETA                = -1
#前進基礎參數
FORWARD_PARAM              = 1
#後退基礎參數
BACK_PARAM                 = -1

class LiftandCarry(API):
#LC主策略
    def __init__(self,edge):
        super().__init__('lift_and_carry_node')
        self.edge = edge
        self.init()
        self.action_status = "初始化中..."

        # 啟動顯示畫面的 Thread
        self.printer_thread = StatusPrinterThread(self)
        self.printer_thread.start()

        self.timer = self.create_timer(0.05, self.main)

    def main(self):
        self.sendHeadMotor(1,self.head_Horizontal,100)#水平
        if self.layer <4:
            self.sendHeadMotor(2,self.head_Vertical,100)#垂直
        else:
            self.sendHeadMotor(2,self.head_Vertical,100)#垂直
            # self.sendHeadMotor(2,self.head_Vertical+20,100)#垂直

        if DRAW_FUNCTION_FLAG:
            self.draw_function()

        if self.is_start == False:
        #關閉策略,初始化設定
            if not self.walkinggait_stop:
                self.sendHeadMotor(1,self.head_Horizontal,100)  #水平
                self.sendHeadMotor(2,self.head_Vertical,100)    #垂直
                self.sendLCWalkParameter(                
                com_y_swing  = float(0),   #起步步態補償
                width_size   = float(0),  #雙腳距離
                period_t     = int(420),  #步態頻率
                t_dsp        = float(0),  #雙支撐時間
                lift_height  = float(5),
                stand_height = float(50), #機器人初始站姿高度
                com_height   = float(40),  #質心高度
                )    
                time.sleep(4)
                self.sendbodyAuto(0)
                time.sleep(1.5)
                self.sendBodySector(29)             #基礎站姿磁區                 
            self.init()
            self.sendSensorReset(True)

        elif self.is_start == True:
        #開啟LC策略
            if self.layer < 7:
                if self.walkinggait_stop and self.first_in:
                    self.sendBodySector(29)             #基礎站姿磁區
                    self.action_status ="站立姿勢"
                    time.sleep(1)
                    if self.board_cfg["stand_correct_enabled"]:
                        self.sendBodySector(self.board_cfg["stand_correct_sector"])
                        self.action_status = "上板看板子站姿調整" if self.board_cfg["group"] == "UP" else "下板看板子站姿調整"
                        time.sleep(1)

                    self.sendBodyAutoCmd(self.forward,0,0,0)
                    # self.sendbodyAuto(1)
                    # self.sendContinuousValue(self.forward,0,0)

                    self.walkinggait_stop = False
                    self.first_in         = False
                    self.route_plan(self.layer)
                elif self.walkinggait_stop and not self.first_in:
                    if self.layer > 3:
                        self.find_board()
                        if (self.distance[0] == 0 and self.distance[1] == 0 and self.distance[2] == 0 and self.distance[3] == 0) or \
                           (self.distance[2] == 0 and self.distance[3] == 0 and self.distance[4] == 0 and self.distance[5] == 0) or \
                           (self.distance[0] == 0 and self.distance[1] == 0 and max(self.distance) < 2) or \
                           (self.distance[4] == 0 and self.distance[5] == 0 and max(self.distance) < 2):
                            self.action_status = "！！！！！！！！！！直接下板！！！！！！！！！！"
                            self.walkinggait(motion = 'continue_to_lc')
                
                    self.sendbodyAuto(0)
                    self.walkinggait(motion = 'walking')
                    self.walkinggait_stop = False
                    self.route_plan(self.layer)
                elif not self.walkinggait_stop:
                    self.find_board()
                    self.walkinggait(motion=self.edge_judge())
                    
    def init(self):
        #狀態
        self.state                 = '停止'
        self.angle                 = '直走'
        self.search                = 'right'
        #步態啟動旗標
        self.walkinggait_stop      = True
        self.first_in              = True  
        #層數       
        self.layer                 = START_LAYER
        self.board_cfg             = self.get_board_config(self.layer)
        #設定頭部馬達
        self.head_Horizontal       = HEAD_HORIZONTAL
        self.head_Vertical         = HEAD_VERTICAL
        #距離矩陣                     [左左,左中,左右 ,右左,右中,右右 ]
        self.distance              = [9999,9999,9999,9999,9999,9999]
        self.next_distance         = [9999,9999,9999,9999,9999,9999]
        #步態參數
        self.forward               = FORWARD["NORMAL"] + FORWARD_CORRECTION
        self.translation           = 0              + TRANSLATION_CORRECTION
        self.theta                 = 0              + THETA_CORRECTION
        self.now_forward           = 0 
        self.now_translation       = 0
        self.now_theta             = 0  
        #建立板子資訊
        self.next_board            = ObjectInfo(BOARD_COLOR[self.layer+1],'Board',self) #設定下一個尋找的板子
        self.now_board             = ObjectInfo(BOARD_COLOR[self.layer], 'Board', self)   #設定當前尋找的板子
        self.last_board            = None                                          #設定前一階板子
        self.edge.color                 = ObjectInfo.color_dict[BOARD_COLOR[self.layer]]
        self.edge.layer = self.layer
        self.v_label_matrix_flatten = 0
        self.current_func = "初始化 (init)"
        self.func_detail = ""

    def find_board(self):
    #獲取板子資訊、距離資訊
        if self.layer < 6:
            self.next_board.update()
        self.now_board.update()
        if self.last_board is not None:
            self.last_board.update()
        #腳與邊緣點距離
        self.distance         = [9999,9999,9999,9999,9999,9999]
        self.next_distance    = [9999,9999,9999,9999,9999,9999]
        #邊緣點
        now_edge_point        = [9999,9999,9999,9999,9999,9999]
        next_edge_point       = [9999,9999,9999,9999,9999,9999]
        #-------距離判斷-------#
        for i in range(6):
            self.distance[i],now_edge_point[i] = self.return_real_board(outset=FOOTBOARD_LINE,x=FOOT[i],board=self.now_board.color_parameter)
        #-----------------#
        if self.layer != 6 or self.layer != 3:
        #除了上最頂層和下最底層以外,偵測上下板空間
            for i in range(6):
                if now_edge_point[i]>240:
                    continue
                else:
                    self.next_distance[i] ,next_edge_point[i]= self.return_real_board(outset=now_edge_point[i],x=FOOT[i],board=self.next_board.color_parameter)
    
    def walkinggait(self,motion):
    #步態函數,用於切換countiue 或 LC 步態
        if motion == 'ready_to_lc' or motion == 'continue_to_lc':
            self.action_status ="對正板子"
            time.sleep(0.25)
            if motion == 'ready_to_lc':
                self.sendbodyAuto(0)
                time.sleep(3)                           #穩定停止後的搖晃
            self.sendSensorReset(True)              #IMU reset 避免機器人步態修正錯誤
            self.sendBodySector(29)                  #這是基本站姿的磁區
            self.action_status ="站立姿勢"
            time.sleep(1)
            
            self.sendLCWalkParameter(
                com_y_swing  = float(self.board_cfg["walk_param"]["com_y_swing"]),
                width_size   = float(self.board_cfg["walk_param"]["width_size"]),
                period_t     = int(self.board_cfg["walk_param"]["period_t"]),
                t_dsp        = float(self.board_cfg["walk_param"]["t_dsp"]),
                clearance    = float(self.board_cfg["walk_param"]["clearance"]),
                board_high   = float(self.board_cfg["walk_param"]["board_high"]),
                stand_height = float(self.board_cfg["walk_param"]["stand_height"]),
                com_height   = float(self.board_cfg["walk_param"]["com_height"]),
            )
            time.sleep(3)
            self.action_status = "準備上板" if self.board_cfg["group"] == "UP" else "準備下板"

            if self.board_cfg["lc_correct_enabled"]:
                self.sendBodySector(self.board_cfg["lc_correct_sector"])
                self.action_status = "上板前姿勢" if self.board_cfg["group"] == "UP" else "下板前姿勢"
                time.sleep(1.5)
            self.now_forward,self.forward = self.board_cfg["forward_value"], self.board_cfg["forward_value"]
            self.now_translation,self.translation = 0,0
            self.now_theta,self.theta = 0,0
            self.sendBodyAutoCmd(x=self.board_cfg["forward_value"], walking_mode=(1 if self.board_cfg["group"]=="UP" else 2))
            time.sleep(3)
            self.sendLCWalkParameter(                
                com_y_swing  = float(0),   #起步步態補償
                width_size   = float(0),  #雙腳距離
                period_t     = int(420),  #步態頻率
                t_dsp        = float(0),  #雙支撐時間
                lift_height  = float(5),
                stand_height = float(50), #機器人初始站姿高度
                com_height   = float(40),  #質心高度
            )           
            time.sleep(3) 
            self.sendBodySector(29)                  #這是基本站姿的磁區
            self.action_status ="站立姿勢"
            time.sleep(1.5)

            if self.board_cfg["stand_correct_enabled"]:
                self.sendBodySector(self.board_cfg["stand_correct_sector"])
                self.action_status = "上板看板子站姿調整" if self.board_cfg["group"] == "UP" else "下板看板子站姿調整"
                time.sleep(1)

            self.forward,self.now_forward = FORWARD_CORRECTION,FORWARD_CORRECTION
            self.translation,self.now_translation = TRANSLATION_CORRECTION,TRANSLATION_CORRECTION
            self.theta,self.now_theta = THETA_CORRECTION,THETA_CORRECTION
            time.sleep(1)
            
            #-初始化-#
            self.forward        = 0
            self.translation    = 0
            self.theta          = 0
            self.layer += 1                          #層數加一
            self.board_cfg = self.get_board_config(self.layer) 
            self.walkinggait_stop   = True
            if self.layer < 7:
                self.edge.color = ObjectInfo.color_dict[BOARD_COLOR[self.layer]]
                self.edge.layer = self.layer
                self.now_board  = ObjectInfo(BOARD_COLOR[self.layer],'Board',self)   #設定當前尋找的板子
                self.last_board = None 
                if self.layer != 4:
                    if self.layer != 6:
                        self.next_board = ObjectInfo(BOARD_COLOR[self.layer+1],'Board',self) #設定下一個尋找的板子
                    self.last_board = ObjectInfo(BOARD_COLOR[self.layer-2],'Board',self) #設定前一個板子
                else:
                    self.next_board = ObjectInfo(BOARD_COLOR[self.layer+1],'Board',self) #設定下一個尋找的板子
            #-------#
        else:
            #前進變化量
            if self.now_forward > self.forward:
                self.now_forward -= BASE_CHANGE
            elif self.now_forward < self.forward:
                self.now_forward += BASE_CHANGE
            else:
                self.now_forward = self.forward
            #平移變化量
            if self.now_translation > self.translation:
                self.now_translation -= BASE_CHANGE
            elif self.now_translation < self.translation:
                self.now_translation += BASE_CHANGE
            else:
                self.now_translation = self.translation
            #旋轉變化量
            if self.now_theta > self.theta:
                self.now_theta -= 1
            elif self.now_theta < self.theta:
                self.now_theta += 1
            else:
                self.now_theta = self.theta
            
            if self.now_translation >1000 and self.now_forward >2000:
                self.now_forward = 2000
            #速度調整
            self.sendContinuousValue(self.now_forward,self.now_translation,self.now_theta)
            self.sendbodyAuto(1)

    def edge_judge(self):
    #邊緣判斷,回傳機器人走路速度與走路模式
        cfg    = self.board_cfg
        offset = cfg["active_offset"]
        base   = cfg["trigger_base"]

        if (self.distance[0] < base+offset[0] and self.distance[1] < base+offset[1] and
            self.distance[2] < base+offset[2] and self.distance[3] < base+offset[3] and
            self.distance[4] < base+offset[4] and self.distance[5] < base+offset[5]):

            if cfg["group"] == "UP":
                self.state = f"上板U{self.layer}" if cfg["u_shape"] else "上板"
            else:
                self.state = f"下板U{self.layer}" if cfg["u_shape"] else "下板"

            return 'ready_to_lc'
        else:
            if (self.layer < 4) and (min(self.distance) <= UP_WARNING_DISTANCE):
                if max(self.distance[0:3])>30:
                    self.forward = BACK["MIN"] + FORWARD_CORRECTION
                    self.translation = RIGHT_THETA * TRANSLATION["NORMAL"] + TRANSLATION_CORRECTION
                    if abs(self.distance[0]-self.distance[2]) < 5:
                        self.theta = 0
                    else:
                        self.theta = RIGHT_THETA*THETA["NORMAL"] + THETA_CORRECTION
                    self.state = "!!!右平移!!!"
                elif max(self.distance[3:6])>30:
                    self.forward = BACK["MIN"] + FORWARD_CORRECTION
                    self.translation = LEFT_THETA * TRANSLATION["NORMAL"] + TRANSLATION_CORRECTION
                    if abs(self.distance[3]-self.distance[5]) < 5:
                        self.theta = 0
                    else:
                        self.theta = LEFT_THETA*THETA["NORMAL"] + THETA_CORRECTION
                    self.state = "!!!左平移!!!"
                else:
                    self.forward = BACK["MIN"] + FORWARD_CORRECTION
                    self.theta_change()
                    self.state = "!!!小心踩板,後退(上板)!!!"
            elif self.layer >= 4 and (min(self.distance) <= DOWN_WARNING_DISTANCE):
                if self.distance[0] < GO_DOWN_DISTANCE and min(self.distance[3:6]) > GO_DOWN_DISTANCE:
                    self.forward = BACK["MIN"] + FORWARD_CORRECTION
                    self.translation = RIGHT_THETA * TRANSLATION["MIN"] + TRANSLATION_CORRECTION
                    self.theta = THETA["MIN"]*LEFT_THETA
                    self.state = "!!!右平移,左旋!!!"
                elif self.distance[5] < GO_DOWN_DISTANCE and min(self.distance[0:3]) > GO_DOWN_DISTANCE:
                    self.forward = BACK["MIN"] + FORWARD_CORRECTION
                    self.translation = LEFT_THETA * TRANSLATION["MIN"] + TRANSLATION_CORRECTION
                    self.theta = THETA["MIN"]*RIGHT_THETA
                    self.state = "!!!左平移,右旋!!!"
                else:
                    self.forward = BACK["MIN"] + FORWARD_CORRECTION
                    self.theta_change()
                    self.state = "!!!小心踩板,後退(下板)!!!"
            else:
                min_dist = min(self.distance)
                if min_dist < FORWORD_CHANGE_LINE["MIN_NORMAL"]:
                    self.forward = FORWARD["MIN"] + FORWARD_CORRECTION
                    self.theta_change()
                    self.state = '小前進'
                elif min_dist < FORWORD_CHANGE_LINE["NORMAL_BIG"]:
                    self.forward = FORWARD["NORMAL"] + FORWARD_CORRECTION
                    self.theta_change()
                    self.state = '前進'
                elif min_dist < FORWORD_CHANGE_LINE["BIG_SUPER"]:
                    self.forward = FORWARD["BIG"] + FORWARD_CORRECTION
                    self.theta_change()
                    self.state = '大前進'
                else:
                    self.theta = THETA_CORRECTION
                    if self.layer == 1:
                        self.forward = FORWARD["SUPER"] + FORWARD_CORRECTION
                        self.state = '超大前進'
                    else:
                        self.forward = FORWARD["BIG"] + FORWARD_CORRECTION
                        self.state = '大前進'
                self.translation = TRANSLATION_CORRECTION           #距離板太遠不須平移
            return 'walking'

    def theta_change(self):
        # 旋轉修正

        slope = self.edge.slope

        # 決定旋轉方向
        if slope > 0:
            decide_theta = LEFT_THETA
            self.angle = '左旋'
        elif slope < 0:
            decide_theta = RIGHT_THETA
            self.angle = '右旋'
        else:
            self.angle = '直走'


        # 依據斜率大小決定旋轉角度與位移
        if abs(slope) > SLOPE["BIG"]:  # 斜率過大，旋轉修正最大
            self.theta = THETA["BIG"] * decide_theta + THETA_CORRECTION
            self.translation = TRANSLATION["NORMAL"] * decide_theta * -1
        elif abs(slope) > SLOPE["NORMAL"]:  # 斜率較大，適中修正
            self.theta = THETA["NORMAL"] * decide_theta + THETA_CORRECTION
            self.translation = TRANSLATION["MIN"] * decide_theta * -1
        elif abs(slope) > SLOPE["MIN"]:  # 斜率較小，微調修正
            self.theta = THETA["MIN"] * decide_theta + THETA_CORRECTION
            self.translation = 0 + THETA_CORRECTION
        else:  # 斜率接近0，不旋轉
            self.translation = 0 + TRANSLATION_CORRECTION
            self.theta = 0 + THETA_CORRECTION
            self.angle = '直走'

        # 特殊情況處理：當 `layer == 4`，且斜率過大時，保持直行
        if slope > 10 and self.layer == 4:
            self.theta = 0 + THETA_CORRECTION

    def return_real_board(self,x,board,outset):
        
    #檢查回傳的物件是否為板子,確認連續10個點為同一色模
        for y in range(outset,10,-1):
            real_distance_flag = (self.edge.new_label_matrix_flatten[320*y+x] == board)
            self.v_label_matrix_flatten=self.edge.new_label_matrix_flatten[320*y+x]
            if real_distance_flag:
                for i in range(1,11):
                    real_distance_flag = (real_distance_flag and self.edge.new_label_matrix_flatten[320*(y-i)+x] == board)
                    
                    if not real_distance_flag:
                        break
            if  real_distance_flag:
                break 
        return (outset - y,y)if real_distance_flag else (9999,9999)

    def route_plan(self, now_layer):
        if ROUTE_PLAN_FLAG:
            self.current_func = "路徑規劃 (route_plan)"
            # 計算總步驟數
            total_steps = len(ROUTE_PLAN[now_layer-1]) // 4
            
            for t in range(total_steps):                
                start = self.get_clock().now().nanoseconds / 1e9
                target_time = ROUTE_PLAN[now_layer-1][3+4*t]
                end = start
                
                time.sleep(1) # 啟動步態後穩定時間

                # 設定目標值
                self.forward     = ROUTE_PLAN[now_layer-1][0+4*t] + FORWARD_CORRECTION
                self.translation = ROUTE_PLAN[now_layer-1][1+4*t] + TRANSLATION_CORRECTION
                self.theta       = ROUTE_PLAN[now_layer-1][2+4*t] + THETA_CORRECTION
                
                # 同步 Now 值，讓儀表板顯示當前發送的速度
                self.now_forward, self.now_translation, self.now_theta = self.forward, self.translation, self.theta

                while (end - start) < target_time:
                    end = self.get_clock().now().nanoseconds / 1e9
                    elapsed = end - start
                    
                    # 更新 func_detail，顯示已過時間
                    self.func_detail = f"層:{now_layer} 步:{t+1}/{total_steps} | 進度:{elapsed:.1f}s / {target_time}s"
                    
                    self.sendContinuousValue(self.forward, self.translation, self.theta)
                    # 這裡可以加極短的 sleep 避免過度佔用 CPU，但維持頻率
                    time.sleep(0.01)
                    
            self.current_func = "主迴圈 (main)"
            self.func_detail = ""
    
    def get_board_config(self, layer):
        group = "UP" if layer < 4 else "DOWN"
        base_cfg = BOARD_LAYER_CONFIG[group]
        override = BOARD_LAYER_CONFIG.get(layer, {})
        forward_key = "LCUP" if group == "UP" else "LCDOWN"

        cfg = {}
        cfg["group"]                 = group
        cfg["forward_value"]         = override.get(forward_key) or base_cfg[forward_key]
        cfg["walk_param"]            = override["WALK_PARAM"] if override.get("WALK_PARAM_YN") else base_cfg["WALK_PARAM"]
        cfg["stand_correct_enabled"] = base_cfg["STAND_CORRECT"]
        cfg["stand_correct_sector"]  = base_cfg["STAND_CORRECT_SECTOR"]
        cfg["lc_correct_enabled"]    = override.get("LC_CORRECT", False)
        cfg["lc_correct_sector"]     = override.get("LC_CORRECT_SECTOR", base_cfg["STAND_CORRECT_SECTOR"])
        cfg["u_shape"]               = override.get("LC_U", False)

        normal_params = override.get("NORMAL_OFFSET", base_cfg["NORMAL_OFFSET"])
        u_params      = override.get("U_OFFSET", base_cfg["U_OFFSET"])
        normal_offset = self.make_offset_profile(*normal_params)
        u_offset      = self.make_offset_profile(*u_params)

        cfg["active_offset"] = u_offset if cfg["u_shape"] else normal_offset
        cfg["trigger_base"]  = GO_UP_DISTANCE if group == "UP" else GO_DOWN_DISTANCE

        return cfg

    def make_offset_profile(self,center_value, edge_value, power=2.0):
        offsets = []
        for i in range(6):
            d = min(abs(i - 2), abs(i - 3)) / 2 
            value = edge_value + (center_value - edge_value) * ((1 - d) ** power)
            offsets.append(round(value))
        return offsets
    
    def draw_function(self):
        #腳的距離判斷線
        self.drawImageFunction(1,1,0,320,FOOTBOARD_LINE,FOOTBOARD_LINE,0,128,255,1)
        self.drawImageFunction(2,1,FOOT[0],FOOT[0],0,240,255,128,128,1)
        self.drawImageFunction(3,1,FOOT[1],FOOT[1],0,240,255,128,128,1)
        self.drawImageFunction(4,1,FOOT[2],FOOT[2],0,240,255,128,128,1)
        self.drawImageFunction(5,1,FOOT[3],FOOT[3],0,240,255,128,128,1)
        self.drawImageFunction(6,1,FOOT[4],FOOT[4],0,240,255,128,128,1)
        self.drawImageFunction(7,1,FOOT[5],FOOT[5],0,240,255,128,128,1)
        #邊緣(目前實際偵測到的距離)
        self.drawImageFunction(8,2,FOOT[0]-5,FOOT[0]+5,FOOTBOARD_LINE-self.distance[0]-5,FOOTBOARD_LINE-self.distance[0]+5,255,0,128,1)
        self.drawImageFunction(9,2,FOOT[1]-5,FOOT[1]+5,FOOTBOARD_LINE-self.distance[1]-5,FOOTBOARD_LINE-self.distance[1]+5,255,0,128,1)
        self.drawImageFunction(10,2,FOOT[2]-5,FOOT[2]+5,FOOTBOARD_LINE-self.distance[2]-5,FOOTBOARD_LINE-self.distance[2]+5,255,0,128,1)
        self.drawImageFunction(11,2,FOOT[3]-5,FOOT[3]+5,FOOTBOARD_LINE-self.distance[3]-5,FOOTBOARD_LINE-self.distance[3]+5,255,0,128,1)
        self.drawImageFunction(12,2,FOOT[4]-5,FOOT[4]+5,FOOTBOARD_LINE-self.distance[4]-5,FOOTBOARD_LINE-self.distance[4]+5,255,0,128,1)
        self.drawImageFunction(13,2,FOOT[5]-5,FOOT[5]+5,FOOTBOARD_LINE-self.distance[5]-5,FOOTBOARD_LINE-self.distance[5]+5,255,0,128,1)
        #第二板邊緣點(維持原樣)
        self.drawImageFunction(14,2,FOOT[0]-5,FOOT[0]+5,FOOTBOARD_LINE-self.distance[0]-self.next_distance[0]-5,FOOTBOARD_LINE-self.distance[0]-self.next_distance[0]+5,0,90,128,1)
        self.drawImageFunction(15,2,FOOT[1]-5,FOOT[1]+5,FOOTBOARD_LINE-self.distance[1]-self.next_distance[1]-5,FOOTBOARD_LINE-self.distance[1]-self.next_distance[1]+5,0,90,128,1)
        self.drawImageFunction(16,2,FOOT[2]-5,FOOT[2]+5,FOOTBOARD_LINE-self.distance[2]-self.next_distance[2]-5,FOOTBOARD_LINE-self.distance[2]-self.next_distance[2]+5,0,90,128,1)
        self.drawImageFunction(17,2,FOOT[3]-5,FOOT[3]+5,FOOTBOARD_LINE-self.distance[3]-self.next_distance[3]-5,FOOTBOARD_LINE-self.distance[3]-self.next_distance[3]+5,0,90,128,1)
        self.drawImageFunction(18,2,FOOT[4]-5,FOOT[4]+5,FOOTBOARD_LINE-self.distance[4]-self.next_distance[4]-5,FOOTBOARD_LINE-self.distance[4]-self.next_distance[4]+5,0,90,128,1)
        self.drawImageFunction(19,2,FOOT[5]-5,FOOT[5]+5,FOOTBOARD_LINE-self.distance[5]-self.next_distance[5]-5,FOOTBOARD_LINE-self.distance[5]-self.next_distance[5]+5,0,90,128,1)

        
        target = [self.board_cfg["trigger_base"] + o for o in self.board_cfg["active_offset"]]
        for i in range(6):
            y = int(round(FOOTBOARD_LINE - target[i]))
            self.drawImageFunction(30+i, 2, FOOT[i]-4, FOOT[i]+4, y-2, y+2, 0, 255, 255, 1)

class Coordinate:
#儲存座標
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ObjectInfo:
#物件的影件資訊
    color_dict = {  'Orange':  0,
                    'Yellow':  1,
                    'Blue'  :  2,
                    'Green' :  3,
                    'Black' :  4,
                    'Red'   :  5,
                    'White' :  6 }
    parameter  = {  'Orange':  2**0,
                    'Yellow':  2**1,
                    'Blue'  :  2**2,
                    'Green' :  2**3,
                    'Black' :  2**4,
                    'Red'   :  2**5,
                    'White' :  2**6 }

    def __init__(self, color, object_type,api_node):
        self.api = api_node # 把 API 存起來
        self.color            = self.color_dict[color]
        self.color_parameter  = self.parameter[color]
        self.edge_max         = Coordinate(0, 0)
        self.edge_min         = Coordinate(0, 0)
        self.center           = Coordinate(0, 0)
        self.get_target       = False
        self.target_size      = 0

        update_strategy = { 'Board': self.get_object,
                            'Ladder': self.get_object,
                            'Ball' : self.get_ball_object}
        self.find_object = update_strategy[object_type]

    def get_object(self):
        # 2. 把 self. 改成 self.api.
        if len(self.api.object_sizes[self.color]) == 0:
            return None # 直接回傳找不到，不要往下算 max()
        max_object_size = max(self.api.object_sizes[self.color])
        max_object_idx = self.api.object_sizes[self.color].index(max_object_size)
        return max_object_idx if max_object_size > 500 else None
    def get_ball_object(self):
        object_idx = None
        # 2. 把 self. 改成 self.api.
        for i in range(self.api.color_counts[self.color]):
            length_width_diff = abs(abs(self.api.object_x_max[self.color][i] - self.api.object_x_min[self.color][i]) - abs(self.api.object_y_max[self.color][i] - self.api.object_y_min[self.color][i]))
            if 100 < self.api.object_sizes[self.color][i] < 2500 and length_width_diff < 8:
                object_idx = i
        return object_idx

    def update(self):
        object_idx = self.find_object()

        if object_idx is not None:
            self.get_target  = True
            self.edge_max.x  = self.api.object_x_max[self.color][object_idx]
            self.edge_min.x  = self.api.object_x_min[self.color][object_idx]
            self.edge_max.y  = self.api.object_y_max[self.color][object_idx]
            self.edge_min.y  = self.api.object_y_min[self.color][object_idx]
            self.center.x    = (self.api.object_x_max[self.color][object_idx] + self.api.object_x_min[self.color][object_idx]) // 2
            self.center.y    = (self.api.object_y_max[self.color][object_idx] + self.api.object_y_min[self.color][object_idx]) // 2
            self.target_size = self.api.object_sizes[self.color][object_idx]

        else:
            self.get_target = False
            self.edge_max.x  = 0
            self.edge_min.x  = 0
            self.edge_max.y  = 0
            self.edge_min.y  = 0
            self.center.x    = 0
            self.center.y    = 0
            self.target_size = 0


import threading

class StatusPrinterThread(threading.Thread):
    def __init__(self, lc_node):
        super().__init__()
        self.lc = lc_node  # 傳入 LiftandCarry 的實體
        self.daemon = True # 設定為 Daemon Thread，這樣主程式結束時它會自動關閉
        self.running = True

    def run(self):
        # 設定更新頻率，例如 0.1 秒更新一次畫面
        while self.running and rclpy.ok():
            self.val_print()
            time.sleep(0.1)

    def val_print(self):
        # 確保初始化完成後才印出，避免一開始讀不到屬性報錯
        if getattr(self.lc, 'distance', None) is None:
            return

        try:
            # 取得當前板子顏色名稱 (避免層數大於 6 爆掉)
            board_color_str = BOARD_COLOR[self.lc.layer] if self.lc.layer < len(BOARD_COLOR) else "完成"
            board_cfg = getattr(self.lc, 'board_cfg', None)
            if board_cfg is not None:
                target_distance = [round(board_cfg["trigger_base"] + o, 1) for o in board_cfg["active_offset"]]
            else:
                target_distance = "尚未初始化"
            # 清除終端機畫面並回到最左上角
            sys.stdout.write("\033[H\033[J")
            
            # 格式化輸出字串
            sys.stdout.write(f"\
#==============系統狀態==============#\n\
is_start         : {self.lc.is_start}\n\
layer            : {self.lc.layer} ({board_color_str})\n\
state            : {self.lc.state}\n\
angle_state      : {self.lc.angle}\n\
#===============速度狀態===============#\n\
Now  (x, y, th)  : {self.lc.now_forward}, {self.lc.now_translation}, {self.lc.now_theta}\n\
Goal (x, y, th)  : {self.lc.forward}, {self.lc.translation}, {self.lc.theta}\n\
#===============感測與影像=============#\n\
slope            : {self.lc.edge.slope:.2f} \n\
imu_yaw          : {self.lc.imu_rpy[2]:.2f}\n\
board_size       : {self.lc.now_board.target_size if self.lc.now_board else 0}\n\
distance         : {self.lc.distance}\n\
next_distance    : {self.lc.next_distance}\n\
target_distance  : {target_distance}\n\
#===============當前動作===============#\n\
action_status    : {getattr(self.lc, 'action_status', 'None')}\n\
current_function : {getattr(self.lc, 'current_func', 'None')}\n\
function_detail  : {getattr(self.lc, 'func_detail', 'None')}\n\
#====================================#\n\
label_matrix_flatten:{self.lc.v_label_matrix_flatten}\n\
")
            sys.stdout.flush()
        except Exception as e:
            # 避免輸出時發生不可預期的錯誤導致 Thread 死掉
            pass



def main(args=None):
    rclpy.init(args=args)
    # global edge
    edge = deep_calculate(5, 1)
    edge.slope = 0
    # lc = LiftandCarry()    

    lc = LiftandCarry(edge) # 把 edge 傳進去
    edge.api = lc

    executor = MultiThreadedExecutor()
    executor.add_node(lc)
    executor.add_node(edge)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        lc.destroy_node()
        edge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# ░░░░░░░░░▄░░░░░░░░░░░░░░▄░░░░░░░  ⠂⠂⠂⠂⠂⠂⠂⠂▀████▀▄▄⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂▄█ 
# ░░░░░░░░▌▒█░░░░░░░░░░░▄▀▒▌░░░░░░  ⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂█▀░░░░▀▀▄▄▄▄▄⠂⠂⠂⠂▄▄▀▀█⠂⠂
# ░░░░░░░░▌▒▒█░░░░░░░░▄▀▒▒▒▐░░░░░░   ⠂⠂⠂▄⠂⠂⠂⠂⠂⠂⠂█░░░░░░░░░░░▀▀▀▀▄░░▄▀ ⠂⠂
# ░░░░░░░▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐░░░░░░  ⠂▄▀░▀▄⠂⠂⠂⠂⠂⠂▀▄░░░░░░░░░░░░░░▀▄▀⠂⠂⠂⠂⠂
# ░░░░░▄▄▀▒░▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐░░░░░░   ▄▀░░░░█⠂⠂⠂⠂⠂⠂█▀░░░▄█▀▄░░░░░░▄█⠂⠂⠂⠂⠂
# ░░░▄▀▒▒▒░░░▒▒▒░░░▒▒▒▀██▀▒▌░░░░░░   ▀▄░░░░░▀▄⠂⠂⠂█░░░░░▀██▀░░░░░██▄█ ⠂⠂⠂⠂
# ░░▐▒▒▒▄▄▒▒▒▒░░░▒▒▒▒▒▒▒▀▄▒▒▌░░░░░  ⠂⠂▀▄░░░░▄▀⠂█░░░▄██▄░░░▄░░▄░░▀▀░█ ⠂⠂⠂⠂
# ░░▌░░▌█▀▒▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐░░░░░  ⠂⠂⠂█░░▄▀⠂⠂█░░░░▀██▀░░░░▀▀░▀▀░░▄▀⠂⠂⠂⠂
# ░▐░░░▒▒▒▒▒▒▒▒▌██▀▒▒░░░▒▒▒▀▄▌░░░░  ⠂⠂█░░░█⠂⠂█░░░░░░▄▄░░░░░░░░░░░▄▀⠂⠂⠂⠂⠂
# ░▌░▒▄██▄▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▌░░░░  ⠂█░░░█⠂⠂█▄▄░░░░░░░▀▀▄░░░░░░▄░█ ⠂⠂⠂⠂⠂
# ▀▒▀▐▄█▄█▌▄░▀▒▒░░░░░░░░░░▒▒▒▐░░░░  ⠂⠂▀▄░▄█▄█▀██▄░░▄▄░░░▄▀░░▄▀▀░░░█ ⠂⠂⠂⠂⠂
# ▐▒▒▐▀▐▀▒░▄▄▒▄▒▒▒▒▒▒░▒░▒░▒▒▒▒▌░░░  ⠂⠂⠂⠂▀███░░░░░░░░░▀▀▀░░░░▀▄░░░▄▀⠂⠂⠂⠂⠂
# ▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒▒▒░▒░▒░▒▒▐░░░░  ⠂⠂⠂⠂⠂⠂▀▀█░░░░░░░░░▄░░░░░░▄▀█▀ ⠂⠂⠂⠂⠂
# ░▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒░▒░▒░▒░▒▒▒▌░░░░  ⠂⠂⠂⠂⠂⠂⠂⠂▀█░░░░░▄▄▄▀░░▄▄▀▀░▄▀ ⠂⠂⠂⠂⠂
# ░▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▄▒▒▐░░░░░  ⠂⠂⠂⠂⠂⠂⠂⠂⠂⠂▀▀▄▄▄▄▀⠂▀▀▀⠂▀▀▄▄▄▀⠂⠂⠂⠂⠂
# ░░▀▄▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▄▒▒▒▒▌░░░░░      ░ ▄ ▌ ▌▀ ⠂

####################################################################################################################################################################                                                                                                                                                                                                      
#                                                                                                             .::-========--:.                                                         
#                                                                                                     :-=+++==---:::::---=++=-.                                                     
#                                                                                                 .-++=-:...              ..:=++:                                                   
#                                                                                                 .=++-:.  ....................  .:=+-                                                 
#                                                                                             -++=:. .............................:++:                                               
#                                                                                         :=+=:. ..................................:*+                                              
#                                                                                         :++-. .......................................-*:                                            
#                                                                                     -*+:........................................... :*-                                           
#                                                                                     -*+:................................................#-                                          
#                                                                                 :*+. ..................................................#-                                         
#                                                                                 :**:.................................................... :#:                                        
#                                                                             .+*-....................................................... =#.                                       
#                                                                             =#=...........................................................*+                                       
#                                                                             :*+..............................................................#-                                      
#                                                                         =*-.............................................................. =#                                      
#                                                                         :*+..................................................................#=                                     
#                                                                         =*-.................................................................. -#.                                    
#                                                                     .*+......................................................................#=                                    
#                                                                     -#=...................................................................... =#.                                   
#                                                                     +*: .......................................................................:%-                                   
#                                                                 .*+.......................................................................... *+                                   
#                                                                 :#=........................................................................... =#.                                  
#                                                                 -#- ............................................................................:#-                                  
#                                                             =#: ..............................................................................#=                                  
#                                                             =#: .............................................................................. +*                                  
#                                                             +#:................................................................................ -#.                                 
#                                                             +*...................................................................................:%:                                 
#                                                         +#.....................................................................................#-                                 
#                                                         +#......................................................................................*=                                 
#                                                         =#...................................................................................... ++                                 
#                                                         -#:...................................................................................... +*                                 
#                                                     :%- ...................................................................................... =*                                 
#                                                     .#- ....................................................................................... =#.                                
#                                                     *+ ........................................................................................ =#.                                
#                                                     =#.......................................................................................... =#.                                
#                                                     :%: .....................................       ............................................. =#.                                
#                                                     #%................................... ..-==+++==-............................................ =*                                 
#                                                 =%#:.............................. ..-+**++======+**-......................................... =*                                 
#                                                 .%=*= ........................... .-+*+=------------=#+........................................ +*                                 
#                                                 +#:+#......................... .-+*+=-----------------*+ ...................................... +*                                 
#                                                 .%=:-%- ..................... .=**=---------------------*+ ......................................*+                                 
#                                                 +#:-:+#.................... .=*+-------------------------#- .....................................*=                                 
#                                                 .%=:---#+ ................ .=*+---------------------------=#. ....................................#-                                 
#                                                 =%:-----#+................-*+-----------------------------:+#: .......-=+++++++=: ................#-                                 
#                                                 *+:------+*-...........:=*+---------------------------------=*+-::-==++=-------+*=...............:#:                                 
#                                             :%-:--------+++=----=++++=-------------------------------------=++++=-------------*+ ............ -#.                                 
#                                             -#-------------=++++==-------------------------------------------------------------*+ ........... =#.                                 
#                     .:-=======-:.             +*:-------------------------------+-------------------------------------------------*+ .......... +*                                  
#                 :=+=-::......:=++=:          *+:------------------------------*=--------------------------------------------------*+.......... *+                                  
#                 -+=:.             .:=++:      .#=:----------------------------:+%----------------------------------------------------**..........#=                                  
#             :*=.                    :=*=.   :%-:-----------------------------%+:---------------------------------------------------:+*: .... .#@-                                  
#             +*:                        .-+=. -%------------------------------+@----------------==-------------------------------------=*=:..:-**#.                                  
#         .*=.                            -+++#:----------------------------:##:---------------*----------------------------------------=++++=-+#                                   
#         -*-            ..................  -%*:-----------------------------%+:--------------+#----------------------------------------------:**                                   
#         -#:          ...................... .#*:-----------------------------@+:--------------**------------------------------*=--------------:*+                                   
#         -*:        ................. ....... .%*:-----------------------------+----------------*+-----------------------------=#----------------#-                                   
#     :#:        ................ .   ..... .%*:-----------------------------------------------------------------------------*+:-------------:-%:                                   
#     .#-       ................          .. .%*:----------------------------------------------------------------------------=#---------------:=#.                                   
#     *= .     ........  ........             ##:----------==================================--------------------------------+=---------------:**                                    
#     =*...    ...................             +#----=====+++++*****####################******++++======----------------------------------------#=                                    
#     .#: .     ...........                     :%=-=++++**#######***********************#@++*****######**+++=====-------------------------------%:                                    
#     ++ .     .........                         *#-+++****+++++++++++++++++++++++++++++++%=:::::----=%#*#######**++====-----------------------:+#.                                    
# .#: .    .....                              .#*=+++++++++++++++++++++++++++++++++++++%=:-------:+%++++++++**######*++===------------------:*+                                     
# =*.     .                                    .+#*++++++++++++++++++++++++++++++++++++%+:--------#%++++++++++++++**#####*++====-------------#-                                     
# *= .   ..                                      .-=****+++++++++++++++++++++++++++++++%+::::::::-%*++++++++++++++++++++*####*++====--------=#.                                     
# .#: .  .                                            .:-=*****++=++++++++++++++++++++++%+.....::.=@+++++++++++++++++++++++++**##**++===----:**                                      
# :#. .                                                    ..:-=#%*****+++++++++++++++++%=        +%+++++++++++++++++++++++++++++*##*++++=--+#.                                      
# =*  ..                                                        .*+.::-=++********++++++%-        ##+++++++++++++++++++++++++++++++++++++++**.                                       
# =*                                              .              .*=        ..:-==++***#@:       .%#+++++++++++++++++++++++++++++++****++=-.                                         
# =*                          .-----:.        .-+++++=-.          .#-                  -%.       -%*********************+**+++====--:..                                              
# =*                       .=+++===+++=-.  .:+*=------=*+.         :#:                 -#        =*         ..........                    ....:::..                                  
# =*                      =*=-::::::::-++++++-::::::::::=*=.        +#.                ++        +=                               .:-========--===++++=:                             
# -#.                   .*+::::--::---:::--:::-::-::----:-=+=:.    -+**                #-        *-                          .:-+++=--:..           ..-=*=.                          
# :#.                  .*=:----:-::-:---::::-::-::--:-::-:::=+++==++:-#-              .#:       .#:                       :=++=-:.                      .-*=                         
# .#:                  ++:-::--::::---:--:----::::-:--:--:--:::----:::+#              -#        :#.                   .-++=-.                             .**                        
# *=                 .#-:::-::-:----:::--:--:-=:-----:--------::::-:::#=             *+        -*.                .=++-:                                  .**                       
# =*                 +*:-------:--:--::-::--:-#=:--::-:-::---:-::-:-:.=*            .%:        -*              :=++-.       ....                           .#=                      
# :#.               :%-:--::--::::::::-:::-:-:=%=:-::---::::--::----::-#:           -#.        -*           .=*+-.   .   ....                       .       =#                      
#     #=               *+:---:-:---::::-:::-----::+*:-::::::::::::::::::::*+           *+         =*         :**-.  ....... ...         ......           ..... :#.                     
#     =*              =#::-:::-::=-:-:--:---::::::::::-----------------:::+#          .#.         -*       -*#=  .......  ...             .....           .    .#:                     
#     .#:            :#-:-:::::::+*::-::--:::::----=======+++++++++++==--:=#.         +*          -*     =*++:   ........       ....           .                *-                     
#     =+           .#=:-::-::--::#+::::::----===+++**############****++=--#:         #-          :*.  :*+-*: ..  .......       ....                            #-                     
#     .#=.        :#=:::::-::::-:-*-:---===++*####@#**++++++++++++++++++=-#-        =#           .*. +*-:*- ....   ....  ...           ...  .              .   #-                     
#     =%+=-::::-=*-:--:--::---:::---==++*##**+=--%*=+++++++++++++++++++==%-        #-            *+*=::++   ..     ...                    ...    .......   . .#-                     
#     *+:-==++=-::-:--:----:::--==+*##%%+--::--:+@++++++++++++++++++++=*#.       -#             *%-::-#. ..                                      ...        .%:                     
#     :#-.:::::::::::-:-:::--==++###*+=*%=:-----:#%++++++++++++++++++=*#:        #=            +*::::+* .                                                   :%.                     
#         =*:::::-:::---:-::--==+*#%*++++++#%-:-:::::%#=++++++++++++++++*#:        -#            +*-::::++ .                                              .    =#                      
#         +*::::---:::-::--=++*%#*+++++++++##:::... :%*++++++++++++++*#=          %-           =#-:::::-#.                                                    ++                      
#         *+::::-::-:::-=++*%#+++++++++++++%*..     :%*+++++++++++*+-           +#           :#-:::::::+*.                                                  .#-                      
#         *+:::----:--=++#%*+++++++++++++++%+       :%*+++++++**=.            .%:          .#=:-:::--::=*=:.....                                           -#.                      
#         .*+:::-:::-=++%%+++++++++++++++++*%-       .##++**+=:               *+           =#:::::-:::-::-======+++==:                                     +*                       
#             .*+:::-:-=++*%=++++++++++++++++++*%:        +%*-.                 =#            +*::::-::::---::::::::::-=*=                                   .#-                       
#             ++::::-=++*#+++++++++++++++++++=##.        -*-                 .%:            +*::--::::--::-::----:--:::*=                                  -#                        
#             +*-::-=+++++++++++++++++++++****%*         .=+-               #=             -%-:--::-----::--:::-:--:-:-#.                                 *+                        
#                 :+*+=-====+++++++++*******++-:..#-          .++-            ++               *#::---:::-:-+*-:--:-:-:-::#-                                :#.                        
#                 .-=++++****+++++==-::.        -#            .=+-.        =*                .**::--::-:=#=:::------:-::+=                                *+                         
#                                                 #-             .-+=:   .-+%+====:.           .**::-:--:--:::::-:-::::-:-*:                    .--==+=-  -#.                         
#                                                 ++                :**===-:.....:=+++:          +#-::-:-::--::--:----:--:=*:                .-++=-----*+.*=                          
#                                                 =*               :++-.             :+*=         -#+::----:-----:--:--::::=*-.          .:-++=-:::::-:-*+*                           
#                                                 -* .:--====-:. .++:                  .=*-        .+*=:::::::--:--::-:---::-+*+=-----==+++=-:::---:-::-#%:                           
#                                                 =%++=-:::::-=+*#-                      .*+.... ..:-*%*+++=--:::-::::=#=::-:::-===+===--::::--::----::-%=                            
#                                             .++-.          :#:                         *+..:=++-:.....:-=++=::--:-#=::--:--::::::::::--:::::-:-::::*+                             
#                                             =+:             *+                          .#=+*-             .=*=:-:--:-----:::::--------::::-::--:::*+                              
#                                             =+               #-                           *@=                 :%+::::-:-:--::----::-::---:--:---::=#+                               
#                                             :#:               %-                           ++                   :%#+=-::::::::::::---::-:::::::::=**:                                
#                                             .#.               #+                          .#:                    =%=++**+==------:::::::-----==+*+:                                  
#                                             *-               :*.                         =*                     :%:.=+=:--=++++++++++++++++=--.                                     
#                                             :+%:               .:                        :#.                     =#=*=.        .:-----:.                                             
#                                         :+=.-#-               .:                      :+.                     .%%*+---==:.-++==----==+=:                                           
#                                         -+.   .=:.                                                            .*+.      .+%=..        .:++.                                         
#                                         .#-       ...                                                         :::          =*...........  :*:                                        
#                                         :#:                                                                  .              -............. -#.                                       
#                                         *-                                                                                  .............. *=                                       
#                                         -#.                                                                                 .............. ++                                       
#                                         -*:                                                                                .............. #=                                       
#                                         :*=.                                                                             .............. =#                                        
#                                             :++                                                                           ............. .+*.                                        
#                                             .*=                                                                              .....   ..=*=                                          
#                                             -*:                                                                            .::....::-=++-                                            
#                                         =*.                                                                              +%+====-:.                                               
#                                         =*.                                                                                =+                                                      
#                                         =*.                                                                                 .#-                                                     
#                                         -#.                  .:.                                                              =*                                                     
#                                     .#-                 +#%%%#:                          ...                               .#:                                                    
#                                     ++                :#%####%%-                       -*###*-                              *=                                                    
#                                     :#:               :%%######%#.                    :#%####%@+                             +*                                                    
#                                     ++          ....  %%########@=                   -%%######%@:                            =#                                                    
#                                     .#-        ...... =@#########%*                  :%%########%+                            =%                                                    
#                                     :#:       ........#%#########%#.                 #%#########%#. ...                       =#                                                    
#                                     -#.       .......:@%#########%%.                :@%#########%%:......                     =#                                                    
#                                     -#.       .......:@%#########%%.                =@##########%%:.......                    +*                                                    
#                                     -%.        .......%%#########%#.                =@##########%%:.......                    #=                                                    
#                                     :#:         ..... +@#########@*                 =@##########%#........                   -#.                                                    
#                                     .#=               .%%#######%@-                 :@%#########@+ ......                    *+                                                     
#                                     **                -@#######@*                   *@########%#. ....                     -%.                                                     
#                                     :#:                =%%####%*.                   .%%######%%:                          .#=                                                      
#                                     +*                 :+###*=                      :#%####%#:                          .#+                                                       
#                                         *=                   ..                          =*##*=.                          .#+                                                        
#                                         .*=                                                                              :#=                                                         
#                                         .*+                                                                            =*-                                                          
#                                         =*:                                                                        :*+.                                                           
#                                             :*+:...                                                                 :+*:                                                             
#                                             -*+:......                                                         .:++:                                                               
#                                                 -*+-.........                                              ....:=*=:                                                                 
#                                                 :+*=:...:::..........                         ...........:-+*+:                                                                    
#                                                     .-+*+-:...:::::.............................:::::...:=++=:                                                                       
#                                                         :=+++-::...:::::::::::::::::::::::::::::..::-=+++-.                                                                          
#                                                         .:=++++=-:::....::::::::::.....::::--==++=-.                                                                              
#                                                                 .:-=++++=---:.::.::::-==++++++=-:.                                                                                   
#                                                                     .:--+#-....::**-:..                                                                                           
#                                                                             ++.:::.-#.                                                                                               
#                                                                             ==.:::.#=                                                                                                
#                                                                             *-.::.-#.                                                                                                
#                                                                         .#:....*=                                                                                                 
#                                                                         -*    :%.                                                                                                 
#                                                                         +=    +*                                                                                                  
#                                                                         .#:    #-                                                                                                  
#                                                                         -*    :#.                                                                                                  
#                                                                         *-    =*                                                                                                   
#                                                                         :#     #=                                                                                                   
#                                                                         +=    .%:                                                                                                   
#                                                                         :#.    :%                                                                                                    
#                                                                         +* ..  :%                                                                                                    
#                                                                     :%:.....:%.                                                                                                   
#                                                                     *+ ......#-                                                                                                   
#                                                                     +*........=#                .-=+=                                                                              
#                                                                     =#:........-**:           .-+++=-#+                                                                             
#                                                                     =#:........==-=**+=-:::-=++++====-**                                                                             
#                                                                 =#:....:... **-====+++++++=======-=%-                                                                             
#                                                                 +*:....*#... =@+==================+#-                                                                              
#                                                                 .**.....*+*= ..:%**+==============+*+.                                                                               
#                                                             -*=....-*= :#... -*:-+***++==+++**+=.                                                                                 
#                                     .:--::.                .=+=.....=*:   +*... -*-  .:-====--:.                                                                                    
#                                 -+=-::--==+====---:---===+-.....-*=      **.....=+-                                                                                               
#                                 ++ ...    ...:::-----::.......-++.        +#- ....=+-.                                                                                            
#                                 .++:.......................:=+=.           :++: ....:=+=:.            .:--.                                                                       
#                                     :++-:................:-=+=-                :++-......:-=++==----=====-:-*=                                                                      
#                                     .-=+==--:--:--=====-:.                     :+*-.........:::---:..     *=                                                                      
#                                         .::--==---:..                            .=++-:.................-*-                                                                       
#                                                                                     .-=++=--::::::::-===-.                                                                        
#                                                                                         .::-=====---:..                                                                           
####################################################################################################################################################################