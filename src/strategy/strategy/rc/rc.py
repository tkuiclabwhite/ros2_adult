#!/usr/bin/env python3
#coding=utf-8
import sys
import rclpy 
from strategy.API import API
import numpy as np
import math
import time


#--校正量--#
#前進量校正
FORWARD_CORRECTION         = -700
#平移校正
TRANSLATION_CORRECTION     = -200
#旋轉校正
THETA_CORRECTION           = -1
#基礎變化量(前進&平移)
BASE_CHANGE                = 300
HEAD_CHANGE_H              = 5
HEAD_CHANGE_V              = 100
# ---微調站姿開關---#
STAND_CORRECT_CW           = False                 #sector(33) CW_stand微調站姿 測就知道
DRAW_FUNCTION_FLAG         = True                  #影像繪圖開關
TARGET_COLOAR_1              = 'Red'                  #'Orange' 'Yellow' 'Blue' 'Green' 'Black' 'Red' 'White'       
TARGET_COLOAR_2              = 'Blue'

W_CENTER = 30  # 距離中心點的權重 
W_ALIGN = 70  # 水平對齊的權重 
W_HEIGHT = -50  #高度權重





#------------------#
HEAD_HORIZONTAL            = 2048  


#------------------#
HEAD_HORIZONTAL            = 2048               #頭水平
HEAD_VERTICAL              = 2048              #頭垂直 #30cm2150

HEAD_LEFT_HAND_H = 2550
HEAD_LEFT_HAND_V = 2100  #2100

HEAD_RIGHT_HAND_H = 1550
HEAD_RIGHT_HAND_V = 2050
 
HEAD_LEFT_LEG_H = 1750
HEAD_LEFT_LEG_V = 1750

HEAD_RIGHT_LEG_H = 2350
HEAD_RIGHT_LEG_V = 1750

MOTOR_LEFT_HAND_Y = 1071
MOTOR_LEFT_HAND_X = 1919

MOTOR_RIGHT_HAND_Y = 2406
MOTOR_RIGHT_HAND_X = 1742

MOTOR_LEFT_LEG_Y = 0
MOTOR_LEFT_LEG_X = 0

MOTOR_RIGHT_LEG_Y = 0
MOTOR_RIGHT_LEG_X = 0

MOTOR_LEFT_HEAP = 0
MOTOR_RIGHT_HEAP = 0


#判斷值
# FOOTLADDER_LINE            = 215                   #上梯基準線
MY_LINE_Y= 120 #攀岩基準線
MY_LINE_X =160 #攀岩基準線
MY_SIZE = 760
ROI_RADIUS = 120

W_FAR = MY_SIZE + 150
RATIO_FAR = -0.90 #2251-2363)/100

W_NEAR  = MY_SIZE + 150
RATIO_NEAR = -1.35 #(2251-2363)/100

#前後值
BACK_MIN                   = -800                #小退後
FORWARD_MIN               = 100
FORWARD_LOW               = 200             #小前進 300
FORWARD_NORMAL             = 300                 #前進
FORWARD_HIGH                = 500               #大前進

#平移值
TRANSLATION_BIG            = 500                  #大平移
TRANSLATION_NORMAL         = 300
#旋轉值
# THETA_MIN                  = 3                     #小旋轉
# THETA_NORMAL               = 1                    #旋轉
# THETA_BIG                  = 5                     #大旋轉
#----------------------------------------手校正
LHEAD_X_CA = 2576
LHEAD_Y_CA = 2101

LHAND_X_CA = 1791
LHAND_Y_CA = 1836

RHEAD_X_CA = 1618
RHEAD_Y_CA = 2357

RHAND_X_CA = 2060
RHAND_Y_CA = 2701
#----------------------------------------腳校正
HEAD_X_CA2 = 2600
HEAD_X_CA2 = 2190

LEG_X_CA = 1
LEG_Y_CA = 1
#---------------------------------------腳高度校正


class WallClimbing(API):
#CW主策略
    def __init__(self):
        super().__init__('wall_climbing_node')
        self.target = ObjectInfo(TARGET_COLOAR_1,TARGET_COLOAR_2,'Ladder',self)
        self.init()
        self.timer = self.create_timer(0.05, self.strategy_loop)
        self.get_logger().info("Wall Climbing Node Initialized")
        
        # self.STAND_CORRECT_CW = STAND_CORRECT_CW

    def strategy_loop(self):
        if self.is_start :
            self.current_strategy = "Wall_Climb_on"
        else:
            self.current_strategy = "Wall_Climb_off" 
        # self.current_strategy = "Wall_Climb_on"            
        strategy = self.current_strategy
        if DRAW_FUNCTION_FLAG:
            self.draw_function()

        sys.stdout.write("\033[J")
        sys.stdout.write("\033[H")

        self.get_logger().info('________________________________________\033[K')
        self.get_logger().info(f'x: {self.now_forward} ,y: {self.now_translation} ,theta: {self.now_theta}\033[K')
        self.get_logger().info(f'Goal_x: {self.forward} ,Goal_y: {self.translation} ,Goal_theta: {self.theta}\033[K')
        self.get_logger().info(f"機器人狀態: {self.state}\033[K")
        self.get_logger().info(f"target_1_cy: {self.target_1_cy}\033[K")
        self.get_logger().info(f"head_h : {int(self.now_head_Horizontal)},head_v: {int(self.now_head_Vertical)},track:{int(self.track)}\033[K")
        self.get_logger().info('￣￣￣￣￣￣')
        
        
        if strategy == "Wall_Climb_off":
        #關閉策略,初始化設定
        
            if (not self.walkinggait_stop and self.climbing):
                self.get_logger().info("🔊CW parameter reset\033[K")
                self.init()
                self.sendbodyAuto(0)
                self.sendSensorReset()           #IMUreset
                time.sleep(2)
                self.sendBodySector(29)             #基礎站姿磁區
                time.sleep(1.5)
                # if STAND_CORRECT_CW:
                #     send.sendBodySector(30)             #CW基礎站姿調整磁區
                #     STAND_CORRECT_CW = False 
                
                self.get_logger().info("reset\033[K")
                self.imu_reset = True
            self.sendHeadMotor(1,HEAD_HORIZONTAL,100)  #水平
            self.sendHeadMotor(2,HEAD_VERTICAL,100)    #垂直
            self.init()
            # self.find_ladder()
            # self.keep_head()
            self.get_logger().info("turn off\033[K")
            self.walkinggait_stop = True

        elif strategy == "Wall_Climb_on":#為甚麼要有兩個參數
        #開啟CW策略
            if self.reset:
                    #self.sendSensorReset()   #!!記得改 debug用
                    self.sendHeadMotor(1,HEAD_HORIZONTAL, 80)#水平
                    self.sendHeadMotor(2,HEAD_VERTICAL, 80)#垂直
                    time.sleep(1)
                    self.reset = False

            if self.state != 'cw_finish':
                # if self.STAND_CORRECT_CW:
                #     send.sendBodySector(102)             #CW基礎站姿調整磁區
                #     while not send.execute:
                #         rospy.logdebug("站立姿勢\033[K")
                #     send.execute = False
                #     self.STAND_CORRECT_CW = False
                #     rospy.sleep(2)
                
                if not self.readyclimb:
                    self.sendbodyAuto(1)            #開始動作
                    self.find_ladder()
                    self.track = (int(self.target_1_cy - MY_LINE_Y))*(38/240)
                    self.now_head_Vertical = self.now_head_Vertical - round(self.track * 4096 / 360 * 0.05)
                    if abs(self.now_head_Vertical - 2048) > 900 :   # 防止馬達過度旋轉
                        if (self.now_head_Vertical - 2048 ) > 0 :
                            self.now_head_Vertical = 2048 + 900
                        elif (self.now_head_Vertical - 2048) < 0 :
                            self.now_head_Vertical = 2048 - 900
                    self.sendHeadMotor(2,self.now_head_Vertical,100)  #像素轉換馬達值
                    # self.sendHeadMotor(2,HEAD_VERTICAL ,20)
                    motion = self.new_edge_judge()
                    self.walkinggait(motion)
                    
                else:
                    if not self.climbing:  # readyclimb == True：開始抓點 + 動作
                        self.action, self.value = self.lambs_select()

                        # 關鍵：找不到點就只掃描，不要進 climbmode 卡死
                        if self.action == 'searching' or self.value == 'no_object' or self.value is None:
                            self.get_logger().info(f"找不到點，持續掃描中... step={getattr(self,'climb_step',0)} scan={getattr(self,'scan_count',0)}\033[K")
                            return
                        
                        # 找到點才真的執行動作
                        self.get_logger().info(f"看完了嘛:{self.lookok}\033[K")
                        if self.lookok:
                            self.state = '已找到攀爬點，準備切換追蹤模式'
                            self.climbing = True
                            self.lookok = False
                    else:
                        if self.move_ok:
                            # 動作做完了，狀態重置，準備換下一隻手
                            self.get_logger().info("動作完成，重置狀態換手")
                            self.beseen_lamb = False
                            self.climbing = False
                            self.move_ok = False
                            self.okcnt = 0
                            self.climb_step += 1
                            self.been_done = False
                        else:
                            self.state = "視覺追蹤與姿態準備中"
                            self.climbmode(self.action, self.value)
                            

                    
                    
                    
    def init(self):
        #狀態
        self.state                 = '停止'
        self.is_start = False
        #步態啟動旗標
        self.walkinggait_stop      = True  
        
        #設定頭部馬達
        self.now_head_Horizontal   = HEAD_HORIZONTAL
        self.now_head_Vertical = HEAD_VERTICAL
        
        #imu_reast
        self.imu_reset             = True

        self.target_1_y = 0
        self.target_1_ym = 0
        self.target_2_y = 0
        self.target_2_ym = 0


        #步態參數
        self.forward               = 0
        self.translation           = 0 
        self.theta                 = 0 
        self.now_forward           = 0 
        self.now_translation       = 0
        self.now_theta             = 0

        self.new_target_xmin       = 0
        self.new_target_xmax       = 0 
        self.size = 0 

        self.lost_target_count = 0

        self.forward_ok = False
        self.dist_ok = False
        self.pos_x_ok = False

        self.head = True
        self.action = None
        self.value = None
        self.reset =True
        self.readyclimb=False 
        self.lookok = False
        self.climbing=False
        self.beseen = False
        self.move_ok = False
        self.track_ok = False
        self.beseen_lamb = False
        self.allontarget = False
        self.been_done = False

        self.object_x = 0  #debug default 0
        self.object_y = 0  # 0
        self.object_y1_len = 0
        self.object_y2_len = 0
        self.target_1_cx = 160
        self.target_2_cx = 160
        self.target_1_cy = 120
        self.target_2_cy = 120
        self.target_1_size = 0
        self.target_2_size = 0

        self.track = 0
        self.stoptime = 0

        self.cnt = 1


        self.best_candidate = None
        self.max_score = -1

       
        self.times = 1

        self.score_left_hand = 0
        self.score_right_hand = 0
        self.score_left_leg = 0
        self.score_right_leg = 0

        head1 = np.array([[LHEAD_X_CA,0],[0,LHEAD_X_CA]])
        hand1 = np.array([[LHAND_X_CA],[LHAND_Y_CA]])
        linear_solution = np.linalg.solve(head1,hand1)

        self.p_x1 = linear_solution[0]
        self.p_y1 = linear_solution[1]


        head2 = np.array([[RHEAD_X_CA,0],[0,RHEAD_X_CA]])
        hand2 = np.array([[RHAND_X_CA],[RHAND_Y_CA]])
        linear_solution = np.linalg.solve(head2,hand2)

        self.p_x2 = linear_solution[0]
        self.p_y2 = linear_solution[1]

        self.search_stage = 0
        # self.p_height = HEAP_HEIGHT / TARGET_SIZE 

    def find_ladder(self):
        sys.stdout.write("\033[K")

        # 1. 確保變數存在，初始化以預防 AttributeError

        color_1 = self.target.color1
        color_2 = self.target.color2


        target_colors = [color_1, color_2]
        candidates_y_max = []

        for color in target_colors:
            obj_count = self.color_counts[color]
            self.get_logger().info(f"color {color} subject cnts: {obj_count}\033[K")
            actual_list_len = len(self.object_sizes[color])
            for i in range(min(obj_count, actual_list_len+1)):
                if self.object_sizes[color][i] > 100:
                    cx = (self.object_x_max[color][i] + self.object_x_min[color][i]) // 2
                    cy = (self.object_y_max[color][i] + self.object_y_min[color][i]) // 2
                    if cx is not None:
                        candidates_y_max.append({
                            'y_max':self.object_y_max[color][i],
                            'color': color,
                            'cx':cx,
                            'cy':cy,
                            'size':self.object_sizes[color][i],
                            'idx': i
                        })
            
            # 修正 2: 修正 key 名稱 (y_max 而非 ymax)
        candidates_y_max.sort(key=lambda x: x['y_max'], reverse=True) 

        if len(candidates_y_max) >= 1:
            c1 = candidates_y_max[0]
            self.target_1_cy = c1['cy']
            self.target_1_cx = c1['cx']
            self.target_1_size =c1['size']

            if len(candidates_y_max) >= 2:
                c2 = candidates_y_max[1]
                if abs(c1['cx'] - c2['cx']) > 30 :
                    self.target_2_cy = c2['cy']
                    self.target_2_cx = c2['cx']
                    self.target_2_size = c2['size']
                else:
                    if len(candidates_y_max) >= 3:
                        c2 = candidates_y_max[2]
                        self.target_2_cy = c2['cy']
                        self.target_2_cx = c2['cx']
                        self.target_2_size =c2['size']
                    else:
                        self.target_2_cx = self.target_1_cx

                self.object_x = abs(self.target_1_cx + self.target_2_cx)//2
                self.size = self.target_1_size + self.target_2_size             #面積
            else:
                self.object_x = 0
            self.get_logger().info(f"鎖定目標 - Y1:cx,cy:{self.target_1_cx},{self.target_1_cy}, X_Dist:{self.object_x}")
            self.get_logger().info(f"鎖定目標 - Y2:cx,cy:{self.target_2_cx},{self.target_2_cy}")
            self.get_logger().info(f"鎖定目標 - Y1:size:{self.target_1_size},鎖定目標 - Y2:size:{self.target_2_size}")
        else:
            self.get_logger().info("丟失目標...\033[K")
            
        
    def new_edge_judge(self):
        # 如果完全沒目標
        if not hasattr(self, 'lost_target_count'): self.lost_target_count = 0

        if self.size == 0 and self.object_x == 0:
            self.lost_target_count += 1
            self.forward = 0.0
            self.translation = 0.0
            if self.lost_target_count > 20: # 連續 1 秒沒看到
                self.state = "目標丟失：搜尋中"
                self.forward= 500 # 緩慢直走找尋
            return 'walking'  
        else:
            self.lost_target_count = 0

        # 偏差計算 ---

        error_x = MY_LINE_X - self.object_x
        error_size = MY_SIZE - self.size

        self.get_logger().info(f"目前面積:{self.size}")
        self.get_logger().info(f"垂直線:{MY_LINE_X},我的面積:{MY_SIZE}")
        self.get_logger().info(f"誤差size:{error_size}, 誤差x:{error_x}")
        
        # 前後距離控制 ---
        if not self.forward_ok :
            if abs(error_size) <= 10:   #前進死區值
                self.forward = 0.0 
                time.sleep(2)
                self.forward_ok = True 
            else:
                if error_size > 0:
                    if error_size > 200 and error_size <400:
                        self.forward = FORWARD_NORMAL
                    elif error_size > 401:
                        self.forward = FORWARD_HIGH
                    else:
                        self.forward = FORWARD_LOW
                else:
                    self.forward = BACK_MIN
        else:
                self.get_logger().info("前進對齊,準備平移")
                if self.object_x > 0:            
                    self.translation = max(min(error_x * 35, 1000), -1000) if abs(error_x) > 10.0 else 0.0
                else:
                    self.translation = 0.0

                self.pos_x_ok = abs(error_x) <= 15 or self.object_x == 0      #平移死區值

        if self.forward_ok  and self.pos_x_ok : 
            if not hasattr(self, 'ready_count'): self.ready_count = 0
            self.get_logger().info("穩定加一")
            time.sleep(0.25)
            self.ready_count += 1
        else:
            self.ready_count = 0

        if  self.ready_count > 2: #  5 幀穩定
            self.state = "對齊完成：切換攀爬模式"
            return "ready_to_cw"
        else:
            self.state = f"對齊中...穩定度:{int(self.ready_count/2*100)}%"
            return "walking"
         


    def walkinggait(self,motion):
    #步態函數
        if motion == 'ready_to_cw':
            self.get_logger().info("正面對齊：準備攀爬\033[K")
            self.forward = 0.0
            self.translation = 0.0

            self.sendbodyAuto(0)        #停止步態
            time.sleep(0.5)
            self.sendSensorReset(True)  #IMU reset 避免機器人步態修正錯誤
            self.state = 'ready_finish'
            self.readyclimb = True  


        else:
            self.now_forward = self.ramp_speed(self.now_forward, self.forward, BASE_CHANGE)
            self.now_translation = self.ramp_speed(self.now_translation, self.translation, BASE_CHANGE)
               
            f = max(min(int(self.now_forward), 501), -501)
            t = max(min(int(self.now_translation), 1001), -1001)

            self.get_logger().info(f"now_forward   :{self.now_forward}")
            self.get_logger().info(f"now_translation  : {self.now_translation}")
            
            self.get_logger().info(f"f   :{f}")
            self.get_logger().info(f"t   :{t}")
            
            if not self.forward_ok:
                self.sendContinuousValue(f, TRANSLATION_CORRECTION ,THETA_CORRECTION)
            else:
                self.sendContinuousValue(-800,t,-1)


    def ramp_speed(self, current, target, step):
        if abs(current - target) < step:
            return target
        if current < target:
            return current + step
        elif current > target:
            return current - step
        return target


    def climbmode(self, action,target_data):        #目前需增加攀爬後重心轉移，攀爬中若脫鉤的應變

        current_target = self.get_best_climbing_target()
        
        if current_target == 'no_object' or current_target is None:
            self.get_logger().info(f"目標丟失！當前追蹤側: {action}，啟動分段搜尋...")
            
            if self.search_stage == 0:
                self.search_stage = 1
                self.cnt = 0
            
            if self.search_stage == 1:
                step_h = -HEAD_CHANGE_H if action == 'left' else HEAD_CHANGE_H
                self.sendHeadMotor(1, self.now_head_Horizontal + step_h, 30)
                self.now_head_Horizontal += step_h 
                
                if abs(self.now_head_Horizontal - 2048) <= abs(HEAD_CHANGE_H):
                    self.get_logger().info("頭部已抵達中心，準備往上掃描")
                    self.search_stage = 2
                    
            elif self.search_stage == 2:
                self.sendHeadMotor(2, HEAD_CHANGE_V, 30) 
                self.now_head_Vertical += HEAD_CHANGE_V
                
                self.get_logger().info("抬頭完成，準備反向掃描")
                self.search_stage = 3
                
            elif self.search_stage == 3:
                step_h = HEAD_CHANGE_H if action == 'left' else -HEAD_CHANGE_H
                
                self.sendHeadMotor(1, self.now_head_Horizontal + step_h, 30)
                self.now_head_Horizontal += step_h

            self.cnt += 1
            if self.cnt > 20 :
                self.get_logger().info("搜尋超時！停止動作確保安全")
                self.okcnt = 0
                self.search_stage = 0  # 重置搜尋狀態
                self.sendbodyAuto(0)   # 讓機器人先暫停動作確保安全
            return
        
        else:
            if self.search_stage != 0:
                self.get_logger().info("重新鎖定目標！恢復正常追蹤")
                self.search_stage = 0
                self.cnt = 0
        
        if target_data == 'no_object' or target_data is None:
            self.get_logger().info("壞了...")
            self.get_logger().info("我閃亮登場...")
            self.sendbodyAuto(0)                        #若攀爬中僵直，會很危險，在之後會改 , 增加脫鉤回正 ，如果可以的話
            return
        
        else:

            config = {
                'left_hand': {
                    'ids': [1, 2], 'base_m': [HEAD_LEFT_HAND_H, HEAD_LEFT_HAND_V],
                    'weight_sector': 871, 'climb_sector_1': 10 ,'climb_sector_2': 11 ,'climb_sector_3': 12 ,'climb_sector_4':16,
                    'ready_climb' : [MOTOR_LEFT_HAND_X , MOTOR_LEFT_HAND_Y],
                },
                'right_hand': {
                    'ids': [5, 6], 'base_m': [HEAD_RIGHT_HAND_H, HEAD_RIGHT_HAND_V],
                    'weight_sector': 873, 'climb_sector_1': 13 , 'climb_sector_2': 14 ,'climb_sector_3': 15 ,'climb_sector_4':17,
                    'ready_climb' : [MOTOR_RIGHT_HAND_X , MOTOR_RIGHT_HAND_Y],
                },
                'left_leg': {
                    'ids': [12, 10], 'base_m': [HEAD_LEFT_LEG_H, HEAD_LEFT_LEG_V],
                    'weight_sector': 875, 'climb_sector_1': 0 ,'climb_sector_2': 0 ,'heap_sector': 0 ,
                    'ready_heap' :[MOTOR_LEFT_HEAP],
                    'ready_climb' : [MOTOR_LEFT_LEG_X , MOTOR_LEFT_LEG_Y]
                },
                'right_leg': {
                    'ids': [18, 16], 'base_m': [HEAD_RIGHT_LEG_H, HEAD_RIGHT_LEG_V],
                    'weight_sector': 877, 'climb_sector_1': 0 ,'climb_sector_2': 0 ,'heap_sector': 0 ,
                    'ready_heap' :[MOTOR_RIGHT_HEAP],
                    'ready_climb' : [MOTOR_RIGHT_LEG_X , MOTOR_RIGHT_LEG_Y]
                }
            }

            # if action == 'left_hand' and not self.been_done :
            #     self.sendSingleMotor(9,-200,20)
            #     self.been_done = True
            # elif action == 'right_hand' and not self.been_done:
            #     self.sendSingleMotor(9,400,20)
            #     time.sleep(1)
            #     self.been_done = True

            if action not in config:
                self.get_logger().error(f"蛤—這是甚摸: {action}")
                return
            
            cfg = config[action]

            if not self.move_ok:
                self.keep_head(current_target['center'])
            

            if self.track_ok:
                if not hasattr(self, 'okcnt'): self.okcnt = 0
                self.okcnt += 1
                self.get_logger().info(f"目標對準中，穩定度:{int(self.okcnt/15*100)}%")
            else:
                self.okcnt = 0

            if self.okcnt > 14 and not self.move_ok:
                self.state = "穩定鎖定！開始攀爬動作"
                self.get_logger().info(f"對準目標中心 x:{current_target['center'][0]} y: {current_target['center'][1]}")
                if not self.allontarget :
                    if action == 'left_hand' or action == 'right_hand':
                        time.sleep(1)
                        self.sendBodySector(cfg['climb_sector_1'])
                        time.sleep(2)
                        self.sendBodySector(cfg['climb_sector_2'])
                        time.sleep(2.5)
                        self.sendBodySector(cfg['climb_sector_3'])
                        time.sleep(2)
                        self.sendBodySector(cfg['climb_sector_4'])
                        time.sleep(2)
                        self.get_logger().info(f"動作中")
                    elif action == 'left_leg' or action == 'right_leg':
                        #動作
                        #動作
                        if action == 'right_leg':
                            self.allontarget = True
                else:
                    self.sendBodySector(cfg['default_sector'])
                    time.sleep(2)
                    self.get_logger().info(f"動作中")


                # current_w = current_target['size']

                # if current_w <= W_FAR:
                #     alpha = 0.0
                # elif current_w >= W_NEAR:
                #     alpha = 1.0
                # else:
                #     alpha = (current_w - W_FAR) / (W_NEAR - W_FAR)

                # # === 3. 算出當下最完美的動態 left_value ===
                # dynamic_left_value = RATIO_FAR + alpha * (RATIO_NEAR - RATIO_FAR)
                
                # # 假設左右手視差對稱，共用同一個動態比例
                # dynamic_right_value = dynamic_left_value

                left_value = (2251-2363)/100
                right_value = left_value
           
                if action == 'left_hand':
                    
                    #motor_value_x = ((self.now_head_Horizontal-2394) * self.p_x1 * DS) - cfg['ready_climb'][0]
                    #motor_value_x = (self.now_head_Horizontal-2550) * dynamic_left_value
                    motor_value_x = (self.now_head_Horizontal-2550) * left_value
                    motor_value_y = ((self.now_head_Vertical * self.p_y1) - cfg['ready_climb'][1]) + 325 #175
                    


                    self.get_logger().info(f"{action} 最終輸出: M1(Y)={int(motor_value_y) }, M2(X)={int(motor_value_x) }")
                    
   

                elif action == 'right_hand':
                    
                    #motor_value_x = abs((self.now_head_Horizontal * self.p_x2 * DS) - cfg['ready_climb'][0] )
                    #motor_value_x = (self.now_head_Horizontal - 1546) * dynamic_right_value  + 100 
                    motor_value_x = (self.now_head_Horizontal - 1546) * right_value  + 100 
                    motor_value_y = ((self.now_head_Vertical * self.p_y1) - cfg['ready_climb'][1]) + 125 #275

                    self.get_logger().info(f"{action} 最終輸出: M1(Y)={int(motor_value_y) }, M2(X)={int(motor_value_x) }")
                    
                if action == 'left_hand' or action == 'right_hand':
                    
                    self.sendSingleMotor(cfg['ids'][1], int(motor_value_x), 20)
                    time.sleep(2)
                    self.sendSingleMotor(cfg['ids'][0], int(motor_value_y), 20)
                    time.sleep(2)
                if action == 'left_leg' :
                    self.sendBodySector(19)
                    time.sleep(2)
                    self.sendBodySector(20)
                    time.sleep(2)
                    self.sendBodySector(21)
                    time.sleep(3)
                    self.sendBodySector(22)
                    time.sleep(2)
                    self.sendBodySector(23)
                    time.sleep(2)
                    self.sendBodySector(24)
                    time.sleep(2)
                    


                self.move_ok = True
                self.state = '攀爬結束'

                
            # if track_data['size']<CAN't str:
            #     cmd backagian


    def keep_head(self,coordinate):

        cx,cy = coordinate

        error_x = cx - MY_LINE_X # 水平誤差
        error_y = cy - MY_LINE_Y # 垂直誤差

        diff_h = (error_x * (51/320)) * (4096/360) *0.05
        diff_v = (error_y * (38/240)) * (4096/360) *0.05

        self.get_logger().info(f"diff_h:{diff_h},diff_v :{diff_v}")

        self.now_head_Horizontal -= round(diff_h)
        self.now_head_Vertical -= round(diff_v)

        self.now_head_Horizontal = max(1000, min(3000, self.now_head_Horizontal))
        self.now_head_Vertical = max(1000, min(3000, self.now_head_Vertical))

        self.sendHeadMotor(1, int(self.now_head_Horizontal), 100) # 水平
        self.sendHeadMotor(2, int(self.now_head_Vertical), 100) # 垂直

        if abs(error_x) < 10 and abs(error_y) < 10 :
            self.track_ok = True 
        else:
            self.track_ok = False


    def imu_angle(self):      #一般 imu修正角度
        imu_ranges = [  (180,  -3),
                        (90,  -3), 
                        (60,  -3), 
                        (45,  -3), 
                        (20,  -3), 
                        (10,  -2), 
                        (5,   -2), 
                        (2,   -1), 
                        (0,    0),
                        (-2,    1),
                        (-5,    2),
                        (-10,   2),
                        (-20,   3),
                        (-45,   3),
                        (-60,   3),
                        (-90,   3),
                        (-180,   3)]
        for imu_range in imu_ranges:           
            if self.imu_rpy[2] >= imu_range[0]:
                return imu_range[1]
        return 0
    def draw_function(self): 
            self.drawImageFunction(1,1,0,320,MY_LINE_Y,MY_LINE_Y,255,255,3) #水平判斷線
            self.drawImageFunction(2,1,MY_LINE_X,MY_LINE_X,0,240,255,255,3) #垂直判斷線
            edge_x = 160-ROI_RADIUS  #最左上角的x
            edge_y = 120-ROI_RADIUS  #最左上角的y
            xmin = edge_x
            xmax = ROI_RADIUS*2 + edge_x
            ymin =edge_y
            ymax =ROI_RADIUS*2 + edge_y

            self.ROI_cx = ROI_RADIUS + edge_x
            self.ROI_cy = ROI_RADIUS + edge_y
           
            self.drawImageFunction(3,1,xmin,xmax,ymin,ymax,0,255,255)
            self.drawImageFunction(4,1,self.ROI_cx,self.ROI_cx,self.ROI_cy,self.ROI_cy,0,255,255)

    def get_best_climbing_target(self):
        best_candidate =[]   # 每一次找攀爬點都重新開始
        color_1 = self.target.color1
        color_2 = self.target.color2
        target_colors = [color_1,color_2]
        
        
        point_r = 10 

        
        for color in target_colors:
            cnts = self.color_counts[color]

            for i in range(min(cnts, len(self.object_sizes[color])+1)):
                # 取得基礎資訊
                size = self.object_sizes[color][i]
                if size < 300: continue # 過濾掉太小的雜訊
                
                # 取得邊界與中心

                cx =(self.object_x_max[color][i] + self.object_x_min[color][i])//2
                cy =(self.object_y_max[color][i] + self.object_y_min[color][i])//2

                if cx is None or cy is None:
                    continue

                xmax = self.object_x_max[color][i]
                xmin = self.object_x_min[color][i]
                ymax = self.object_y_max[color][i]
                ymin = self.object_y_min[color][i]
                
                # # --- 限制 1: 邊緣截斷過濾 (不可碰到畫面邊緣) ---
                if xmin <= 0 or xmax >= 320 or ymin <= 0 or ymax >= 240:
                    continue

                # --- 限制 2: 嚴格方形 ROI 判定 ---
                dist = math.sqrt((cx - self.ROI_cx)**2 + (cy - self.ROI_cy)**2)
                if (dist + point_r) > ROI_RADIUS:
                    continue

                center_ratio = 1.0 - (dist / ROI_RADIUS)
                center_score = center_ratio * W_CENTER

                align_ratio = 1.0 - (abs(cx - self.ROI_cx) / ROI_RADIUS)
                alignment_score = align_ratio * W_ALIGN

                height_ratio = (240.0 - cy) / 240.0
                height_score = height_ratio * W_HEIGHT

                height_score = int((240 - cy) * 5)
                alignment_score = int((1 - abs(cx - self.ROI_cx) / ROI_RADIUS) * 55)  #評分參數測試改
                
                total_score = int(center_score + alignment_score + height_score)
               
                best_candidate.append ({
                    'center': (cx, cy),
                    'score': total_score,
                    'size': size,
                    'bbox': (xmin, ymin, xmax, ymax),
                    'id': {i}
                })
                # 找出最高分
        if len(best_candidate) == 0:
            self.get_logger().info("畫面中沒有符合條件的攀爬點...")
            return 'no_object'

        best_candidate.sort(key=lambda x: x['score'], reverse=True)

        best = best_candidate[0]

        best_score =best['score']
        best_size = best['size']
        best_cx = best['center'][0]
        best_cy = best['center'][1]
        best_xmin = best['bbox'][0]
        best_xmax = best['bbox'][2]
        best_ymin = best['bbox'][1]
        best_ymax =  best['bbox'][3]

        
        self.get_logger().info(f"total_score,{best_score}")
        self.get_logger().info(f"size:{best_size}") 
        self.get_logger().info(f"cx:{best_cx},cy:{best_cy}")     
        self.get_logger().info(f"xmin,{best_xmin},ymin,{best_ymin},xmax,{best_xmax},ymax,{best_ymax},")        
        
        return best
        

    def lambs_select(self):
        # --- 1. 狀態初始化 (只會執行一次) ---
        if not hasattr(self, 'climb_step'): self.climb_step = 1  # 紀錄目前是第幾次動作
        if not hasattr(self, 'last_limb'): self.last_limb = None # 紀錄上一次動的部位
        if not hasattr(self, 'any_lamb'): self.any_lamb = None
        if not hasattr(self, 'scan_count'): self.scan_count = 0  # 紀錄自動掃描的次數
        if not hasattr(self, 'times'): self.times = 1

        if self.climb_step == 1:   #左手
            now_limb = 'left_hand'
            h_pos, v_pos = HEAD_LEFT_HAND_H, HEAD_LEFT_HAND_V
        elif self.climb_step == 2: #右手
            now_limb = 'right_hand'
            h_pos, v_pos = HEAD_RIGHT_HAND_H, HEAD_RIGHT_HAND_V
        elif self.climb_step == 3: #左腳
            now_limb = 'left_leg'
            h_pos, v_pos = HEAD_LEFT_LEG_H, HEAD_LEFT_LEG_V
        elif self.climb_step == 4: #右腳
            now_limb = 'right_leg'
            h_pos, v_pos = HEAD_RIGHT_LEG_H, HEAD_RIGHT_LEG_V
        else:                      
            now_limb = 'any' # 隨意
        
        self.get_logger().info(f"當前模式: {now_limb}")

        if now_limb != 'any':
            if not self.beseen_lamb:
                self.get_logger().info(f"頭部轉向 {now_limb} 尋找點位")
                current_v = int(v_pos - (self.scan_count * HEAD_CHANGE_V))
                self.sendHeadMotor(1, int(h_pos), 40)  
                self.sendHeadMotor(2, current_v, 40)

                self.now_head_Horizontal = int(h_pos)
                self.now_head_Vertical = current_v
                time.sleep(1)
                self.beseen_lamb = True
                return ('searching', 'no_object')
                
            target_info = self.get_best_climbing_target()

            if target_info == 'no_object' or target_info is None:
                self.lookok = False

                if now_limb == 'left_leg' or now_limb == "right_leg":
                    return ('searching', 'no_object')
                self.scan_count += 1
                if (self.now_head_Vertical - (self.scan_count * HEAD_CHANGE_V)) <= 2000: 
                    self.scan_count = 0
                self.beseen_lamb = False
                return ('searching', 'no_object')
            else:
                self.lookok = True
                self.scan_count = 0
                return (now_limb, target_info)
                

        else:
            if self.lookok:
                if self.any_lamb is None:
                    return ('none', 'no_object')
                if self.any_lamb == 'left_hand':
                    val = self.val_left_hand
                elif self.any_lamb == 'right_hand':
                    val = self.val_right_hand
                elif self.any_lamb == 'left_leg':
                    val = self.val_left_leg 
                elif self.any_lamb == 'right_leg':
                    val = self.val_right_leg
                    
                return (self.any_lamb, val)

            # 還沒決定，開始四處看
            else:
                if self.times == 1:
                    self.sendHeadMotor(1, HEAD_LEFT_HAND_H, 40)
                    self.sendHeadMotor(2, HEAD_LEFT_HAND_V, 40)
                    time.sleep(1.5)
                    self.val_left_hand = self.get_best_climbing_target()
                    self.score_left_hand = -1 if self.val_left_hand in [None, 'no_object'] else self.val_left_hand['score']
                    self.times = 2
                    return ('searching', 'no_object')

                elif self.times == 2:
                    self.sendHeadMotor(1, HEAD_RIGHT_HAND_H, 40)
                    self.sendHeadMotor(2, HEAD_RIGHT_HAND_V, 40)
                    time.sleep(1.5)
                    self.val_right_hand = self.get_best_climbing_target()
                    self.score_right_hand = -1 if self.val_right_hand in [None, 'no_object'] else self.val_right_hand['score']
                    self.times = 3
                    return ('searching', 'no_object')

                elif self.times == 3:
                    self.sendHeadMotor(1, HEAD_LEFT_LEG_H, 40)
                    self.sendHeadMotor(2, HEAD_LEFT_LEG_V, 40)
                    time.sleep(1.5)
                    self.val_left_leg = self.get_best_climbing_target()
                    self.score_left_leg = -1 if self.val_left_leg in [None, 'no_object'] else self.val_left_leg['score']
                    self.times = 4
                    return ('searching', 'no_object')

                elif self.times == 4:
                    self.sendHeadMotor(1, HEAD_RIGHT_LEG_H, 40)
                    self.sendHeadMotor(2, HEAD_RIGHT_LEG_V, 40)
                    time.sleep(1.5)
                    self.val_right_leg = self.get_best_climbing_target()
                    self.score_right_leg = -1 if self.val_right_leg in [None, 'no_object'] else self.val_right_leg['score']
                    self.times = 5
                    return ('searching', 'no_object')

                elif self.times >= 5:
                    self.times = 1 
                    
                    if self.last_limb == 'left_hand': self.score_left_hand = -1
                    elif self.last_limb == 'right_hand': self.score_right_hand = -1
                    elif self.last_limb == 'left_leg': self.score_left_leg = -1
                    elif self.last_limb == 'right_leg': self.score_right_leg = -1

                    # 找出最高分
                    scores = {
                        'left_hand': self.score_left_hand,
                        'right_hand': self.score_right_hand,
                        'left_leg': self.score_left_leg,
                        'right_leg': self.score_right_leg
                    }
                    
                    best_limb = max(scores, key=scores.get)
                    best_score = scores[best_limb]

                    if best_score == -1:
                        self.get_logger().info("四個方向都找不到點...")
                        return ('none', 'no_object')
                    
                    self.any_lamb = best_limb
                    self.lookok = True # 標記已經看好目標
                    self.get_logger().info(f"Any 模式選定目標: {best_limb}, 分數: {best_score}")
                
                    # 準備進入追蹤與攀爬階段，回傳對應的目標
                    return self.lambs_select() 

    
class Coordinate:
#儲存座標
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ObjectInfo():  #最大物件的訊息
    # 顏色與編號對照
    def __init__(self, color1,color2, object_type,api_node):
        
        self.api = api_node
        self.color_dict = {
        'Orange': 0, 'Yellow': 1, 'Blue': 2, 'Green': 3,
        'Black': 4, 'Red': 5, 'White': 6
        } 
        self.color1= self.color_dict[color1]
        self.color1_str = color1
        self.color2 = self.color_dict[color2]
        self.color2_str = color2

        self.edge_max = Coordinate(0, 0)
        self.edge_min = Coordinate(0, 0)
        self.center = Coordinate(0, 0)
        self.get_target = False
        self.target_size = 0

        # 設定物件尋找策略
        update_strategy = {
            'Board': self.get_object,
            'Ladder': self.get_object,
            'Target': self.get_object,
        }
        self.find_object = update_strategy[object_type]

    def get_object(self):
        """獲取最大面積物件的編號"""

        color_type = [self.color1,self.color2]

        for color in color_type :
            
            count = self.api.color_counts[color]

            if count == 0:
                return None
            else:
                for i in range(1,count+1):
                    max_size = 0
                    max_idx = None
                    size = self.api.object_sizes[color][i]

                    if size > max_size:
                        max_size = size
                        max_idx = i
                        self.max_size_color = color
            
            # 門檻判斷 (面積 > 100 像素才視為目標)
            return max_idx if max_size > 100 else None

    def update(self):
        """更新目標物的所有詳細資訊"""
        object_idx = self.find_object() #max_idx

        if object_idx is not None:
            self.get_target = True
            # 更新邊界座標 (API: object_x_min/max, object_y_min/max)
            self.edge_max.x = self.api.object_x_max[self.max_size_color][object_idx]
            self.edge_min.x = self.api.object_x_min[self.max_size_color][object_idx]
            self.edge_max.y = self.api.object_y_max[self.max_size_color][object_idx]
            self.edge_min.y = self.api.object_y_min[self.max_size_color][object_idx]
            
            # 更新中心點 (API: get_object_cx/cy)
            self.center.x = self.api.get_object_cx(self.max_size_color, object_idx)
            self.center.y = self.api.get_object_cy(self.max_size_color, object_idx)
            
            # 更新面積 (API: object_sizes)
            self.target_size = self.api.object_sizes[self.max_size_color][object_idx]
        else:
            self.get_target = False

def main(args=None):
    rclpy.init(args=args)
    node = WallClimbing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


