#!/usr/bin/env python3
#coding=utf-8
import sys
import rclpy 
from strategy.API import API
import numpy as np
import math
import time


#--校正量--#
FORWARD_CORRECTION         = -300
TRANSLATION_CORRECTION     = -100
THETA_CORRECTION           = 0

#基礎變化量(前進&平移)
BASE_CHANGE                = 500
HEAD_CHANGE_H              = 50    
HEAD_CHANGE_V              = 50

# ---微調站姿開關---#
STAND_CORRECT_CW           = False
DRAW_FUNCTION_FLAG         = True
TARGET_COLOAR_1            = 'Red'
TARGET_COLOAR_2            = 'Blue'


#------------------#
HEAD_HORIZONTAL            = 2040
HEAD_VERTICAL              = 2040

HEAD_LEFT_HAND_H = 2300
HEAD_LEFT_HAND_V = 2450
HEAD_RIGHT_HAND_H = 2300
HEAD_RIGHT_HAND_V = 2450
HEAD_LEFT_LEG_H = 2300
HEAD_LEFT_LEG_V = 2450
HEAD_RIGHT_LEG_H = 2300
HEAD_RIGHT_LEG_V = 2450

HEAD_LOOK_LINE = 1600

MOTOR_LEFT_HAND_Y = 2000
MOTOR_LEFT_HAND_X = 2000
MOTOR_RIGHT_HAND_Y = 2000
MOTOR_RIGHT_HAND_X = 2000
MOTOR_LEFT_LEG_Y = 2000
MOTOR_LEFT_LEG_X = 2000
MOTOR_RIGHT_LEG_Y = 2000
MOTOR_RIGHT_LEG_X = 2000

#判斷值
MY_LINE_Y= 180
MY_LINE_X =160
ROI_RADIUS = 120

#前後值（只用在「接近」）
BACK_MIN                   = -500
FORWARD_MIN               = 500
FORWARD_LOW               = 1000
FORWARD_NORMAL             = 1500
FORWARD_HIGH                = 2000

#平移值（只做平移，不旋轉）
TRANSLATION_BIG            = 800
TRANSLATION_NORMAL         = 400

# ✅ 用 size 判斷何時停
APPROACH_SIZE_STOP = 3500    # 到這個 size 就停（上機要調）
APPROACH_SIZE_SLOW = 2000    # 超過這個 size 開始慢慢走（上機要調）
APPROACH_FWD_FAST  = 1800
APPROACH_FWD_SLOW  = 700

# 平移對齊參數
KP_X = 1.0                   # 平移P
TRANS_LIMIT = 500            # 平移限幅
X_DEADBAND = 6               # 低於這個誤差不平移
ALIGN_STABLE_N = 3           # 連續幾次對齊才算OK


class WallClimbing(API):
    def __init__(self):
        super().__init__('wall_climbing_node')
        self.target = ObjectInfo(TARGET_COLOAR_1,TARGET_COLOAR_2,'Ladder',self)
        self.init()
        self.timer = self.create_timer(0.05, self.strategy_loop)
        self.current_strategy = "Wall_Climb_on"
        self.get_logger().info("Wall Climbing Node Initialized")

    def strategy_loop(self):
        strategy = self.current_strategy
        if DRAW_FUNCTION_FLAG:
            self.draw_function()

        sys.stdout.write("\033[J")
        sys.stdout.write("\033[H")

        self.get_logger().info('________________________________________\033[K')
        self.get_logger().info(f'x: {self.now_forward} ,y: {self.now_translation} ,theta: {self.now_theta}\033[K')
        self.get_logger().info(f'Goal_x: {self.forward} ,Goal_y: {self.translation} ,Goal_theta: {self.theta}\033[K')
        self.get_logger().info(f"機器人狀態: {self.state}\033[K")
        self.get_logger().info(f"head_h : {int(self.now_head_Horizontal)},head_v: {int(self.now_head_Vertical)}\033[K")
        self.get_logger().info(f"target_1_size : {int(getattr(self,'target_1_size',0))}\033[K")
        self.get_logger().info('￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣')
        
        if strategy == "Wall_Climb_off":
            if (not self.walkinggait_stop and self.climbing):
                self.get_logger().info("🔊CW parameter reset\033[K")
                self.init()
                self.sendbodyAuto(0)
                self.sendSensorReset()
                time.sleep(2)
                self.sendBodySector(29)
                time.sleep(1.5)
                self.get_logger().info("reset\033[K")
                self.imu_reset = True

            self.sendHeadMotor(1,HEAD_HORIZONTAL,100)
            self.sendHeadMotor(2,HEAD_VERTICAL,100)
            self.init()
            self.get_logger().info("turn off\033[K")
            self.walkinggait_stop = True

        elif strategy == "Wall_Climb_on":
            if self.state != 'cw_finish':
                if self.imu_reset:
                    self.sendHeadMotor(1,HEAD_HORIZONTAL, 100)
                    self.sendHeadMotor(2,HEAD_VERTICAL, 100)
                    if STAND_CORRECT_CW:
                        self.sendBodySector(102)
                        time.sleep(1)
                    self.sendbodyAuto(0)
                    self.imu_reset = False
                
                # ====== 走到牆前：用 size 判斷停下 ======
                if not self.readyclimb:
                    self.sendHeadMotor(2,HEAD_LOOK_LINE,20)

                    self.find_ladder()
                    motion = self.new_edge_judge_size_only_stop_then_translate()

                    self.sendbodyAuto(1)
                    self.walkinggait(motion)

                # ====== 停下並平移對齊完成後，才進攀爬 ======
                else:
                    if (not self.climbing):
                        self.action, self.value = self.lambs_select()

                        # 找不到點就掃描
                        if self.action == 'searching' or self.value == 'no_object' or self.value is None:
                            self.get_logger().info(
                                f"找不到點，持續掃描中... step={getattr(self,'climb_step',0)} scan={getattr(self,'scan_count',0)}\033[K"
                            )
                            return
                        
                        self.climbing = True
                        self.climbmode(self.action, self.value)
                        time.sleep(0.5)
                        self.climbing = False

    def init(self):
        self.state                 = '停止'
        self.walkinggait_stop      = True  
        
        self.now_head_Horizontal   = HEAD_HORIZONTAL
        self.now_head_Vertical     = HEAD_VERTICAL
        
        self.imu_reset             = True

        self.target_1_y = 0
        self.target_1_ym = 0
        self.target_2_y = 0
        self.target_2_ym = 0

        self.movetotal = 0

        self.forward               = 0
        self.translation           = 0
        self.theta                 = 0  # ✅永遠不用旋轉
        self.now_forward           = 0 
        self.now_translation       = 0
        self.now_theta             = 0  # ✅永遠不用旋轉

        self.lost_target_count     = 0

        self.head = True
        self.action = None
        self.value = None

        self.readyclimb=False 
        self.climbing=False

        self.object_x = 0
        self.object_y = 0
        self.object_y1_len = 0
        self.object_y2_len = 0
        self.target_1_cx = 160
        self.target_2_cx = 160

        self.best_candidate = None
        self.max_score = -1

        self.target_1_size = 0  # ✅用 size 判斷停下
        self.align_ready_count = 0  # ✅平移對齊穩定度計數

    def find_ladder(self):
        sys.stdout.write("\033[K")

        color_1 = self.target.color1
        color_2 = self.target.color2

        target_colors = [color_1, color_2]
        candidates_y_max = []

        for color in target_colors:
            obj_count = self.color_counts[color]
            self.get_logger().info(f"color {color} subject cnts: {obj_count}\033[K")
            actual_list_len = len(self.object_sizes[color])

            # ⚠️ 你的 API 陣列多半是 [0..count] 這種，所以保留 +1 但要保護 i 範圍
            for i in range(1, min(obj_count, actual_list_len-1) + 1):
                if self.object_sizes[color][i] > 100:
                    cx = (self.object_x_max[color][i] + self.object_x_min[color][i]) // 2
                    candidates_y_max.append({
                        'y_max': self.object_y_max[color][i],
                        'color': color,
                        'cx': cx,
                        'idx': i,
                        'size': self.object_sizes[color][i],  # ✅記住 size
                    })

        candidates_y_max.sort(key=lambda x: x['y_max'], reverse=True) 

        if len(candidates_y_max) >= 1:
            c1 = candidates_y_max[0]
            self.target_1_y  = self.object_y_min[c1['color']][c1['idx']]
            self.target_1_ym = self.object_y_max[c1['color']][c1['idx']]
            self.target_1_cx = c1['cx']
            self.target_1_size = c1['size']  # ✅主要 size 參考

            if len(candidates_y_max) >= 2:
                c2 = candidates_y_max[1]
                if abs(c1['cx'] - c2['cx']) > 30:
                    self.target_2_y  = self.object_y_min[c2['color']][c2['idx']]
                    self.target_2_ym = self.object_y_max[c2['color']][c2['idx']]
                    self.target_2_cx = c2['cx']
                else:
                    if len(candidates_y_max) >= 3:
                        c2 = candidates_y_max[2]
                        self.target_2_y  = self.object_y_min[c2['color']][c2['idx']]
                        self.target_2_ym = self.object_y_max[c2['color']][c2['idx']]
                        self.target_2_cx = c2['cx']
                    else:
                        self.target_2_cx = self.target_1_cx

                # ✅ 你真正用來「對齊牆中心」的是兩柱中點，所以用 mid_cx
                self.object_x = (self.target_1_cx + self.target_2_cx) // 2
                self.object_y2_len = self.target_2_ym - self.target_2_y
            else:
                self.object_x = self.target_1_cx
                self.object_y2_len = 0

            self.object_y = self.target_1_y
            self.object_y1_len = self.target_1_ym - self.target_1_y

            self.get_logger().info(f"鎖定目標 - Y1:{self.target_1_y}, mid_cx:{self.object_x}, size:{self.target_1_size}")
        else:
            self.get_logger().info("丟失目標...\033[K")
            self.object_x = 0
            self.object_y = 0
            self.target_1_size = 0

    def new_edge_judge_size_only_stop_then_translate(self):
        """
        ✅ 不旋轉（theta 永遠 0）
        ✅ 用 size 判斷何時停下來
        ✅ 停下後才做平移對齊
        """
        # 沒目標：慢慢走找
        if (self.object_y == 0 and self.object_x == 0):
            self.lost_target_count += 1
            self.theta = 0.0
            self.translation = 0.0
            # 丟失就慢慢前進找
            self.forward = 800
            self.state = f"目標丟失：慢走找({self.lost_target_count})"
            return "walking"
        else:
            self.lost_target_count = 0

        size_now = int(getattr(self, 'target_1_size', 0))

        # ---- 1) 接近：用 size 控制前進，先不平移 ----
        if size_now < APPROACH_SIZE_STOP:
            self.theta = 0.0
            self.translation = 0.0

            if size_now >= APPROACH_SIZE_SLOW:
                self.forward = APPROACH_FWD_SLOW
                self.state = f"接近中(慢) size={size_now}"
            else:
                self.forward = APPROACH_FWD_FAST
                self.state = f"接近中(快) size={size_now}"

            self.align_ready_count = 0
            return "walking"

        # ---- 2) 停下：size 達標，forward=0 ----
        self.forward = 0.0
        self.theta = 0.0  # ✅ 永遠不旋轉

        # ---- 3) 停下後平移對齊 ----
        error_x = MY_LINE_X - self.object_x

        if abs(error_x) <= X_DEADBAND:
            self.translation = 0.0
            x_ok = True
        else:
            self.translation = max(min(error_x * KP_X, TRANS_LIMIT), -TRANS_LIMIT)
            x_ok = False

        # 穩定判定
        if x_ok:
            self.align_ready_count += 1
        else:
            self.align_ready_count = 0

        self.state = f"已停下：平移對齊中 size={size_now} err_x={error_x} stable={self.align_ready_count}/{ALIGN_STABLE_N}"

        if self.align_ready_count >= ALIGN_STABLE_N:
            self.sendContinuousValue(0, 0, 0)
            self.state = "停下+平移對齊完成：準備攀爬"
            return "ready_to_cw"

        return "walking"

    def walkinggait(self,motion):
        if motion == 'ready_to_cw':
            self.get_logger().info("停下+平移對齊完成：準備攀爬\033[K")
            self.forward = 0.0
            self.translation = 0.0
            self.theta = 0.0

            self.sendContinuousValue(0,0,0)
            self.sendbodyAuto(0)
            time.sleep(0.5)
            self.sendSensorReset(True)
            self.sendBodySector(29)
            time.sleep(2)

            self.status = 'ready_finish'
            self.readyclimb = True  
            return

        # ✅ walking：只送 forward / translation，theta 永遠 0
        self.now_forward      = self.ramp_speed(self.now_forward, self.forward, BASE_CHANGE)
        self.now_translation  = self.ramp_speed(self.now_translation, self.translation, BASE_CHANGE)
        self.now_theta        = 0.0

        f = max(min(int(self.now_forward), 2001), -2001)
        t = max(min(int(self.now_translation), 501), -501)
        r = 0

        self.sendContinuousValue(f, t, r)

    def ramp_speed(self, current, target, step):
        if abs(current - target) < step:
            return target
        if current < target:
            return current + step
        elif current > target:
            return current - step
        return target

    def climbmode(self, action, target_data):
        if target_data == 'no_object' or target_data is None:
            self.get_logger().info("壞了...")
            self.sendbodyAuto(0)
            return
        
        cx, cy = target_data['center']
        self.object_error_x = cx - 160
        self.object_error_y = cy - 120

        config = {
            'left_hand': {
                'ids': [1, 2], 'base_m': [MOTOR_LEFT_HAND_Y, MOTOR_LEFT_HAND_X],
                'weight_sector': 871, 'climb_sector': 872
            },
            'right_hand': {
                'ids': [5, 6], 'base_m': [MOTOR_RIGHT_HAND_Y, MOTOR_RIGHT_HAND_X],
                'weight_sector': 873, 'climb_sector': 874
            },
            'left_leg': {
                'ids': [12, 10], 'base_m': [MOTOR_LEFT_LEG_Y, MOTOR_LEFT_LEG_X],
                'weight_sector': 875, 'climb_sector': 876
            },
            'right_leg': {
                'ids': [18, 16], 'base_m': [MOTOR_RIGHT_LEG_Y, MOTOR_RIGHT_LEG_X],
                'weight_sector': 877, 'climb_sector': 878
            }
        }

        if action not in config:
            self.get_logger().error(f"蛤—這是甚摸: {action}")
            return

        cfg = config[action]

        self.get_logger().info(f"正在調整重心以移動 {action}...")
        self.sendBodySector(cfg['weight_sector'])
        time.sleep(1.0)

        # 注意：這裡你原本就這樣寫，我先不大改你的邏輯
        self.now_head_Vertical = cfg['base_m'][1]
        self.now_head_Horizontal = cfg['base_m'][0]
        
        self.keep_head(target_data['center'])
        KP_X2, KP_Y2 = 1.0, 1.0
        
        motor_val_y = int(cfg['base_m'][0] - (self.object_error_y * KP_Y2))
        motor_val_x = int(cfg['base_m'][1] - (self.object_error_x * KP_X2))

        self.sendBodySector(cfg['climb_sector'])
        time.sleep(0.5)
        
        self.sendSingleMotor(cfg['ids'][0], motor_val_y, 20)
        self.sendSingleMotor(cfg['ids'][1], motor_val_x, 20)
        
        self.get_logger().info(f"{action} 動作執行完畢: M1={motor_val_y}, M2={motor_val_x}")

    def keep_head(self,coordinate):
        cx,cy = coordinate
        self.object_error_x = cx - 160
        self.object_error_y = cy - 120

        KP_H = 1
        KP_V = 1

        new_head_h = self.now_head_Horizontal - (self.object_error_x * KP_H)
        new_head_v = self.now_head_Vertical   - (self.object_error_y * KP_V)

        self.sendHeadMotor(1, int(new_head_h), 40)
        self.sendHeadMotor(2, int(new_head_v), 40)

        self.now_head_Horizontal = new_head_h
        self.now_head_Vertical   = new_head_v

    def imu_angle(self):
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
        self.drawImageFunction(1,1,0,320,MY_LINE_Y,MY_LINE_Y,255,255,0)
        self.drawImageFunction(2,1,MY_LINE_X,MY_LINE_X,0,240,255,255,0)
        
        edge_x = 160-ROI_RADIUS
        edge_y = 120-ROI_RADIUS
        xmin = edge_x
        xmax = ROI_RADIUS*2 + edge_x
        ymin = edge_y
        ymax = ROI_RADIUS*2 + edge_y

        self.ROI_cx = ROI_RADIUS + edge_x
        self.ROI_cy = ROI_RADIUS + edge_y
       
        self.drawImageFunction(3,1,xmin,xmax,ymin,ymax,0,255,255)
        self.drawImageFunction(4,1,self.ROI_cx,self.ROI_cx,self.ROI_cy,self.ROI_cy,0,255,255)

    def get_best_climbing_target(self):
        self.best_candidate = None
        self.max_score = -1
        color_1 = self.target.color1
        color_2 = self.target.color2
        target_colors = [color_1,color_2]
        
        point_r = 10 

        for color in target_colors:
            cnts = self.color_counts[color]
            for i in range(1, min(cnts, len(self.object_sizes[color])-1) + 1):
                size = self.object_sizes[color][i]
                if size < 100: 
                    continue
                
                cx =(self.object_x_max[color][i] + self.object_x_min[color][i])//2
                cy =(self.object_y_max[color][i] + self.object_y_min[color][i])//2
                if cx is None or cy is None:
                    continue

                xmax = self.object_x_max[color][i]
                xmin = self.object_x_min[color][i]
                ymax = self.object_y_max[color][i]
                ymin = self.object_y_min[color][i]

                if xmin <= 0 or xmax >= 320 or ymin <= 0 or ymax >= 240:
                    continue

                dist = math.sqrt((cx - self.ROI_cx)**2 + (cy - self.ROI_cy)**2)
                if (dist + point_r) > ROI_RADIUS:
                    continue

                edge_score = int((1-(dist // ROI_RADIUS)) * 40)
                height_score = int((240 - cy) * 5)
                alignment_score = int((1 - abs(cx - self.ROI_cx) / ROI_RADIUS) * 55)
                total_score = edge_score + height_score + alignment_score

                if total_score > self.max_score:
                    self.max_score = total_score
                    self.best_candidate = {
                        'center': (cx, cy),
                        'score': total_score,
                        'size': size,
                        'bbox': (xmin, ymin, xmax, ymax)
                    }

        return self.best_candidate if self.best_candidate is not None else 'no_object'

    def lambs_select(self):
        if not hasattr(self, 'climb_step'): self.climb_step = 1
        if not hasattr(self, 'last_limb'): self.last_limb = None
        if not hasattr(self, 'scan_count'): self.scan_count = 0

        if self.climb_step == 1:
            now_limb = 'left_hand'
            h_pos, v_pos = HEAD_LEFT_HAND_H, HEAD_LEFT_HAND_V
        elif self.climb_step == 2:
            now_limb = 'right_hand'
            h_pos, v_pos = HEAD_RIGHT_HAND_H, HEAD_RIGHT_HAND_V
        elif self.climb_step == 3:
            now_limb = 'left_leg'
            h_pos, v_pos = HEAD_LEFT_LEG_H, HEAD_LEFT_LEG_V
        elif self.climb_step == 4:
            now_limb = 'right_leg'
            h_pos, v_pos = HEAD_RIGHT_LEG_H, HEAD_RIGHT_LEG_V
        else:
            now_limb = 'any'

        if now_limb != 'any':
            self.sendHeadMotor(1, int(h_pos), 40)
            current_v = int(v_pos - (self.scan_count * HEAD_CHANGE_V))
            self.sendHeadMotor(2, current_v, 40)
            
            target_info = self.get_best_climbing_target()
            if target_info == 'no_object':
                self.scan_count += 1
                if current_v <= 2100:
                    self.scan_count = 0 
                return ('searching', 'no_object')
            else:
                self.climb_step += 1
                self.last_limb = now_limb
                self.scan_count = 0 
                return (now_limb,target_info)

        # any: 找最高分（保留你原本概念）
        best_choice = None
        max_score = -1
        
        limbs = {
            'left_hand':  (HEAD_LEFT_HAND_H, HEAD_LEFT_HAND_V),
            'right_hand': (HEAD_RIGHT_HAND_H, HEAD_RIGHT_HAND_V),
            'left_leg':   (HEAD_LEFT_LEG_H, HEAD_LEFT_LEG_V),
            'right_leg':  (HEAD_RIGHT_LEG_H, HEAD_RIGHT_LEG_V)
        }
        
        for limb, pos in limbs.items():
            if limb == self.last_limb: 
                continue
            self.sendHeadMotor(1, int(pos[0]), 40)
            self.sendHeadMotor(2, int(pos[1]), 40)
            
            val = self.get_best_climbing_target()
            score = val['score'] if val != 'no_object' else -1
            
            if score > max_score:
                max_score = score
                best_choice = (limb, val)
        
        if best_choice and max_score > 0:
            self.last_limb = best_choice[0]
            self.climb_step += 1
            return best_choice
        
        return ('none', 'no_object')


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ObjectInfo():
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

        update_strategy = {
            'Board': self.get_object,
            'Ladder': self.get_object,
            'Target': self.get_object,
        }
        self.find_object = update_strategy[object_type]

    def get_object(self):
        color_type = [self.color1,self.color2]
        max_size = 0
        max_idx = None
        self.max_size_color = self.color1

        for color in color_type:
            count = self.api.color_counts[color]
            if count == 0:
                continue
            for i in range(1, min(count, len(self.api.object_sizes[color])-1) + 1):
                size = self.api.object_sizes[color][i]
                if size > max_size:
                    max_size = size
                    max_idx = i
                    self.max_size_color = color

        return max_idx if max_size > 100 else None

    def update(self):
        object_idx = self.find_object()
        if object_idx is not None:
            self.get_target = True
            self.edge_max.x = self.api.object_x_max[self.max_size_color][object_idx]
            self.edge_min.x = self.api.object_x_min[self.max_size_color][object_idx]
            self.edge_max.y = self.api.object_y_max[self.max_size_color][object_idx]
            self.edge_min.y = self.api.object_y_min[self.max_size_color][object_idx]
            self.center.x = self.api.get_object_cx(self.max_size_color, object_idx)
            self.center.y = self.api.get_object_cy(self.max_size_color, object_idx)
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

