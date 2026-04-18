#!/usr/bin/env python3
#coding=utf-8
from collections import Counter, deque
import rospy
import numpy as np
import sys 
# sys.path.append('/home/iclab/Desktop/kid_hurocup/src/strategy')
from Python_API import Sendmessage
import time
# import cv2
import math

ORIGIN_THETA = 0
ORIGIN_SPEED = 4000
send = Sendmessage()

# 2025.8.5

class Coordinate: # 計算
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __add__(self, other):
        return Coordinate((self.x + other.x), (self.y + other.y))
    def __sub__(self, other):
        return Coordinate((self.x - other.x), (self.y - other.y))
    def __floordiv__(self, other):
        return Coordinate((self.x // other), (self.y // other))

class Mar:
    def __init__(self):
        rospy.init_node('mar', anonymous=True, log_level=rospy.DEBUG)
        self.initial()
        self.stright_head_flag =False
        self.avoid_mode = False
        self.avoid_timer = 0
        self.speed_x = 0
        self.speed_y = 0
        self.theta = 0
        self.seek_line = Seek_line()
        self.arrow_center = Coordinate(0, 0)
        self.arrow_temp = deque(['None', 'None', 'None', 'None', 'None'], maxlen = 5) 
        self.search_done = False   # 控制搜尋只做一次
        self.search_num = 0              # 紀錄目前搜尋步驟
        self.search_flag = 0             # 紀錄搜尋方向
        self.head_horizon = 2048         # 頭部水平初始位置
        self.head_vertical = 2048        # 頭部垂直初始位置
        self.reg = 1                     # 搜尋方向 (正/反順序)
    
    def initial(self):
        self.status = 'First'
        self.can_turn_flag = False                                         
        self.arrow_flag = False
        self.turn_now_flag = False
        self.arrow_cnt_times = 0
        self.yaw_temp = 0       
        self.arrow_to_line_flag = False                           
        self.line_status = 'online'
        send.sendHeadMotor(2, 2048, 50)
        send.sendHeadMotor(1, 2048, 50)
        send.sendSensorReset(1, 1, 1)


    def view_search(self, right_place, left_place, up_place, down_place, speed, delay):   
        if self.reg > 0:
            turn_order = [3, 4, 1, 2]   # 右→上→左→下
        else:
            turn_order = [1, 4, 3, 2]   # 左→上→右→下

        if self.search_num >= len(turn_order):
            self.search_num = 0

        self.search_flag = turn_order[self.search_num]
        rospy.loginfo("搜尋箭頭")

        # === 檢查是否已經找到箭頭 ===
        if self.arrow_center.x != 0 and self.arrow_center.y != 0:
            rospy.loginfo("找到箭頭，切換到追蹤模式1")
            self.search_done = True
            self.lock_arrow()
            return  

        # === 沒找到 → 繼續掃描 ===
        if self.search_flag == 1:   # 左尋
            if self.head_horizon >= left_place:
                self.move_head(1, self.head_horizon, 880, 880, speed)
                self.head_horizon -= speed
                time.sleep(delay)
            else:
                self.search_num += 1

        elif self.search_flag == 4: # 上尋
            if self.head_vertical <= up_place:
                self.move_head(2, self.head_vertical, 880, 880, speed)
                self.head_vertical += speed
                time.sleep(delay)
            else:
                self.search_num += 1  

        elif self.search_flag == 3: # 右尋
            if self.head_horizon <= right_place:
                self.move_head(1, self.head_horizon, 880, 880, speed)
                self.head_horizon += speed
                time.sleep(delay) 
            else:
                self.search_num += 1      

        elif self.search_flag == 2: # 下尋
            if self.head_vertical >= down_place:
                self.move_head(2, self.head_vertical, 880, 880, speed)
                self.head_vertical -= speed
                time.sleep(delay)   
            else:
                self.search_num = 0
        
        if self.arrow_yolo():
            self.search_done = True
            rospy.loginfo("箭頭搜尋完成，進入追蹤模式2")

    def move_head(self, ID, Position, head_max_x, head_max_y, Speed):
        send.sendHeadMotor(ID, Position, Speed)
        if ID == 1:
            self.head_horizon = Position
            if abs(self.head_horizon - 2048) > head_max_x:
                self.head_horizon = 2048 + head_max_x if (self.head_horizon - 2048) > 0 else 2048 - head_max_x
        else:
            self.head_vertical = Position
            if abs(self.head_vertical - 2048) > head_max_y:
                self.head_vertical = 2048 + head_max_y if (self.head_vertical - 2048) > 0 else 2048 - head_max_y
        rospy.loginfo("head_horizon={self.head_horizon},head_vertical{self.head_vertical}")
        

    def lock_arrow(self, speed=10):
            """ 讓頭部追蹤箭頭中心 (整合 trace_revise)，箭頭越近頭越低 """
            self.arrow_temp.append(send.yolo_Label)
            if self.arrow_center.x != 0 and self.arrow_center.y != 0:
                # === 基本參數 ===
                img_center_x, img_center_y = 160, 120  # 影像中心
                x_diff = self.arrow_center.x - img_center_x
                y_diff = self.arrow_center.y - img_center_y

                # === 誤差轉角度 ===
                x_degree = x_diff * (38.0 / 320.0)   # 水平視角 ±65°
                y_degree = y_diff * (55.0 / 240.0)   # 垂直視角 ±38°

                # === 轉換為馬達刻度 ===
                horizon_target = 2048 - round(x_degree * 4096 / 360 )
                vertical_target = 2048 - round(y_degree * 4096 / 360 )

                # === 發送命令 ===
                send.sendHeadMotor(1, horizon_target, speed)
                send.sendHeadMotor(2, vertical_target, speed)
                if vertical_target <= 2000:
                    self.stright_head_flag =True
            
                rospy.logdebug(
                    f"[鎖定箭頭] X:{self.arrow_center.x}, Y:{self.arrow_center.y}, "
                    f"hor:{horizon_target}, ver:{vertical_target}"
                )
            else:
                rospy.logdebug("miss_target -> 需重新尋找箭頭")


    def arrow_yolo(self): # 檢測箭頭透過yolo_Label判斷出箭頭方向
        self.arrow_center.x, self.arrow_center.y = 0, 0
        self.arrow_temp.append(send.yolo_Label)
        rospy.loginfo(f" arrow list: {self.arrow_temp}")
    
        arrow_cnt_temp = len(set(self.arrow_temp)) # 用set去除重複，保證5項全部都一樣
        if arrow_cnt_temp == 1 and self.arrow_temp[0] != 'None':
            self.can_turn_flag = False
            if self.arrow_temp[0] == 'left' or self.arrow_temp[0] == 'right':
                rospy.logwarn("!!!!!!! set can_tuen_flag")
                self.can_turn_flag = True
            self.arrow_center.y = send.yolo_Y
            self.arrow_center.x = send.yolo_X
            send.drawImageFunction(1, 1, send.yolo_XMin, send.yolo_XMax, send.yolo_YMin, send.yolo_YMax, 255, 0, 0)
            return True
        return False


    def arrow_turn(self): # 遇到箭頭轉彎
        self.yaw = send.imu_value_Yaw
        print("yaw = ", send.imu_value_Yaw)
        # self.yaw_calculate()
        if self.arrow_temp[0] == 'right':
            rospy.logdebug(f'箭頭：右轉')
            send.sendContinuousValue(2000, 0, 0, -8 + ORIGIN_THETA, 0)
        elif self.arrow_temp[0] == 'left':
            rospy.logdebug(f'箭頭：左轉')
            send.sendContinuousValue(2000, 0, 0, 8 + ORIGIN_THETA, 0)
        if  abs(self.yaw) > 85: # 成功轉90度
            send.sendSensorReset(0, 0, 1)    # imu Reset
            rospy.logdebug(f'箭頭轉彎結束')
            # self.yaw_temp = self.yaw
            self.turn_now_flag = False # 轉彎完將旗標關閉
            self.can_turn_flag = False
        #elif self.arrow_temp[0] == 'stright':
            
    def stright_avoid(self):
    # ========== 避開模式 ==========
        self.yaw = send.imu_value_Yaw
        rospy.loginfo("開始避開箭頭")
        rospy.logdebug(f"self.yaw = {self.yaw}")
        self.speed_x = 2000 
        if  self.yaw > 0 : 
            if abs(self.yaw) < 50:
                self.theta = -8
                send.sendContinuousValue(self.speed_x, -500, 0, ORIGIN_THETA + self.theta, 0)
                rospy.logdebug('➡ 右避開中')
            else:
                send.sendSensorReset(0, 0, 1) 
                rospy.loginfo("✅ 避開完成，回到基準線")
                
        elif  self.yaw < 0: 
            if abs(self.yaw) < 50:
                self.theta = 8
                send.sendContinuousValue(self.speed_x, 500, 0, ORIGIN_THETA + self.theta, 0)
                rospy.logdebug('➡ 左避開中')
            else:
                send.sendSensorReset(0, 0, 1) 
                rospy.loginfo("✅ 避開完成，回到基準線")
                
        self.stright_head_flag =False
        self.avoid_mode = False
                
    def imu_go(self):  # 直走   yaw 可以知道機器人面向 
        self.yaw = send.imu_value_Yaw
        self.speed_x = 2000  # 保證有值
        self.arrow_temp.append(send.yolo_Label)

        rospy.logdebug(f"x={self.arrow_center.x}, y={self.arrow_center.y}")

        # ========== 沒看到箭頭 ==========
        
        # ========== 修正模式 ==========
        if 0 < self.arrow_center.x <= 100:
            self.theta = 5
            send.sendContinuousValue(self.speed_x, 100, 0, self.theta + ORIGIN_THETA, 0)
        elif self.arrow_center.x >= 220:
            self.theta = -5
            send.sendContinuousValue(self.speed_x, 100, 0, self.theta + ORIGIN_THETA, 0)
        else:
            if self.yaw > 5:
                self.theta = -3 + ORIGIN_THETA
                rospy.logdebug('imu修正：右轉')
            elif self.yaw < -5:
                self.theta = 3 + ORIGIN_THETA
                rospy.logdebug('imu修正：左轉')
    
        send.sendContinuousValue(self.speed_x, 100, 0, self.theta, 0)
        


    ###############  main  ###############
    def main(self):
        if send.is_start: # 確認機器人啟動
            # while True:
            #     print(send.imu_value_Yaw)
            print(self.status)
            
            if self.status == 'First': #初始化
                self.initial()
                time.sleep(1)
                send.sendBodyAuto(0, 0, 0, 0, 1, 0) # mode = 1為continue步態
                self.status = 'Arrow_Part'  if send.DIOValue == 48 else 'Arrow_Part'
                send.sendHeadMotor(2, 2048, 50) # 馬達編號1為水平，2為垂直調整   位置為馬達目標刻度，2048為正朝前方
                self.search_done = False

        

            elif self.status == 'Arrow_Part':
                rospy.loginfo(f"{self.avoid_mode}")
                rospy.loginfo(f"{self.stright_head_flag}")
                
                if not self.search_done:
                    self.view_search(2700, 1400, 2500, 1500, 20, 0.1)
                else :
                    self.lock_arrow()
                    if self.arrow_temp[0] == 'stright': 
                        self.avoid_mode = True  

                    if self.turn_now_flag:
                        self.arrow_turn()
                    elif self.avoid_mode and self.stright_head_flag :
                        self.stright_avoid()
                    else: 
                        self.arrow_yolo()
                        
                        if self.can_turn_flag:

                            
                            rospy.loginfo('can turn !!!')
                            rospy.loginfo(f"{self.arrow_center.y}")
                            if self.arrow_center.y >= 140:#小白90
                                self.arrow_cnt_times += 1
                            if self.arrow_cnt_times >= 10:
                                self.turn_now_flag = True
                                self.arrow_cnt_times = 0
                    
                        
                        self.imu_go()    
                            
        else:
            if self.status != 'First':
                # self.initial()
                self.status = 'First'
                send.sendBodyAuto(0, 0, 0, 0, 1, 0) #關閉



class Seek_line:
    def __init__(self):
        self.upper_center = Coordinate(0, 0)
        self.lower_center = Coordinate(0, 0)
    #功能:對我們的陣列做補0
    def cvt_list2d2numpy(self, list2d):   #將一個2D的Python列表（list2d）轉換為NumPy的2D數組
        max_len = max([len(sub_lst) for sub_lst in list2d])
        np_array = np.vstack([np.pad(np.array(lst), (0, (max_len - len(lst)))) for lst in list2d])
        return np_array

    def update(self):
        img_size = self.cvt_list2d2numpy(send.color_mask_subject_size)
        img_xmin = self.cvt_list2d2numpy(send.color_mask_subject_XMin)
        img_xmax = self.cvt_list2d2numpy(send.color_mask_subject_XMax)
        img_ymin = self.cvt_list2d2numpy(send.color_mask_subject_YMin)
        img_ymax = self.cvt_list2d2numpy(send.color_mask_subject_YMax)
        # img_size = np.array(send.color_mask_subject_size)
        # img_xmin = np.array(send.color_mask_subject_XMin)
        # img_xmax = np.array(send.color_mask_subject_XMax)
        # img_ymin = np.array(send.color_mask_subject_YMin)
        # img_ymax = np.array(send.color_mask_subject_YMax)
        filter_img_size = img_size > 380   #濾雜訊
        has_object = filter_img_size.any() #filter_img_size 輸出是布林值
        send.data_check = False
        if not has_object:   #確認沒有看到物體
            rospy.logdebug(f'no object')
            self.upper_center.x, self.upper_center.y = 0, 0
            self.lower_center.x, self.lower_center.y = 0, 0
            return
        
        #print(img_ymin[filter_img_size]) 
        img_xmin_new = int(img_xmin[filter_img_size].min()) 
        img_xmax_new = int(img_xmax[filter_img_size].max()) 
        img_ymin_new = int(img_ymin[filter_img_size].min())
        img_ymax_new = int(img_ymax[filter_img_size].max())
        send.drawImageFunction(7, 1, img_xmin_new, img_xmax_new, img_ymin_new, img_ymax_new, 255, 0, 255)
        #影像輸出為一維陣列，8bits(0~255)
        img_data = np.frombuffer(send.Label_Model, dtype = np.uint8)
        img_data = img_data.reshape(240, 320)
        img_data = img_data[img_ymin_new : img_ymax_new, img_xmin_new : img_xmax_new] #
        
        y_coord, x_coord = np.where(img_data != 0) #抓取不等於 0 的

        if len(x_coord) == 0:
            self.upper_center.x, self.upper_center.y = 0, 0
            self.lower_center.x, self.lower_center.y = 0, 0
            return
        
        middle_y = (np.max(y_coord) + np.min(y_coord)) // 2 #中心點
        upper_filter = y_coord <= middle_y
        upper_x, upper_y = x_coord[upper_filter], y_coord[upper_filter]
        self.upper_center.x = np.mean(upper_x) + img_xmin_new #因為抓色模區間會重新定義原點 + img_xmin_new 可以知道其真實位置
        self.upper_center.y = np.mean(upper_y) + img_ymin_new
        # upper_xmin = upper_x.min() + img_xmin_new
        # upper_xmax = upper_x.max() + img_xmin_new
        # upper_ymin = upper_y.min() + img_ymin_new
        # upper_ymax = upper_y.max() + img_ymin_new
        # send.drawImageFunction(5, 1, upper_xmin, upper_xmax, upper_ymin, upper_ymax, 255, 0, 0)
        lower_filter = y_coord > middle_y
        lower_x, lower_y = x_coord[lower_filter], y_coord[lower_filter]
        self.lower_center.x = np.mean(lower_x) + img_xmin_new
        self.lower_center.y = np.mean(lower_y) + img_ymin_new
        # lower_xmin = lower_x.min() + img_xmin_new
        # lower_xmax = lower_x.max() + img_xmin_new
        # lower_ymin = lower_y.min() + img_ymin_new
        # lower_ymax = lower_y.max() + img_ymin_new
        # send.drawImageFunction(6, 1, lower_xmin, lower_xmax, lower_ymin, lower_ymax, 0, 255, 0)
        send.drawImageFunction(2, 0, int(self.upper_center.x), int(self.lower_center.x), int(self.upper_center.y), int(self.lower_center.y), 0, 0, 0)
        #send.drawImageFunction(編號, 型態, xmin, xmax, ymin, ymax, r, g, b)
    def calculate_slope(self):
        delta = self.upper_center - self.lower_center
        if delta.x == 0:
            return float("inf")
        return delta.y / delta.x

if __name__ == '__main__':
    try:
        mar = Mar()
        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            mar.main()
            r.sleep()
    except rospy.ROSInterruptException: #捕獲 Ctrl+C 的中斷信號
        pass
