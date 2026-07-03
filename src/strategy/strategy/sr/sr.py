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
import cv2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


STAND_CORRECT_BB = False #站姿微調
CROUCH_BB        = False #平衡姿勢
 
CONTINUOUS_STEP = True

#--校正量--#
FORWARD_CORRECTION         = -600 #前進量校正
TRANSLATION_CORRECTION     = 0 #平移校正
THETA_CORRECTION           = -1 #旋轉校正

TRANSLATION_SPEED          = 1500 #平移速度
# COLOR                      = 6 #*'Orange': 0, 'Yellow': 1, 'Blue': 2, 'Green': 3,
#                                #'Black': 4, 'Red': 5, 'White': 6
FOOTBOARD_LINE             = 14 #腳尖對準基準線

HEAD_HORIZONTAL            = 2048    #頭水平
HEAD_VERTICAL              = 1250    #頭垂直

# PD 控制器參數
FORWARD_SCALE_FACTOR = 150 #X軸倍率
THETA_SCALE_FACTOR   = 1.5 #theta倍率

FORWARD_VISION = False #啟用x視覺計算
THETA_VISION   = False #啟用theta視覺計算
THETA_PD       = False #theta計算用pd還是範圍表(THETA_VISION)

KPX     = 0.5          # X軸(前後) 比例常數
KDX     = 0.1          # X軸(前後) 微分常數

KPTHETA = 0.6      # Theta(旋轉) 比例常數
KDTHETA = 0.2      # Theta(旋轉) 微分常數

FORWARD_MAX = 500  # 前進量最大最小值(無正負)
THETA_MAX   = 3.0  # 旋轉量最大最小值(無正負)

class BalanceBeam(API):
    def __init__(self):
        super().__init__('balance_beam_node')
        self.init()
        self.bridge = CvBridge()
        self.Image_compress_sub = self.create_subscription(
            Image,
            'processed_image',
            self.calculate_step,
            10
        )
        self.timer = self.create_timer(0.05, self.main)
        self.get_logger().info("Balance Beam Node Initialized")
        
        # self.is_start = True

        # self.sendWalkParameter(
        #     mode         = 0,    #walking介面步態方式
        #     com_y_swing  = -3,   #起步步態補償
        #     width_size   = 4.5,  #雙腳距離
        #     period_t     = 330,  #步態頻率
        #     t_dsp        = 0.3,  #雙支撐時間
        #     lift_height  = 4,    #抬腳高度
        #     stand_height = 23.5, #機器人初始站姿高度
        #     com_height   = 27.5  #質心高度
        # )

    def init(self):
        self.walkinggait_stop      = True
        self.first_in              = True  
        self.forward = FORWARD_CORRECTION
        self.translation = TRANSLATION_CORRECTION+TRANSLATION_SPEED
        self.theta = THETA_CORRECTION
        # PD 控制器參數 
        self.kp_x = KPX          # X軸(前後) 比例常數
        self.kd_x = KDX          # X軸(前後) 微分常數
        self.prev_err_x = 0.0

        self.kp_theta = KPTHETA      # Theta(旋轉) 比例常數
        self.kd_theta = KDTHETA      # Theta(旋轉) 微分常數
        self.prev_err_theta = 0.0

        self.slope = 0
        self.degree = 0
        self.Xmin, self.Ymin = 0, 24

        self.found_any_white = False
        
    def main(self):
        if self.is_start:
            time.sleep(1)
            self.sendBodyAutoCmd(0,0,0,0)
            time.sleep(3)
            self.sendbodyAuto(0)
            time.sleep(2)
            self.sendBodyAutoCmd(30000,0,0,1)
            time.sleep(3)
        else:
            if not self.walkinggait_stop:
                self.sendHeadMotor(1, 2048, 30)
                self.sendHeadMotor(2, 2048, 30)
                self.sendbodyAuto(0)
                time.sleep(1.5)
                # self.sendBodySector(29)             #基礎站姿磁區
                # self.loginfo("resetOKOKOK\033[K")                
            self.init()
            self.sendSensorReset(True)
            # self.loginfo("turn off\033[K")
            self.val_print()

    def calculate_step(self, imgmsg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
            cv_image = cv2.resize(cv_image, (320, 240))
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # 縮小畫面 (32x24)
        cv_image_small = cv2.resize(cv_image, (32, 24), interpolation=cv2.INTER_NEAREST)

        self.white_depth_matrix = []
        self.found_any_white = False
        
        self.Xmin, self.Ymin = 0, 24
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0
        first_white = True

        # 掃描邏輯
        for w in range(0, 32):
            found_in_column = False
            self.white_depth_matrix.append(24) # 預設為沒找到

            for h in range(0, 24):
                b, g, r = cv_image_small[h, w]

                # 偵測青綠色 (目標物顏色 BGR: 0, 255, 255)
                if b < 50 and g > 200 and r > 200: 
                    current_y = h # 使用影像座標 h
                    self.found_any_white = True
                    
                    if not found_in_column:
                        self.white_depth_matrix[w] = current_y
                        found_in_column = True
                        
                        # 紀錄掃描到的第一個白色像素點（左邊界）
                        if first_white:
                            self.x1, self.y1 = w, current_y
                            first_white = False
                        
                        # 持續更新右邊界點
                        self.x2, self.y2 = w, current_y
                        
                        # 尋找全域「最遠/最高」的點 (y 最小的點)
                        if current_y < self.Ymin:
                            self.Xmin, self.Ymin = w, current_y
                    
                    # 找到該行第一個像素（上緣）後就跳出 h 迴圈，換下一行 w
                    break

        if (not self.found_any_white) and (FORWARD_VISION or THETA_VISION):#(沒看到任何白色)
            self.forward = 0
            self.theta = 0
        else:
            # --- PD 控制部分 ---
            
            # 1. 前進控制 (Forward/X)
            err_x = (FOOTBOARD_LINE - self.Ymin) * FORWARD_SCALE_FACTOR
            
            p_x = self.kp_x * err_x
            d_x = self.kd_x * (err_x - self.prev_err_x)
            
            # 標準 PD 公式為 P + D
            if FORWARD_VISION:
                self.forward = p_x + d_x
            else:
                self.forward = 0
            self.prev_err_x = err_x

            # 2. 轉向控制 (Theta)
            # 計算斜率：由左至右的趨勢
            dx = (self.x2 - self.x1)
            if dx != 0:
                self.slope = (self.y2 - self.y1) / dx
                self.degree = int(math.degrees(math.atan(self.slope)))
            else:
                self.degree = 0

            # 轉向誤差 (目標角度為 0)
            if THETA_VISION:
                err_theta = self.degree - self.imu_rpy[2]
            else:
                err_theta = -self.imu_rpy[2]
            err_theta = (err_theta + 180) % 360 - 180
            
            p_theta = self.kp_theta * err_theta
            d_theta = self.kd_theta * (err_theta - self.prev_err_theta)
            
            if (not THETA_PD) and (not THETA_VISION):
                self.theta = self.imu_angle()
            else:
                self.theta = (p_theta + d_theta)*THETA_SCALE_FACTOR                
            self.prev_err_theta = err_theta

            #最大最小值限制
            self.forward = max(min(self.forward, FORWARD_MAX), (-FORWARD_MAX)) + FORWARD_CORRECTION
            self.theta = max(min(self.theta, THETA_MAX), (-THETA_MAX)) + THETA_CORRECTION
        
    def val_print(self):
        # sys.stdout.write("\033[H\033[J")
        self.get_logger().info("\033[H\033[J")

        self.get_logger().info(f"\
#==========================================#\n\
ContinuousValue : {self.forward} {self.translation} {self.theta}\n\
imu             : {self.imu_rpy[0]} {self.imu_rpy[1]} {self.imu_rpy[2]}\n\
Slope           : {self.slope}\n\
Degree          : {self.degree}\n\
Xmin            : {self.Xmin}\n\
Ymin            : {self.Ymin}\n\
WhiteYN         : {self.found_any_white}\n\
KPX KDX PEX     : {self.kp_x} {self.kd_x} {self.prev_err_x}\n\
KPT KDT PET     : {self.kp_theta} {self.kd_theta} {self.prev_err_theta}\n\
        ")
        self.drawImageFunction(1, 1, 0, 320, FOOTBOARD_LINE*10, FOOTBOARD_LINE*10, 255, 0, 0, 2)


    def imu_angle(self):      #一般 imu修正角度
        imu_ranges = [  (180,  -3),
                        (90,  -3), 
                        (60,  -3), 
                        (45,  -3), 
                        (20,  -3), 
                        (10,  -3), 
                        (5,   -3), 
                        (2,   -1), 
                        (0,    0),
                        (-2,    1),
                        (-5,    3),
                        (-10,   3),
                        (-20,   3),
                        (-45,   3),
                        (-60,   3),
                        (-90,   3),
                        (-180,   3)]
        for imu_range in imu_ranges:           
            if self.imu_rpy[2] >= imu_range[0]:
                return imu_range[1]
        return 0

def main(args=None):
    rclpy.init(args=args)
    node = BalanceBeam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()