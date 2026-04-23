#!/usr/bin/env python3
# coding=utf-8
# ros2 topic pub --once /Zoom_In_Topic tku_msgs/msg/Zoom "{zoomin: 1.0}"
#colcon build --packages-select strategy

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import math
import numpy as np

# 引用 API
from strategy.API import API

# 定義常數
HORIZON_HEAD = 3010
HEAD_CHECK = 2080
HAND_BACK = 222 
LEG_BACK = 1812
VERTICAL_HEAD = 2155
X_BENCHMARK = [210, 200, 189, 178, 169] # [最左,中左,中間,中右,最右]    #改大射左

Y_BENCHMARK = 74 #改大射高
SHOOT_DELAY = 0.78

# motion sector
PREPARE = 9999   
SHOOT = 456       
HAND_UP = 111
LEG_DOWN1 = 1218 
LEG_DOWN2 = 12188  

#========================================================
RIGHT_TURN = 2.4
LEFT_TURN = 2.4
#========================================================

# 視覺找靶
class ArcheryTarget:
    def __init__(self, api_instance):
        self.api = api_instance     # 綁定 API 實例，用來獲取視覺數據
        self.red_x = 0              # 儲存找到的靶心 X 座標
        self.red_y = 0              # 儲存找到的靶心 Y 座標
        self.found = False          # 標記是否成功找到標靶
        self.tolerance = 40         # 容許誤差值(像素)：判斷色塊中心點距離是否夠近，藉此確認是不是同心圓
        self.log_counter = 0        # 用來控制印出 Log 的頻率，避免洗版
        
    def find(self):
        self.found = False          # 每次尋找前先將標記設為 False
        
        # 安全檢查：確保 API 已經抓到顏色物件的座標陣列
        if not hasattr(self.api, 'color_counts') or \
           not hasattr(self.api, 'object_x_min') or \
           self.api.color_counts is None:
            return                  # 如果還沒抓到資料，就直接跳出

        self.log_counter += 1       # 計數器加 1
        show_log = (self.log_counter % 20 == 0) # 每 20 次才印一次 Log (顯示偵測狀態)
        self.api.drawImageFunction(1, 1, 0, 320, 120, 120, 0, 0, 0)
        self.api.drawImageFunction(2, 1, 160, 160, 0, 240, 0, 0, 0)
        try:
            # API 定義的顏色索引:Yellow=1, Blue=2, Red=5
            # 遍歷藍色 (Index 2)
            for j in range(self.api.color_counts[2]): 
                # 遍歷黃色 (Index 1)
                for k in range(self.api.color_counts[1]): 
                    # 遍歷紅色 (Index 5)
                    for m in range(self.api.color_counts[5]): 
                        
                        try:
                            # 【關鍵修正】計算中心點： (Min + Max) / 2
                            # 藍色 B
                            bx_min = self.api.object_x_min[2][j]
                            bx_max = self.api.object_x_max[2][j]
                            by_min = self.api.object_y_min[2][j]
                            by_max = self.api.object_y_max[2][j]
                            blue_x = (bx_min + bx_max) / 2
                            blue_y = (by_min + by_max) / 2

                            # 黃色 Y
                            yx_min = self.api.object_x_min[1][k]
                            yx_max = self.api.object_x_max[1][k]
                            yy_min = self.api.object_y_min[1][k]
                            yy_max = self.api.object_y_max[1][k]
                            yellow_x = (yx_min + yx_max) / 2
                            yellow_y = (yy_min + yy_max) / 2

                            # 紅色 R
                            rx_min = self.api.object_x_min[5][m]
                            rx_max = self.api.object_x_max[5][m]
                            ry_min = self.api.object_y_min[5][m]
                            ry_max = self.api.object_y_max[5][m]
                            red_x = (rx_min + rx_max) / 2
                            red_y = (ry_min + ry_max) / 2

                            # 計算距離差 (絕對值)
                            diff_by_x = abs(blue_x - yellow_x)
                            diff_by_y = abs(blue_y - yellow_y)
                            diff_yr_x = abs(yellow_x - red_x)
                            diff_yr_y = abs(yellow_y - red_y)
                            
                            # Log 偵錯 (只印出第一組組合，協助你調整 tolerance)
                            if show_log and j==0 and k==0 and m==0:
                                print(f"--- 距離檢測 ---")
                                print(f"B({int(blue_x)},{int(blue_y)}) Y({int(yellow_x)},{int(yellow_y)}) R({int(red_x)},{int(red_y)})")
                                print(f"B-Y 差: ({int(diff_by_x)}, {int(diff_by_y)})")
                                print(f"Y-R 差: ({int(diff_yr_x)}, {int(diff_yr_y)})")
                                print(f"容許值: {self.tolerance}")
                                print("----------------")

                            # 判斷邏輯
                            # 1. 藍黃距離是否在誤差內
                            if diff_by_x <= self.tolerance and diff_by_y <= self.tolerance:
                                # 2. 黃紅距離是否在誤差內
                                if diff_yr_x <= self.tolerance and diff_yr_y <= self.tolerance:
                                    
                                    self.red_x = red_x
                                    self.red_y = red_y
                                    self.found = True
                                    
                                    print(f" ======== 鎖定靶心! R:({int(red_x)},{int(red_y)}) ========")
                                    return # 找到一組就直接回傳，鎖定目標

                        except IndexError:
                            # 防止陣列變動導致索引錯誤
                            continue 
                        except Exception as e:
                            print(f"計算錯誤: {e}")
                            continue

        except Exception as e:
            pass

        # 沒找到則歸零
        if not self.found:
            self.red_x, self.red_y = 0, 0


class Archery(Node):
    def __init__(self):
        super().__init__('ar') 
        
        # 初始化 API
        self.send = API() 
        
        # 初始化 Target
        self.archery_target = ArcheryTarget(self.send)

        # 初始化變數
        self.stand = 0
        self.x_points = []
        self.y_points = []
        self.first_point = False
        self.ctrl_status = 'find_period'
        self.lowest_x = 0
        self.lowest_y = 0
        self.turn_right = 0
        self.turn_left = 0
        self.hand_move_cnt = 0
        self.start_time = 0
        self.end_time = 0
        self.init_cnt = 0
        self.archery_action_ready = False
        self.timer = None
        self.back_flag = False
        self.turn_left_cnt = 0
        self.turn_right_cnt = 0
        self.hand_back_cnt = 0
        self.leg_back_cnt = 0
        self.x_benchmark_type = 0

    def shoot(self):
        """射擊 Timer 回調"""
        self.get_logger().error("###### in SHOOT func #####")
        
        if self.archery_action_ready:
            # 確保使用 float 進行 sleep
            delay = float(self.end_time - self.start_time - SHOOT_DELAY)
            if delay < 0: delay = 0
            time.sleep(delay)
            self.get_logger().error("!!!!!! SHOOT !!!!!!!")
            
            self.send.sendBodySector(SHOOT) 
            self.send.drawImageFunction(6, 1, int(self.lowest_x-1), int(self.lowest_x+1), int(self.lowest_y-1), int(self.lowest_y+1), 255, 0, 255) 
            time.sleep(2)
            
            # self.send.sendBodySector(999)
            time.sleep(2)
            
            if self.timer is not None:
                self.timer.cancel()
                self.destroy_timer(self.timer)
                self.timer = None
                
            self.archery_action_ready = False
            self.back_flag = True

    def initial(self):
        self.x_points = []
        self.y_points = []
        self.first_point = False
        self.ctrl_status = 'find_period'
        self.lowest_x = 0
        self.lowest_y = 0
        self.hand_move_cnt = 0
        self.start_time = 0
        self.end_time = 0
        self.archery_action_ready = False
        self.timer = None
        self.back_flag = False
        self.x_benchmark_type = 0

    def main_strategy(self):
        """主策略迴圈"""
        self.get_logger().info("策略開始執行 Loop...")
        # self.send.is_start = True
        
        while rclpy.ok():
            # [狀態監控] 每2秒印一次,確認 API 有收到數據
            try:
                self.get_logger().info(
                    f"Status:{self.ctrl_status}, Found:{self.archery_target.found}, Cnts:{self.send.color_counts}, tolerance = {self.archery_target.tolerance}", 
                    throttle_duration_sec=2.0
                )
            except AttributeError:
                pass

            # 畫十字sssssssssssssssssssssssss
            self.send.drawImageFunction(4, 0, 0, 320, 120, 120, 0, 0, 0)
            self.send.drawImageFunction(5, 0, 160, 160, 0, 240, 0, 0, 0)

            if self.send.is_start:
                self.stand = 0
                # ------------------- 初始化動作 -------------------
                if self.init_cnt == 1: 
                    self.initial() 
                    print()
                    self.init_cnt = 0 
                    self.send.sendHeadMotor(2, HEAD_CHECK, 80)
                    time.sleep(0.05)
                    self.send.sendHeadMotor(2, HEAD_CHECK, 80)
                    time.sleep(0.05)
                    self.send.sendHeadMotor(2, HEAD_CHECK, 80)
                    time.sleep(0.05)
                    self.send.sendHeadMotor(2, VERTICAL_HEAD, 80)
                    time.sleep(0.05)
                    self.send.sendHeadMotor(2, VERTICAL_HEAD, 80)
                    time.sleep(2)
                
                # ------------------- 視覺找靶 -------------------
                if self.ctrl_status != 'wait_shoot':
                    self.archery_target.find() 

                # ------------------- 狀態機邏輯 -------------------
                if self.ctrl_status == 'find_period': 
                    if self.archery_target.found:
                        self.x_points.append(self.archery_target.red_x)
                        self.y_points.append(self.archery_target.red_y)
                        
                        if not self.first_point: 
                            if (len(self.x_points) > 0 and self.x_points[0] != 0) and \
                               (len(self.y_points) > 0 and self.y_points[0] != 0):
                                time.sleep(0.2)
                                self.first_point = True
                        
                        self.archery_target.found = False 

                        if len(self.x_points) > 1:
                            dis = math.hypot(self.archery_target.red_x-self.x_points[0], self.archery_target.red_y-self.y_points[0])
                            if dis <= 1.5: 
                                self.end_time = time.time()
                                self.lowest_y = max(self.y_points)
                                self.lowest_x = self.x_points[self.y_points.index(self.lowest_y)]
                                self.get_logger().info(f"鎖定! period = {self.end_time - self.start_time}")
                                self.ctrl_status = 'wait_lowest_point'
                        else:
                            self.start_time = time.time()
                            self.get_logger().info(f"Start Timing... {self.start_time}")

                elif self.ctrl_status == 'wait_lowest_point':
                    dis = math.hypot(self.archery_target.red_x-self.lowest_x, self.archery_target.red_y-self.lowest_y)
                    if dis <= 1.5:
                        duration = self.end_time - self.start_time
                        if duration <= 0: duration = 0.01
                        # 啟動射擊計時器
                        self.timer = self.create_timer(duration, self.shoot)
                        
                        self.send.drawImageFunction(6, 1, int(self.lowest_x-2), int(self.lowest_x+2), int(self.lowest_y-2), int(self.lowest_y+2), 0, 0, 255)
                        self.get_logger().info("到達最低點 (At Lowest Y)")
                        
                        if self.init_cnt != 2: 
                            self.ctrl_status = 'archery_action'
                        else:
                            self.archery_action_ready = True
                            self.ctrl_status = 'wait_shoot'

                elif self.ctrl_status == 'archery_action':
                    # 決定要轉多少
                    if 0 < self.lowest_x <= 110: 
                        self.x_benchmark_type = 4 # 最右(97,1794)
                        print("44444444444444444444444444")
                    elif 110 < self.lowest_x <= 150: 
                        self.x_benchmark_type = 3 # 中右(146,1914)(150,1928)
                        print("33333333333333333333333333")
                    elif self.lowest_x >= 190: 
                        self.x_benchmark_type = 0 # 最左(219,2093)
                        print("00000000000000000000000000")
                    elif 190 > self.lowest_x >= 170 :   
                        self.x_benchmark_type = 1 # 中左(197,2042)
                        print("11111111111111111111111111")
                    else:   
                        self.x_benchmark_type = 2  # 中間(166,1960)(168,1971)
                        print("22222222222222222222222222") 
                    
                    self.get_logger().info(f'Action Type: {self.x_benchmark_type}')
                    
                    # 轉腰(左正右負)
                    if self.lowest_x - X_BENCHMARK[self.x_benchmark_type] > 0: 
                        self.turn_right = X_BENCHMARK[self.x_benchmark_type] - self.lowest_x
                        self.get_logger().info(f"向右轉: {int(RIGHT_TURN*self.turn_right)}")
                        self.send.sendSingleMotor(15,(int(RIGHT_TURN*self.turn_right)),15)
                
                        self.turn_right_cnt = 1
                        time.sleep(3)
                    else: 
                        self.turn_left = X_BENCHMARK[self.x_benchmark_type] - self.lowest_x
                        self.get_logger().info(f"向左轉: {int(LEFT_TURN*self.turn_left)}")
                        self.send.sendSingleMotor(15,(int(LEFT_TURN*self.turn_left)),15)
                        
                        self.turn_left_cnt = 1
                        time.sleep(3)
                    # 手腳動作
                    if self.lowest_y - Y_BENCHMARK > 0:
                        self.leg_move_cnt = abs(int((Y_BENCHMARK - self.lowest_y) / 2))
                        self.leg_back_cnt = self.leg_move_cnt
                        self.get_logger().info("向下蹲")
                        self.get_logger().info(f"向下蹲 {self.leg_move_cnt}")

                        for i in range(0,self.leg_move_cnt,1):
                            if i < 2:
                                self.send.sendBodySector(LEG_DOWN1) #蘿菠蹲
                            else:
                                self.send.sendBodySector(LEG_DOWN2) #蘿菠蹲
                            time.sleep(0.5)

                    else:
                        self.hand_move_cnt = abs(int((self.lowest_y - Y_BENCHMARK) / 2))
                        self.hand_back_cnt = self.hand_move_cnt
                        while self.hand_move_cnt != 0:
                            self.get_logger().info(f"HAND_UP")
                            # self.send.sendBodySector(HAND_UP)
                            self.hand_move_cnt -= 1
                            time.sleep(0.5)
                    
                    # 關閉 timer (以防萬一)
                    if self.timer is not None:
                        self.timer.cancel()
                        self.destroy_timer(self.timer)
                        self.timer = None
                        
                    time.sleep(0.1)
                    self.initial()
                    time.sleep(0.1)
                    self.init_cnt = 2
                    self.ctrl_status = 'find_period'

                elif self.ctrl_status == 'wait_shoot':
                    time.sleep(1)

            else:
                # ------------------- 預備動作 & 復原 -------------------
                if self.stand == 0: 
                    self.send.sendHeadMotor(1, HORIZON_HEAD, 80)
                    time.sleep(3)
                    self.send.sendBodySector(PREPARE) 
                    time.sleep(2.8)
                    self.stand = 1 
                    self.get_logger().info('預備動作 Done')
                
                # if self.back_flag:
                #     print("###### in BACK func #####")
                #     if self.turn_right_cnt != 0:
                #         self.send.sendSingleMotor(15,int(-(RIGHT_TURN*self.turn_right)),15)
                #         self.get_logger().info(f"{RIGHT_TURN*self.turn_right}")
                #         time.sleep(2.5)
                #     elif self.turn_left_cnt != 0:
                #         self.send.sendSingleMotor(15,int(-(LEFT_TURN*self.turn_left)),15)
                #         self.get_logger().info(f"{self.end_time - self.start_time}")
                #         time.sleep(2.5)
                    
                #     for i in range(0, self.hand_back_cnt):
                #         # self.send.sendBodySector(HAND_BACK)
                #         time.sleep(0.5)
                #     self.hand_back_cnt = 0 

                #     for i in range(0, self.leg_back_cnt):
                #         # self.send.sendBodySector(LEG_BACK)
                #         time.sleep(0.5)
                #     self.leg_back_cnt = 0 
                    
                #     self.back_flag = False

                time.sleep(1) 


def main(args=None):
    rclpy.init(args=args) # 初始化 ROS 2 環境
    
    strategy = Archery()  # 建立 Archery 策略節點
    executor = MultiThreadedExecutor() # 建立多執行緒執行器
    executor.add_node(strategy)        # 將策略節點加入執行器
    
    # 將 API 節點也加入執行器 (非常重要：這樣 API 才能非同步接收攝影機回傳的影像 Topic)
    executor.add_node(strategy.send)
    
    # 開啟一個背景執行緒去讓 ROS 2 不斷接收與發送資料 (spin)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        strategy.main_strategy() # 啟動我們上面定義的主策略無窮迴圈
    except KeyboardInterrupt:
        pass                     # 如果使用者按下 Ctrl+C 就略過並準備關閉
    finally:
        strategy.destroy_node()  # 銷毀節點
        rclpy.shutdown()         # 關閉 ROS 2 系統

if __name__ == '__main__':
    main() # 呼叫 main 函式