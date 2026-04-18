#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
import time
import sys

# 假設你的 API 檔案名叫做 API.py
from API import API

class WalkTestStrategy(API):
    def __init__(self):
        # 1. 繼承初始化：這行一定要做，才能拿到 self.sendContinuousValue 等功能
        super().__init__('WalkTestNode')
        
        self.get_logger().info("策略啟動：準備走路 5 秒...")

        # 2. 設定狀態變數
        self.state = "INIT"         # 狀態機：INIT -> WALKING -> STOP
        self.start_time = 0.0       # 用來記下開始走路的時間點
        self.walk_duration = 5.0    # 目標走幾秒
        self.x_speed = 0.02         # 前進速度 (請先用小數值測試，以免暴衝)
        
        # 3. 建立策略計時器 (Timer)
        # 每 0.1 秒 (100ms) 執行一次 main_logic
        # 這就像是你的「大腦運轉週期」
        self.create_timer(0.1, self.main_logic)

    def main_logic(self):
        """
        這是主要的策略迴圈，每 0.1 秒會被呼叫一次。
        """
        current_time = time.time()

        # --- 狀態機邏輯 ---
        
        if self.state == "INIT":
            # 剛啟動時，先做一些準備（例如確保已經站穩）
            # 這裡我們簡單做，直接切入走路狀態
            self.start_time = current_time
            self.state = "WALKING"
            self.get_logger().info("開始走路！")

        elif self.state == "WALKING":
            # 計算已經走了多久
            elapsed_time = current_time - self.start_time
            
            if elapsed_time < self.walk_duration:
                # 時間還沒到：繼續發送前進指令
                # sendContinuousValue(x, y, theta)
                self.sendContinuousValue(self.x_speed, 0.0, 0.0)
                
                # 顯示倒數計時 (除錯用)
                remaining = self.walk_duration - elapsed_time
                # print 比較好讀，但在 ROS 建議用 get_logger
                # self.get_logger().info(f"走路中... 剩餘 {remaining:.1f} 秒") 
            else:
                # 時間到了：切換到停止狀態
                self.state = "STOP"
                self.get_logger().info("時間到！停車！")

        elif self.state == "STOP":
            # 停止狀態：持續發送 0 速度，確保機器人停下來
            # 為什麼要持續送？因為底層走路節點通常有 Timeout 機制，
            # 如果不持續送 0，它可能會以為失聯而進入保護模式（或維持最後速度暴衝）
            self.sendContinuousValue(0.0, 0.0, 0.0)
            
            # (可選) 如果想讓程式完全結束，可以打開下面這行：
            # sys.exit(0) 

def main(args=None):
    rclpy.init(args=args)
    
    # 建立策略節點
    node = WalkTestStrategy()
    
    # 使用 MultiThreadedExecutor 是為了讓 API 內部的 ReentrantCallbackGroup 生效
    # 這樣你的 main_logic 在跑的時候，背景的 sensor callback 也能同時收資料
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        # 按 Ctrl+C 時的優雅退出
        node.get_logger().info("使用者強制中斷，嘗試停止機器人...")
        node.sendContinuousValue(0.0, 0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()