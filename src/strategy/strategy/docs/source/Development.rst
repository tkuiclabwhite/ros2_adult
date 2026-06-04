新策略開發
==========

每個子策略只需繼承 ``API``，在主迴圈回呼中實作邏輯。

程式骨架
--------

.. code-block:: python

   #!/usr/bin/env python3
   import rclpy
   from rclpy.executors import MultiThreadedExecutor
   from strategy.API import API

   class MyStrategy(API):
       def __init__(self):
           super().__init__('my_strategy')
           self.create_timer(0.1, self.main)  # 10Hz 主迴圈

       def main(self):
           if not self.is_start:   # 硬體開關未開啟時不執行
               return

           # --- 視覺 ---
           # self.color_counts[0]          → 橘色物件數量
           # self.object_sizes[0]          → 橘色各物件面積列表
           # self.object_x_min/max[0][i]   → 第 i 個橘色物件邊界

           # --- ZED YOLO 3D 定位 ---
           # self.pose_x, self.pose_y, self.pose_z  → 目標物 3D 座標 (來自 /zed_yolo_ball)

           # --- 步態 ---
           # self.sendBodyAutoCmd(x=500, y=0, theta=0)     # 啟動步態並前進
           # self.sendContinuousValue(x=500, y=0, theta=0) # 只更新步長
           # self.sendbodyAuto(0)                           # 停止步態

           # --- 動作 ---
           # self.sendBodySector(29)   # 執行站姿

           # --- 頭部 ---
           # self.sendHeadMotor(1, 2048, 100)  # 水平居中（絕對刻度）
           # self.set_head(pan=0.0, tilt=0.0)  # 比例控制，0.0 為中心

           # --- IMU ---
           # self.yaw, self.pitch, self.roll

   def main(args=None):
       rclpy.init(args=args)
       node = MyStrategy()
       executor = MultiThreadedExecutor()
       executor.add_node(node)
       try:
           executor.spin()
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()

setup.py 設定
-------------

在 ``src/strategy/setup.py`` 的 ``console_scripts`` 加入一行，才能用 ``ros2 run`` 啟動：

.. code-block:: python

   'my_strategy = strategy.my_folder.my_file:main',

API._strategy_map 註冊
-----------------------

在 ``API.py`` 的 ``_strategy_map`` 中加入名稱，讓網頁介面可以切換到此策略：

.. code-block:: python

   "my": ("strategy", "my_strategy"),

---

網頁操作介面
============

位於 ``hurocup_interface/``，使用瀏覽器開啟對應 HTML 檔案。

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - 頁面
     - 說明
   * - ``index.html``
     - 主頁，策略選擇與啟動
   * - ``WalkingInterface.html``
     - 步態參數調整（步長/步寬/質心高度等）
   * - ``ImageProcessInterface.html``
     - HSV 色彩校正，即時預覽
   * - ``MotionControlInterface.html``
     - 動作序列錄製與播放
   * - ``NodeMonitor.html``
     - 節點狀態監控
   * - ``miniDRC.html``
     - 精簡遙控介面

---

設定檔位置
==========

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - 路徑
     - 說明
   * - ``src/walking/walking/Parameter.py``
     - 步態初始參數（COM_HEIGHT=40, STAND_HEIGHT=50 等）
   * - ``src/strategy/strategy/{策略}/Parameter/WalkingParameter.ini``
     - 各策略步態設定
   * - ``src/strategy/strategy/{策略}/Parameter/hsv.ini``
     - 各策略顏色偵測門檻
   * - ``src/strategy/strategy/{策略}/Parameter/{N}.ini``
     - 各策略動作序列（Sector 編號對應）
   * - ``src/strategy/strategy/strategy.ini``
     - 記錄當前載入哪個策略的參數路徑
   * - ``src/usb_cam/config/CameraSet.ini``
     - 相機縮放等設定
