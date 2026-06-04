系統架構總覽
============

ros2_adult 是淡江大學（TKU）HuroCup 人形機器人 Adultsize 的完整 ROS2 控制系統，
涵蓋底層馬達驅動、步態生成、影像視覺，以及各賽項策略。

Package 架構圖
--------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────┐
   │                   hurocup_interface                      │
   │             (瀏覽器網頁操作與監控介面)                   │
   └───────────────┬──────────────────────────────────────────┘
                   │ HTTP / WebSocket / ROS Topics
   ┌───────────────▼──────────────────────────────────────────┐
   │                    strategy / API                        │
   │   (賽項策略主程式：ar/bb/sp/obs/sr/wl/mar/pk/rc)         │
   │   繼承 API base class，整合視覺、IMU、步態控制            │
   └───────┬────────────────────────────┬─────────────────────┘
           │                            │
   ┌───────▼──────┐           ┌─────────▼───────────┐
   │   walking    │           │    imageprocess      │
   │  (步態生成)  │           │    (HSV 色彩偵測)    │
   └───────┬──────┘           └─────────┬────────────┘
           │                            │
   ┌───────▼──────┐           ┌─────────▼────────────┐
   │motionpackage │           │      usb_cam          │
   │(動作序列執行)│           │    (USB 相機輸入)     │
   └───────┬──────┘           └──────────────────────┘
           │
   ┌───────▼──────────────────┐
   │      motor_control       │
   │   DynamixelDriver        │
   │  (混合 X/P 系列馬達底層) │
   │  U2D2 ×3 /dev/U2D2_P*   │
   └──────────────────────────┘

訊號流向
--------

.. code-block:: text

   相機 → imageprocess → /detections/{color} → strategy (API)
                                                     ↓
   IMU (/dev/ttyTHS1) → /package/sensorpackage → strategy (API)
   GPIO PIN 32 開關 ─────────────────────────→ strategy (API)
                                                     ↓
   strategy → /SendBodyAuto_Topic ──→ walking → /joint_commands → motor_control → Dynamixel
   strategy → /package/Sector ──────→ motionpackage → /joint_commands → motor_control
   strategy → /Head_Topic ─────────────────────────────────────→ motor_control

Package 一覽
------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Package
     - 說明
   * - ``motor_control``
     - 混合 X-Series + H42 Pro 馬達底層驅動，管理 29 顆馬達（3 個 U2D2 USB-RS485），20Hz 控制迴圈，各 Port 並列讀寫
   * - ``walking``
     - LIPM 步態生成器（50Hz）＋混合系列逆運動學（X-Series TPR=4096/2π；H-Series TPR=303750/2π）
   * - ``motionpackage``
     - 固定動作序列播放（INI 格式）＋ Jetson GPIO PIN 32 硬體開關讀取
   * - ``imageprocess``
     - HSV 色彩切割視覺節點，輸出各顏色物件 bbox/centroid 及二值化遮罩（20Hz）
   * - ``usb_cam``
     - USB 相機驅動，發布 ``/camera1/image_raw``
   * - ``strategy``
     - 所有賽項策略（繼承 ``API``）及 API 基底類別（含 ZED YOLO 整合）
   * - ``tku_msgs``
     - 自定義 msg/srv 定義

賽項策略
--------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - 指令
     - 說明
   * - ``ar``
     - 射箭
   * - ``bb``
     - 籃球
   * - ``sp``
     - 競走
   * - ``obs``
     - 避障
   * - ``sr``
     - 斯巴達
   * - ``wl``
     - 舉重
   * - ``mar``
     - 馬拉松
   * - ``pk``
     - 罰踢
   * - ``rc``
     - 攀岩
   * - ``strategy_example``
     - 複製用範本

主要 ROS2 Topic 一覽
--------------------

**步態相關**

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Topic
     - Type
     - 說明
   * - ``/ContinousMode_Topic``
     - Int16
     - 步態 on/off（1=啟動）
   * - ``/SendBodyAuto_Topic``
     - Interface
     - 步態指令（x/y/theta）＋啟動
   * - ``/ChangeContinuousValue_Topic``
     - Interface
     - 步長即時調整
   * - ``/joint_commands``
     - JointState
     - walking/motionpackage → motor_control
   * - ``/joint_states``
     - JointState
     - motor_control → walking（馬達現在位置）

**視覺相關**

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Topic
     - Type
     - 說明
   * - ``/camera1/image_raw``
     - Image
     - USB 相機原始影像
   * - ``/detections/{color}``
     - String
     - 各顏色物件偵測 JSON（8 個 topic）
   * - ``/{color}_mask``
     - UInt8MultiArray
     - 各顏色二值化遮罩
   * - ``/label_matrix``
     - Image
     - 二值化 mask
   * - ``/zed_yolo_ball``
     - Point
     - ZED 相機 YOLO 球體 3D 座標

**感測器**

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Topic
     - Type
     - 說明
   * - ``/package/sensorpackage``
     - SensorPackage
     - IMU yaw/pitch/roll（來自 ``/dev/ttyTHS1``）
   * - ``/package/dioarray``
     - Dio
     - Jetson GPIO PIN 32 開關狀態

**動作執行**

.. list-table::
   :widths: 55 45
   :header-rows: 1

   * - Topic / Service
     - 說明
   * - ``/package/Sector`` (Int16)
     - 執行固定動作編號
   * - ``/Head_Topic`` (HeadPackage)
     - 頭部馬達控制（映射 ID 1→28, 2→29）
   * - ``/package/SingleMotorData``
     - 單軸相對位置控制
   * - ``/package/SingleAbsolutePosition``
     - 單軸絕對位置控制
   * - ``/set_torque`` (Int16MultiArray)
     - 馬達扭力開關

**策略切換**

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Topic
     - Type
     - 說明
   * - ``/strategy/name``
     - String
     - 選擇策略（"ar"、"bb" ...）
   * - ``/strategy/start``
     - Bool
     - 啟動/停止選定的策略
   * - ``/strategy/status``
     - String
     - 目前策略狀態回報

啟動方式
--------

.. code-block:: bash

   # Build（第一次或修改程式後）
   cd ~/ros2_adult
   colcon build --symlink-install
   source install/setup.bash

   # 主系統（相機、視覺、驅動、步態、IMU 等）
   ros2 launch usb_cam camera.launch.py

   # 策略（擇一執行）
   ros2 run strategy ar   # 射箭
   ros2 run strategy bb   # 籃球

   # 快捷別名
   cb   # colcon build --symlink-install && source install/setup.bash
   cbp  # colcon build --symlink-install --packages-select strategy && source install/setup.bash   

設定檔位置
----------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - 路徑
     - 說明
   * - ``src/walking/walking/Parameter.py``
     - 步態初始參數（COM_HEIGHT=40, STAND_HEIGHT=50 等）
   * - ``src/strategy/strategy/{策略}/Parameter/WalkingParameter.ini``
     - 各策略步態設定
   * - ``src/strategy/strategy/{策略}/Parameter/hsv.ini``
     - 各策略顏色偵測門檻
   * - ``src/strategy/strategy/strategy.ini``
     - 記錄當前載入哪個策略的參數路徑
   * - ``src/usb_cam/config/CameraSet.ini``
     - 相機縮放等設定
