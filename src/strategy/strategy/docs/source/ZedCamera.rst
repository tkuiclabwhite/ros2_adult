ZED 相機功能 (ZED Camera)
--------------------------------------------------
.. currentmodule:: API

.. note::
   本章節功能為 **大人型專用**。小人型機器人未配備 ZED 深度相機，以下 API 不存在。

.. important::
   **影像座標系為 960 x 600。**
   其他章節標示的 0~320 / 0~240 為小人型的解析度。
   色模、深度、疊合三套系統共用同一組座標，可以直接互相對照 ——
   例如拿色模算出的物體中心去查 :meth:`API.depth_at`，得到的就是該物體的距離。

三套偵測系統的索引結構完全相同：``[類別索引][該類別的第幾個物體]``。
第二層的順序是連通元件的掃描順序（由上而下、由左而右），
**不代表大小或遠近，也不保證跨幀穩定**，需要「最大的」或「最近的」請自行挑選。

存取前務必先檢查對應的 ``*_counts``，數量為 0 時清單是空的。

色模擴充欄位
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

色模本身的用法請見 :doc:`VisionSensors`，以下三個欄位為大人型額外提供。

.. autoattribute:: API.object_cx
.. autoattribute:: API.object_cy
.. autoattribute:: API.object_ratio

深度偵測 (Depth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

依距離區間 D1~D8 將畫面分類，區間範圍在網頁的深度模式中設定。
除了與色模相同的邊界、面積等欄位外，額外提供每個物體的實際距離。

.. automethod:: API.depth_at

.. autoattribute:: API.distance_counts
.. autoattribute:: API.distance_cm
.. autoattribute:: API.distance_min_cm
.. autoattribute:: API.distance_max_cm
.. autoattribute:: API.distance_x_min
.. autoattribute:: API.distance_x_max
.. autoattribute:: API.distance_y_min
.. autoattribute:: API.distance_y_max
.. autoattribute:: API.distance_cx
.. autoattribute:: API.distance_cy
.. autoattribute:: API.distance_sizes
.. autoattribute:: API.distance_ratio
.. autoattribute:: API.new_distance_info

疊合偵測 (Overlap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set1~Set8 各自綁定一個顏色類別與一個距離類別，取兩者的交集區域，
用於「**這個顏色、而且在這個距離**」的複合條件判斷。
組合內容在網頁的疊合模式中設定。

.. autoattribute:: API.overlap_counts
.. autoattribute:: API.overlap_cm
.. autoattribute:: API.overlap_min_cm
.. autoattribute:: API.overlap_max_cm
.. autoattribute:: API.overlap_x_min
.. autoattribute:: API.overlap_x_max
.. autoattribute:: API.overlap_y_min
.. autoattribute:: API.overlap_y_max
.. autoattribute:: API.overlap_cx
.. autoattribute:: API.overlap_cy
.. autoattribute:: API.overlap_sizes
.. autoattribute:: API.overlap_ratio
.. autoattribute:: API.new_overlap_info

ZED 內建 IMU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZED 相機內建一顆 IMU，與機身上的 Arduino IMU（:attr:`API.imu_rpy`）是兩顆獨立的感測器。

.. warning::
   ZED 安裝在**頭部雲台**上，頭部轉動時即使機身未移動 yaw 也會改變，
   兩顆 IMU 的數值不能直接互相比較。

.. automethod:: API.sendZedSensorReset

.. autoattribute:: API.zed_imu_rpy
.. autoattribute:: API.zed_imu_abs_rpy
.. autoattribute:: API.zed_gyro
.. autoattribute:: API.zed_accel
.. autoattribute:: API.zed_imu_zeroed

視覺里程計 (Odometry)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZED 以視覺特徵搭配 IMU 融合估算相機的位移。

.. warning::
   目前為**驗證階段**，尚未做頭部雲台的補償。回傳的是相機而非機身的位移，
   只轉頭不移動時數值同樣會變化；走路時的晃動與場地特徵稀少也會影響精度。
   **請勿讓策略依賴此數值。**

.. automethod:: API.sendOdomReset

.. autoattribute:: API.odom_xyz
.. autoattribute:: API.odom_yaw

內部處理函式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下為 API 內部使用的回呼與統計函式，策略端不需要直接呼叫。

.. automethod:: API._depth_det_callback
.. automethod:: API._overlap_det_callback
.. automethod:: API._json_det_callback
.. automethod:: API._fill_stats
.. automethod:: API._recompute_distance_stats
.. automethod:: API._recompute_overlap_stats
.. automethod:: API._depth_mm_cb
.. automethod:: API._zed_imu_cb
.. automethod:: API._odom_cb
