網頁操作介面教學
================

各節的 **▶ 開啟 DEMO** 按鈕開啟模擬版，不需連接機器人即可互動試用。

點選圖片中虛線匡內可直接跳轉至該功能說明。

----

.. _shared-elements:

共用介面元素
------------

各介面頁面共用以下控制項，操作方式一致。

.. _strategy:

策略狀態顯示
^^^^^^^^^^^^^

左上角的 Strategy 框顯示目前載入的策略名稱，訂閱自 ``/locationBack`` 話題，連線後自動更新。

正常狀態顯示策略簡稱（例如 ``us``、``ar``、``lc``）；未連線時顯示 ``---``。

.. _shared-address:

連線地址（Address）
^^^^^^^^^^^^^^^^^^^^

下拉選單選擇機器人 IP，點擊 **Enter Address** 後重新建立 WebSocket 連線（``ws://{IP}:9090``）。

預設 IP：``172.17.121.10``

連線成功後頁面顯示 **Connected**\ （綠色），背景切換為正常背景圖；連線中斷則切換為紅色背景。

----

.. _interface-image:

ImageProcessInterface
--------------------------------------

.. raw:: html

   <a href="_static/ImageProcess_demo.html" target="_blank"
      style="background:#03e9f4;color:#000;padding:5px 14px;border-radius:4px;
             text-decoration:none;font-weight:bold;font-family:monospace;">
     &#9654; 開啟 DEMO
   </a>

.. raw:: html

   <div class="iface-map-wrap">
     <img src="_static/screenshots/ImageProcess.png" alt="ImageProcess 介面截圖">

    <!-- Address(各網頁共用) -->
    <a class="area-overlay" href="#shared-address"
      style="top:18.5%;left:36%;width:12.75%;height:4.25%;color:#ffcc00"></a>

    <!-- 策略選擇 -->
    <a class="area-overlay" href="#image-strategy-select"
      style="top:18.5%;left:48.75%;width:10.75%;height:4.25%;color:#ffcc00"></a>

    <!-- SaveAll -->
    <a class="area-overlay" href="#image-save-all"
      style="top:18.5%;left:59.5%;width:5%;height:4.25%;color:#ffcc00"></a>

    <!-- 策略指撥開關 -->
    <a class="area-overlay" href="#image-strategy-toggle"
      style="top:22.5%;left:46%;width:8.5%;height:4.25%;color:#ffcc00"></a>

    <!-- 原始影像 -->
    <a class="area-overlay" href="#image-camera"
      style="top:30%;left:27.5%;width:19%;height:24.25%;color:#ffcc00"></a>

    <!-- HSV色模 -->
    <a class="area-overlay" href="#image-model-view"
      style="top:30%;left:53.8%;width:19%;height:24.25%;color:#ffcc00"></a>

    <!-- 頭部馬扭力 -->
    <a class="area-overlay" href="#image-head-torque"
      style="top:57%;left:27.5%;width:12.5%;height:3.5%;color:#ffcc00"></a>

    <!-- 儲存相機參數 -->
    <a class="area-overlay" href="#image-save-camera"
      style="top:57%;left:40.5%;width:5.75%;height:3.5%;color:#ffcc00"></a>

    <!-- 頭部馬達控制 -->
    <a class="area-overlay" href="#image-head-motor"
      style="top:60.25%;left:27.5%;width:19%;height:14%;color:#ffcc00"></a>

    <!-- 相機參數設定 -->
    <a class="area-overlay" href="#image-camera-settings"
      style="top:73.75%;left:27.5%;width:19%;height:23.5%;color:#ffcc00"></a>

    <!-- Build table -->
    <a class="area-overlay" href="#image-build-table"
      style="top:57.25%;left:53.8%;width:19%;height:3.5%;color:#ffcc00"></a>

    <!-- Color Select -->
    <a class="area-overlay" href="#image-color-labels"
      style="top:60.5%;left:53.8%;width:10.25%;height:3.5%;color:#ffcc00"></a>

    <!-- Reset/Close -->
    <a class="area-overlay" href="#image-reset-close"
      style="top:60.5%;left:66%;width:6.75%;height:3.5%;color:#ffcc00"></a>

    <!-- 色模參數設定 -->
    <a class="area-overlay" href="#image-hsv"
      style="top:63.75%;left:53.8%;width:19%;height:20.5%;color:#ffcc00"></a>

    <!-- SaveHsv -->
    <a class="area-overlay" href="#image-save-hsv"
      style="top:84%;left:53.8%;width:9.5%;height:4%;color:#ffcc00"></a>

    <!-- BuildHsv -->
    <a class="area-overlay" href="#image-build-hsv"
      style="top:84%;left:63.3%;width:9.5%;height:4%;color:#ffcc00"></a>


   </div>

.. _image-strategy-select:

策略選擇
^^^^^^^^

下拉選單，選擇目前要使用的策略模組。

**可用策略：** ``us``\ （足球）、``ar``\ （射箭）、``bb``\ （籃球）、``sp``\ （競走）、``obs``\ （避障）、``sr``\ （斯巴達）、``mar``\ （馬拉松）、``pk``\ （PK）、``rc``\ （遙控）

選定後點擊 ``location_load`` 按鈕，載入該策略所有參數。

.. _image-save-all:

SaveAll
^^^^^^^

一鍵儲存所有設定的快捷按鈕，等同於同時執行 :ref:`image-save-camera` 與 :ref:`image-save-hsv`。
調整完相機參數與 HSV 色模後，按此確保所有變更都寫入檔案。

.. _image-strategy-toggle:

策略指撥開關
^^^^^^^^^^^^^

模擬實體 DIO 撥動開關，控制 API.py 內的 ``is_start`` 旗標。

- **開（Start）**：發送 ``true``
- **關（Stop）**：發送 ``false``

.. _image-camera:

原始影像
^^^^^^^^

顯示相機的原始影像，畫面大小為320*240。

.. _image-model-view:

HSV 色模顯示
^^^^^^^^^^^^^

顯示經過 HSV 遮罩處理後的影像，畫面大小為320*240，顯示各顏色抓取HSV遮罩的結果。

各顏色在 HSV 色模顯示畫面中對應的標示顏色（OpenCV BGR 轉換）：

.. list-table::
   :header-rows: 1

   * - 偵測目標
     - Label ID
     - 畫面標示顏色
     - BGR 值
   * - Black
     - 1
     - 洋紅（Magenta）
     - ``[255, 0, 255]``
   * - Blue
     - 2
     - 紫色（Purple）
     - ``[128, 0, 128]``
   * - Green
     - 3
     - 深紅（Dark Red）
     - ``[0, 0, 128]``
   * - Orange
     - 4
     - 深藍（Dark Blue）
     - ``[128, 0, 0]``
   * - Red
     - 5
     - 青色（Cyan）
     - ``[255, 255, 0]``
   * - Yellow
     - 6
     - 暗青（Dark Teal）
     - ``[128, 128, 0]``
   * - White
     - 7
     - 黃色（Yellow）
     - ``[0, 255, 255]``
   * - Others
     - 8
     - 粉紫（Violet）
     - ``[255, 0, 128]``

.. _image-head-torque:

頭部馬達扭力
^^^^^^^^^^^^^

切換頭部馬達（水平/垂直軸）的扭力開關。

.. _image-save-camera:

儲存相機參數
^^^^^^^^^^^^^

將目前 :ref:`image-camera-settings` 的所有數值（亮度、對比、飽和度、白平衡、自動曝光、縮放），寫入設定檔(CameraSet.ini)以供下次啟動時自動載入。

.. _image-head-motor:

頭部馬達控制
^^^^^^^^^^^^^

滑桿控制頭部相機的水平（Horizontal）與垂直（Vertical）方向。

.. list-table::
   :header-rows: 1

   * - 參數
     - 說明
     - 預設值
     - 最小值
     - 最大值
   * - Horizontal Position
     - 水平刻度
     - 2048（中心）
     - 1024
     - 3096
   * - Horizontal Speed
     - 水平轉速
     - 1
     - 1
     - 200
   * - Vertical Position
     - 垂直刻度
     - 2048（中心）
     - 1023
     - 2548
   * - Vertical Speed
     - 垂直轉速
     - 1
     - 1
     - 200

.. _image-camera-settings:

相機參數設定
^^^^^^^^^^^^^

即時調整相機的影像擷取參數，調整後畫面立即更新：

.. list-table::
   :header-rows: 1

   * - 參數
     - 說明
   * - ``brightness``
     - 亮度
   * - ``contrast``
     - 對比度
   * - ``saturation``
     - 飽和度
   * - ``white_balance``
     - 色溫（手動白平衡值）
   * - ``auto_white_balance``
     - 自動白平衡開關
   * - ``auto_exposure``
     - 自動曝光開關
   * - ``zoomin``
     - 數位縮放倍率（預設 1.0）

.. _image-build-table:

Build Table
^^^^^^^^^^^

選擇色模重建的範圍後觸發建表：

- ``All_color_build``：重建所有顏色的偵測模型，上方 :ref:`image-model-view` 所有顏色遮罩結果。
- ``Single_color_build``：僅重建目前 :ref:`image-color-labels` 所選的顏色，上方 :ref:`image-model-view` 目前顏色遮罩與原始畫面疊加結果。

.. _image-color-labels:

Color Select
^^^^^^^^^^^^

下拉選單，選擇目前要編輯的顏色標籤（Orange、Yellow、Blue、Red、green、White、Other）。

.. _image-reset-close:

Reset / Close
^^^^^^^^^^^^^

- **Reset**：將 HSV 滑桿重設為全範圍（H: 0–180、S/V: 0–255），並即時發布至 HSV topic，
  適合重新開始調色時使用。
- **Close**：將所有 HSV 值設為 0，相當於關閉該顏色的偵測（偵測不到任何像素）。

.. _image-hsv:

色模參數設定（HSV）
^^^^^^^^^^^^^^^^^^^^

六組滑桿定義顏色偵測的 HSV 範圍，調整後即時發布至 ROS topic 並更新 :ref:`image-model-view` 顯示：

.. list-table::
   :header-rows: 1

   * - 參數
     - 範圍
     - 說明
   * - ``hmin`` / ``hmax``
     - 0 – 180
     - 色相下/上限
   * - ``smin`` / ``smax``
     - 0 – 255
     - 飽和度下/上限
   * - ``vmin`` / ``vmax``
     - 0 – 255
     - 明度下/上限

.. tip::

   先調整 H 範圍鎖定目標顏色，再縮小 S/V 範圍排除背景雜訊。

.. _image-save-hsv:

SaveHsv
^^^^^^^

將目前 :ref:`image-hsv` 的參數寫入檔案(hsv.ini)。

若未執行 SaveHsv 而直接 BuildHsv，使用的仍是上次儲存的舊參數。

.. _image-build-hsv:

BuildHsv
^^^^^^^^

讀出hsv.ini內的參數，並更新 :ref:`image-model-view` 與 :ref:`image-hsv`。

.. note::

   建議流程：調整 HSV → :ref:`image-save-hsv` 儲存 → :ref:`image-build-hsv` 重建 → 確認 :ref:`image-model-view` 結果。

----

.. _interface-motion:

MotionControlInterface
--------------------------------------

.. raw:: html

   <a href="_static/MotionControl_demo.html" target="_blank"
      style="background:#03e9f4;color:#000;padding:5px 14px;border-radius:4px;
             text-decoration:none;font-weight:bold;font-family:monospace;">
     &#9654; 開啟 DEMO
   </a>

.. raw:: html

   <style>
   .iface-map-wrap{position:relative;display:inline-block;width:100%;margin:12px 0 20px 0;}
   .iface-map-wrap img{width:100%;display:block;border:1px solid #444;border-radius:4px;}
   .area-overlay{
     position:absolute;background:rgba(3,233,244,0.12);border:1.5px dashed #03e9f4;
     cursor:pointer;border-radius:3px;display:flex;align-items:center;
     justify-content:center;font-size:11px;font-weight:bold;color:#03e9f4;
     text-decoration:none;transition:background 0.2s;
   }
   .area-overlay:hover{background:rgba(3,233,244,0.28);}
   </style>

   <div class="iface-map-wrap">
    <img src="_static/screenshots/MotionControl.png" alt="MotionControl 介面截圖">

    <!-- 現在策略顯示(各網頁共用) -->
    <a class="area-overlay" href="#strategy"
      style="top:7.5%;left:0.5%;width:9.5%;height:9%;color:#ffcc00"></a>

    <!-- 歷史紀錄與UndoRedo -->
    <a class="area-overlay" href="#motion-history"
      style="top:18%;left:0.5%;width:9.5%;height:10%;color:#ffcc00"></a>
    <a class="area-overlay" href="#motion-history"
      style="top:21.25%;left:10%;width:16.5%;height:3.5%;color:#ffcc00"></a>

    <!-- 檔案SaveRead -->
    <a class="area-overlay" href="#motion-file-ops"
      style="top:15%;left:10%;width:20.25%;height:3.5%;color:#ffcc00"></a>

    <!-- SaveRead Stand與Lock -->
    <a class="area-overlay" href="#motion-stand-file-ops"
      style="top:15%;left:30.25%;width:9.5%;height:3.5%;color:#ffcc00"></a>
    <a class="area-overlay" href="#motion-stand-file-ops"
      style="top:18%;left:25.5%;width:12.75%;height:3.5%;color:#ffcc00"></a>

    <!-- Reset -->
    <a class="area-overlay" href="#motion-reset"
      style="top:15%;left:39.75%;width:5%;height:3.5%;color:#ffcc00"></a>

    <!-- Send Sector -->
    <a class="area-overlay" href="#motion-exec-ops"
      style="top:18%;left:10%;width:15.7%;height:3.5%;color:#ffcc00"></a>

    <!-- ExecuteStand -->
    <a class="area-overlay" href="#motion-execute"
      style="top:18%;left:38.25%;width:9.7%;height:3.5%;color:#ffcc00"></a>

    <!-- Multiple -->
    <a class="area-overlay" href="#motion-multiple"
      style="top:24.5%;left:10%;width:18.5%;height:3.5%;color:#ffcc00"></a>

    <!-- Merge -->
    <a class="area-overlay" href="#motion-merge"
      style="top:24.5%;left:29.5%;width:19%;height:3.5%;color:#ffcc00"></a>

    <!-- Address(各網頁共用) -->
    <a class="area-overlay" href="#shared-address"
      style="top:28%;left:10%;width:13.5%;height:3.75%;color:#ffcc00"></a>

    <!-- label/stand_label -->
    <a class="area-overlay" href="#motion-label"
      style="top:28%;left:23.5%;width:25.5%;height:6%;color:#ffcc00"></a>

    <!-- add/delete/reverse/copy -->
    <a class="area-overlay" href="#motion-table-ops"
      style="top:37%;left:1.5%;width:8%;height:8%;color:#ffcc00"></a>

    <!-- CheckSum -->
    <a class="area-overlay" href="#motion-checksum"
      style="top:46%;left:1.5%;width:7.5%;height:27%;color:#ffcc00"></a>

    <!-- 動作資料/模式切換 -->
    <a class="area-overlay" href="#motion-data-tables"
      style="top:34%;left:10%;width:39%;height:34%;color:#ffcc00"></a>

    <!-- ID上下調整 -->
    <a class="area-overlay" href="#motion-id-adjust"
      style="top:68%;left:10%;width:14%;height:4.25%;color:#ffcc00"></a>

    <!-- 總扭力開關 -->
    <a class="area-overlay" href="#motion-torque-all"
      style="top:17%;left:51%;width:17.75%;height:6.5%;color:#ffcc00"></a>

    <!-- 馬達編號與單顆控制 -->
    <a class="area-overlay" href="#motion-robot-visual"
      style="top:23%;left:51%;width:17.5%;height:42%;color:#ffcc00"></a>

    <!-- X/Pro Series -->
    <a class="area-overlay" href="#motion-xpro-conversion"
      style="top:65%;left:51%;width:17.5%;height:7.5%;color:#ffcc00"></a>

    <!-- 各部位扭力開關 -->
    <a class="area-overlay" href="#motion-part-torque"
      style="top:15.5%;left:71%;width:24.25%;height:2.75%;color:#ffcc00"></a>

    <!-- 單顆扭力控制 -->
    <a class="area-overlay" href="#motion-single-torque"
      style="top:18%;left:71%;width:26.5%;height:4.25%;color:#ffcc00"></a>

    <!-- 即時馬達刻度顯示 -->
    <a class="area-overlay" href="#motion-motor-detail"
      style="top:22%;left:71%;width:26.5%;height:44%;color:#ffcc00"></a>

    <!-- 創建新ID 複製全部馬達刻度 -->
    <a class="area-overlay" href="#motion-create-id"
      style="top:67.5%;left:75.75%;width:8.5%;height:4%;color:#ffcc00"></a>

    <!-- 複製指定馬達至指定ID -->
    <a class="area-overlay" href="#motion-copy-motor"
      style="top:67.5%;left:84.4%;width:8.5%;height:4%;color:#ffcc00"></a>
    <a class="area-overlay" href="#motion-copy-motor"
      style="top:73%;left:71.25%;width:26.5%;height:26%;color:#ffcc00"></a>
   </div>

.. _motion-history:

History 面板與 Undo/Redo
^^^^^^^^^^^^^^^^^^^^^^^^^

左側 History 面板記錄所有對表格的操作，支援多步(20步)撤銷與重做：

- ``↩ Undo``：撤銷最後一次操作，恢復前一個狀態。
- ``↪ Redo``：重新執行被撤銷的操作。
- ``Clear History``：清除全部操作記錄，不影響目前表格內容。

.. note::

   切換策略或存檔會清除所有紀錄

.. _motion-file-ops:

檔案操作（Save / Read）
^^^^^^^^^^^^^^^^^^^^^^^^

在檔名輸入框填入名稱後操作：

- **Save**：彈出確認對話框，確認後將以指定檔名儲存到目前策略Parameter資料夾內。
- **Read**：讀取目前策略Parameter資料夾內指定檔名檔案。

.. note::

   Save 前會清除 :ref:`motion-history` 記錄。

.. _motion-stand-file-ops:

Save Stand / Read Stand / Locked
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. danger::
   專題生請勿擅自使用此功能

站立姿勢（stand）的獨立儲存/讀取，檔名固定為 ``stand``，不受檔名輸入框影響：

- **SaveStand**：將29.ini內的資料複製至stand.ini，請在使用前先send Sector 29。需先取消勾選 ``Locked Stand`` 才能執行，避免誤覆蓋。
- **ReadStand**：載入 stand.ini 的姿勢資料。
- ``Locked Stand`` 勾選框：勾選時禁止 SaveStand/ReadStand，預設勾選作為保護。
- ``Locked 29`` 勾選框：勾選時禁止 Send 使用 Sector 29，避免誤改站姿。

.. note::

  * ReadStand會清除目前網頁上所有資料並讀取Stand.ini，使用前請確保已存檔或另開新頁。
  * 操作順序：取消勾選 ``Locked Stand`` → ``ReadStand`` → 調整Absolute ID29內的數值 → 取消勾選 ``Locked 29`` → send Sector 29 → execute確認調整結果 → ``SaveStand`` → 測試stand是否與Sector 29動作一致。

.. _motion-reset:

Reset
^^^^^

重新啟用所有操作按鈕。

執行任何動作卡住或無回應時可 ``Reset`` 恢復。

.. _motion-exec-ops:

Send / Sector
^^^^^^^^^^^^^

指定 ID 與 Sector 後送出動作：

- **ID**：要送出的動作序號（網頁中的號碼）。
- **Sector**：動作 Sector 編號（程式/檔案中的號碼，Sector 29 預設被 :ref:`motion-stand-file-ops` 的 Locked 29 保護）。
- **Send**：將現在ID(網頁)的內容send到Sector(檔案)。

.. note::
  需要釐清ID與Sector的差異，ID是網頁上的值，Sector是現在系統檔案內的值，兩者不一定相等。

.. _motion-execute:

execute / stand
^^^^^^^^^^^^^^^

- **execute**：先驗證輸入的Sector號碼檔案是否存在後執行動作。
- **stand**：執行stand.ini內的動作，是共同預設站姿。

.. note::
  execute只會讀取現在Sector輸入的值，ID可不輸入

.. _motion-multiple:

Multiple
^^^^^^^^

將指定 ID 列的所有馬達數值乘以倍數：

- **ID**：要縮放的動作 ID。
- **multipy × times**：乘以幾倍（支援小數）。

.. warning::

   僅在 RelativePosition / RelativeSpeed 有效，MotionList 與 Absolute 不支援。

.. _motion-merge:

Merge
^^^^^

將 ID1 的馬達數值加到 ID2 上，並刪除 ID1 列：

- **ID1 merge to ID2**：兩組相對位移合併為一個 ID2 動作。
- RelativeSpeed 的對應列也會同步處理（非零值保留，零值覆蓋為 20）。

.. note::
  執行後會只剩ID2，ID1會刪除並無法復原，請謹慎使用。

.. warning::

  僅在 RelativePosition / RelativeSpeed 檢視下有效，MotionList 與 Absolute 不支援。

.. _motion-table-ops:

表格列操作（Add / Delete / Reverse / Copy）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

對目前表格檢視的列進行操作，所有操作都會被記錄到 :ref:`motion-history`：

.. list-table::
   :header-rows: 1

   * - 按鈕
     - 說明
   * - ``Add``
     - 在目前表格新增一列空白/預設資料列。
   * - ``Delete : ID``
     - 刪除指定 ID 的列（Position/Speed 同步刪除）。
   * - ``Reverse : ID``
     - 將指定 ID 列的所有馬達數值正負號反轉（僅限 RelativePosition）。
   * - ``Copy : ID``
     - 複製指定 ID 的列，新增一個內容相同的列。

.. _motion-checksum:

CheckSum
^^^^^^^^

輸入 ID 後顯示該列所有馬達（M1–M27）各自的數值總和，主要用於檢查MotionList。
結果顯示在下方 CheckSum 欄位。

.. _motion-label:

label / stand_label
^^^^^^^^^^^^^^^^^^^^

介面頂部的狀態訊息列，顯示最近一次操作的結果：

- ``label``：一般操作回饋，如 ``"Add is successful !!"``、``"Sector 29 is Locked"``。
- ``stand_label``：Stand 相關操作的訊息。

.. _motion-data-tables:

動作資料表格（分頁）
^^^^^^^^^^^^^^^^^^^^^

以 Radio button 切換五種資料檢視，各檢視共用同一組 ID：

.. list-table::
   :header-rows: 1

   * - 分頁
     - 說明
   * - MotionList
     - 動作串（ID、Name、A1/D1 … A8/D8 動作/延遲參數）
   * - RelativePosition
     - 各馬達相對目前刻度的位移量（正/負刻度差）
   * - RelativeSpeed
     - 各馬達執行相對位移的速度
   * - AbsolutePosition
     - 各馬達的絕對刻度
   * - AbsoluteSpeed
     - 各馬達執行絕對位置的速度

.. note::
  絕對刻度內輸入-1會跳過該馬達並維持當前刻度

.. _motion-id-adjust:

ID 上下調整
^^^^^^^^^^^

``Move Up`` / ``Move Down`` 搭配 ID 輸入框，將指定 ID 的列在目前表格中向上或向下移動一格，調整ID的顯示順序。

.. _motion-torque-all:

總扭力開關
^^^^^^^^^^

``扭力關閉`` / ``扭力已開啟`` 切換按鈕：

- 向所有馬達發送扭力開/關封包。
- 同步更新 :ref:`motion-robot-visual` 圖示的顏色狀態。

.. warning::

   關閉全體扭力後機器人身體會洩力，需有人在旁扶持。

.. _motion-robot-visual:

馬達圖示控制
^^^^^^^^^^^^^

機器人正面圖示（大人型），各馬達編號按鈕疊加在對應關節位置，點擊可切換單顆馬達扭力。
按鈕綠色代表扭力 ON，紅色代表扭力 OFF。

.. note::
  重開網頁時不會同步更新，請重新觸發總扭力開關來確保狀態正確

馬達配置（共 29 顆）：

.. list-table::
   :header-rows: 1

   * - 部位
     - ID
   * - 左手
     - 1–7
   * - 右手
     - 8–14
   * - 腰
     - 15
   * - 左腿
     - 16–21
   * - 右腿
     - 22–27
   * - 頭部
     - 28–29

.. _motion-xpro-conversion:

X_Series / Pro_Series 換算面板
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

頭部馬達同時存在 X 系列與 Pro 系列，兩種系列刻度不相容。介面提供獨立的 Position 與 Speed
欄位，輸入後可分別送出換算指令，避免直接混用刻度導致問題。

.. _motion-part-torque:

各部位扭力開關
^^^^^^^^^^^^^^

依部位批次切換扭力的群組按鈕：``left_hand``、``right_hand``、``waist``、``left_leg``、``right_leg``。

- 點擊 → 該部位所有馬達 **洩力（OFF）**，方便手動調整關節角度。
- 再次點擊 → 恢復 **鎖死（ON）**。
- 同步更新 :ref:`motion-robot-visual` 圖示狀態。

.. _motion-single-torque:

單顆扭力控制
^^^^^^^^^^^^^

輸入馬達 ID（1–29），點擊 ``開啟 (ON)`` 或 ``關閉 (OFF)`` 控制單顆馬達扭力，
同步更新 :ref:`motion-robot-visual` 圖示狀態。

.. _motion-motor-detail:

即時馬達刻度顯示
^^^^^^^^^^^^^^^^^

訂閱 ``/joint_states`` topic，即時顯示所有馬達（ID 1–27）目前的絕對刻度。
可用於確認機器人目前姿勢，配合 :ref:`motion-create-id` 擷取當前姿勢。

.. _motion-create-id:

創建新 ID（複製全部馬達刻度）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Copy To Absolute`` 按鈕：

1. 自動切換至 AbsolutePosition 檢視。
2. 新增一列空白資料列。
3. 將 :ref:`motion-motor-detail` 的即時刻度值填入該列（ID 1–27），速度預設 50。

用於快速將機器人目前姿勢記錄為一個 Absolute 動作。

.. _motion-copy-motor:

複製指定馬達至指定 ID
^^^^^^^^^^^^^^^^^^^^^^

``Copy Selected →`` 按鈕展開選擇面板：

1. 勾選要複製的馬達（可按部位群組全選/全不選）。
2. 輸入目標 Row ID（AbsolutePosition 表格中已存在的 ID）。
3. 點擊 ``Execute``：將勾選馬達的即時刻度值覆寫到目標列。

.. note::

   來源為 :ref:`motion-motor-detail` 的即時數值；目標列必須已存在於 AbsolutePosition 表格中。

----

.. _interface-walking:

WalkingInterface
---------------------------------

.. raw:: html

   <a href="_static/Walking_demo.html" target="_blank"
      style="background:#03e9f4;color:#000;padding:5px 14px;border-radius:4px;
             text-decoration:none;font-weight:bold;font-family:monospace;">
     &#9654; 開啟 DEMO
   </a>

.. raw:: html

   <div class="iface-map-wrap">
     <img src="_static/screenshots/Walking.png" alt="Walking 介面截圖">

    <!-- 現在策略顯示(各網頁共用) -->
    <a class="area-overlay" href="#strategy"
      style="top:9%;left:0.5%;width:7.5%;height:13%;color:#ffcc00"></a>

    <!-- Address(各網頁共用) -->
    <a class="area-overlay" href="#shared-address"
      style="top:28.25%;left:44%;width:12%;height:5%;color:#ffcc00"></a>

    <!-- IMU -->
    <a class="area-overlay" href="#walking-imu"
      style="top:35%;left:27.25%;width:11%;height:53%;color:#ffcc00"></a>

    <!-- Walking Mode 調整 -->
    <a class="area-overlay" href="#walking-mode"
      style="top:35%;left:38.5%;width:21.25%;height:8.5%;color:#ffcc00"></a>

    <!-- 步長調整 -->
    <a class="area-overlay" href="#walking-step-size"
      style="top:43%;left:38.5%;width:21.25%;height:12.5%;color:#ffcc00"></a>

    <!-- Load -->
    <a class="area-overlay" href="#walking-load"
      style="top:55%;left:38.5%;width:10.625%;height:6.25%;color:#ffcc00"></a>

    <!-- Generate -->
    <a class="area-overlay" href="#walking-generate"
      style="top:55%;left:49.125%;width:10.625%;height:6.25%;color:#ffcc00"></a>

    <!-- 方向控制 -->
    <a class="area-overlay" href="#walking-direction"
      style="top:61%;left:38.5%;width:21.25%;height:14.25%;color:#ffcc00"></a>

    <!-- 步態參數 -->
    <a class="area-overlay" href="#walking-params"
      style="top:35%;left:60%;width:18.5%;height:25.5%;color:#ffcc00"></a>

    <!-- Save -->
    <a class="area-overlay" href="#walking-save"
      style="top:60.25%;left:60%;width:9.2%;height:6%;color:#ffcc00"></a>

    <!-- Send -->
    <a class="area-overlay" href="#walking-send"
      style="top:60.25%;left:69.25%;width:9.2%;height:6%;color:#ffcc00"></a>

   </div>

.. _walking-imu:

IMU 顯示
^^^^^^^^

左側 Sensor_Value 區塊顯示即時 IMU 數值。

.. list-table::
   :header-rows: 1

   * - 欄位
     - 說明
   * - Roll
     - 左右傾斜角（°）
   * - Pitch
     - 前後傾斜角（°）
   * - Yaw
     - 旋轉角（°）

**IMU Reset** 按鈕：將 IMU 數值重置歸零。

.. _walking-mode:

Walking Mode 調整
^^^^^^^^^^^^^^^^^

選擇步態模式的單選按鈕。切換模式後會自動呼叫 `Load`_ 重新載入對應參數。

.. list-table::
   :header-rows: 1

   * - 模式
     - statetype
     - 說明
   * - Continuous
     - 0
     - 平地連續行走（預設）
   * - LC_up
     - 1
     - 上樓梯步態
   * - LC_down
     - 2
     - 下樓梯步態

.. note::
   LC_up / LC_down 選項只有在目前策略為 ``lc`` 時才會顯示。

.. _walking-step-size:

步長調整
^^^^^^^^

輸入步態方向向量，數值單位為整數（內部以 1/100 cm 換算）。

.. list-table::
   :header-rows: 1

   * - 欄位
     - 說明
   * - X
     - 前進（正）／後退（負）
   * - Y
     - 左移（正）／右移（負）
   * - Theta
     - 左轉（正）／右轉（負），單位：步

.. note::
   可通過 :ref:`walking-direction` 即時控制數值

.. _walking-load:

Load
^^^^

從機器人後端讀取目前存檔的步態參數，填回畫面上的參數欄位。

.. note::
  在啟動步態前建議先按下 ``Load`` 來確保數值正確

.. _walking-generate:

Generate
^^^^^^^^

啟動或切換步行狀態，並發布當前方向指令。

- **連續模式**：作為開關，第一次點擊發送 ``1``\ （開始），再次點擊發送 ``0``\ （停止）。
- **單步模式（LC_up/LC_down）**：每次點擊均發送 ``1``，執行一次步行動作。

.. _walking-direction:

方向控制
^^^^^^^^

六個方向按鈕，每次點擊使對應值增減一個步長，並於連續模式下即時更新。

.. list-table::
   :header-rows: 1

   * - 按鈕
     - 鍵盤捷徑
     - 動作
   * - Forward
     - ``W``
     - X += 100
   * - Backward
     - ``S``
     - X -= 100
   * - Left Move
     - ``A``
     - Y += 100
   * - Right Move
     - ``D``
     - Y -= 100
   * - Left Turn
     - ``Q``
     - Theta += 1
   * - Right Turn
     - ``E``
     - Theta -= 1
   * - （Reset）
     - ``R``
     - X=Y=Theta=0

.. _walking-params:

步態參數
^^^^^^^^

參數面板依選取的 Walking Mode 顯示不同欄位組。

**Continuous Step（平地連續步態）：**

.. list-table::
   :header-rows: 1

   * - 參數
     - 說明
     - 預設值
   * - ``com_y_swing``
     - 質心起步偏移（cm）
     - 0
   * - ``width_size``
     - 腳寬（cm）
     - 0
   * - ``period_t``
     - 週期（20一單位）
     - 420
   * - ``T_DSP``
     - 雙支撐時間比例（0–1）
     - 0
   * - ``lift_height``
     - 抬腳高度（cm）
     - 5
   * - ``STAND_HEIGHT``
     - 站姿高度（cm）
     - 50
   * - ``COM_HEIGHT``
     - 質心高度（cm）
     - 40

**LCup / LCdown Step（樓梯步態）：** 額外包含 ``Board_High`` （階梯高度）、``Clearance`` （越障餘裕）參數。

預設值：``com_height`` = 40 cm、``stand_height`` = 50 cm、
``period_t`` = 420 ms、``lift_height`` = 5 cm。

.. _walking-save:

Save
^^^^

將畫面上的參數儲存至目前策略Parameter資料夾中的參數檔案。

- 平地模式：儲存至WalkingParameter.ini。
- LC 模式：儲存至UpStair.ini/DownStair.ini。

.. _walking-send:

Send
^^^^

將畫面上的參數即時套用至行走節點（不寫入檔案）。

.. note::
   Send 僅更新執行中的節點，不寫入檔案。若要永久保存請改用 `Save`_。

----

.. _interface-minidrc:

miniDRC
-------------------

.. raw:: html

   <a href="_static/miniDRC_demo.html" target="_blank"
      style="background:#03e9f4;color:#000;padding:5px 14px;border-radius:4px;
             text-decoration:none;font-weight:bold;font-family:monospace;">
     &#9654; 開啟 DEMO
   </a>

.. raw:: html

   <div class="iface-map-wrap">
     <img src="_static/screenshots/miniDRC.png" alt="miniDRC 介面截圖">

    <!-- 現在策略顯示(各網頁共用) -->
    <a class="area-overlay" href="#strategy"
      style="top:8%;left:0.5%;width:7.75%;height:10%;color:#ffcc00"></a>

    <!-- Address(各網頁共用) -->
    <a class="area-overlay" href="#shared-address"
      style="top:14.5%;left:43.3%;width:13%;height:4.25%;color:#ffcc00"></a>

    <!-- 相機畫面 -->
    <a class="area-overlay" href="#image-camera"
      style="top:22%;left:7.25%;width:62.5%;height:73.5%;color:#ffcc00"></a>

    <!-- 步態調整面板 -->
    <a class="area-overlay" href="#minidrc-walking-panel"
      style="top:20.5%;left:72.5%;width:6.75%;height:3.25%;color:#ffcc00"></a>

    <!-- 相機調整面板 -->
    <a class="area-overlay" href="#minidrc-camera-panel"
      style="top:20.5%;left:86.9%;width:6.75%;height:3.25%;color:#ffcc00"></a>

    <!-- 頭部馬達控制 -->
    <a class="area-overlay" href="#image-head-motor"
      style="top:23.5%;left:72.5%;width:19%;height:13%;color:#ffcc00"></a>

    <!-- 切換面板 -->
    <a class="area-overlay" href="#minidrc-panel"
      style="top:36.25%;left:72.5%;width:19%;height:17%;color:#ffcc00"></a>

    <!-- Walking Mode 調整 -->
    <a class="area-overlay" href="#walking-mode"
      style="top:58%;left:70.5%;width:17%;height:4%;color:#ffcc00"></a>

    <!-- 步長調整 -->
    <a class="area-overlay" href="#walking-step-size"
      style="top:61.75%;left:70.5%;width:17%;height:8.5%;color:#ffcc00"></a>

    <!-- Generate -->
    <a class="area-overlay" href="#walking-generate"
      style="top:70%;left:77.5%;width:6.5%;height:4.5%;color:#ffcc00"></a>

    <!-- IMU -->
    <a class="area-overlay" href="#walking-imu"
      style="top:74.25%;left:76%;width:8%;height:11.25%;color:#ffcc00"></a>

    <!-- 執行指定motion -->
    <a class="area-overlay" href="#minidrc-sector"
      style="top:85.5%;left:70.5%;width:23%;height:14%;color:#ffcc00"></a>
   </div>

將 WalkingInterface、ImageProcessInterface 與MotionControlInterface整合成單一精簡頁面。

.. _minidrc-walking-panel:

步態調整面板按鈕（Walking Setup）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

點擊後切換右側面板至步態參數調整畫面，功能與 WalkingInterface 的步態參數區塊相同。

.. seealso::
   :ref:`walking-params`、:ref:`walking-load`、:ref:`walking-send`

.. _minidrc-camera-panel:

相機調整面板按鈕（Camera Setup）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

點擊後切換右側面板至相機設定畫面，功能與 ImageProcessInterface 的相機設定區塊相同。

.. seealso::
   :ref:`image-camera-settings`

.. _minidrc-panel:

切換面板（三種狀態）
^^^^^^^^^^^^^^^^^^^^^

右側面板依點擊按鈕切換三種顯示模式：

.. tabs::

   .. tab:: ① 預設（頭部位置按鈕）

      點擊按鈕將兩顆頭部馬達轉至指定預設刻度

      .. image:: _static/screenshots/MiniDRCNormal.png
         :width: 100%

   .. tab:: ② Camera Setup

      相機參數調整，功能與 :ref:`image-camera-settings` 相同

      .. image:: _static/screenshots/miniDRCCamera.png
         :width: 100%

   .. tab:: ③ Walking Setup

      步態參數調整，功能與 WalkingInterface 的步態參數區塊相同。
      :ref:`walking-params`、:ref:`walking-load`、:ref:`walking-send`。

      .. image:: _static/screenshots/miniDRCWalk.png
         :width: 100%

.. _minidrc-sector:

執行指定 Motion
^^^^^^^^^^^^^^^^

頁面底部提供 11 個快捷動作按鈕（motion1–motion11），各自對應一個可編輯的 Sector 編號欄位。

點擊按鈕後，讀取對應輸入欄的 Sector 編號，呼叫 ``CheckSectorDRC(sector)`` 向機器人下達執行指令。

預設 Sector 對應表：

.. list-table::
   :header-rows: 1

   * - 按鈕
     - 預設 Sector
   * - motion1
     - 209
   * - motion2
     - 208
   * - motion3
     - 111
   * - motion4
     - 333
   * - motion5
     - 1111
   * - motion6
     - 2222
   * - motion7
     - 3333
   * - motion8
     - 103
   * - motion9
     - 871
   * - motion10
     - 102
   * - motion11
     - 872

另有 ``stand`` 與 ``notice`` 按鈕。

.. note::
  * Sector 編號可直接在欄位中修改，無需重新整理頁面，方便比賽現場臨時調整動作。
  * 預設編號可根據當年度負責人更改。

----

其他界面
--------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - 頁面
     - 說明
   * - ``NodeMonitor.html``
     - 節點狀態監控：即時顯示所有 ROS2 節點的 ``/rosout`` log，支援節點篩選與多視窗並排
   * - ``API.html``
     - ROS2 API 瀏覽器：查看所有可用 topic / service / action
