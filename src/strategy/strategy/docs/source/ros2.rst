ROS2 內建輔助功能
=================

訊息輸出與畫面控制
------------------

.. py:method:: get_logger()

   獲取節點的日誌紀錄器。此功能繼承自 ``rclpy.node.Node``。

   * **所需套件**：``import rclpy``
   * **備註**：這是 ROS 2 最標準的訊息輸出方式，會自動附帶時間戳記與節點名稱。

   .. rubric:: 日誌層級詳解表

   .. list-table::
      :widths: 15 15 25 45
      :header-rows: 1

      * - 層級
        - 預設顏色
        - 顯示範例
        - 輸出時機
      * - **DEBUG**
        - 藍/灰色
        - [DEBUG] [node]:
        - 開發除錯用。預設不顯示。
      * - **INFO**
        - 綠/白色
        - [INFO] [node]:
        - **最常用**。回報機器人正常運作狀態。
      * - **WARN**
        - 黃色
        - [WARN] [node]:
        - 異常但暫不影響系統運作。
      * - **ERROR**
        - 紅色
        - [ERROR] [node]:
        - 功能失效，但程式尚未崩潰。
      * - **FATAL**
        - 底色紅
        - [FATAL] [node]:
        - 嚴重錯誤，系統即將停止運作。

.. py:method:: sys.stdout.write(string)

   將字串直接寫入標準輸出流。與 ``print()`` 不同，它不會自動換行，適合用於「原地更新」數據。

   * **所需套件**：``import sys``

.. py:method:: sys.stdout.flush()

   強制刷新輸出緩衝區。在 ROS 2 高頻迴圈中，若不呼叫此功能，輸出可能會因為緩衝機制而延遲顯示。

   * **所需套件**：``import sys``

   **範例: 實現「儀表板模式」原地更新數據**
      >>> import sys
      >>> # \r 回行首，\033[K 清除殘影
      >>> sys.stdout.write(f"\r目前馬達角度: {angle:.2f}\033[K")
      >>> sys.stdout.flush()

.. rubric:: 終端機畫面控制碼 (僅限 sys.stdout)

這些代碼用於控制游標位置與螢幕狀態，通常不建議放在 ``get_logger()`` 中使用。

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - 轉義碼
     - 功能說明
     - 建議用途
   * - ``\033[H\033[J``
     - 回到原點並清空全螢幕
     - 初始化或重繪儀表板
   * - ``\r``
     - 歸位 (Carriage Return)
     - 回到當前行首進行覆蓋
   * - ``\033[K``
     - 清除從游標到行尾的內容
     - 避免長短數字重疊殘留
   * - ``\033[?25l`` / ``\033[?25h``
     - 隱藏 / 顯示游標
     - 提升監控畫面美觀度

.. rubric:: 萬用 ANSI 顏色與樣式字典 (通用於 get_logger 與 stdout)

格式公式：``\033[模式;前景色;背景色m 內容 \033[0m``。**必須以 0m 結尾以重置樣式。**

**1. 文字樣式 (Styles)**
   * ``0``：重置 | ``1``：**粗體** | ``2``：微暗 | ``4``：底線 | ``5``：閃爍 | ``7``：反白

**2. 前景色與背景色對照表**

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - 顏色名稱
     - 前景 (標準)
     - 前景 (亮色)
     - 背景 (標準)
     - 背景 (亮色)
   * - **黑色**
     - 30
     - 90
     - 40
     - 100
   * - **紅色**
     - 31
     - 91
     - 41
     - 101
   * - **綠色**
     - 32
     - 92
     - 42
     - 102
   * - **黃色**
     - 33
     - 93
     - 43
     - 103
   * - **藍色**
     - 34
     - 94
     - 44
     - 104
   * - **洋紅**
     - 35
     - 95
     - 45
     - 105
   * - **青色**
     - 36
     - 96
     - 46
     - 106
   * - **白色**
     - 37
     - 97
     - 47
     - 107

**混合使用範例:**
   >>> # 加粗(1); 亮綠字(92); 紅底(41)
   >>> msg = "\033[1;92;41m [系統重啟] \033[0m"
   >>> self.get_logger().info(msg)
時間管理與任務排程
------------------

.. py:method:: get_clock().now()

   獲取當前系統時鐘的精確時間。此方法最適合用於「非阻塞式」的時間差計算。

   * **所需套件**：``import rclpy``

   **範例:**
      **如何實現「等待 1 秒」的邏輯判斷：**
      
      >>> # 紀錄起始時間
      >>> self.start_time = self.get_clock().now()
      >>> # 在策略迴圈中不斷檢查是否已過 1 秒
      >>> now = self.get_clock().now()
      >>> if (now - self.start_time).nanoseconds >= 1e9: 
      >>>     self.get_logger().info("已過 1 秒，執行下一步動作！")

.. py:method:: create_timer(timer_period_sec: float, callback: Callable)

   建立非同步定時器。

   * **所需套件**：``import rclpy``, ``from typing import Callable``

   **範例:**
      **如何實現「每 1 秒」執行一次任務：**

      >>> # 設定週期為 1.0 秒
      >>> self.timer = self.create_timer(1.0, self.my_callback)
      >>>
      >>> def my_callback(self):
      >>>     self.get_logger().info("計時器：1 秒時間到！")

.. py:method:: create_rate(frequency: float)

   建立固定頻率控制器。

   * **所需套件**：``import rclpy``

   **範例:**
      **如何實現「每 1 秒」循環一次的迴圈：**

      >>> # 設定頻率為 1 Hz (即每秒一次)
      >>> rate = self.create_rate(1.0)
      >>> while rclpy.ok():
      >>>     # ... 執行你的控制邏輯 ...
      >>>     rate.sleep()  # 自動補償運算時間，確保每次循環間隔剛好 1 秒

.. py:function:: time.sleep(seconds: float)

   強制暫停當前執行緒。

   * **所需套件**：``import time``

   .. danger::
      **警告**：嚴禁在 ROS 2 的回呼函式（Callback）中使用，這會導致節點失去反應！

   **範例:**
      **簡單強制的等待 1 秒：**

      >>> import time
      >>> # 直接讓當前程式碼暫停 1 秒
      >>> time.sleep(1.0)

.. rubric:: 時間管理工具對照表

.. list-table::
   :widths: 20 20 30 30
   :header-rows: 1

   * - 工具名稱
     - 運作類型
     - 「等待 1 秒」的實現方式
     - 系統安全性
   * - **get_clock**
     - 手動比對
     - 計算 ``now - start >= 1.0s``
     - **最高 (不受卡頓影響)**
   * - **create_timer**
     - 非同步事件
     - 設定週期為 ``1.0`` 秒觸發
     - 高 (ROS 2 內建標準)
   * - **create_rate**
     - 頻率控制
     - 設定頻率為 ``1.0`` Hz 循環
     - 中 (需搭配執行緒)
   * - **time.sleep**
     - 強制阻塞
     - 呼叫 ``time.sleep(1.0)``
     - **低 (會卡死節點回呼)**