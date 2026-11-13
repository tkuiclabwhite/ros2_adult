Terminal 指令
=============

機器人 IP 與 SSH
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 12 24 24

   * - 機器人
     - 型號
     - 有線 IP
     - 無線 IP
   * - 大人型主機
     - ADULT
     - ``172.17.121.10``
     - ``10.10.10.10``
   * - 大人型測試機
     - ADULT
     - ``172.17.121.11``
     - （不使用）
   * - KID
     - KID
     - ``192.168.1.58``
     - ``192.168.10.58``
   * - 小黑
     - KID
     - ``192.168.1.59``
     - ``192.168.10.59``

使用者名稱都是 ``iclab``\ ：

.. code-block:: bash

   ssh iclab@172.17.121.10      # 大人型主機
   ssh iclab@172.17.121.11      # 大人型測試機

大人型測試機（.11）是主機（.10）的鏡像，兩台都是 Jetson AGX Orin，
測試完成後才同步到 .10。

要跑會開視窗的程式（例如 LinkerHand 官方的 PyQt GUI）要帶 ``-X``\ ：

.. code-block:: bash

   ssh -X iclab@172.17.121.10

開啟主系統
----------

.. code-block:: bash

   cd ~/ros2_adult
   source install/setup.bash
   ros2 launch usb_cam camera.launch.py

啟動參數
~~~~~~~~

``camera.launch.py`` 吃幾個參數，不帶就是用預設值。

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - 參數
     - 可用值
     - 說明
   * - ``camera_source``
     - ``zed``\ （預設）/ ``usb``
     - 相機來源。``usb`` 時不啟動深度、疊合與 ZED 參數橋接那幾個節點
   * - ``imu_source``
     - ``xsens``\ （預設）/ ``arduino``
     - 機身 IMU。兩者都發 ``/package/sensorpackage``\ ，同時開會互相蓋掉，所以二擇一
   * - ``xsens_port``
     - ``/dev/ttyTHS1``\ （預設）
     - Xsens 的序列埠。接 40-pin 排針的 UART；改用 USB 轉接板時填
       ``/dev/serial/by-id/...``
   * - ``hands``
     - ``both``\ （預設）/ ``left`` / ``right`` / ``none``
     - 要啟動哪幾隻靈巧手

.. code-block:: bash

   ros2 launch usb_cam camera.launch.py                      # 全部用預設
   ros2 launch usb_cam camera.launch.py camera_source:=usb   # 用一般 USB 相機
   ros2 launch usb_cam camera.launch.py hands:=left          # 只接了左手
   ros2 launch usb_cam camera.launch.py imu_source:=arduino  # Xsens 壞掉時回退

開啟策略
--------

.. code-block:: bash

   cd ~/ros2_adult
   source install/setup.bash
   ros2 run strategy <策略名稱>

**可用策略：** ``ar``\ （射箭）、``bb``\ （籃球）、``sp``\ （競走）、``obs``\ （避障）、``sr``\ （斯巴達）、``wl``\ （舉重）、``mar``\ （馬拉松）、``pk``\ （罰踢）、``rc``\ （攀岩）

編譯
----

以下為 ``~/.bashrc`` 中定義的 alias，可直接在終端機輸入：

.. code-block:: bash

   cb     # 完整編譯所有 package
   cbp    # 只編譯 strategy package（速度較快）

``cb`` 展開為：

.. code-block:: bash

   colcon build --symlink-install
   source install/setup.bash

``cbp`` 展開為：

.. code-block:: bash

   colcon build --symlink-install --packages-select strategy
   source install/setup.bash

.. note::

   只有改策略的 Python 檔時，**不需重新編譯**，重開策略即可。

   有改到 motion 相關的東西則需要執行 ``cbp`` 編譯一次。
