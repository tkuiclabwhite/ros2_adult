套件安裝
========

安裝時**建議指定版本**（寫法見每一節）。不指定的話裝到新版出問題時，
會很難回推上次能動的是哪一版。

系統底層
--------

這幾項不是 apt / pip 裝的，要照各自的安裝程序走，換版前先確認其他東西跟得上。

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - 項目
     - 已驗證版本
     - 備註
   * - **JetPack**
     - ``6.2.1+b38``
     - ``dpkg -l nvidia-jetpack``
   * - **L4T**
     - ``R36.4.3``
     - ``cat /etc/nv_tegra_release``
   * - Ubuntu
     - ``22.04.5 LTS``
     - kernel ``5.15.148-tegra``
   * - **ZED SDK**
     - ``5.0.6``
     - ``/usr/local/zed/tools/ZED_Explorer --version``\ 。
       搭配 ``zed-ros2-wrapper`` 5.0.0（已 vendor 進 ``src/``\ ）
   * - **PEAK CAN 驅動**
     - ``Release_20240116_n``\ （chardev 8.17.0）
     - 靈巧手走 CAN 要裝。``modinfo pcan`` /
       ``cat /sys/class/pcan/version``\ 。
       驅動會自己裝 ``/etc/udev/rules.d/45-pcan.rules``\ ，不用手寫

rosdep 初始化
-------------

.. code-block:: bash

   sudo apt install python3-rosdep
   sudo rosdep init
   rosdep update
   rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"

apt 套件
--------

指定版本的寫法是 ``套件名=版本``\ ：

.. code-block:: bash

   sudo apt install ros-humble-rosbridge-server=2.0.4-1jammy.20260217.150402

.. warning::

   ROS 的 apt repo **會汰除舊版本**\ ，精確釘版過一段時間可能裝不到。
   建議只對「換新版曾經出過問題」的套件釘版本，其他照已驗證版本當參考就好。

.. code-block:: bash

   sudo apt install python3-colcon-common-extensions
   sudo apt install python3-tk
   sudo apt install byobu
   sudo apt install -y \
     ros-humble-rosbridge-server \
     ros-humble-cv-bridge \
     ros-humble-image-transport-plugins \
     ros-humble-camera-info-manager \
     ros-humble-ament-cmake-auto \
     ros-humble-web-video-server
   sudo apt install -y \
     cmake pkg-config \
     libavcodec-dev libavutil-dev libswscale-dev \
     libv4l-dev v4l-utils

.. list-table:: 已驗證版本
   :header-rows: 1
   :widths: 46 26 28

   * - 套件
     - 版本
     - 用途
   * - ``python3-colcon-common-extensions``
     - ``0.3.0-100``
     - 編譯
   * - ``python3-tk``
     - ``3.10.8-1~22.04``
     - matplotlib 的 Tk 後端
   * - ``byobu``
     - ``5.133-1``
     - 終端機分頁（方便用，不是相依套件）
   * - ``ros-humble-rosbridge-server``
     - ``2.0.4``
     - 網頁 ↔ ROS 的橋接（port 9090）
   * - ``ros-humble-cv-bridge``
     - ``3.2.1``
     - 影像與 ROS 訊息互轉
   * - ``ros-humble-image-transport-plugins``
     - ``2.5.4``
     - 影像壓縮傳輸
   * - ``ros-humble-camera-info-manager``
     - ``3.1.12``
     - usb_cam 用
   * - ``ros-humble-ament-cmake-auto``
     - ``1.3.12``
     - 編譯 usb_cam 用
   * - ``ros-humble-web-video-server``
     - ``3.0.0``
     - 網頁看即時影像

pip 套件
--------

.. important::

   **``torch`` 不可以直接 ``pip install torch``。**

   Jetson 要的是 NVIDIA 提供的 CUDA 版 wheel。實機上裝的是
   ``torch 2.8.0``\ （CUDA 12.6、wheel tag ``cp310-cp310-linux_aarch64``\ ）、
   ``torchvision 0.23.0``\ ，``torch.cuda.is_available()`` 為 True。
   PyPI 上的 aarch64 wheel **沒有 CUDA**\ ，裝錯的話 ``ultralytics``
   會掉到 CPU 跑，慢到不能用而且**不會有任何錯誤訊息**\ 。

   安裝來源要跟 JetPack 版本對應（JetPack 6.2 → CUDA 12.6）。
   確切的安裝指令請向系統負責人確認後補上這一節。

   裝完一定要驗：

   .. code-block:: bash

      python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
      # 預期：2.8.0 12.6 True

.. note::

   ``numpy`` 需限制在 2.0 以下，避免與其他套件不相容。

整段複製貼上即可，每一行都是獨立可執行的指令。

.. code-block:: bash

   pip install "numpy==1.26.4"
   pip install "opencv-python==4.8.1.78"
   pip install "Pillow==9.0.1"
   pip install "pyserial==3.5"
   pip install "PyYAML==6.0.2"
   pip install "pydantic==1.10.26"
   pip install "dynamixel-sdk==3.8.4"
   pip install "pupil-apriltags==1.0.4.post11"
   pip install "matplotlib==3.5.1"
   pip install "mediapipe==0.10.13"
   pip install "ultralytics==8.4.34"
   pip install "scipy==1.8.0"
   pip install "flask==3.1.3"

.. note::

   ``torch`` **不在上面那串裡** —— 它要用 NVIDIA 的 Jetson wheel，見上方的警告。

.. list-table:: 用途對照
   :header-rows: 1
   :widths: 28 72

   * - 套件
     - 誰在用
   * - ``numpy`` / ``opencv-python`` / ``Pillow``
     - 影像處理全線
   * - ``pyserial``
     - 馬達與感測器的序列埠通訊
   * - ``PyYAML``
     - 讀各種 ``.yaml`` 設定
   * - ``pydantic``
     - 設定驗證
   * - ``dynamixel-sdk``
     - Dynamixel 馬達
   * - ``pupil-apriltags``
     - AprilTag 偵測
   * - ``matplotlib``
     - 波形圖、除錯用圖表
   * - ``mediapipe``
     - 人體姿態
   * - ``ultralytics`` / ``torch``
     - YOLO 推論
   * - ``scipy``
     - 逆運動學（``strategy/bb/ik_solve.py``\ ）
   * - ``flask``
     - 熱點管理 API（``usb_cam/launch/hotspot_api.py``\ ）

GPIO
----

使用 Jetson AGX 的實體 GPIO，需額外安裝：

.. code-block:: bash

   sudo pip install Jetson.GPIO==2.1.7
   sudo groupadd -f -r gpio
   sudo usermod -a -G gpio $USER

機身 IMU (Xsens MTi)
--------------------

ROS 相依
~~~~~~~~

.. code-block:: bash

   sudo apt install -y ros-humble-nmea-msgs ros-humble-mavros-msgs

已驗證版本：``ros-humble-nmea-msgs`` ``2.0.0``\ 、``ros-humble-mavros-msgs`` ``2.14.0``\ 。

driver
~~~~~~

官方 driver **放在獨立的 workspace**\ ，不在 ``ros2_adult`` 裡：

.. code-block:: bash

   cd ~
   git clone --branch ros2 https://github.com/xsenssupport/Xsens_MTi_ROS_Driver_and_Ntrip_Client.git
   mv Xsens_MTi_ROS_Driver_and_Ntrip_Client xsens_ws
   cd ~/xsens_ws
   colcon build --symlink-install

.. note::

   實機上的目錄名稱是 ``~/xsens_ws``\ 。``camera.launch.py`` 用
   ``get_package_share_directory('xsens_mti_ros2_driver')`` 找它，
   所以建完要 ``source ~/xsens_ws/install/setup.bash``\ 。

參數設定
~~~~~~~~

編 ``~/xsens_ws/src/xsens_mti_ros2_driver/param/xsens_mti_node.yaml``\ ，
改成下面四個值（其餘保持原樣）：

.. code-block:: yaml

   scan_for_devices: false                      # 務必 false，見下方警告
   port: '/dev/ttyTHS1'                         # AGX 40-pin 排針 UART（pin 8 TX / pin 10 RX）
   baudrate: 115200                             # MTi-620 出廠鮑率
   port_config_hardware_flow_control: false     # GPIO UART 只接 TX/RX，沒接 RTS/CTS

.. danger::

   **``scan_for_devices`` 一定要是 ``false``。**
   全域掃描會對每一個 ``ttyUSB`` 輪流送 Xsens 的握手封包，
   **包括接著 Dynamixel 匯流排的 U2D2** —— 等於對馬達亂送資料。

改完驗一下沒打壞格式：

.. code-block:: bash

   cd ~/xsens_ws/src/xsens_mti_ros2_driver/param
   python3 -c "import yaml; yaml.safe_load(open('xsens_mti_node.yaml')); print('YAML valid')"
   grep -nE "scan_for_devices:|^\s+port:|^\s+baudrate:|flow_control" xsens_mti_node.yaml

40-pin UART
~~~~~~~~~~~

預設 ``/dev/ttyTHS1`` 被系統的除錯主控台佔用，要先關掉：

.. code-block:: bash

   sudo systemctl stop nvgetty
   sudo systemctl disable nvgetty
   sudo usermod -aG dialout $USER
   sudo reboot

重開後確認裝置存在：

.. code-block:: bash

   ls -l /dev/ttyTHS*
   # 預期看到 /dev/ttyTHS1（group 為 dialout）

.. note::

   **沒有 ``/dev/ttyTHS0``。** AGX Orin 上 40-pin 排針的 UART 是 ``ttyTHS1``\ ，
   填 ``ttyTHS0`` driver 會一直重試連不上。

   改用 USB 轉接板時才需要下面這條 udev 規則（走 GPIO UART 不需要）：

   .. code-block:: bash

      sudo tee /etc/udev/rules.d/99-ftdi-latency.rules << 'EOF'
      ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
      EOF
      sudo udevadm control --reload

靈巧手 (LinkerHand O6)
----------------------

PEAK USB-CAN 驅動
~~~~~~~~~~~~~~~~~

手走 CAN，需要 PEAK 的核心模組。**已驗證版本 8.17.0**\ ：

.. code-block:: bash

   sudo apt install -y build-essential libpopt-dev
   cd ~
   wget https://www.peak-system.com/fileadmin/media/linux/files/peak-linux-driver-8.17.0.tar.gz
   tar -xzvf peak-linux-driver-8.17.0.tar.gz
   cd peak-linux-driver-8.17.0
   make NET=NETDEV_SUPPORT
   sudo make install
   sudo modprobe pcan

.. important::

   **一定要 netdev 版。** 我們是用 ``ip link`` 操作 ``can2`` / ``can3`` 這種網路介面，
   不是 chardev 的 ``/dev/pcan*``\ 。裝完這樣驗：

   .. code-block:: bash

      modinfo pcan | grep -E "^version|^description"
      # 預期：version: Release_20240116_n
      #       description: Netdev driver for PEAK-System CAN interfaces

      ls /dev/pcan* 2>/dev/null
      # 預期：沒有東西（有的話是裝成 chardev 模式了）

驅動會自己裝 ``/etc/udev/rules.d/45-pcan.rules``\ ，不用手寫。

CAN 介面設定
~~~~~~~~~~~~

插上 USB-CAN 盒後確認介面有出現：

.. code-block:: bash

   ip -br link show type can

.. note::

   AGX 內建的 CAN 是 ``can0`` / ``can1``\ （沒接東西），
   **PEAK USB-CAN 會是 ``can2`` / ``can3``**\ 。
   換機器前先用上面那行確認編號，再用 ``left_can:=`` / ``right_can:=`` 覆蓋。

設定比特率並啟用（左右手各一條，比特率固定 1 Mbps）：

.. code-block:: bash

   sudo ip link set can2 type can bitrate 1000000
   sudo ip link set can2 up
   sudo ip link set can3 type can bitrate 1000000
   sudo ip link set can3 up

.. note::

   平常不用手動下這幾行 —— SDK 啟動時會自己用 ``setting.yaml`` 裡的
   ``PASSWORD`` 去 ``sudo ip link set ... up``\ 。
   但**那段失敗時是靜默的**\ （例外被吞掉），現象是「手完全不動但沒有任何錯誤訊息」。
   ``hands.launch.py`` 有加啟動檢查，CAN 沒 UP 會印警告。

Python 套件
~~~~~~~~~~~

.. code-block:: bash

   pip install python-can==4.6.1

.. note::

   * LinkerHand 的 SDK **沒有用 pip 版**\ ，是 vendor 在
     ``src/linkerhand-ros2-sdk``\ 。不要去 ``pip install linker-hand-sdk``\ 。
   * 走 RS485 才需要 ``pymodbus``\ 。我們走 CAN，實機上沒裝也完全正常。
   * 編譯 SDK **必須加** ``--symlink-install``\ ，否則 ``setting.yaml``
     與 ``O6_positions.yaml`` 不會被安裝，程式讀設定時會掛掉。

官方 GUI（選用）
~~~~~~~~~~~~~~~~

平常用網頁介面操作，這兩個只有要跑 LinkerHand 官方的 PyQt GUI
（``gui_control`` / ``graphic_display`` / ``pressure_diagram``\ ）才需要：

.. code-block:: bash

   pip install PyQt5==5.15.6
   pip install pyqtgraph==0.12.4

.. note::

   以上套件清單可能不完整，若執行時出現 ``ModuleNotFoundError``\ ，
   請依錯誤訊息補裝對應套件，並回報系統負責人更新本頁。
