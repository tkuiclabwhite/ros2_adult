套件安裝
========

rosdep 初始化
-------------

.. code-block:: bash

   sudo apt install python3-rosdep
   sudo rosdep init
   rosdep update
   rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"

apt 套件
--------

.. code-block:: bash

   sudo apt install python3-colcon-common-extensions
   sudo apt install python3-pydantic
   sudo apt install byobu
   sudo apt install python3-tk
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

Intel RealSense
---------------

需先加入 Intel 官方 APT 來源再安裝：

.. code-block:: bash

   sudo mkdir -p /etc/apt/keyrings
   curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
     sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

   echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
   https://librealsense.intel.com/Debian/apt-repo \
   $(lsb_release -cs) main" | \
     sudo tee /etc/apt/sources.list.d/librealsense.list

   sudo apt update
   sudo apt install -y librealsense2-dev

pip 套件
--------

.. note::

   ``numpy`` 需限制版本在 2.0 以下，避免與其他套件不相容。

.. code-block:: bash

   pip install "numpy<2"
   pip install opencv-python
   pip install Pillow
   pip install pyserial
   pip install PyYAML
   pip install pydantic
   pip install dynamixel-sdk
   pip install pupil-apriltags
   pip install matplotlib
   pip install mediapipe
   pip install ultralytics
   pip install torch

GPIO（ADULT 專用）
------------------

ADULT 使用 Jetson AGX 的實體 GPIO，需額外安裝：

.. code-block:: bash

   sudo pip install Jetson.GPIO
   sudo groupadd -f -r gpio
   sudo usermod -a -G gpio $USER

.. note::

   以上套件清單可能不完整，若執行時出現 ``ModuleNotFoundError``，請依錯誤訊息補裝對應套件，並回報系統負責人。
