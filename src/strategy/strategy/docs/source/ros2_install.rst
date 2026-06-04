ROS2 安裝（Humble）
====================

兩台機器人均使用 **ROS2 Humble**，安裝步驟相同。

設定 Locale
-----------

.. code-block:: bash

   locale   # 確認目前設定

   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8

   locale   # 驗證設定

加入 ROS2 套件庫
----------------

.. code-block:: bash

   sudo apt install software-properties-common
   sudo add-apt-repository universe

   sudo apt update && sudo apt install curl -y

   export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
   curl -L -o /tmp/ros2-apt-source.deb \
     "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
   sudo dpkg -i /tmp/ros2-apt-source.deb

安裝 ROS2 Humble Desktop
-------------------------

.. code-block:: bash

   sudo apt update
   sudo apt install ros-humble-desktop
