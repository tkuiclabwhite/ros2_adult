~/.bashrc 設定
==============

在 ``~/.bashrc`` 末尾加入以下自定義設定：

.. code-block:: bash

   export PATH="$HOME/.local/bin:$PATH"

   # ROS 2 環境
   source /opt/ros/humble/setup.bash

   # 自動補完
   source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash

   # ROS Domain
   export ROS_DOMAIN_ID=1

   #限定接線設備連接
   export ROS_LOCALHOST_ONLY=1

   # 自定義快捷鍵
   alias cb='colcon build --symlink-install && source install/setup.bash'
   alias cbp='colcon build --symlink-install --packages-select strategy && source install/setup.bash'
   alias gitpush='bash ~/ros2_adult/git_sync.sh'
   alias wifi='python3 /home/iclab/ros2_adult/wifi_manager.py'
   alias hotspotup='sudo nmcli connection up Hotspot'
   alias hotspotdown='sudo nmcli connection down Hotspot'

套用設定：

.. code-block:: bash

   source ~/.bashrc

.. note::

   ``cb`` 與 ``cbp`` 會在當前目錄下執行，使用前請先 ``cd ~/ros2_adult``\ 。

   若有安裝 Miniconda，conda 初始化區段會由 ``conda init bash`` 自動加入 ``~/.bashrc``\ 。
