Terminal 指令
=============

開啟主系統
----------

.. code-block:: bash

   cd ~/ros2_adult
   source install/setup.bash
   ros2 launch usb_cam camera.launch.py

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
