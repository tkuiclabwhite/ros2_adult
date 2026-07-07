WiFi 管理
=========

使用 ``wifi_manager.py`` 提供互動式 WiFi 管理介面。

安裝依賴套件
------------

.. code-block:: bash

   pip install textual --break-system-packages

sudoers 設定（讓 nmcli 免密碼執行）
-------------------------------------

.. code-block:: bash

   sudo visudo -f /etc/sudoers.d/iclab-nmcli

填入以下內容並存檔：

.. code-block:: text

   iclab ALL=(ALL) NOPASSWD: /usr/bin/nmcli

確認 nmcli 路徑正確：

.. code-block:: bash

   which nmcli

若輸出不是 ``/usr/bin/nmcli``\ ，需回去修改 sudoers 成實際路徑。

檔案部署
--------

.. code-block:: bash

   chmod +x /home/iclab/ros2_adult/wifi_manager.py

tmux / byobu 滑鼠設定
----------------------

.. code-block:: bash

   echo "set -g mouse on" >> ~/.byobu/keybindings.tmux
   tmux source-file ~/.byobu/keybindings.tmux
   echo "set -g mouse on" >> ~/.tmux.conf
   tmux source-file ~/.tmux.conf

驗證
----

.. code-block:: bash

   source ~/.bashrc
   wifi

畫面正常顯示、按 ``r`` 重新掃描不需要輸入密碼，即代表安裝成功。
