機器人命令設定
==============

udev 規則
---------

設定 udev 規則可讓裝置在每次連接時都使用固定的設備名稱。

U2D2
^^^^

U2D2 是 Dynamixel 馬達的 USB 轉接器，共 3 顆，需依序號綁定固定名稱。

.. code-block:: bash

   sudo nano /etc/udev/rules.d/99-u2d2.rules

填入以下內容（序號依實際情況填寫）：

.. code-block:: text

   SUBSYSTEM=="tty", ATTRS{serial}=="<ADULT_U2D2_SERIAL_1>", SYMLINK+="U2D2_P1", GROUP="dialout", MODE="0660", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"
   SUBSYSTEM=="tty", ATTRS{serial}=="<ADULT_U2D2_SERIAL_2>", SYMLINK+="U2D2_P2", GROUP="dialout", MODE="0660", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"
   SUBSYSTEM=="tty", ATTRS{serial}=="<ADULT_U2D2_SERIAL_3>", SYMLINK+="U2D2_P3", GROUP="dialout", MODE="0660", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"

查詢 U2D2 序號（接上後執行）：

.. code-block:: bash

   udevadm info -a -n /dev/ttyUSB0 | grep serial
   udevadm info -a -n /dev/ttyUSB1 | grep serial
   udevadm info -a -n /dev/ttyUSB2 | grep serial

套用規則：

.. code-block:: bash

   sudo udevadm control --reload-rules && sudo udevadm trigger

~/.bashrc 設定
--------------

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

套用設定：

.. code-block:: bash

   source ~/.bashrc

.. note::

   ``cb`` 與 ``cbp`` 會在當前目錄下執行，使用前請先 ``cd ~/ros2_adult``\ 。

   若有安裝 Miniconda，conda 初始化區段會由 ``conda init bash`` 自動加入 ``~/.bashrc``\ 。

git_sync.sh
-----------

``git_sync.sh`` 放在專案根目錄，支援以暱稱或路徑指定要上傳的範圍，預設上傳全部。

用法：

.. code-block:: bash

   gitpush           # 上傳全部
   gitpush strategy  # 只上傳 strategy package
   gitpush obs       # 只上傳 obs 策略

.. code-block:: bash

   #!/bin/bash
   cd "$(dirname "$0")"

   # --- 1. 設定身分 ---
   git config user.name "tkuiclabwhite"
   git config user.email "tkuiclabwhite@gmail.com"

   # --- 2. 定義暱稱映射表 (在這邊增加你的縮寫) ---
   # 格式為： [暱稱]="實際完整路徑"
   declare -A NICKNAMES
   NICKNAMES=(
       ["ar"]="src/strategy/strategy/ar"
       ["bb"]="src/strategy/strategy/bb"
       ["mar"]="src/strategy/strategy/mar"
       ["obs"]="src/strategy/strategy/obs"
       ["pk"]="src/strategy/strategy/pk"
       ["rc"]="src/strategy/strategy/rc"
       ["sp"]="src/strategy/strategy/sp"
       ["sr"]="src/strategy/strategy/sr"
       ["wl"]="src/strategy/strategy/wl"
       ["strategy"]="src/strategy"
       ["image"]="src/imageprocess"
       ["motion"]="src/motionpackage"
       ["motor"]="src/motor_control"
       ["msgs"]="src/tku_msgs"
       ["usb_cam"]="src/usb_cam"
       ["walking"]="src/walking"
       ["all"]="."
   )

   # --- 3. 處理輸入參數 ---
   INPUT=$1

   # 如果沒輸入參數，預設上傳全部 (.)
   if [ -z "$INPUT" ]; then
       TARGET="."
   # 如果輸入的字在暱稱表裡有對應
   elif [[ -n "${NICKNAMES[$INPUT]}" ]]; then
       TARGET="${NICKNAMES[$INPUT]}"
   # 如果輸入的不是暱稱，就當作它是原始路徑
   else
       TARGET="$INPUT"
   fi

   # --- 4. 執行 Git 流程 ---
   echo ":open_file_folder: 目標路徑：$TARGET"

   # 檢查資料夾是否存在，防止打錯字
   if [ ! -d "$TARGET" ] && [ "$TARGET" != "." ]; then
       echo ":x: 錯誤：找不到路徑 '$TARGET'，請檢查暱稱或路徑是否正確。"
       exit 1
   fi

   git add "$TARGET"

   # 檢查是否有變動
   if git diff-index --quiet HEAD --; then
       echo ":information_source:  沒有偵測到變動，取消上傳。"
       exit 0
   fi

   current_date=$(date +"%Y-%m-%d %H:%M")
   git commit -m "Update ($TARGET): $current_date"

   git push origin main

   echo "-------------------------------"
   echo "上傳成功！[push package: ${INPUT:-all}] -> $TARGET"
   echo "上傳成功！日期: $current_date"
   echo "-------------------------------"

賦予執行權限：

.. code-block:: bash

   chmod +x ~/ros2_adult/git_sync.sh

GitHub 初始化
-------------

前置步驟：SSH 金鑰設定
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir -p ~/.ssh
   ssh-keygen -t ed25519 -C "tkuiclabwhite@gmail.com"
   # Enter file in which to save the key: /home/iclab/.ssh/ros2_adult
   # Enter passphrase: （直接按 Enter，不設密碼）

   ls -l ~/.ssh/ros2_adult*
   cat ~/.ssh/ros2_adult.pub

將 ``cat`` 輸出的公鑰完整複製，前往 **GitHub → Settings → SSH and GPG keys → New SSH key**，貼上後儲存。

設定 ``~/.ssh/config``\ ：

.. code-block:: bash

   nano ~/.ssh/config

填入以下內容：

.. code-block:: text

   Host github.com
       HostName github.com
       User git
       IdentityFile ~/.ssh/ros2_adult

測試連線（出現 ``Hi tkuiclabwhite!`` 即成功）：

.. code-block:: bash

   ssh -T git@github.com

情況一：全新 repo（GitHub 上還沒有）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

先在 **GitHub 網站建立空白倉庫** （不要勾選 Initialize this repository with a README）。

.. code-block:: bash

   cd ~/ros2_adult
   git init
   git config user.name "tkuiclabwhite"
   git config user.email "tkuiclabwhite@gmail.com"
   echo "build/" >> .gitignore
   echo "install/" >> .gitignore
   echo "log/" >> .gitignore
   git status
   git add .
   git commit -m "Initial commit: first push"
   git remote add origin git@github.com:tkuiclabwhite/ros2_adult.git
   git branch -M main
   git push -u origin main

情況二：已有 repo（換機器人，直接從 GitHub clone）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSH 金鑰設定完成後，直接 clone 即可，不需要 ``git init``\ 。

.. code-block:: bash

   cd ~
   git clone git@github.com:tkuiclabwhite/ros2_adult.git
