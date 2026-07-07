GitHub 設定
===========

GitHub 初始化
-------------

前置步驟：SSH 金鑰設定
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSH 金鑰設定完成後，直接 clone 即可，不需要 ``git init``\ 。

.. code-block:: bash

   cd ~
   git clone git@github.com:tkuiclabwhite/ros2_adult.git

git_sync.sh
-----------

``git_sync.sh`` 放在專案根目錄，支援以暱稱或路徑指定要上傳的範圍，預設上傳全部。

用法：

.. code-block:: bash

   gitpush           # 上傳全部
   gitpush strategy  # 只上傳 strategy package
   gitpush obs       # 只上傳 obs 策略

.. code-block:: bash

   nano ~/ros2_adult/git_sync.sh

填入以下內容：

.. code-block:: bash

   #!/bin/bash
   cd "$(dirname "$0")"

   # --- 1. 設定身分 ---
   git config user.name "tkuiclabwhite"
   git config user.email "tkuiclabwhite@gmail.com"

   BRANCH="main"

   # --- 2. 定義暱稱映射表 ---
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

   if [ -z "$INPUT" ]; then
       TARGET="."
   elif [[ -n "${NICKNAMES[$INPUT]}" ]]; then
       TARGET="${NICKNAMES[$INPUT]}"
   else
       TARGET="$INPUT"
   fi

   echo "📂 目標路徑：$TARGET"

   if [ ! -d "$TARGET" ] && [ "$TARGET" != "." ]; then
       echo "❌ 錯誤：找不到路徑 '$TARGET'，請檢查暱稱或路徑是否正確。"
       exit 1
   fi

   # --- 4. 檢查網路是否能連到這個 repo 的 remote ---
   echo "🌐 檢查與 GitHub 的連線..."
   if ! git ls-remote origin "$BRANCH" &> /dev/null; then
       echo "❌ 無法連接到 GitHub remote (origin)。"
       echo "   可能原因：未連接 WiFi、WiFi 無法上網、或 DNS/防火牆問題。"
       echo "   請確認網路後再執行一次。"
       exit 1
   fi
   echo "✅ 連線正常"

   # --- 5. 加入變動並視情況 commit ---
   git add "$TARGET"

   if git diff-index --quiet HEAD --; then
       echo "ℹ️  沒有偵測到新變動。"
   else
       current_date=$(date +"%Y-%m-%d %H:%M")
       git commit -m "Update ($TARGET): $current_date"
   fi

   # --- 6. 跟遠端比對，看本地是否領先 ---
   git fetch origin "$BRANCH" --quiet

   LOCAL=$(git rev-parse HEAD)
   REMOTE=$(git rev-parse "origin/$BRANCH")

   if [ "$LOCAL" = "$REMOTE" ]; then
       echo "✅ 本地與遠端已同步，沒有需要上傳的內容。"
       exit 0
   fi

   # --- 7. 執行 push 並檢查結果 ---
   git push origin "$BRANCH"
   PUSH_STATUS=$?

   echo "-------------------------------"
   if [ $PUSH_STATUS -eq 0 ]; then
       echo "✅ 上傳成功！[push package: ${INPUT:-all}] -> $TARGET"
       echo "✅ 日期: $(date +"%Y-%m-%d %H:%M")"
   else
       echo "❌ 上傳失敗！git push 回傳錯誤，請檢查上方訊息。"
       echo "（commit 已存在本機，下次有網路時重跑這支腳本即可補推）"
       exit 1
   fi
   echo "-------------------------------"

賦予執行權限：

.. code-block:: bash

   chmod +x ~/ros2_adult/git_sync.sh
