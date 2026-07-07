熱點設定
========

一般熱點設定
------------

步驟 1：確認 WiFi 介面名稱
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   nmcli device status

介面名稱：``wlP1p1s0``\ 。

步驟 2：建立熱點
^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo nmcli device wifi hotspot ifname wlP1p1s0 ssid "TKU_TOWEN" password "humanoidmaster"

步驟 3：設定固定 IP
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo nmcli connection modify Hotspot ipv4.addresses 10.10.10.10/24

步驟 4：套用設定
^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo nmcli connection down Hotspot
   sudo nmcli connection up Hotspot

快捷指令
^^^^^^^^

``hotspotup``\ / ``hotspotdown`` 別名已設定於 :doc:`bashrc_setup`\ ，套用後可直接使用：

.. code-block:: bash

   hotspotdown                    # 關閉熱點
   hotspotup                      # 開啟熱點
   nmcli connection show --active # 確認熱點是否正在運作

開機自動開啟
------------

.. note::

   需先完成上方「一般熱點設定」各步驟，確保 ``Hotspot`` profile 已建立完畢。
   自動腳本只負責開機時觸發 profile，不會重新設定加密方式。

步驟 1：建立 auto_hotspot.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   nano ~/auto_hotspot.sh

填入以下內容：

.. code-block:: bash

   #!/bin/bash
   # auto_hotspot.sh
   # 開機後等 60 秒，判斷有線網卡狀態，分成三種情況：
   #   1. 有 LOWER_UP（插線且協商成功）→ 進入穩定監看，確保不被熱點動作干擾，穩定後才建熱點
   #   2. 沒有 LOWER_UP 但有 UP + NO-CARRIER（網卡驅動正常，只是沒插線）→ 不需要救援，直接建熱點
   #   3. 沒有 UP（網卡驅動/PHY 初始化失敗，例如 Aquantia PHY 卡死）→ 嘗試 ip link down/up + ethtool 救援

   WIFI_INTERFACE="wlP1p1s0"
   WIRED_INTERFACE="eno1"
   HOTSPOT_SSID="TKU_TOWEN"
   HOTSPOT_PASSWORD="humanoidmaster"
   HOTSPOT_IP="10.10.10.10/24"

   INITIAL_WAIT=60      # 開機後先等幾秒，讓系統完全啟動
   STABLE_REQUIRED=30   # 插線且協商成功時，需要連續穩定幾秒才開熱點
   RESCUE_AFTER=30      # 網卡開不起來時，超過幾秒就嘗試手動重置
   CHECK_INTERVAL=1     # 每幾秒檢查一次

   LOG_FILE="/var/log/auto_hotspot.log"

   check_wired_state() {
       local link_info
       link_info=$(ip link show "$WIRED_INTERFACE")

       if echo "$link_info" | grep -q "LOWER_UP"; then
           echo "connected"
       elif echo "$link_info" | grep -qE "<[^>]*\bUP\b[^>]*>" && echo "$link_info" | grep -q "NO-CARRIER"; then
           echo "no_cable"
       else
           echo "not_ready"
       fi
   }

   echo "$(date): 開機後等待 ${INITIAL_WAIT} 秒..." >> "$LOG_FILE"
   sleep "$INITIAL_WAIT"

   STATE=$(check_wired_state)
   echo "$(date): ${WIRED_INTERFACE} 初始狀態判斷為: ${STATE}" >> "$LOG_FILE"

   if [ "$STATE" = "no_cable" ]; then
       echo "$(date): ${WIRED_INTERFACE} 網卡正常但未插線，跳過保護等待，直接建立熱點" >> "$LOG_FILE"
   else
       echo "$(date): 開始監看 ${WIRED_INTERFACE} 狀態..." >> "$LOG_FILE"

       stable_count=0
       rescue_timer=0

       while true; do
           STATE=$(check_wired_state)

           case "$STATE" in
               connected)
                   stable_count=$((stable_count + 1))
                   rescue_timer=0

                   if [ "$stable_count" -eq "$STABLE_REQUIRED" ]; then
                       echo "$(date): ${WIRED_INTERFACE} 已連續穩定 ${STABLE_REQUIRED} 秒，開始建立熱點" >> "$LOG_FILE"
                       break
                   fi
                   ;;

               no_cable)
                   echo "$(date): ${WIRED_INTERFACE} 監看中偵測到線已拔除（網卡本身正常），改為直接建立熱點" >> "$LOG_FILE"
                   break
                   ;;

               not_ready)
                   if [ "$stable_count" -gt 0 ]; then
                       echo "$(date): ${WIRED_INTERFACE} 在穩定 ${stable_count} 秒後異常，重置計數器" >> "$LOG_FILE"
                   fi
                   stable_count=0
                   rescue_timer=$((rescue_timer + 1))

                   if [ "$rescue_timer" -ge "$RESCUE_AFTER" ]; then
                       echo "$(date): ${WIRED_INTERFACE} 已 ${rescue_timer} 秒未開成功，嘗試手動重置..." >> "$LOG_FILE"
                       ip link set "$WIRED_INTERFACE" down 2>/dev/null
                       sleep 2
                       ip link set "$WIRED_INTERFACE" up 2>/dev/null
                       sleep 3

                       NEW_STATE=$(check_wired_state)
                       if [ "$NEW_STATE" = "not_ready" ]; then
                           ethtool -s "$WIRED_INTERFACE" speed 1000 duplex full autoneg on 2>/dev/null
                           echo "$(date): 已嘗試 ethtool 救援" >> "$LOG_FILE"
                       else
                           echo "$(date): ip link 重置成功，${WIRED_INTERFACE} 狀態變為 ${NEW_STATE}" >> "$LOG_FILE"
                       fi
                       rescue_timer=0
                   fi
                   ;;
           esac

           sleep "$CHECK_INTERVAL"
       done
   fi

   nmcli connection up Hotspot

   if [ $? -eq 0 ]; then
       echo "$(date): 熱點啟動成功，SSID=$HOTSPOT_SSID, IP=$HOTSPOT_IP" >> "$LOG_FILE"
   else
       echo "$(date): 熱點啟動失敗" >> "$LOG_FILE"
   fi

賦予執行權限：

.. code-block:: bash

   chmod +x ~/auto_hotspot.sh

步驟 2：建立 systemd service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo nano /etc/systemd/system/auto-hotspot.service

填入以下內容：

.. code-block:: text

   [Unit]
   Description=Auto hotspot after boot with wired interface detection
   After=NetworkManager.service
   Wants=NetworkManager.service

   [Service]
   Type=simple
   ExecStart=/home/iclab/auto_hotspot.sh
   RemainAfterExit=yes

   [Install]
   WantedBy=multi-user.target

步驟 3：啟用服務
^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo systemctl daemon-reload
   sudo systemctl enable auto-hotspot.service

步驟 4：重開機驗證
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo reboot

重開機後：

.. code-block:: bash

   tail -f /var/log/auto_hotspot.log
   nmcli connection show --active
