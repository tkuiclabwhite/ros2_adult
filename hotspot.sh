#!/bin/bash
# auto_hotspot.sh
# 開機時直接建立熱點，不等待、不判斷 WiFi 連線狀態

WIFI_INTERFACE="wlP1p1s0"
HOTSPOT_SSID="TKU_TOWEN"
HOTSPOT_PASSWORD="humanoidmaster"
HOTSPOT_IP="10.10.10.10/24"

LOG_FILE="/var/log/auto_hotspot.log"
echo "$(date): 開機，直接建立熱點" >> "$LOG_FILE"

# 如果之前已經有 Hotspot 這個 profile，先確保乾淨重建，避免設定殘留
nmcli connection delete Hotspot &> /dev/null

nmcli device wifi hotspot ifname "$WIFI_INTERFACE" ssid "$HOTSPOT_SSID" password "$HOTSPOT_PASSWORD"

if [ $? -eq 0 ]; then
    nmcli connection modify Hotspot ipv4.addresses "$HOTSPOT_IP"
    nmcli connection down Hotspot
    nmcli connection up Hotspot
    echo "$(date): 熱點建立成功，SSID=$HOTSPOT_SSID, IP=$HOTSPOT_IP" >> "$LOG_FILE"
else
    echo "$(date): 熱點建立失敗，請檢查錯誤訊息" >> "$LOG_FILE"
fi
