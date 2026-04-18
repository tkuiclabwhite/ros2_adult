#!/bin/bash

# 進入程式碼所在目錄 (避免在錯的地方執行)
cd "$(dirname "$0")"

# 1. 確保身分正確
git config user.name "tkuiclabwhite"
git config user.email "tkuiclabwhite@gmail.com"

# 2. 暫存並提交
git add .
current_date=$(date +"%Y-%m-%d %H:%M")
git commit -m "Robot Update: $current_date"

# 3. 上傳
# 確保 SSH Agent 知道這把金鑰 (若沒設 config 才需要這行)
# ssh-add ~/.ssh/ros2_adult 2>/dev/null 
git push origin main

echo "-------------------------------"
echo "上傳成功！日期: $current_date"
echo "-------------------------------"
