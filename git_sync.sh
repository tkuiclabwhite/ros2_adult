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
    ["bm"]="src/strategy/strategy/bm"
    ["lc"]="src/strategy/strategy/lc"
    ["mar"]="src/strategy/strategy/mar"
    ["obs"]="src/strategy/strategy/obs"
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
