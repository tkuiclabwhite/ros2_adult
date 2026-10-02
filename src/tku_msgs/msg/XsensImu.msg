# Xsens MTi-620 的姿態與運動資料（由 xsens_bridge 轉換自 driver 的 filter/imu 系列 topic）
#
# roll/pitch/yaw      : 套用歸零後的相對角度，單位為度
# abs_*               : 未歸零的絕對角度；roll/pitch 以重力為基準，
#                       yaw 為 VRU 模式無磁力計校正，會隨時間漂移
# angular_velocity    : [x, y, z]，來自 /imu/angular_velocity，原生 rad/s 已轉為 deg/s
# free_acceleration   : [x, y, z]，來自 /filter/free_acceleration，
#                       已扣除重力，單位 m/s^2，可作為質心加速度的近似
# zeroed              : 是否已設定過零點
# rate_hz             : 實測 /filter/euler 的進來頻率，用來判斷頻寬是否吃滿
# alive               : watchdog，最近 watchdog_sec 秒內有收到資料才為 true
float64 roll
float64 pitch
float64 yaw
float64 abs_roll
float64 abs_pitch
float64 abs_yaw
float64[3] angular_velocity
float64[3] free_acceleration
bool zeroed
float64 rate_hz
bool alive
