# ZED 相機內建 IMU 的姿態與運動資料（由 zed_imu_node 轉換自 sensor_msgs/Imu）
#
# roll/pitch/yaw   : 套用歸零後的相對角度
# abs_*            : 未歸零的絕對角度；roll/pitch 以重力為基準，yaw 無絕對基準會漂移
# angular_velocity : [x, y, z]，原生 rad/s 已轉為 deg/s
# linear_acceleration: [x, y, z]，含重力，靜止水平時 z 約 9.81
# zeroed           : 是否已設定過零點
float64 roll
float64 pitch
float64 yaw
float64 abs_roll
float64 abs_pitch
float64 abs_yaw
float64[3] angular_velocity
float64[3] linear_acceleration
bool zeroed
