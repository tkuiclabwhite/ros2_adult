# Parameter.py
# 只放「設定值/可調參數」，不要放步態迴圈的狀態

import math
from typing import Dict

# ---- IK計算的站姿 (實際上走路演算主要看這裡的9-21，但為了初始化，全身都要有) ----
STAND_GP: Dict[int, int] = {
    # --- 上半身 (ttyUSB0) ---
    1: 1069, 2: 1775,  3: -37,  4: 2290,  5: 3121,  6: 2102,  7: 2081,          # 左手 (請自行填入設定值)
    8: 3044, 9: 2303, 10: 1998, 11: 1797, 12: 997, 13: 2003, 14: 1919,          # 右手 (請自行填入設定值)
    
    # --- 下半身 (ttyUSB2) ---
    15: 2026,                                  # 腰

    16: 18448, 17: 1605, 18: 1309, 19: -24967, 20: 2159, 21: 1859,  # 左腳
    22: -15214, 23: -2813, 24: -709, 25: 24278, 26: 1911, 27: 2217,  # 右腳
    
    # --- 頭部/配件 (ttyUSB1) ---
    28: 2048, 29: 2048                             # 頭部 (請自行填入設定值)
}


DIR: Dict[int, int] = {
    16:  1,   # L_HIP_YAW
    17: -1 ,  # L_HIP_ROLL
    18:  1,   # L_HIP_PITCH
    19: -1,   # L_KNEE
    20:  1,   # L_ANKLE_PITCH
    21:  1,   # L_ANKLE_ROLL

    22: -1,   # R_HIP_YAW
    23: -1,   # R_HIP_ROLL
    24:  1,   # R_HIP_PITCH
    25:  1,   # R_KNEE
    26: -1,   # R_ANKLE_PITCH
    27: -1,   # R_ANKLE_ROLL
}

# ---- 時序參數 ----
period_t    = 420   # 單步週期 (ms)
sample_time = 20    # 取樣時間 (ms)
Tdsp        = 0.0   # 雙支撐比例: 0 <= Tdsp < 1

# ---- 幾何/物理 ----
COM_HEIGHT   = 40  # 質心高度 (cm)
STAND_HEIGHT = 50  # 站姿高度 (cm)
LENGTH_PELVIS= 19.8   # 骨盆寬 (cm)
G            = 981.0 # 重力 (cm/s^2)
Tc_          = math.sqrt(COM_HEIGHT / G)  # LIPM 時間常數

# ---- 步態形狀 ----
step_length  = 0      # x
shift_length = 0.0    # y
theta_       = 0.0    # theta
width_size   = 0    # 半步寬 (cm)
lift_height  = 5
com_y_swing  = 0      # 質心側擺幅度 (cm)
hip_roll     = 0
ankle_roll   = 0
SPEED_SCALE  = 1

# ---- 新增：樓梯與姿態補償參數 ----
walking_mode = 0     # 0:平地, 1:上樓, 2:下樓, 3:SR_Continuous
Board_High   = 0.0   # 階梯高度 (cm)
Clearance    = 3.0   # 越障安全餘裕 (cm)

# ---- SR_Continuous 自適應平衡參數 ----
imukp          = 0.1   # Pitch/Roll 誤差比例增益
imukd          = 0.01  # Pitch/Roll 誤差微分增益
target_pitch   = 0.0   # 目標俯仰角 (°)
target_roll    = 0.0   # 目標側傾角 (°)
yaw_kp         = 0.05  # Yaw 偏移比例增益
max_correction = 5.0   # 補正量上限 (cm)
pitch_deadband = 3.0   # Pitch 死區 (°)：誤差小於此值不補償（平地搖擺約 ±2.7°，故設略高）
roll_deadband  = 5.0   # Roll  死區 (°)：誤差小於此值不補償（平地搖擺約 ±3.5°，故設略高）

# ---- SR_Continuous 關節空間 pitch 反射補償（只作用於 SR_Continuous，疊在 px_u 之上）----
# 依 IMU pitch 誤差(與 px_u 共用 pitch_deadband，已過死區)與角速度，直接對 pitch 軸關節加角度偏移。
# 各關節獨立增益：想髖補多、踝補少、膝不補，就分別設值（設 0 = 該關節不參與）。
# 增益全 0 → 行為與現在完全相同。kd(角速度項)=製造角動量、抗倒主力；kp≈移重心(與 px_u 同性質)。
hip_reflex_kp    = 0.0   # 髖/腰(16/22) 比例增益 (deg偏移 / deg誤差)
hip_reflex_kd    = 0.0   # 髖/腰(16/22) 微分增益 (deg偏移 / (deg·s⁻¹))：腰前傾抗倒主力
knee_reflex_kp   = 0.0   # 膝(19/25) 比例增益（不建議用，膝管站高/吸震，預設 0）
knee_reflex_kd   = 0.0   # 膝(19/25) 微分增益（預設 0）
ankle_reflex_kp  = 0.0   # 踝(20/21,26/27) 比例增益（小擾動微調）
ankle_reflex_kd  = 0.0   # 踝(20/21,26/27) 微分增益
reflex_max_deg   = 15.0  # 每顆關節偏移上限 (°)，防暴衝/超限位