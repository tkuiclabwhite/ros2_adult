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
    16: -1,   # L_HIP_YAW
    17: -1 ,  # L_HIP_ROLL
    18:  1,   # L_HIP_PITCH
    19: -1,   # L_KNEE
    20:  1,   # L_ANKLE_PITCH
    21:  1,   # L_ANKLE_ROLL

    22:  1,   # R_HIP_YAW
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