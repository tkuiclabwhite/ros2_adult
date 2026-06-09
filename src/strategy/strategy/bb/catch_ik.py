"""
catch_ball IK 整合模組 (右臂, 馬達 ID 8~14)

把這個檔放在 bb.py 同目錄，匯入後在 catch_ball 裡呼叫。
依賴 arm_ik.py (放同目錄) 與 adult_size_urdf.urdf。

刻度換算: tick = 2048 + angle_rad * 4096 / (2*pi)   (Dynamixel, 2048=0度)
"""

import os
import numpy as np
from .arm_ik import ArmChain, solve_ik, make_T

_BB_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
#  右臂馬達 / 關節設定
# ============================================================

ARM_MOTOR_IDS = [8, 9, 10, 11, 12, 13, 14]
ARM_JOINTS = ['RshouderJoint', 'RarmJoint', 'Rarm1Joint', 'RforarmJoint',
              'Rforarm1Joint', 'Rforarm2Joint', 'RhandJoint']

CATCH_ORI_WEIGHT = 0.0   # 0 = 純位置 IK，較容易收斂

# 各關節實際可動範圍 (rad)，比 ±90° 寬以允許低位夾球
ARM_Q_LIMITS = np.array([
    [0.0,        np.pi],    # RshouderJoint  axis=-y → 正角=往前，只允許往前
    [-np.pi,     np.pi],    # RarmJoint      肩膀俯仰
    [-np.pi,     np.pi],    # Rarm1Joint     上臂扭轉
    [-np.pi,     np.pi],    # RforarmJoint   肘關節
    [-np.pi,     np.pi],    # Rforarm1Joint  前臂扭轉
    [-np.pi,     np.pi],    # Rforarm2Joint  手腕俯仰
    [-np.pi/2,   np.pi/2],  # RhandJoint     手掌
])

# 夾球初始猜測（SIGN 已改 +1 後對應同一物理位置，q_init 符號需反轉）
# ID8: q=1.77 → tick=3202+1154=4356→4000; ID9: q=-0.36 → tick=2284-235=2049
CATCH_Q_INIT = np.array([1.77, -0.36, 0.02, -0.35, 1.61, 0.05, -0.05])

ARM_MOVE_SPEED = 30

# MOTOR_SIGN: 物理安裝方向與 URDF q 的對應關係
# +1 = tick 增加 ↔ q 增加 (物理方向與 URDF 相同)
# -1 = tick 增加 ↔ q 減少 (物理方向與 URDF 相反)
# ID8  實測: tick 增加 → 手臂往前 = q>0 往前 → sign=+1
# ID9  實測: tick 增加 → 手臂往外(左) → 向右(q<0)使 tick 下降 → sign=+1
MOTOR_SIGN      = np.array([1, 1, -1, -1, 1, 1, 1])
MOTOR_ZERO_TICK = np.array([3202, 2284, 2061, 1818, 1000, 2016, 2018])

# 每個馬達的刻度安全範圍 [min, max]，對應 ID 8~14
TICK_LIMITS = np.array([
    [2048, 4000],   # ID 8  RshouderJoint   tick↑=往前
    [2000, 3500],   # ID 9  RarmJoint       tick↑=往外(左), tick↓=往右; 向右夾球 tick<2284
    [1024, 3072],   # ID 10 Rarm1Joint
    [1600, 3000],   # ID 11 RforarmJoint
    [ 700, 2048],   # ID 12 Rforarm1Joint   sign=-1 → tick<2048 = 正q = 手朝前
    [1200, 2600],   # ID 13 Rforarm2Joint   sign=-1
    [1536, 2560],   # ID 14 RhandJoint
])

# ============================================================
#  頭部幾何校正
#
#  HEAD_HEIGHT         : 夾球姿態下攝影機距地面高度 (公尺)  <<<< 量測
#  BASE_HEIGHT         : 夾球姿態下 base_link 原點距地面高度 (公尺)  <<<< 量測
#  HEAD_X_IN_BASE      : 攝影機在 base_link frame 的 X 偏移 (公尺，向前為正)
#  HEAD_Y_IN_BASE      : 攝影機在 base_link frame 的 Y 偏移 (公尺，向左為正)
#  HEAD_HORIZON_CENTER : 夾球前頭部水平馬達的刻度
#  HEAD_TILT_ZERO_RAD  : head_vertical=2048 時的額外俯仰偏移 (rad，往下為正)
#  BALL_RADIUS         : 籃球半徑 (公尺)
#  BALL_POLE_HEIGHT    : 球放在柱子上時，柱頂離地高度 (公尺)；球放地上時設 0
# ============================================================

HEAD_HEIGHT         = 1.09   # <<< 量測: 夾球姿態攝影機距地面 (m)  (球在柱上不需深蹲)
BASE_HEIGHT         = 0.567  # <<< 量測: base_link 原點距地面 (m)  = HEAD_HEIGHT - 0.523
HEAD_X_IN_BASE      = 0.0    # 攝影機在 base_link frame X 偏移 (前向)
HEAD_Y_IN_BASE      = -0.069 # 攝影機在 base_link frame Y 偏移: 肩膀y(0.061) - 頭肩y差(0.13)
HEAD_HORIZON_CENTER = 1780   # 夾球前頭部水平中心刻度
HEAD_TILT_ZERO_RAD  = 0.0    # head_vertical=2048 的俯仰偏差 (rad)
BALL_RADIUS         = 0.07   # 標準籃球半徑 (m)
BALL_POLE_HEIGHT    = 0.67   # 柱子高度 (m)；球放地上時設為 0  (= 球心高 - 球半徑)

# 球心離地高度（直接量測值 74 cm）
BALL_CENTER_HEIGHT  = BALL_POLE_HEIGHT + BALL_RADIUS   # 0.62 + 0.12 = 0.74 m

# 舊名稱相容 (避免其他地方引用到 HEAD_HEIGHT_CROUCH / BASE_HEIGHT_CROUCH 出錯)
HEAD_HEIGHT_CROUCH  = HEAD_HEIGHT
BASE_HEIGHT_CROUCH  = BASE_HEIGHT

# RshouderJoint origin in base_link (from URDF) — 僅供除錯參考，IK 不需要額外減去
# xyz = [0.00198, 0.06146, 0.34309]

TICK_PER_RAD = 4096.0 / (2.0 * np.pi)


def angle_to_tick(q_rad):
    """關節角(rad, 7,) -> 馬達刻度(int, 7,)，並夾在 TICK_LIMITS 內"""
    ticks = MOTOR_ZERO_TICK + MOTOR_SIGN * q_rad * TICK_PER_RAD
    ticks = np.clip(ticks, TICK_LIMITS[:, 0], TICK_LIMITS[:, 1])
    return np.round(ticks).astype(int)


def ball_xyz_in_base(head_horizon_tick, head_vertical_tick):
    """
    從頭部馬達刻度估算球的位置 (base_link frame, 公尺)。

    假設:
      - 球心離地 = BALL_CENTER_HEIGHT (球在地板時=BALL_RADIUS, 在柱上時=BALL_POLE_HEIGHT+BALL_RADIUS)
      - 攝影機離地 = HEAD_HEIGHT
      - head_vertical 增加 = 頭往下看
      - head_horizon 相對 HEAD_HORIZON_CENTER 的偏移 = 水平轉角

    回傳 np.array([x, y, z]) in base_link frame，
    若仰角幾乎為零 (頭幾乎水平) 則回傳 None。
    """
    pan_rad  = -(head_horizon_tick - HEAD_HORIZON_CENTER) * (2.0 * np.pi / 4096.0)
    tilt_rad = (head_vertical_tick - 2048) * (2.0 * np.pi / 4096.0) + HEAD_TILT_ZERO_RAD

    dz = HEAD_HEIGHT - BALL_CENTER_HEIGHT   # 攝影機比球心高多少
    if abs(tilt_rad) < 1e-3:
        return None

    horiz_dist = dz / np.tan(tilt_rad)

    cam_fwd  = horiz_dist * np.cos(pan_rad)
    cam_left = horiz_dist * np.sin(pan_rad)

    ball_x = HEAD_X_IN_BASE + cam_fwd
    ball_y = HEAD_Y_IN_BASE + cam_left
    ball_z = BALL_CENTER_HEIGHT - BASE_HEIGHT   # 球心相對 base_link 的高度

    return np.array([ball_x, ball_y, ball_z])


class ArmIK:
    """封裝右臂 IK。在 BasketBall.__init__ 裡建一次。"""

    def __init__(self, urdf_path=None):
        if urdf_path is None:
            urdf_path = os.path.join(_BB_DIR, "adult_size_urdf.urdf")
        elif not os.path.isabs(urdf_path):
            urdf_path = os.path.join(_BB_DIR, urdf_path)
        self.chain = ArmChain(urdf_path, ARM_JOINTS)
        self.q_last = CATCH_Q_INIT.copy()

    def compute_catch_ticks(self, head_horizon=None, head_vertical=None, ori_weight=None):
        """
        算夾球姿態的馬達刻度。
        傳入頭部馬達刻度時自動計算球的 3D 位置；
        省略則退回靜態目標點 (需校正 HEAD_* 常數後才準確)。
        回傳 (motor_ids, ticks, info)。info['success'] 失敗時不要送馬達。
        """
        if ori_weight is None:
            ori_weight = CATCH_ORI_WEIGHT

        target_xyz = None
        if head_horizon is not None and head_vertical is not None:
            target_xyz = ball_xyz_in_base(head_horizon, head_vertical)

        if target_xyz is None:
            # 幾何法失敗或未傳頭部刻度 -> 用零姿估算末端位置作 placeholder
            _, T0 = self.chain.fk(np.zeros(self.chain.n))
            target_xyz = T0[:3, 3]

        T_target = make_T(np.eye(3), np.array(target_xyz, dtype=float))
        q, info = solve_ik(
            self.chain, T_target,
            q_init=self.q_last,
            ori_weight=ori_weight,
            q_limits=ARM_Q_LIMITS,
        )
        if info['success']:
            self.q_last = q
        ticks = angle_to_tick(q)
        info['target_xyz'] = target_xyz.tolist()
        return ARM_MOTOR_IDS, ticks, info
