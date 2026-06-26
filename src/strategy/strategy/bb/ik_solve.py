#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
right_arm_fk_ik.py
==================
adult_size_urdf 右手手臂 (7 DOF) 的正向運動學 (FK) 與逆向運動學 (IK)。
並提供從「相機世界座標 (cm)」轉成「base_link IK 目標 (m)」的介面:

    ik_target = pixel_world_to_ik_target(
        ball_world_x_cm=ball_world_x,
        ball_world_y_cm=ball_world_y,
        ball_h_m=0.74,
    )

座標定義
--------
相機世界座標 (來自你的 pixel_to_world / homography):
    ball_world_x_cm : 側向 (相機座標 x)。+x = 相機畫面右側。
    ball_world_y_cm : 地面上、自相機正下方 (nadir) 起算的前方距離。
    ball_h_m        : 球心離地高度 (公尺)。

相對幾何 (公尺):
    shoulder_h            = 0.90   右肩離地高度
    camera_h              = 1.05   相機離地高度
    shoulder_right_of_cam = 0.07   右肩在相機右側多少 (m)
    cam_center_x_cm       = 0.0    相機光心側向校正 (cm)

base_link 慣例: x=前, y=左, z=上 (URDF 內右肩位於 z≈0.343 m)。
"""

import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------
# 右手手臂運動鏈 (直接取自 adult_size_urdf.urdf)
#   每個元素: (joint_name, origin_xyz, origin_rpy, axis)
# ----------------------------------------------------------------------------
RIGHT_ARM_CHAIN = [
    ("RshouderJoint",  [0.0019757906382302, 0.0614589192815765, 0.343088951852611],
                       [3.14159265358979, 0.0, 3.10802059661545], [0, 1, 0]),
    ("RarmJoint",      [0.0220000000000001, 0.0325, 0.0220000000000001],
                       [0.0, 0.0, 0.0], [-1, 0, 0]),
    ("Rarm1Joint",     [-0.022, 0.0, 0.1139999999999],
                       [3.14159265358979, -1.5707963267949, 0.0], [-1, 0, 0]),
    ("RforarmJoint",   [0.0650000000001, 0.0220000000000001, 0.0],
                       [0.0, 1.5707963267949, 0.0], [0, -1, 0]),
    ("Rforarm1Joint",  [-0.000500000000000209, -0.0219999999999999, 0.0625],
                       [0.0, 1.5707963267949, 0.0], [-1, 0, 0]),
    ("Rforarm2Joint",  [-0.05625, 0.017, 0.0],
                       [0.0, 1.5707963267949, 0.0], [0, -1, 0]),
    ("RhandJoint",     [-0.00399999999999998, 0.0, -0.0612500000000001],
                       [-3.14159265358979, 0.0, -3.14159265358979], [0, -1, 0]),
]
JOINT_NAMES = [c[0] for c in RIGHT_ARM_CHAIN]

# 手掌工具點 (RhandLink 質心，當作末端 / 抓球點)。需要指尖可自行調整。
TOOL_LOCAL = np.array([0.0161223890513385, -0.0169997919450656, 0.0537939592650262])

# 由 random-pose 取樣量到的最大可達半徑 (自右肩算起)。超過此值無解。
MAX_REACH_M = 0.45


# ----------------------------------------------------------------------------
# 基礎變換
# ----------------------------------------------------------------------------
def _rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = _rpy_to_R(*rpy)
    M[:3, 3] = xyz
    return M


def _axis_R(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _joint_T(xyz, rpy, axis, q):
    M = _T(xyz, rpy)
    Rj = np.eye(4)
    Rj[:3, :3] = _axis_R(axis, q)
    return M @ Rj


# 右肩在 base_link 中的固定位置
SHOULDER_BASE = _T(RIGHT_ARM_CHAIN[0][1], RIGHT_ARM_CHAIN[0][2])[:3, 3]


# ----------------------------------------------------------------------------
# 正向運動學
# ----------------------------------------------------------------------------
def forward_kinematics(q, return_all=False):
    """
    q : 長度 7 的關節角 (rad)，順序 = JOINT_NAMES。
    回傳 (T_hand_4x4, tool_tip_xyz)。return_all=True 時另回傳各 link 的 4x4。
    """
    q = np.asarray(q, float).reshape(-1)
    assert q.shape[0] == 7, "右手手臂需要 7 個關節角"
    M = np.eye(4)
    frames = []
    for i, (_, xyz, rpy, axis) in enumerate(RIGHT_ARM_CHAIN):
        M = M @ _joint_T(xyz, rpy, axis, q[i])
        frames.append(M.copy())
    tip = (M @ np.array([*TOOL_LOCAL, 1.0]))[:3]
    if return_all:
        return M, tip, frames
    return M, tip


# ----------------------------------------------------------------------------
# 逆向運動學 (位置, 阻尼最小平方 DLS)
# ----------------------------------------------------------------------------
def _numeric_jacobian(q, eps=1e-6):
    _, p0 = forward_kinematics(q)
    J = np.zeros((3, 7))
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        _, p = forward_kinematics(dq)
        J[:, i] = (p - p0) / eps
    return J, p0


def inverse_kinematics(target_xyz, q_init=None, iters=300, tol=1e-4,
                       lam=0.05, max_step=0.3):
    """
    解出讓 tool tip 到 target_xyz (base_link, m) 的 7 個關節角。
    回傳 dict: q, error_m, iters, converged, clamped。
    超出可達範圍時，目標會被夾到 MAX_REACH_M 球面上 (clamped=True)。
    """
    target = np.asarray(target_xyz, float).copy()

    # 可達性夾制
    v = target - SHOULDER_BASE
    dist = np.linalg.norm(v)
    clamped = False
    if dist > MAX_REACH_M:
        target = SHOULDER_BASE + v / dist * MAX_REACH_M
        clamped = True

    q = np.zeros(7) if q_init is None else np.asarray(q_init, float).copy()
    converged = False
    for k in range(iters):
        J, p = _numeric_jacobian(q)
        e = target - p
        if np.linalg.norm(e) < tol:
            converged = True
            break
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + (lam ** 2) * np.eye(3), e)
        m = np.max(np.abs(dq))
        if m > max_step:
            dq *= max_step / m
        q = q + dq
        q = (q + np.pi) % (2 * np.pi) - np.pi

    _, p = forward_kinematics(q)
    return {
        "q": q,
        "joint_names": JOINT_NAMES,
        "error_m": float(np.linalg.norm(target - p)),
        "iters": k + 1,
        "converged": converged,
        "clamped": clamped,
    }


# ----------------------------------------------------------------------------
# 相機世界座標 (cm) -> base_link IK 目標 (m)
# ----------------------------------------------------------------------------
def pixel_world_to_ik_target(ball_world_x_cm, ball_world_y_cm, ball_h_m,
                             shoulder_h=0.90, camera_h=1.05,
                             shoulder_right_of_cam=0.1, cam_center_x_cm=0.0,
                             return_relative=False):
    """
    將相機世界座標轉成 base_link 下的 IK 目標 (x,y,z, 單位 m)。

    參數
    ----
    ball_world_x_cm : 側向 (相機座標 x)，+x = 畫面右側 (cm)
    ball_world_y_cm : 地面前方距離，自相機正下方算起 (cm)
    ball_h_m        : 球心離地高度 (m)
    shoulder_h, camera_h, shoulder_right_of_cam : 幾何 (m)
    cam_center_x_cm : 相機光心側向校正 (cm)

    回傳
    ----
    np.array([x, y, z])  (base_link, m)；若 return_relative，另回傳
    [前, 左, 上] 相對右肩的向量。
    """
    # cm -> m，套用光心校正
    bx = (ball_world_x_cm - cam_center_x_cm) / 100.0   # 側向 (相機 +x = 右)
    by = ball_world_y_cm / 100.0                       # 前方
    bz = float(ball_h_m)                               # 離地高度

    # 以右肩為參考，base_link 軸向 (x=前, y=左, z=上):
    rel_forward = by
    # 球側向 bx 是相對相機；右肩在相機右側 shoulder_right_of_cam，
    # 所以球相對右肩往左 = bx - shoulder_right_of_cam (相機右為 +x，左肩為 +y)
    rel_left = bx - shoulder_right_of_cam
    rel_up = bz - shoulder_h

    tx = SHOULDER_BASE[0] + rel_forward
    ty = SHOULDER_BASE[1] + rel_left
    tz = SHOULDER_BASE[2] + rel_up
    target = np.array([tx, ty, tz])
    if return_relative:
        return target, np.array([rel_forward, rel_left, rel_up])
    return target


# ----------------------------------------------------------------------------
# 一步到位: 相機座標 -> 關節角
# ----------------------------------------------------------------------------
def solve_arm_from_ball(ball_world_x_cm, ball_world_y_cm, ball_h_m,
                        q_init=None, **geom):
    """相機世界座標直接解出右手 7 軸角度。回傳 IK dict (含 target 與 relative)。"""
    target, rel = pixel_world_to_ik_target(
        ball_world_x_cm, ball_world_y_cm, ball_h_m,
        return_relative=True, **geom)
    res = inverse_kinematics(target, q_init=q_init)
    res["target_base_link"] = target
    res["relative_to_shoulder"] = rel
    return res


# ============================================================================
# Isaac Sim world-frame offset
# Isaac Sim world origin in base_link coords = ISAAC_OFFSET
# To convert:  isaac_xyz = base_link_xyz - ISAAC_OFFSET  (use to_isaac / from_isaac)
# ============================================================================

ISAAC_OFFSET = np.array([0.0, -0.03848, -0.5462])


def to_isaac(base_link_xyz):
    """Convert base_link coordinates (m) to Isaac Sim world coordinates."""
    return np.asarray(base_link_xyz) - ISAAC_OFFSET


def from_isaac(isaac_xyz):
    """Convert Isaac Sim world coordinates to base_link coordinates (m)."""
    return np.asarray(isaac_xyz) + ISAAC_OFFSET


# ============================================================================
# 6-DOF API — channels 8-13, ch14 = RhandJoint passthrough
# Used by bb.py: solve_ik, q_to_ticks, world_to_target, fk, CHANNELS, HAND_TICK
# Robot faces -X; right=+Y; up=+Z  (matches adult_size_urdf convention)
# ============================================================================

_CHAIN6 = [
    ([0.0019757906382302, 0.0614589192815765, 0.343088951852611],
     [3.14159265358979, 0.0, 3.10802059661545], [0, 1, 0]),   # ch8  RshouderJoint
    ([0.0220000000000001, 0.0325, 0.0220000000000001],
     [0.0, 0.0, 0.0], [-1, 0, 0]),                            # ch9  RarmJoint
    ([-0.022, 0.0, 0.1139999999999],
     [3.14159265358979, -1.5707963267949, 0.0], [1, 0, 0]),   # ch10 Rarm1Joint
    ([0.0650000000001, 0.0220000000000001, 0.0],
     [0.0, 1.5707963267949, 0.0], [0, 1, 0]),                 # ch11 RforarmJoint
    ([-0.000500000000000209, -0.0219999999999999, 0.0625],
     [0.0, 1.5707963267949, 0.0], [1, 0, 0]),                 # ch12 Rforarm1Joint
    ([-0.05625, 0.017, 0.0],
     [0.0, 1.5707963267949, 0.0], [0, 1, 0]),                 # ch13 Rforarm2Joint
]

_SHOULDER6 = np.array([0.0019757906382302, 0.0614589192815765, 0.343088951852611])
_GRASP_OFFSET6 = np.array([-0.02, 0.0, -0.11])
_HAND_ORIGIN_RPY6 = [-3.14159265358979, 0.0, -3.14159265358979]
_HAND_AXIS6 = [0, -1, 0]

_STAND_Q6 = np.array([0.0, 0.4, 1.57, 0.35, 1.57, 0.0])
_STAND_TICK6 = np.array([3202, 2284, 2061, 1817, 1000, 2016])
_TICKS_PER_REV = 4096
_TPR = _TICKS_PER_REV / (2 * np.pi)
_TICK_SIGN6 = np.array([+1, +1, +1, -1, -1, +1])

HAND_TICK = 2700
CHANNELS = [8, 9, 10, 11, 12, 13]
JOINT_NAMES = ['RshouderJoint', 'RarmJoint', 'Rarm1Joint',
               'RforarmJoint', 'Rforarm1Joint', 'Rforarm2Joint']


def _rpy6(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T6(xyz, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = xyz
    return M


def _axis_rot6(axis, th):
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    x, y, z = a
    c, s = np.cos(th), np.sin(th)
    C = 1 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def fk(q):
    """Return (grasp_pos, R_forarm2, axis13_world, hand_axis_world) in Isaac Sim coords."""
    M = np.eye(4)
    axis13_world = None
    for i, ((xyz, r, ax), th) in enumerate(zip(_CHAIN6, q)):
        M = M @ _T6(xyz, _rpy6(*r))
        if i == 5:
            axis13_world = M[:3, :3] @ (np.array(ax, float) / np.linalg.norm(ax))
        M = M @ _T6([0, 0, 0], _axis_rot6(ax, th))
    R2 = M[:3, :3]
    grasp = to_isaac((M @ _T6(_GRASP_OFFSET6, np.eye(3)))[:3, 3])
    hand_axis_local = np.array(_HAND_AXIS6, float)
    hand_axis_local /= np.linalg.norm(hand_axis_local)
    hand_axis_world = (R2 @ _rpy6(*_HAND_ORIGIN_RPY6)) @ hand_axis_local
    return grasp, R2, axis13_world, hand_axis_world


def world_to_target(ball_h=0.75, shoulder_h=0.88929, camera_h=1.0512,
                    shoulder_right_of_cam=0.09994, ball_front_of_cam=0.30,
                    ball_right_of_cam=0.0):
    """Camera-relative ball position (metres) -> Isaac Sim IK target."""
    dx = -ball_front_of_cam          # robot faces -X
    dy = ball_right_of_cam - shoulder_right_of_cam
    dz = ball_h - shoulder_h
    return to_isaac(_SHOULDER6 + np.array([dx, dy, dz]))


def pixel_world_to_ik_target(ball_world_x_cm, ball_world_y_cm,
                              ball_h_m,
                              shoulder_h=0.88929,
                              camera_h=1.0512,
                              shoulder_right_of_cam=0.09994,
                              cam_center_x_cm=0.0,
                              return_relative=False):
    """PixelToWorld ground coords (cm) -> Isaac Sim IK target (m)."""
    ball_front_of_cam = ball_world_y_cm / 100.0
    ball_right_of_cam = (ball_world_x_cm - cam_center_x_cm) / 100.0
    target = world_to_target(
        ball_h=ball_h_m,
        shoulder_h=shoulder_h,
        camera_h=camera_h,
        shoulder_right_of_cam=shoulder_right_of_cam,
        ball_front_of_cam=ball_front_of_cam,
        ball_right_of_cam=ball_right_of_cam,
    )
    if return_relative:
        rel = np.array([-ball_front_of_cam,
                        ball_right_of_cam - shoulder_right_of_cam,
                        ball_h_m - shoulder_h])
        return target, rel
    return target


def solve_ik(target, constraint=None, seed=None, bound=1.4, weight=2.0):
    """Solve 6-DOF IK. target in Isaac Sim coords (m). Returns q (radians) for channels 8-13."""
    target = np.asarray(target, float)
    if seed is None:
        seed = _STAND_Q6.copy()
    lo, hi = seed - bound, seed + bound

    def residual(q):
        g, R2, ax13, hand_axis = fk(q)
        r = list(g - target)
        if constraint == 'grasp_perp_z':
            grasp_dir = R2 @ np.array([0, 0, -1])
            r.append(weight * grasp_dir[2])
        elif constraint == 'axis13_perp_z':
            r.append(weight * ax13[2])
        elif constraint == 'hand_axis_z':
            r.append(weight * (hand_axis[2] - 1.0))
        elif constraint == 'rforarm2_z_up':
            z_w = R2[:, 2]
            r += [weight * z_w[0], weight * z_w[1], weight * (z_w[2] - 1.0)]
        elif constraint == 'rforarm2_z_down':
            z_w = R2[:, 2]
            r += [weight * z_w[0], weight * z_w[1], weight * (z_w[2] + 1.0)]
        return np.array(r)

    sol = least_squares(residual, seed, bounds=(lo, hi))
    return sol.x


def q_to_ticks(q):
    """Convert 6-DOF joint angles (rad) to servo ticks (integer, 0-4095 range)."""
    dq = np.asarray(q) - _STAND_Q6
    return np.round(_STAND_TICK6 + _TICK_SIGN6 * dq * _TPR).astype(int)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("右肩 (base_link):", SHOULDER_BASE, " 高度=%.3f m" % SHOULDER_BASE[2])
    print("最大可達半徑     :", MAX_REACH_M, "m\n")

    # 你 prompt 的呼叫格式
    ball_world_x, ball_world_y = 0.0, 30.0
    ik_target = pixel_world_to_ik_target(
        ball_world_x_cm=ball_world_x,
        ball_world_y_cm=ball_world_y,
        ball_h_m=0.74,
        shoulder_h=0.88929, camera_h=1.0481,
        shoulder_right_of_cam=0.1, cam_center_x_cm=0.0,
    )
    print("ik_target (base_link, m):", ik_target)

    res = solve_arm_from_ball(ball_world_x, ball_world_y, 0.74)
    print("converged :", res["converged"], " clamped:", res["clamped"])
    print("error     : %.3f mm" % (res["error_m"] * 1000))
    print("關節角(rad):", res["q"])
    print("關節順序   :", res["joint_names"])