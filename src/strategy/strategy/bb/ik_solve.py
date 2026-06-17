#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KID robot right-arm IK solver.

Chain (channel = joint):
    8  RshouderJoint
    9  RarmJoint
    10 Rarm1Joint
    11 RforarmJoint
    12 Rforarm1Joint
    13 Rforarm2Joint
    14 RhandJoint  (not solved, passthrough)

Grasp point = Rforarm2 frame extended -Z by 15 cm.
Coordinate frame = URDF base_link, which matches Isaac Sim:
    robot faces -X, right shoulder toward +Y, up +Z.

Tick conversion derived from the user's +100-tick probe directions:
    sign per joint = [+1, +1, +1, -1, -1, +1]
    resolution     = 4096 ticks / rev  (change TICKS_PER_REV if geared)

Requires: numpy, scipy
"""

import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# URDF-derived kinematic chain: (origin_xyz, origin_rpy, joint_axis)
# Values copied verbatim from adult_size_urdf.urdf
# ----------------------------------------------------------------------
CHAIN = [
    ([0.0019757906382302, 0.0614589192815765, 0.343088951852611],
     [3.14159265358979, 0.0, 3.10802059661545], [0, 1, 0]),   # 8  RshouderJoint
    ([0.0220000000000001, 0.0325, 0.0220000000000001],
     [0.0, 0.0, 0.0], [-1, 0, 0]),                            # 9  RarmJoint
    ([-0.022, 0.0, 0.1139999999999],
     [3.14159265358979, -1.5707963267949, 0.0], [1, 0, 0]),   # 10 Rarm1Joint
    ([0.0650000000001, 0.0220000000000001, 0.0],
     [0.0, 1.5707963267949, 0.0], [0, 1, 0]),                 # 11 RforarmJoint
    ([-0.000500000000000209, -0.0219999999999999, 0.0625],
     [0.0, 1.5707963267949, 0.0], [1, 0, 0]),                 # 12 Rforarm1Joint
    ([-0.05625, 0.017, 0.0],
     [0.0, 1.5707963267949, 0.0], [0, 1, 0]),                 # 13 Rforarm2Joint
]

SHOULDER = np.array([0.0019757906382302, 0.0614589192815765, 0.343088951852611])
GRASP_OFFSET = np.array([0.0, 0.02, -0.17])  # -Z 15 cm in Rforarm2 frame

# RhandJoint (channel 14, continuous, not solved) - from adult_size_urdf.urdf
HAND_ORIGIN_RPY = [-3.14159265358979, 0.0, -3.14159265358979]
HAND_AXIS = [0, -1, 0]

# Standing pose (radians) and matching servo ticks
STAND_Q = np.array([0.0, 0.4, 1.57, 0.35, 1.57, 0.0])
STAND_TICK = np.array([3202, 2284, 2061, 1817, 1000, 2016])

TICKS_PER_REV = 4096
TPR = TICKS_PER_REV / (2 * np.pi)
TICK_SIGN = np.array([+1, +1, +1, -1, -1, +1])
HAND_TICK = 2016

JOINT_NAMES = ['RshouderJoint', 'RarmJoint', 'Rarm1Joint',
               'RforarmJoint', 'Rforarm1Joint', 'Rforarm2Joint']
CHANNELS = [8, 9, 10, 11, 12, 13]


# ----------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------
def rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T(xyz, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = xyz
    return M


def axis_rot(axis, th):
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
    """Return (grasp_pos, R2, axis13_world, hand_axis_world) in base_link frame."""
    M = np.eye(4)
    axis13_world = None
    for i, ((xyz, r, ax), th) in enumerate(zip(CHAIN, q)):
        M = M @ T(xyz, rpy(*r))
        if i == 5:
            axis13_world = M[:3, :3] @ (np.array(ax, float) / np.linalg.norm(ax))
        M = M @ T([0, 0, 0], axis_rot(ax, th))
    R2 = M[:3, :3]
    grasp = (M @ T(GRASP_OFFSET, np.eye(3)))[:3, 3]

    # RhandJoint rotation axis in world frame. Channel 14 only rolls about
    # this axis, so its own (unsolved) angle doesn't change the axis direction.
    hand_axis_local = np.array(HAND_AXIS, float)
    hand_axis_local /= np.linalg.norm(hand_axis_local)
    hand_axis_world = (R2 @ rpy(*HAND_ORIGIN_RPY)) @ hand_axis_local

    return grasp, R2, axis13_world, hand_axis_world


# ----------------------------------------------------------------------
# Coordinate conversion: PixelToWorld output (cm) -> IK target (m)
# ----------------------------------------------------------------------
def pixel_world_to_ik_target(ball_world_x_cm, ball_world_y_cm,
                              ball_h_m=0.9,
                              shoulder_h=0.90,
                              camera_h=1.05,
                              shoulder_right_of_cam=0.07,
                              cam_center_x_cm=0.0):
    """
    將 PixelToWorld 輸出的地面座標 (cm) 轉為 IK 所需的 base_link 目標點。

    ball_world_x_cm  : PixelToWorld 輸出的 X (左右方向，cm)
    ball_world_y_cm  : PixelToWorld 輸出的 Y (前後方向，cm)
    ball_h_m         : 球離地高度 (m)，預設 0.74
    cam_center_x_cm  : 攝影機中心對應的世界 X 座標 (cm)，
                       正值=相機偏右，需依標定調整
    """
    ball_front_of_cam = ball_world_y_cm / 100.0
    ball_right_of_cam = (ball_world_x_cm - cam_center_x_cm) / 100.0
    return world_to_target(
        ball_h=ball_h_m,
        shoulder_h=shoulder_h,
        camera_h=camera_h,
        shoulder_right_of_cam=shoulder_right_of_cam,
        ball_front_of_cam=ball_front_of_cam,
        ball_right_of_cam=ball_right_of_cam,
    )


def world_to_target(ball_h=0.75, shoulder_h=0.90, camera_h=1.09,
                    shoulder_right_of_cam=0.09, ball_front_of_cam=0.30,
                    ball_right_of_cam=0.0):
    """
    Heights are from the ground. Robot faces -X, right=+Y, up=+Z.
    Returns the grasp target in base_link coordinates.
    """
    dx = -ball_front_of_cam
    dy = ball_right_of_cam - shoulder_right_of_cam
    dz = ball_h - shoulder_h
    return SHOULDER + np.array([dx, dy, dz])


# ----------------------------------------------------------------------
# IK
# ----------------------------------------------------------------------
def solve_ik(target, constraint=None, seed=None, bound=1.4, weight=2.0):
    if seed is None:
        seed = STAND_Q.copy()
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
            # drive RhandJoint's axis to world +Z (unit vector, so z->1 forces x,y->0)
            r.append(weight * (hand_axis[2] - 1.0))
        return np.array(r)

    sol = least_squares(residual, seed, bounds=(lo, hi))
    return sol.x


def q_to_ticks(q):
    dq = q - STAND_Q
    return np.round(STAND_TICK + TICK_SIGN * dq * TPR).astype(int)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def report(target, constraint=None):
    q = solve_ik(target, constraint=constraint)
    g, R2, ax13, hand_axis = fk(q)
    err = np.linalg.norm(g - target)
    ticks = q_to_ticks(q)
    grasp_dir = R2 @ np.array([0, 0, -1])

    print("=" * 60)
    print(f"constraint        : {constraint}")
    print(f"target (base_link): {np.round(target, 4)}")
    print(f"achieved          : {np.round(g, 4)}")
    print(f"position error (m): {err:.6f}")
    print(f"grasp dir (world) : {np.round(grasp_dir, 4)}  z={grasp_dir[2]:+.5f}")
    print(f"joint13 axis world: {np.round(ax13, 4)}  z={ax13[2]:+.5f}")
    print(f"hand(ch14) axis   : {np.round(hand_axis, 4)}  z={hand_axis[2]:+.5f}")
    print("-" * 60)
    dq = q - STAND_Q
    for ch, n, th, d, t in zip(CHANNELS, JOINT_NAMES, q, dq, ticks):
        print(f"ch{ch:<2d} {n:14s} theta={th:+.4f} d={d:+.4f}  tick={t}")
    print(f"ch14 RhandJoint     (passthrough)               tick={HAND_TICK}")
    ok = all(0 <= t <= 4095 for t in ticks)
    print("-" * 60)
    print(f"all ticks in [0,4095]: {ok}")
    print("=" * 60)
    return q, ticks


if __name__ == "__main__":
    target = world_to_target(ball_h=1.4, shoulder_h=0.96, camera_h=1.05,
                             shoulder_right_of_cam=0.09, ball_front_of_cam=0.30)
    print("\n### Position-only IK ###")
    report(target, constraint=None)
