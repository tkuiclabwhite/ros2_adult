#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU 共用的姿態數學：四元數與歐拉角互轉、歸零用的四元數運算。

四元數格式一律為 (x, y, z, w)，與 ROS geometry_msgs/Quaternion 的欄位順序相同。
歐拉角一律為 (roll, pitch, yaw) 度數，ZYX 內旋順序（先 yaw、再 pitch、最後 roll），
與 Xsens /filter/euler 及 ROS 的 RPY 慣例一致。

歸零要用 q_rel = q0⁻¹ ⊗ q_now 而不是歐拉角相減：
歐拉角相減在大角度時會出錯（旋轉不可交換、萬向鎖），只在小角度下剛好夠用。
"""
import math


def quat_conjugate(q):
    """(x, y, z, w) 的共軛；單位四元數的共軛即為其逆。"""
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_multiply(a, b):
    """四元數乘法 a ⊗ b，格式皆為 (x, y, z, w)。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_to_euler_deg(q):
    """(x, y, z, w) -> (roll, pitch, yaw) 角度，ZYX 內旋順序。"""
    x, y, z, w = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 接近 ±90° 時 asin 的定義域會超出，夾住避免例外（萬向鎖）
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def euler_deg_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) 角度 -> (x, y, z, w)，ZYX 內旋順序，為 quat_to_euler_deg 的反函式。"""
    hr = math.radians(roll) * 0.5
    hp = math.radians(pitch) * 0.5
    hy = math.radians(yaw) * 0.5
    cr, sr = math.cos(hr), math.sin(hr)
    cp, sp = math.cos(hp), math.sin(hp)
    cy, sy = math.cos(hy), math.sin(hy)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def wrap_deg(deg):
    """把角度包回 (-180, 180]。"""
    d = math.fmod(deg + 180.0, 360.0)
    if d <= 0.0:
        d += 360.0
    return d - 180.0
