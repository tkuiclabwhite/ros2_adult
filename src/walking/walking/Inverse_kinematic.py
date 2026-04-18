# -*- coding: utf-8 -*-
"""
Inverse_kinematic.py (Unified Baseline Version)
-----------------------------------------------
- 不分軟/硬體模式
- 由外部指定 baseline（角度與刻度）
- 每次 IK 計算產生絕對刻度 → 減 baseline → 取得相對轉動量
"""

import math
import time
import threading
from dataclasses import dataclass
from typing import Dict, List

from . import Walkinggait as wg
from . import Parameter as parameter
from .Parameter import STAND_GP, DIR

# ---------- 基本參數 ----------
@dataclass
class Parameters:
    l1: float = 12.5
    l2: float = 12.5

PI = math.pi
TPR = 4096.0 / (2.0 * PI)

LEFT_IDS = [16, 17, 18, 19, 20, 21]  # 16-19: H, 20-21: X
RIGHT_IDS = [22, 23, 24, 25, 26, 27] # 22-25: H, 26-27: X
IK_IDS = LEFT_IDS + RIGHT_IDS
ALL_IDS = list(STAND_GP.keys())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ticks_diff_signed(a: float, b: float, mod: int = 4096) -> float:
    """回傳有號差值，確保差距在 (-mod/2, +mod/2) 範圍內"""
    d = (a - b) % mod
    if d > mod / 2:
        d -= mod
    return d

def make_transform_matrix(roll, pitch, yaw, px, py, pz):
    """
    對標 C++ control::makeTransformMatrix
    建立 4x4 齊次變換矩陣
    """
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    # ZYX 旋轉矩陣 (與 C++ 邏輯一致)
    #
    T = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, px],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, py],
        [-sy,     cy * sx,                cy * cx,                pz],
        [0,       0,                      0,                      1] # 強制補上最後一列
    ]
    return T

# ---------- IK 計算 ----------
def compute_leg_ik(Lx, Ly, Lz, Lt, Rx, Ry, Rz, Rt, rotate_body_l_=0.0, flag_=0):
    """
    回傳: 12 個馬達的目標弧度 (List[float])
    索引 0~5 為左腿 (ID 16~21), 6~11 為右腿 (ID 22~27)
    """
    def solve_single_leg(px, py, pz, yaw, is_left, roll=0.0, pitch=0.0):
        # 1. 呼叫新函數生成 4x4 矩陣
        T = make_transform_matrix(roll, pitch, yaw, px, py, pz)

        # 2. 從 4x4 矩陣中提取方向餘弦 (Orientation)
        #
        nx, ny, nz = T[0][0], T[1][0], T[2][0]
        ox, oy, oz = T[0][1], T[1][1], T[2][1]
        ax, ay, az = T[0][2], T[1][2], T[2][2]
        
        # 3. 幾何解算 (保持原樣，但此時 px, py, pz 是從 T 矩陣的結構中抽離出來的)
        L6 = -5.8
        l_arm = 25.0
        
        # 計算腕部中心點 (Wrist Center)
        px2 = px + L6 * ax
        py2 = py + L6 * ay
        pz2 = pz + L6 * az
        L_val = math.sqrt(px2**2 + py2**2 + pz2**2)
        
        # 防止 math domain error
        cos_val = (L_val**2) / (2 * l_arm**2) - 1
        cos_val = max(-1.0, min(1.0, cos_val))
        
        theta_knee = math.acos(cos_val) # theta[3]
        a = math.acos(max(-1.0, min(1.0, L_val / (2 * l_arm))))
        
        theta_5_raw = math.atan2(py + l_arm * ay, pz + l_arm * az)
        theta_4_raw = -math.atan2(px + l_arm * ax, math.sqrt((py + l_arm * ay)**2 + (pz + l_arm * az)**2)) - a
        
        # 軸向變換矩陣運算
        s6, c6 = math.sin(theta_5_raw), math.cos(theta_5_raw)
        s45, c45 = math.sin(theta_knee + theta_4_raw), math.cos(theta_knee + theta_4_raw)
        
        R_21 = ny * c45 + oy * s6 * s45 + ay * c6 * s45
        R_22 = oy * c6 - ay * s6
        R_13 = -nx * s45 + ox * s6 * c45 + ax * c6 * c45
        R_23 = -ny * s45 + oy * s6 * c45 + ay * c6 * c45
        R_33 = -nz * s45 + oz * s6 * c45 + az * c6 * c45
        
        theta_0 = math.atan2(R_13, R_33)
        s1, c1 = math.sin(theta_0), math.cos(theta_0)
        
        theta_1 = math.atan2(-R_23, R_13 * s1 + R_33 * c1)
        theta_2 = math.atan2(R_21, R_22)

        # --- 3. 移植 rad2motor 連桿耦合邏輯 ---
        l_link, d_link = 5.7, 4.0
        x_link = d_link * math.sin(theta_5_raw)
        # 避免 asin 超出範圍
        theta56_new = math.asin(max(-1.0, min(1.0, x_link / l_link)))
        
        if is_left:
            # RL == 0 的邏輯
            theta_4 = theta56_new - theta_4_raw
            theta_5 = theta56_new + theta_4_raw
        else:
            # RL != 0 的邏輯
            theta_4 = -theta56_new - theta_4_raw
            theta_5 = -theta56_new + theta_4_raw
            
        return [theta_0, theta_1, theta_2, theta_knee, theta_4, theta_5]

    # 計算左右腳
    left_leg = solve_single_leg(Lx, Ly, Lz, Lt, is_left=True)
    right_leg = solve_single_leg(Rx, Ry, Rz, Rt, is_left=False)
    
    return left_leg + right_leg

# ---------- IKService ----------
class IKService:
    def __init__(self, baseline_ticks=None, baseline_ang=None, min_pv=0, speed_scale=1.0):
        """
        baseline_ticks: Dict[int,int]  → 馬達 baseline 絕對刻度
        baseline_ang:   List[float]    → 對應 baseline 的角度
        """
        self.min_pv = min_pv
        self.speed_scale = float(speed_scale)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._base_ticks = dict(baseline_ticks) if baseline_ticks else dict(STAND_GP)
        self._base_ang = list(baseline_ang) if baseline_ang else [0.0]*len(IK_IDS)
        self._prev_ang = list(self._base_ang)

        self._latest_gp = dict(self._base_ticks)
        self._latest_pv = {mid: 0 for mid in ALL_IDS}
        self._rel_ticks = {mid: 0 for mid in ALL_IDS}

    # ====== 啟動 / 停止 ======
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ====== 主循環 ======
    def _loop(self):
        walking = getattr(wg, "walking", wg.WalkingGaitByLIPM())
        while not self._stop.is_set():
            dt = float(getattr(walking, "sample_time_", 30)) / 1000.0
            if hasattr(walking, "process"):
                walking.process()

            Lx, Ly, Lz, Lt, Rx, Ry, Rz, Rt = (
                walking.end_point_lx_, walking.end_point_ly_, walking.end_point_lz_, walking.end_point_lthta_,
                walking.end_point_rx_, walking.end_point_ry_, walking.end_point_rz_, walking.end_point_rthta_
            )
            ang_now = compute_leg_ik(Lx, Ly, Lz, Lt, Rx, Ry, Rz, Rt)

            gp_abs = self._calc_gp(ang_now, self._base_ang)
            rel = {mid: ticks_diff_signed(gp_abs[mid], self._base_ticks[mid]) for mid in gp_abs}
            pv = self._calc_pv(ang_now, self._prev_ang, dt)
		
            with self._lock:
                self._latest_gp.update(gp_abs)
                self._latest_pv.update(pv)
                self._rel_ticks.update(rel)
                self._prev_ang = list(ang_now)
            
            time.sleep(dt)

    # ====== 工具 ======
    def _calc_gp(self, ang_now, base_ang):
        gp = {}
        for idx, mid in enumerate(IK_IDS):
            # 判斷馬達型號給予不同 TPR
            if mid in [20, 21, 26, 27]:
                tpr = 4096.0 / (2.0 * math.pi)       # X 系列
            else:
                tpr = 303750.0 / (2.0 * math.pi)     # H 系列 (PRO)
                
            delta = ang_now[idx] - base_ang[idx]
            delta_ticks = DIR[mid] * tpr * delta
            ticks = int(round(self._base_ticks[mid] + delta_ticks))

            if mid in [20, 21, 26, 27]:
                gp[mid] = ticks % 4096
            else:
                gp[mid] = ticks
        return gp

    def _calc_pv(self, ang_now, ang_prev, dt):
        pv = {}
        for idx, mid in enumerate(IK_IDS):
            d_ang = abs(ang_now[idx] - ang_prev[idx])
            if d_ang <= 1e-9:
                v = 0
            else:
                # 根據不同馬達計算 ticks_per_s 與對應的 RPM 單位
                if mid in [20, 21, 26, 27]:
                    tpr = 4096.0 / (2.0 * math.pi)
                    # X 系列單位: 0.229 RPM per unit
                    unit_factor = 60.0 / (4096.0 * 0.229) 
                else:
                    tpr = 303750.0 / (2.0 * math.pi)
                    # H 系列 (PRO) 單位通常是 0.01 RPM per unit (請根據手冊確認)
                    unit_factor = 60.0 / (303750.0 * 0.01)
                
                v = int(round((d_ang / dt) * unit_factor * self.speed_scale))
                v = max(v, self.min_pv)
            pv[mid] = clamp(v, 0, 32767)
        return pv

    # ====== 介面 ======
    def latest_gp(self):
        with self._lock:
            return dict(self._latest_gp)

    def latest_rel_gp(self):
        with self._lock:
            return dict(self._rel_ticks)

    def latest_pv(self):
        with self._lock:
            return dict(self._latest_pv)

    def latest_rel_deg(self):
        with self._lock:
            # 這裡的 tick_to_deg 也需要分開算，否則 H 系列的度數會破萬
            res = {}
            for mid in self._rel_ticks:
                if mid in [20, 21, 26, 27]:
                    res[mid] = self._rel_ticks[mid] * (360.0 / 4096.0)
                else:
                    res[mid] = self._rel_ticks[mid] * (360.0 / 303750.0)
            return res