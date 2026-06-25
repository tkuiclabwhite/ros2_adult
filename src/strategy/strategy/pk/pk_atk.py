#!/usr/bin/env python3
# coding=utf-8
import sys
import time
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from strategy.API import API

# ── 頭部馬達絕對位置 ───────────────────────────────────────────────────────────
# HEAD_HORIZONTAL  (ID 1)：2048 = 正中央，> 2048 = 左轉，< 2048 = 右轉
# HEAD_VERTICAL (ID 2)：2048 = 正中央，> 2048 = 往下看，< 2048 = 往上看
PAN_CENTER  = 2048
TILT_SEARCH = 2750   # 往下看地面球的角度
PAN_MIN,  PAN_MAX  = 1024, 3096
TILT_MIN, TILT_MAX = 1450, 2900
HEAD_TRACK_GAIN = 0.15   # 頭部追蹤增益（越大越靈敏）
X_FOV_DEG = 64.5          # 水平視角（度）
Y_FOV_DEG = 40.0          # 垂直視角（度）

# ── 步態速度參數 ───────────────────────────────────────────────────────────────
# 正 x = 前進，正 y = 左平移，正 theta = 左旋轉
FWD_FAST  =  5000   # 快速前進
FWD_MED   =  3000   # 中速前進
FWD_SLOW  =   400   # 慢速前進
FWD_STOP  =     0   # 停止
FWD_BACK  =  -800   # 後退

TRANS_L   =  1200   # 左平移
TRANS_S   =     0   # 不平移
TRANS_R   = -1200   # 右平移

ROT_L     =   4     # 左旋轉
ROT_S     =   0     # 不旋轉
ROT_R     =  -4     # 右旋轉

# ── 狀態轉換閾值 ───────────────────────────────────────────────────────────────
BALL_STABLE_FRAMES  = 5      # 連續偵測到球幾幀才確認球已找到
BALL_APPROACH1_SIZE = 2500   # 第一次接近的停止 球像素面積
BALL_CLOSE_SIZE     = 2500   # 	APF 導航的停止 球像素面積
BALL_ALIGN_X_TOL    = 25     # 水平誤差容忍值（像素）：SECOND_SEARCH_AND_ALIGN 對準
BALL_KICK_X_TOL     = 40     # 水平誤差容忍值（像素）：WEAK_KICK 前置中檢查
PAN_CENTER_TOL      = 150    # 頭部 pan 偏移容忍值（馬達單位）：超過此值視為頭未正
BALL_ALIGN_FRAMES   = 4      # 連續對準幾幀才進入 FINAL_SHOT
BALL_LOST_FRAMES    = 5      # 連續失蹤幾幀才退回搜尋

# ── 障礙物迴避調整參數 ─────────────────────────────────────────────────────────
BLUE_SIDE_THRESH    = 600    # 側欄藍色面積超過此值才開始閃避
BLUE_CENTER_THRESH  = 1500   # 中欄藍色面積超過此值才減速

# ── IMU 對準參數（SECOND_SEARCH_AND_ALIGN 繞球對準用）────────────────────────────
# yaw > 0 = 機器人向左偏，< 0 = 向右偏（相對於開場歸零時的方向）
# ORBIT_ROT / ORBIT_TRANS 的正負號請實測後確認
YAW_TOL     =  5.0   # yaw 誤差容忍值（度），在此範圍內視為對準球門
CY_TARGET   = 180    # 射門準備距離對應的球畫面 y（待實測調整，0~240）
CY_TOL      =  20    # cy 誤差容忍值（像素）
ORBIT_ROT   =   2    # 繞球旋轉速度（ROT 單位，待實測）
ORBIT_TRANS = 300    # 繞球平移速度（TRANS 單位，待實測）

# ── 顏色索引（對應 API.COLORS 陣列順序）──────────────────────────────────────────
_COLOR_IDX = {
    'Orange': 0, 'Yellow': 1, 'Blue': 2,
    'Green':  3, 'Black':  4, 'Red':  5, 'White': 6,
}

# ── 限制掃描象限對應表 ─────────────────────────────────────────────────────────
# 鍵值：(start_side, ball_relative_pos)
# 對應：(pan_lo, pan_hi, tilt)
BALL_COLOR        = 'Yellow'                    # 球的顏色'Orange', 'Yellow', 'Blue',
                                                        #'Green', 'Black', 'Red', 'White',
START_SIDE        = 'center'                      # 機器人在場上的起始位置（'left' / 'center' / 'right'）
BALL_RELATIVE_POS = 'left'                     # 球相對於機器人的方向（'left' / 'front' / 'right'）
START_STATE = 1   # 測試用：從第幾個狀態開始（0~5）
STOP_STATE  = 2  # 測試用：執行完此狀態後停步收尾（-1 = 不提前停止）

# ── 原地步態校正量 ─────────────────────────────────────────────────────────────
STAY_X     = -1800   # 原地步態 X 校正（加在所有 sendContinuousValue 的 x 上）
STAY_Y     = 0      # 原地步態 Y 校正
STAY_THETA = 2      # 原地步態 Theta 校正

# ── 站姿微調 ────────────────────────────────────────────────────────────────────
STAND_CORRECT_ATK    = True   # 是否在比賽開始時執行站姿微調
STAND_CORRECT_SECTOR = 201     # 站姿微調 sector 編號

# 狀態編號對照表
_STATE_NAMES = [
    'INIT_DIRECTIONAL_SEARCH', #0
    'APPROACH_BALL_1',         #1
    'ALIGN_TO_GOAL',           #2
    'WEAK_KICK',               #3
    'VISUAL_GUIDED_APPROACH',  #4
    'SECOND_SEARCH_AND_ALIGN', #5
    'FINAL_SHOT',              #6
]
_STATE_DESCS = [
    '在限制象限內掃頭，找到球後前進',
    '走向球，保持球在畫面中央，距離夠近後停步',
    '到位後繞球修正 yaw，確認朝向球門方向',
    '輕踢第一腳，將球推入射門準備區',
    'APF 人工勢場導航，繞過藍色守門員往球靠近',
    '低頭重新鎖定球，慢速對準準備射門',
    '依球的位置選腳，全力射門',
]
# 各狀態對應的 ANSI 前景色（亮色系）
_STATE_COLORS = [
    '\033[96m',   # 0 INIT_DIRECTIONAL_SEARCH  — 亮青色
    '\033[92m',   # 1 APPROACH_BALL_1           — 亮綠色
    '\033[97m',   # 2 ALIGN_TO_GOAL             — 亮白色
    '\033[93m',   # 3 WEAK_KICK                 — 亮黃色
    '\033[95m',   # 4 VISUAL_GUIDED_APPROACH    — 亮洋紅
    '\033[94m',   # 5 SECOND_SEARCH_AND_ALIGN   — 亮藍色
    '\033[91m',   # 6 FINAL_SHOT                — 亮紅色
]
SCAN_MAP = {
    ('left',   'left'):  (2048, 3072, TILT_SEARCH),
    ('left',   'front'): (1748, 2348, TILT_SEARCH),
    ('left',   'right'): (1024, 2048, TILT_SEARCH),
    ('center', 'left'):  (2048, 2748, TILT_SEARCH),
    ('center', 'front'): (1748, 2348, TILT_SEARCH),
    ('center', 'right'): (1348, 2048, TILT_SEARCH),
    ('right',  'left'):  (2048, 3072, TILT_SEARCH),
    ('right',  'front'): (1748, 2348, TILT_SEARCH),
    ('right',  'right'): (1024, 2048, TILT_SEARCH),
}
SCAN_FALLBACK = (PAN_MIN, PAN_MAX, TILT_SEARCH)   # 找不到對應組合時的備用掃描範圍


class PenaltyKickAtk(API):

    def __init__(self):
        super().__init__('pk_atk_node')

        # 頭部當前位置（馬達絕對值）
        self._pan  = PAN_CENTER
        self._tilt = TILT_SEARCH

        # 掃描範圍（由 start_side + ball_relative_pos 決定）
        self._scan_pan_lo  = PAN_MIN
        self._scan_pan_hi  = PAN_MAX
        self._scan_tilt    = TILT_SEARCH
        self._scan_dir     = 1       # +1 = pan 增加（向左掃），-1 = pan 減少（向右掃）

        # 狀態機
        self._state        = 'INIT_DIRECTIONAL_SEARCH'
        self._stable_count = 0   # 連續偵測到球的幀數計數
        self._align_count  = 0   # 連續對準的幀數計數
        self._lost_count   = 0   # 連續失蹤幀數計數
        self._stance_done  = False   # 本次 is_start=True 後是否已執行站姿調整

        # 步態旗標
        self._walk_active  = False

        # 顯示用快取（由各 handler 更新，display thread 讀取）
        self._disp_ball       = None               # (cx, cy, area) 或 None
        self._disp_blue       = (0.0, 0.0, 0.0)   # (left, center, right)
        self._disp_walk_cmd   = (0, 0, 0)          # (x, y, theta)
        self._disp_last_event = ''                  # 最後一個重要事件訊息

        # 每 0.1 秒執行一次主迴圈
        self.create_timer(0.1, self._tick)
        self._init_state()

        # 啟動獨立的顯示執行緒（不阻塞 ROS callback）
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True
        )
        self._display_thread.start()

    # ─────────────────────────────────────────────────────────── 初始化 ─────────

    def _init_state(self):
        """讀取 ROS2 參數，設定掃描範圍，並重置所有狀態變數。"""
        start_side = START_SIDE
        ball_rel   = BALL_RELATIVE_POS
        
        lo, hi, tilt = SCAN_MAP.get((start_side, ball_rel), SCAN_FALLBACK)
        self._scan_pan_lo = lo
        self._scan_pan_hi = hi
        self._scan_tilt   = tilt

        self._pan  = lo
        self._tilt = tilt
        self._scan_dir = 1

        self._state        = _STATE_NAMES[START_STATE]
        self._stable_count = 0
        self._align_count  = 0
        self._lost_count   = 0
        self._walk_active  = False
        self._stance_done  = False

        self.sendSensorReset(True)
        self.sendHeadMotor(1, self._pan,  30)
        self.sendHeadMotor(2, self._tilt, 30)

        # 從特定狀態跳入時的前置初始化
        if _STATE_NAMES[START_STATE] == 'SECOND_SEARCH_AND_ALIGN':
            self._tilt = TILT_MAX - 150   # 直接往下看找球
            self.sendHeadMotor(2, self._tilt, 30)

        # self.get_logger().info(
        #     f'[pk_atk] 初始化  start_side={start_side}  ball_rel={ball_rel}  '
        #     f'scan_pan=[{lo},{hi}]'
        # )

    # ────────────────────────────────────────────────────── 視覺輔助函式 ────────

    def _get_ball(self):
        """回傳面積最大的球色物件 (cx, cy, area)，若無則回傳 None。"""
        idx = _COLOR_IDX[BALL_COLOR]
        best_i    = None
        best_size = 100   # 最小面積門檻
        for i in range(self.color_counts[idx]):
            if self.object_sizes[idx][i] > best_size:
                best_size = self.object_sizes[idx][i]
                best_i = i
        if best_i is None:
            self._disp_ball = None
            return None
        cx = (self.object_x_max[idx][best_i] + self.object_x_min[idx][best_i]) // 2
        cy = (self.object_y_max[idx][best_i] + self.object_y_min[idx][best_i]) // 2
        result = float(cx), float(cy), float(self.object_sizes[idx][best_i])
        self._disp_ball = result
        return result

    def _get_blue_areas(self):
        """
        將藍色障礙物依畫面 x 位置分為左、中、右三欄，各欄寬約 107px。
        回傳 (blue_left, blue_center, blue_right)，單位為像素面積總和。
        """
        idx   = _COLOR_IDX['Blue']
        left  = center = right = 0.0
        col_w = 320 / 3.0
        for i in range(self.color_counts[idx]):
            size = float(self.object_sizes[idx][i])
            cx   = (self.object_x_max[idx][i] + self.object_x_min[idx][i]) // 2
            if cx < col_w:
                left   += size
            elif cx < col_w * 2:
                center += size
            else:
                right  += size
        self._disp_blue = left, center, right
        return left, center, right

    # ────────────────────────────────────────────────────────── 頭部控制 ────────

    def _head_track(self, cx, cy):
        """依目標位置 (cx, cy) 調整頭部，使目標保持在畫面中心。"""
        x_err = cx - 160
        y_err = cy - 120
        # 像素誤差換算為角度誤差
        x_deg = x_err * (X_FOV_DEG / 320)
        y_deg = y_err * (Y_FOV_DEG / 240)
        # 角度誤差換算為馬達步數並加上增益
        self._pan  -= round(x_deg * 4096 / 360 * HEAD_TRACK_GAIN)
        self._tilt += round(y_deg * 4096 / 360 * HEAD_TRACK_GAIN)
        # 限制在安全範圍內
        self._pan  = max(PAN_MIN,  min(PAN_MAX,  self._pan))
        self._tilt = max(TILT_MIN, min(TILT_MAX, self._tilt))

        self.sendHeadMotor(1, self._pan,  25)
        self.sendHeadMotor(2, self._tilt, 25)

    def _head_scan(self):
        """在限制掃描象限內左右來回掃描，尋找球。"""
        STEP = 40   # 每次掃描步進量（馬達單位）
        self._pan += STEP * self._scan_dir
        if self._pan >= self._scan_pan_hi:
            self._pan      = self._scan_pan_hi
            self._scan_dir = -1   # 碰到右邊界，改為向左掃
        elif self._pan <= self._scan_pan_lo:
            self._pan      = self._scan_pan_lo
            self._scan_dir = 1    # 碰到左邊界，改為向右掃

        self._tilt = self._scan_tilt
        self.sendHeadMotor(1, self._pan,  30)
        self.sendHeadMotor(2, self._tilt, 30)

    # ─────────────────────────────────────────────────────────── 步態控制 ────────

    def _walk(self, x, y, theta):
        """送出步態速度指令：x=前後, y=左右平移, theta=旋轉。"""
        self._disp_walk_cmd = (int(x), int(y), round(theta))
        self.sendContinuousValue(int(x) + STAY_X, int(y) + STAY_Y, round(theta) + STAY_THETA)

    def _start_walk(self):
        """若步態尚未啟動，則啟動步態引擎。"""
        if not self._walk_active:
            self.sendbodyAuto(1)
            self._walk_active = True

    def _stop_walk(self):
        """停止步態：先送出零速度，等待 0.4 秒後關閉步態引擎。"""
        self._walk(FWD_STOP, TRANS_S, ROT_S)
        time.sleep(0.4)
        self.sendbodyAuto(0)
        self._walk_active = False

    # ─────────────────────────────────────────────── 人工勢場（APF）計算 ────────

    def _compute_apf(self, ball_cx, blue_left, blue_center, blue_right):
        """
        以人工勢場法（APF）計算步態指令。

        引力向量：球的 x 位置控制 Theta（轉向球），前進速度固定為 FWD_MED
        斥力向量：各欄藍色面積控制 y 平移（閃避障礙物）

        回傳 (x_cmd, y_cmd, theta_cmd)。
        """
        # ── 引力計算 ──────────────────────────────────────────────────────────
        x_cmd = FWD_MED

        ball_err      = ball_cx - 160          # 正值 = 球在畫面右側
        # 朝球轉向：球在右 → 右轉（theta 為負）
        theta_attract = -ball_err * (abs(ROT_R) / 160)

        # ── 斥力計算 ──────────────────────────────────────────────────────────
        y_repulse = 0.0

        if blue_left > BLUE_SIDE_THRESH:
            # 左側有障礙物 → 推機器人向右（y 為負）
            strength   = min(blue_left / BLUE_SIDE_THRESH, 3.0)
            y_repulse -= strength * abs(TRANS_R)

        if blue_right > BLUE_SIDE_THRESH:
            # 右側有障礙物 → 推機器人向左（y 為正）
            strength   = min(blue_right / BLUE_SIDE_THRESH, 3.0)
            y_repulse += strength * abs(TRANS_L)

        if blue_center > BLUE_CENTER_THRESH:
            # 正前方有障礙物 → 減速並往空曠側閃避
            x_cmd = FWD_SLOW
            if blue_left < blue_right:
                y_repulse += abs(TRANS_L)    # 左側空間較大，向左閃
            else:
                y_repulse -= abs(TRANS_R)    # 右側空間較大，向右閃

        # ── 數值限幅 ──────────────────────────────────────────────────────────
        x_cmd     = max(FWD_BACK, min(FWD_FAST,      x_cmd))
        y_cmd     = max(TRANS_R * 3, min(TRANS_L * 3, y_repulse))
        theta_cmd = max(ROT_R * 2,   min(ROT_L * 2,   theta_attract))

        return x_cmd, y_cmd, theta_cmd

    # ──────────────────────────────────────────────────────── 各狀態處理函式 ────

    def _handle_init_directional_search(self):
        """
        狀態 1 — INIT_DIRECTIONAL_SEARCH（定向搜尋）
        僅在由 start_side + ball_relative_pos 決定的象限內掃描頭部。
        連續偵測到球 BALL_STABLE_FRAMES 幀後，轉移至 APPROACH_BALL_1。
        """
        ball = self._get_ball()

        if ball is None:
            self._head_scan()       # 未看到球，繼續掃描
            self._stable_count = 0
            return

        cx, cy, _ = ball
        self._head_track(cx, cy)    # 看到球，追蹤並計數
        self._stable_count += 1

        if self._stable_count >= BALL_STABLE_FRAMES:
            self._disp_last_event = '找到球 → APPROACH_BALL_1'
            self._stable_count = 0
            self._state = 'APPROACH_BALL_1'
            self._start_walk()

    def _handle_approach_ball_1(self):
        """
        狀態 2 — APPROACH_BALL_1（第一次接近球）
        邊走邊追蹤球，同時以橫向平移修正讓球保持在畫面中央。
        當球的像素面積 > BALL_APPROACH1_SIZE（距離夠近），停步進入 WEAK_KICK。
        """
        ball = self._get_ball()

        if ball is None:
            self._lost_count += 1
            if self._lost_count >= BALL_LOST_FRAMES:
                self._walk(FWD_STOP, TRANS_S, ROT_S)
                self._lost_count  = 0
                self._stable_count = 0
                self._state = 'INIT_DIRECTIONAL_SEARCH'
            return

        self._lost_count = 0
        cx, cy, area = ball
        self._head_track(cx, cy)
        self._start_walk()   # 確保步態引擎已啟動（從此狀態跳入時需要）

        x_err = cx - 160

        # 旋轉修正：依頭部 pan 偏移量讓身體轉向球，頭才能回到中心
        # pan > PAN_CENTER = 頭朝左 = 球在左 → 身體左轉（theta 正）
        pan_err   = self._pan - PAN_CENTER
        theta_cmd = pan_err * (abs(ROT_L) / (PAN_MAX - PAN_CENTER)) * 3
        theta_cmd = max(ROT_R * 2, min(ROT_L * 2, theta_cmd))

        # 橫向微調：輔助細部置中
        y_cmd = -x_err * (abs(TRANS_R) / 160) * 4.0

        # 前進速度：頭偏越大 x 越小，頭未正時不往前
        pan_abs_err = abs(pan_err)
        x_cmd = int(FWD_MED * max(0.0, 1.0 - pan_abs_err / PAN_CENTER_TOL))

        if area < BALL_APPROACH1_SIZE or abs(x_err) > BALL_KICK_X_TOL or pan_abs_err > PAN_CENTER_TOL:
            self._walk(x_cmd, int(y_cmd), theta_cmd)
        else:
            self._stop_walk()
            self._disp_last_event = '已到位且置中 → ALIGN_TO_GOAL'
            self._state = 'ALIGN_TO_GOAL'

    def _handle_align_to_goal(self):
        """
        狀態 2 — ALIGN_TO_GOAL（對準球門）
        到位後原地繞球修正 yaw，確認機器人朝向球門後才進入 WEAK_KICK。
        連續對準 BALL_ALIGN_FRAMES 幀才轉移，避免瞬間誤判。
        """
        ball = self._get_ball()

        if ball is None:
            self._align_count = 0
            self._disp_last_event = '失去球，退回 APPROACH_BALL_1'
            self._state = 'APPROACH_BALL_1'
            self._start_walk()
            return

        cx, cy, _ = ball
        self._head_track(cx, cy)
        self._start_walk()

        yaw = self.imu_rpy[2]

        if abs(yaw) > YAW_TOL:
            rot_dir = -1 if yaw > 0 else 1
            self._walk(FWD_STOP, int(-rot_dir * ORBIT_TRANS), rot_dir * ORBIT_ROT)
            self._disp_last_event = f'yaw 修正中 {yaw:.1f}° → 目標 ±{YAW_TOL}°'
            self._align_count = 0
        else:
            self._walk(FWD_STOP, TRANS_S, ROT_S)
            self._align_count += 1
            self._disp_last_event = f'yaw 對準 {yaw:.1f}°  [{self._align_count}/{BALL_ALIGN_FRAMES}]'
            if self._align_count >= BALL_ALIGN_FRAMES:
                self._stop_walk()
                self._align_count = 0
                self._disp_last_event = '對準球門完成 → WEAK_KICK'
                self._state = 'WEAK_KICK'

    def _handle_weak_kick(self):
        """
        狀態 3 — WEAK_KICK（輕踢）
        第一腳：將球輕推約 120 公分進入射門準備區。
        正確的輕踢 sector 編號需在硬體上實際測試確認。
        """
        self._disp_last_event = '執行輕踢（第一腳）'
        self.sendBodySector(29)    # 先站直
        time.sleep(1.5)

        # self.sendBodySector(1000)  # TODO: 替換為正確的輕踢 sector 編號
        # time.sleep(3.0)

        self.sendBodySector(29)
        time.sleep(1.5)

        # 將頭部重置為向前看的位置，準備下一段導航
        self._pan  = PAN_CENTER
        self._tilt = TILT_SEARCH
        self.sendHeadMotor(1, self._pan,  30)
        self.sendHeadMotor(2, self._tilt, 30)

        self.sendbodyAuto(1)
        self._walk_active = True
        self._disp_last_event = '輕踢完成 → VISUAL_GUIDED_APPROACH'
        self._state = 'VISUAL_GUIDED_APPROACH'

    def _handle_visual_guided_approach(self):
        """
        狀態 4 — VISUAL_GUIDED_APPROACH（視覺引導接近 / 人工勢場）

        核心導航邏輯：
          引力 — 依球的 x 位置調整 theta（轉向球）
          斥力 — 依各欄藍色面積調整 y 平移（繞開障礙物）

        當球的像素面積 > BALL_CLOSE_SIZE（障礙物已繞過、球非常近），
        停步進入 SECOND_SEARCH_AND_ALIGN。
        """
        ball                   = self._get_ball()
        blue_l, blue_c, blue_r = self._get_blue_areas()

        if ball is None:
            # 球暫時被遮擋：假設球在正前方，僅執行障礙物迴避
            self._start_walk()
            _, y_repulse, _ = self._compute_apf(160, blue_l, blue_c, blue_r)
            self._walk(FWD_SLOW, int(y_repulse), ROT_S)
            return

        cx, cy, area = ball
        self._head_track(cx, cy)
        self._start_walk()   # 確保步態引擎已啟動（從此狀態跳入時需要）

        x_cmd, y_cmd, theta_cmd = self._compute_apf(cx, blue_l, blue_c, blue_r)
        self._walk(x_cmd, int(y_cmd), theta_cmd)

        if area > BALL_CLOSE_SIZE:
            self._stop_walk()
            self._disp_last_event = '障礙物已繞過 → SECOND_SEARCH_AND_ALIGN'
            self._state   = 'SECOND_SEARCH_AND_ALIGN'
            self._tilt    = TILT_MIN + 150   # 往下看，重新定位腳邊的球
            self._align_count = 0
            self.sendHeadMotor(2, self._tilt, 30)

    def _handle_second_search_and_align(self):
        """
        狀態 5 — SECOND_SEARCH_AND_ALIGN（第二次搜尋與對準）
        往下看重新鎖定被推進射門準備區的球。
        對準順序：
          1. IMU yaw 修正（繞球旋轉平移，讓機器人面向球門方向）
          2. cy 距離修正（前後調整到射門準備距離）
          3. cx 橫向微調（讓球在畫面正中央）
        三者都在容忍範圍內後計入對準幀數，達標後進入 FINAL_SHOT。
        """
        ball = self._get_ball()

        if ball is None:
            self._tilt = min(TILT_MAX, self._tilt + 15)
            self.sendHeadMotor(2, self._tilt, 20)
            self._align_count = 0
            return

        cx, cy, area = ball
        self._head_track(cx, cy)
        self._start_walk()

        yaw   = self.imu_rpy[2]
        x_err = cx - 160
        cy_err = cy - CY_TARGET

        # 優先 1：yaw 對準（繞球旋轉平移）
        # yaw > 0 = 機器人向左偏 → 需右轉（theta 負）並左移（y 正）
        # yaw < 0 = 機器人向右偏 → 需左轉（theta 正）並右移（y 負）
        # ORBIT_ROT / ORBIT_TRANS 正負號請實測確認
        if abs(yaw) > YAW_TOL:
            rot_dir = -1 if yaw > 0 else 1
            self._walk(FWD_STOP, int(-rot_dir * ORBIT_TRANS), rot_dir * ORBIT_ROT)
            self._disp_last_event = f'yaw 修正中 {yaw:.1f}° → 目標 ±{YAW_TOL}°'
            self._align_count = 0
            return

        # 優先 2：前後距離修正（cy）
        # cy 太小 = 球在畫面上方 = 距離太遠 → 前進
        # cy 太大 = 球在畫面下方 = 距離太近 → 後退
        if abs(cy_err) > CY_TOL:
            x_cmd = FWD_SLOW if cy_err < 0 else -FWD_SLOW
            self._walk(x_cmd, 0, ROT_S)
            self._disp_last_event = f'距離修正中 cy={cy:.0f} → 目標 {CY_TARGET}±{CY_TOL}'
            self._align_count = 0
            return

        # 優先 3：橫向微調（cx）
        y_cmd = -x_err * (abs(TRANS_R) / 160)
        self._walk(FWD_STOP, int(y_cmd), ROT_S)

        if abs(x_err) < BALL_ALIGN_X_TOL and area > BALL_APPROACH1_SIZE:
            self._align_count += 1
        else:
            self._align_count = 0

        if self._align_count >= BALL_ALIGN_FRAMES:
            self._stop_walk()
            self._disp_last_event = '對準完成 → FINAL_SHOT'
            self._state = 'FINAL_SHOT'

    def _handle_final_shot(self):
        """
        狀態 6 — FINAL_SHOT（最終射門）
        全力踢球進門。
        依最後一幀球的橫向位置決定用左腳或右腳。
        """
        self._disp_last_event = '執行最終射門'
        self.sendBodySector(29)
        time.sleep(1.5)

        ball = self._get_ball()
        if ball and ball[0] < 160:
            # self.sendBodySector(2000)
            # time.sleep(1.5)
            self._disp_last_event = '左腳射門'
        else:
            # self.sendBodySector(1000)
            # time.sleep(1.5)
            self._disp_last_event = '右腳射門'

        time.sleep(5.0)
        self.sendBodySector(29)
        time.sleep(2.0)
        self._disp_last_event = '全序列完成 → FINISH'
        self._state = 'FINISH'

    # ────────────────────────────────────────────────────────── 狀態顯示 ────────

    def _print_status(self):
        """將目前所有狀態資訊一次印到終端機（清除後重寫）。"""
        R = '\033[0m'   # Reset
        B = '\033[1m'   # Bold
        D = '\033[2m'   # Dim

        # ── 狀態 ──────────────────────────────────────────────────────────────
        state_idx  = _STATE_NAMES.index(self._state) if self._state in _STATE_NAMES else -1
        state_name = _STATE_NAMES[state_idx] if state_idx >= 0 else self._state
        state_desc = _STATE_DESCS[state_idx] if state_idx >= 0 else ''
        state_col  = _STATE_COLORS[state_idx] if 0 <= state_idx < len(_STATE_COLORS) else '\033[97m'
        state_idx_str = str(state_idx) if state_idx >= 0 else '?'

        is_start_str = (
            f'{B}\033[92mTrue{R}' if self.is_start
            else f'{B}\033[91mFalse (未開始){R}'
        )

        # ── 球 ────────────────────────────────────────────────────────────────
        if self._disp_ball:
            cx, cy, area = self._disp_ball
            ball_str = f'\033[93mcx={cx:.0f}  cy={cy:.0f}  area={area:.0f}{R}'
        else:
            ball_str = f'{D}\033[33m未偵測到{R}'

        # ── 藍色障礙物（依數值動態上色）────────────────────────────────────────
        def _bv(val, thresh):
            if val > thresh:
                return f'{B}\033[91m{val:.0f}{R}'   # 超閾值 — 亮紅
            if val > thresh * 0.5:
                return f'\033[93m{val:.0f}{R}'       # 接近閾值 — 黃
            return f'\033[94m{val:.0f}{R}'           # 安全 — 藍

        bl, bc, br = self._disp_blue
        blue_str = (
            f'左={_bv(bl, BLUE_SIDE_THRESH)}  '
            f'中={_bv(bc, BLUE_CENTER_THRESH)}  '
            f'右={_bv(br, BLUE_SIDE_THRESH)}'
        )

        # ── 步態指令（依正負上色）──────────────────────────────────────────────
        wx, wy, wt = self._disp_walk_cmd
        walk_eng = f'{B}\033[92m啟動中{R}' if self._walk_active else f'\033[91m停止{R}'
        xc = '\033[92m' if wx > 0 else ('\033[91m' if wx < 0 else '\033[90m')
        yc = '\033[94m' if wy > 0 else ('\033[91m' if wy < 0 else '\033[90m')
        tc = '\033[95m' if wt != 0 else '\033[90m'
        walk_cmd_str = f'x={xc}{wx}{R}  y={yc}{wy}{R}  θ={tc}{wt}{R}'

        # ── 站姿微調狀態 ──────────────────────────────────────────────────────────
        stand_str = (
            f'{B}\033[92m啟用{R}  \033[90msector={STAND_CORRECT_SECTOR}{R}'
            if STAND_CORRECT_ATK
            else f'\033[90m停用{R}'
        )
        stance_done_str = f'{B}\033[92m已完成{R}' if self._stance_done else f'\033[93m等待中{R}'

        # ── 進度條 ────────────────────────────────────────────────────────────
        def _bar(val, total, col='\033[92m'):
            filled = min(int(val / total * 8), 8) if total > 0 else 0
            return f'{col}{"█" * filled}{R}\033[90m{"░" * (8 - filled)}{R} {val}/{total}'

        stable_bar = _bar(self._stable_count, BALL_STABLE_FRAMES)
        align_bar  = _bar(self._align_count,  BALL_ALIGN_FRAMES, '\033[94m')

        # ── IMU yaw（依誤差上色）─────────────────────────────────────────────
        yaw = self.imu_rpy[2]
        yaw_col = '\033[91m' if abs(yaw) > YAW_TOL else '\033[92m'
        yaw_str = f'{yaw_col}{yaw:.1f}°{R}  \033[90m容忍 ±{YAW_TOL}°{R}'

        # ── cy 目標距離（依誤差上色）─────────────────────────────────────────
        if self._disp_ball:
            cy_val = self._disp_ball[1]
            cy_err = abs(cy_val - CY_TARGET)
            cy_col = '\033[91m' if cy_err > CY_TOL else '\033[92m'
            cy_str = f'{cy_col}{cy_val:.0f}px{R}  \033[90m目標 {CY_TARGET}±{CY_TOL}{R}'
        else:
            cy_str = f'{D}—{R}'

        # ── 分隔線 ────────────────────────────────────────────────────────────
        SEP  = f'\033[2;33m{"─" * 40}{R}'
        def sec(title):
            return f'{B}\033[33m▌ {title}{R}'

        sys.stdout.write('\033[H\033[J')
        sys.stdout.write(
            f'\033[1;97;44m  PK 進攻狀態                       {R}\n'
            f'  is_start   : {is_start_str}\n'
            f'{SEP}\n'
            f'{sec("目前狀態")}\n'
            f'  {state_col}{B}[{state_idx_str}] {state_name}{R}\n'
            f'  {D}{state_desc}{R}\n'
            f'  \033[96m最後事件 : {self._disp_last_event}{R}\n'
            f'{SEP}\n'
            f'{sec("設定")}\n'
            f'  起始位置    : \033[97m{START_SIDE}{R}\n'
            f'  球相對方向  : \033[97m{BALL_RELATIVE_POS}{R}\n'
            f'  掃描(pan)   : \033[90m[{self._scan_pan_lo}, {self._scan_pan_hi}]{R}\n'
            f'  原地校正    : \033[90mX={STAY_X}  Y={STAY_Y}  θ={STAY_THETA}{R}\n'
            f'  站姿微調    : {stand_str}  [{stance_done_str}]\n'
            f'{SEP}\n'
            f'{sec("視覺資訊")}\n'
            f'  球 (黃色)   : {ball_str}\n'
            f'  藍色障礙物  : {blue_str}\n'
            f'{SEP}\n'
            f'{sec("頭部位置")}\n'
            f'  Pan   : \033[36m{self._pan}{R}\n'
            f'  Tilt  : \033[36m{self._tilt}{R}\n'
            f'{SEP}\n'
            f'{sec("IMU 對準（Shooting Zone 用）")}\n'
            f'  Yaw   : {yaw_str}\n'
            f'  cy    : {cy_str}\n'
            f'{SEP}\n'
            f'{sec("步態")}\n'
            f'  步態引擎 : {walk_eng}\n'
            f'  步態指令 : {walk_cmd_str}\n'
            f'{SEP}\n'
            f'{sec("計數器")}\n'
            f'  stable  [{stable_bar}]\n'
            f'  align   [{align_bar}]\n'
            f'  lost    \033[90m{self._lost_count}/{BALL_LOST_FRAMES}{R}\n'
            f'{SEP}\n'
        )
        sys.stdout.flush()

    def _display_loop(self):
        """背景顯示執行緒：每 0.1 秒刷新一次終端機狀態板。"""
        while rclpy.ok():
            try:
                self._print_status()
            except Exception:
                pass
            time.sleep(0.1)

    def _handle_stopped(self):
        """比賽未開始或中斷時呼叫：停止步態、站直，並重置狀態機。"""
        self._get_ball()
        self._get_blue_areas()
        if self._walk_active:
            self.sendbodyAuto(0)
            time.sleep(1.0)
            self.sendBodySector(29)
        self._init_state()

    # ────────────────────────────────────────────────────────── 主計時器迴圈 ────

    def _tick(self):
        """每 0.1 秒觸發一次，根據目前狀態呼叫對應的處理函式。"""
        if not self.is_start:
            self._handle_stopped()
            return

        if not self._stance_done:
            self._disp_last_event = '站立 → 基本站姿'
            self.sendBodySector(29)
            time.sleep(1.0)
            if STAND_CORRECT_ATK:
                self._disp_last_event = '站姿微調中...'
                self.sendBodySector(STAND_CORRECT_SECTOR)
                time.sleep(1.0)
            self._stance_done = True
            self._disp_last_event = '站姿完成，開始執行'
            return

        match self._state:
            case 'INIT_DIRECTIONAL_SEARCH':
                self._handle_init_directional_search()
            case 'APPROACH_BALL_1':
                self._handle_approach_ball_1()
            case 'ALIGN_TO_GOAL':
                self._handle_align_to_goal()
            case 'WEAK_KICK':
                self._handle_weak_kick()
            case 'VISUAL_GUIDED_APPROACH':
                self._handle_visual_guided_approach()
            case 'SECOND_SEARCH_AND_ALIGN':
                self._handle_second_search_and_align()
            case 'FINAL_SHOT':
                self._handle_final_shot()
            case 'FINISH':
                pass

        # ── STOP_STATE 提早結束檢查 ──────────────────────────────────────────────
        if STOP_STATE >= 0 and self._state in _STATE_NAMES:
            if _STATE_NAMES.index(self._state) > STOP_STATE:
                self._disp_last_event = f'STOP_STATE={STOP_STATE} 已達，停步收尾'
                if self._walk_active:
                    self._stop_walk()
                self.sendBodySector(29)
                self._state = 'FINISH'


def main(args=None):
    rclpy.init(args=args)
    node = PenaltyKickAtk()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('pk_atk 節點停止中...')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
