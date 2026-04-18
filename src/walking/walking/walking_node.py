#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WalkingNode.py — [Snapshot Baseline + Landing Fix + Dynamic Stand Update]
功能：
1. Generate ON: 抓取當前位置作為 Baseline。
2. Generate OFF: 執行落地緩衝，確保雙腳踩平，並回復到 Baseline。
3. Idle (Stand): 若調整 width_size/height，即時重算 IK 並驅動馬達，同時更新 Baseline。
"""

import sys
import time
import threading
import math
import json
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, String, Bool
from sensor_msgs.msg import JointState
from tku_msgs.msg import Interface

# ==== 專案內部匯入 ====
try:
    from . import Parameter as parameter
    from .Parameter import DIR 
    from .Inverse_kinematic import compute_leg_ik, IK_IDS, TPR
    from .Walkinggait import WalkingGaitByLIPM, StopStep, StartStep, FirstStep, Repeat
except ImportError as e:
    print(f"[Error] Import failed: {e}. Make sure Parameter.py exists.")
    sys.exit(1)

# ============================================================
# 馬達 ID 設定
# ============================================================
IDS_ARM  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
IDS_HEAD = [28,29]
IDS_LEG  = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
LEG_IDS = IDS_LEG 
DEFAULT_TICKS = 2048

# ============================================================
# 全域變數：儲存從 Driver 收到的馬達資訊
# ============================================================
current_motor_state: Dict[int, int] = {}
motor_state_lock = threading.Lock()

# IK 自動相容層
import inspect
try:
    _IK_NPARAM = len(inspect.signature(compute_leg_ik).parameters)
except Exception:
    _IK_NPARAM = 8

def compute_ik_auto(lx, ly, lz, lyaw, rx, ry, rz, ryaw, lroll=0.0, lpitch=0.0, rroll=0.0, rpitch=0.0):
    if _IK_NPARAM >= 12:
        return compute_leg_ik(lx, ly, lz, lroll, lpitch, lyaw, rx, ry, rz, rroll, rpitch, ryaw)
    else:
        return compute_leg_ik(lx, ly, lz, lyaw, rx, ry, rz, ryaw)

walking = WalkingGaitByLIPM()
# _gait_dt = float(getattr(walking, "sample_time_", 30)) / 1000.0
_gait_dt = float(getattr(walking, "sample_time_", 20)) / 1000.0
DT = _gait_dt if _gait_dt > 0 else 1.0 / 33.333
CONTROL_RATE_HZ = 1.0 / DT

STATE_NAME = {StartStep: "StartStep", FirstStep: "FirstStep", Repeat: "Repeat", StopStep: "StopStep"}

# ============================================================
# 輔助函式
# ============================================================
def apply_ankle_compensation(ang_list: List[float]) -> List[float]:
    comp_deg = float(getattr(parameter, "compensation_swing_ankle", 0.0))
    if abs(comp_deg) < 1e-9: return ang_list
    comp_rad = math.radians(comp_deg)
    new_ang = list(ang_list)
    new_ang[5]  += comp_rad
    new_ang[11] -= comp_rad
    return new_ang

def calc_rel_gp_from_ang(ang_now: List[float], baseline_ang: List[float]) -> Dict[int, int]:
    rel: Dict[int, int] = {}
    for idx, mid in enumerate(IK_IDS):
        # 根據馬達型號判定解析度
        if mid in [20, 21, 26, 27]:
            tpr_val = 4096.0 / (2.0 * math.pi)       # X 系列
        else:
            tpr_val = 303750.0 / (2.0 * math.pi)     # H 系列 (PRO)
            
        delta = float(ang_now[idx]) - float(baseline_ang[idx])
        rel[mid] = int(round(DIR[mid] * tpr_val * delta))
    return rel

# ============================================================
# Parameter 管理
# ============================================================
PARAM_KEYS = ["period_t","Tdsp","COM_HEIGHT","STAND_HEIGHT","lift_height",
              "com_y_swing","COM_Y_shift","width_size","step_length","shift_length",
              "theta","THTA","compensation_swing_ankle"]

def get_param_dict():
    out = {}
    for k in PARAM_KEYS:
        if hasattr(parameter, k): out[k] = float(getattr(parameter, k))
    return out

def apply_param_dict(d):
    # 1. 先嘗試套用所有傳進來的參數
    for k, v in d.items():
        if hasattr(parameter, k):
            try: setattr(parameter, k, float(v))
            except Exception: pass
    
    # 2. 處理特殊計算 (Tc_)
    if hasattr(parameter, "COM_HEIGHT") and hasattr(parameter, "G"):
        try:
            if float(parameter.COM_HEIGHT) > 0.0 and float(parameter.G) > 0.0:
                parameter.Tc_ = math.sqrt(float(parameter.COM_HEIGHT) / float(parameter.G))
        except Exception: pass
    
    # 3. 更新 walking 物件的參數
    if hasattr(walking, "period_t"): walking.period_t = float(parameter.period_t)
    if hasattr(walking, "T_DSP_"):
        try: walking.T_DSP_ = max(0.0, min(1.0, float(parameter.Tdsp)))
        except Exception: pass
    if hasattr(walking, "width_size_"): walking.width_size_ = float(getattr(parameter, "width_size", 0.0))
    if hasattr(walking, "lift_height_"): walking.lift_height_ = float(getattr(parameter, "lift_height", 0.0))
    if hasattr(walking, "stand_height"): walking.stand_height = float(getattr(parameter, "STAND_HEIGHT", 0.0))
    if hasattr(walking, "com_height"): walking.com_height = float(getattr(parameter, "COM_HEIGHT", 0.0))
    if hasattr(walking, "step_length_"): walking.step_length_ = float(getattr(parameter, "step_length", 0.0))
    if hasattr(walking, "shift_length_"): walking.shift_length_ = float(getattr(parameter, "shift_length", 0.0))
    
    # =========================================================================
    # ★ 關鍵修改：強制鎖定 sample_time 為 30
    # =========================================================================
    # 不管上面讀到了什麼 (0 或亂碼)，這裡直接覆寫回 30
    # FIXED_SAMPLE_TIME = 30.0
    FIXED_SAMPLE_TIME = 20.0
    
    # 更新 Parameter 物件
    if hasattr(parameter, "sample_time"):
        parameter.sample_time = FIXED_SAMPLE_TIME
        
    # 更新 WalkingGait 物件 (這是防止 ZeroDivisionError 的關鍵)
    if hasattr(walking, "sample_time_"):
        walking.sample_time_ = FIXED_SAMPLE_TIME
        # print(f"[DEBUG] sample_time enforced to {FIXED_SAMPLE_TIME}") # 需要除錯時可打開
    # =========================================================================

    if hasattr(walking, "var_theta_"):
        theta_val = None
        if "THTA" in d: 
            try: theta_val = float(d["THTA"])
            except: pass
        elif "theta" in d:
            try: theta_val = float(d["theta"])
            except: pass
        if theta_val is None: theta_val = float(getattr(parameter, "THTA", 0.0))
        try: parameter.THTA = float(theta_val)
        except: pass
        walking.var_theta_ = float(theta_val)

# ============================================================
# Node 類別
# ============================================================
class WalkingNode(Node):
    def __init__(self):
        super().__init__('walking_strategy_node')
        self.cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # Web Interface Topics
        self.create_subscription(Int16, '/ContinousMode_Topic', self._mode_cb, 10)
        self.create_subscription(Interface, '/SendBodyAuto_Topic', self._cmd_cb, 10)
        self.create_subscription(Interface, '/ChangeContinuousValue_Topic', self._cmd_cb, 10)
        self.param_pub = self.create_publisher(String, '/walking_params', 10)
        self.create_subscription(String, '/walking_params_update', self._param_update_cb, 10)
        self.create_subscription(Bool, '/walking_params_request', self._param_request_cb, 10)

        self.req_reset_anchor: bool = False
        self.create_subscription(Bool, '/walking_reset_anchor', self._reset_anchor_cb, 10)

        self.walk_active: bool = False
        self.last_width_size: float = float(getattr(parameter, "width_size", 0.0))
        self.width_dirty: bool = False

        self.get_logger().info("Walking Node Initialized. Waiting for Driver Data...")
        self.publish_params()

    def _joint_state_cb(self, msg: JointState):
        with motor_state_lock:
            for i, name in enumerate(msg.name):
                try:
                    mid = int(name)
                    pos = int(msg.position[i])
                    current_motor_state[mid] = pos
                except ValueError:
                    pass

    def publish_command(self, target_ticks: Dict[int, int]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = []
        msg.position = []
        msg.velocity = []
        for mid, tick in target_ticks.items():
            msg.name.append(str(mid))
            msg.position.append(float(tick))
            msg.velocity.append(float(0))
        self.cmd_pub.publish(msg)

    def _mode_cb(self, msg: Int16):
        self.walk_active = (msg.data == 1)
        self.get_logger().info(f'recv /ContinousMode_Topic={msg.data} -> walk_active={self.walk_active}')

    # Callback 函式
    def _reset_anchor_cb(self, msg: Bool):
        if msg.data:
            self.req_reset_anchor = True
            self.get_logger().warn("[Request] Received Anchor Reset Flag! Next 'Generate' will update baseline.")

    def _cmd_cb(self, msg: Interface):
        scale = 0.001; theta_scale = 0.01
        parameter.step_length  = float(msg.x) * scale
        parameter.shift_length = float(msg.y) * scale
        parameter.THTA         = float(math.radians(msg.theta)) * theta_scale
        parameter.X = float(msg.x) * scale; parameter.Y = float(msg.y) * scale; parameter.theta = float(msg.theta) * theta_scale
        walking.step_length_  = float(msg.x) * scale
        walking.shift_length_ = float(msg.y) * scale
        walking.var_theta_    = float(math.radians(msg.theta)) * theta_scale

    def publish_params(self):
        d = get_param_dict()
        msg = String()
        msg.data = json.dumps(d)
        self.param_pub.publish(msg)

    def _param_request_cb(self, msg: Bool):
        if msg.data: self.publish_params()

    def _param_update_cb(self, msg: String):
        try: d = json.loads(msg.data)
        except Exception as e: return
        apply_param_dict(d)
        if bool(d.get("generate", False)):
            self.walk_active = not self.walk_active
            self.get_logger().info(f"[generate] toggle walk_active -> {self.walk_active}")
        if "width_size" in d:
            try:
                new_w = float(d["width_size"])
                if abs(new_w - self.last_width_size) > 1e-6:
                    self.last_width_size = new_w
                    self.width_dirty = True
            except: pass
        self.publish_params()

# ============================================================
# Main Loop
# ============================================================
def main():
    print(f"[info] Control Loop DT={DT*1000:.1f} ms ({CONTROL_RATE_HZ:.2f} Hz)")
    rclpy.init(args=None)
    node = WalkingNode()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 變數宣告
    baseline_ticks: Dict[int, int] = {} # 這是目前的基準 (走路用的)
    baseline_ang: List[float] = []      # 這是目前的角度 (走路用的)
    
    # ★ 新增：永遠不變的初始錨點 (用來計算 Idle 站姿變化)
    initial_ticks: Dict[int, int] = {}  
    initial_ang: List[float] = []       
    
    was_walking = False 
    stopping_active = False       
    saved_lift_height = None
    
    landing_mode = False       
    landing_counter = 0        
    LANDING_LIMIT = 20         

    # 參數追蹤
    # ★ 設定初始預設值 (請確保這裡跟 Parameter.py 預設一樣)
    start_w = 4.5 
    start_h = 23.5
    last_applied_width = start_w
    last_applied_height = start_h
    last_applied_com = float(getattr(parameter, "COM_HEIGHT", 0.0))
    WIDTH_EPS = 1e-6

    t_next = time.perf_counter() + DT
    last_dbg_t = 0.0

    try:
        print("[setup] Waiting for motor data...")
        while rclpy.ok():
            with motor_state_lock:
                if len([mid for mid in LEG_IDS if mid in current_motor_state]) == len(LEG_IDS):
                    print(f"[setup] Connection Established!")
                    break
            time.sleep(0.1)
        
        # ---------------------------------------------------------
        # ★ 修改：這裡不再抓取 initial_ticks 了！
        # 我們留到按下 Generate 之後才抓，這樣你才有機會手動扶正機器人
        # ---------------------------------------------------------
        print(f"[setup] Ready. Please enable 'Generate' to set the Anchor Pose.")

        while rclpy.ok():
            is_walking = node.walk_active
            now = time.perf_counter()
            ws = getattr(walking, "walking_state", None)

            # Debug Log
            # if now - last_dbg_t > 1.0:
            #     last_dbg_t = now
            #     mode_str = "LANDING_Override" if landing_mode else ("WALKING" if (is_walking or stopping_active) else "IDLE")
            #     state_name = STATE_NAME.get(ws, str(ws))
            #     current_w = float(getattr(walking, "width_size_", 0.0))
            #     # 顯示目前是否已經定錨
            #     anchor_status = "SET" if initial_ticks else "NOT_SET"
            #     # print(f"[Loop] Mode={mode_str} | Anchor={anchor_status} | Width={current_w:.2f}", flush=True)

            # ==========================================================
            # 1. Generate ON (走路開始) --> ★ 在這裡定錨！
            # ==========================================================
            if is_walking and not was_walking:
                print("\n[Generate ON] STARTING...")
                
                # 判斷是否需要重抓錨點
                if (not initial_ticks) or node.req_reset_anchor:
                    if node.req_reset_anchor:
                        print("[Reset] Forced Anchor Reset triggered by Motion!")
                        node.req_reset_anchor = False

                    with motor_state_lock:
                        current_snapshot = dict(current_motor_state)
                        baseline_ticks = dict(current_snapshot)
                        initial_ticks  = dict(current_snapshot)
                    
                    cur_w_start = walking.width_size_
                    cur_h_start = walking.stand_height
                    
                    start_ik_ang = compute_ik_auto(
                        0, cur_w_start/2, cur_h_start, 0,
                        0, -cur_w_start/2, cur_h_start, 0,
                        0, 0, 0, 0
                    )
                    start_ik_ang = apply_ankle_compensation(start_ik_ang)
                    initial_ang = list(start_ik_ang)
                    
                    print(f"[Generate ON] Anchor Updated. W={cur_w_start}, H={cur_h_start}")
                else:
                    print("Resuming: Using existing Baseline (Fixes Drift Issue).")
                    pass  # else 結束在這裡

                # =========================================================
                # ★ 修正重點：這一段必須「跳出」else，向左縮排！
                # 確保無論是 Reset 還是 Resume，這裡都會被執行到
                # =========================================================
                walking.final_step()
                b_ang = compute_ik_auto(
                    walking.end_point_lx_, walking.end_point_ly_, walking.end_point_lz_, walking.end_point_lthta_,
                    walking.end_point_rx_, walking.end_point_ry_, walking.end_point_rz_, walking.end_point_rthta_,
                    lroll=getattr(walking, "end_point_lroll_", 0.0),
                    rroll=getattr(walking, "end_point_rroll_", 0.0),
                )
                baseline_ang = apply_ankle_compensation(b_ang)
                stopping_active = False
                landing_mode = False
                landing_counter = 0
                if saved_lift_height is not None:
                    try: walking.lift_height_ = float(saved_lift_height)
                    except: pass
                    saved_lift_height = None
                    
            # ==========================================================
            # 2. Generate OFF (走路停止)
            # ==========================================================
            if not is_walking and was_walking:
                # ... (維持原樣) ...
                print("[Generate OFF] Stopping requested...")
                stopping_active = True
                landing_mode = False
                try:
                    walking.ready_to_stop_ = True
                    walking.step_ = walking.now_step_ + 1
                except: pass
                node.width_dirty = False

            was_walking = is_walking

            # 3. Landing Logic ... (維持原樣) ...
            if stopping_active and not landing_mode:
                rts = bool(getattr(walking, "ready_to_stop_", False))
                if (ws == StartStep) and (not rts):
                    landing_mode = True
                    landing_counter = 0

            # ==========================================================
            # 分支 A: 走路中
            # ==========================================================
            if (is_walking or stopping_active) and not landing_mode:
                if stopping_active and saved_lift_height is None:
                     saved_lift_height = float(getattr(walking, "lift_height_", 0.0))

                walking.process()

                ang_now = compute_ik_auto(
                    walking.end_point_lx_, walking.end_point_ly_, walking.end_point_lz_, walking.end_point_lthta_,
                    walking.end_point_rx_, walking.end_point_ry_, walking.end_point_rz_, walking.end_point_rthta_,
                    lroll=getattr(walking, "end_point_lroll_", 0.0), rroll=getattr(walking, "end_point_rroll_", 0.0)
                )
                ang_now = apply_ankle_compensation(ang_now)
                
                # 走路時使用動態更新的 baseline
                if baseline_ang:
                    rel_cmd = calc_rel_gp_from_ang(ang_now, baseline_ang)
                    gp_abs = {}
                    for mid in LEG_IDS:
                        base = baseline_ticks.get(mid, DEFAULT_TICKS)
                        delta = rel_cmd.get(mid, 0)
                        
                        if mid in [20, 21, 26, 27]:
                            gp_abs[mid] = int((base + delta) % 4096) # X 系列維持取餘數
                        else:
                            gp_abs[mid] = int(base + delta)           # H 系列不取餘數
                    node.publish_command(gp_abs)

            # ==========================================================
            # 分支 B: 落地模式
            # ==========================================================
            elif landing_mode:
                if baseline_ticks:
                    node.publish_command(baseline_ticks)
                landing_counter += 1
                if landing_counter >= LANDING_LIMIT:
                    print(f"[state] Landing finished. Restored Baseline.")
                    stopping_active = False
                    landing_mode = False
                    if saved_lift_height is not None:
                        try: walking.lift_height_ = float(saved_lift_height)
                        except: pass
                        saved_lift_height = None

            # ==========================================================
            # 分支 C: Idle 站姿調整
            # ==========================================================
            else:
                cur_w = float(getattr(parameter, "width_size", 0.0))
                cur_h = float(getattr(parameter, "STAND_HEIGHT", 0.0))
                cur_com = float(getattr(parameter, "COM_HEIGHT", 0.0))
                is_dirty = node.width_dirty

                width_changed = abs(cur_w - last_applied_width) > WIDTH_EPS
                height_changed = abs(cur_h - last_applied_height) > WIDTH_EPS
                com_changed = abs(cur_com - last_applied_com) > WIDTH_EPS

                if width_changed or height_changed or com_changed or is_dirty:
                    # ★ 防呆：如果還沒按過 Generate，initial_ticks 會是空的，這時不准調整
                    if not initial_ticks:
                        # print(f"[Idle] Ignored adjustment. Please toggle 'Generate' at least once to set Anchor.")
                        # 這裡要更新 last_applied 以免迴圈一直印 Log
                        last_applied_width = cur_w
                        last_applied_height = cur_h
                        last_applied_com = cur_com
                        continue

                    print(f"[Idle] Recomputing Stand... Target: W={cur_w}, H={cur_h}")
                    node.width_dirty = False
                    
                    # 1. 更新參數
                    walking.width_size_ = cur_w
                    walking.stand_height = cur_h if cur_h > 1.0 else 23.5
                    
                    # 2. 計算目標 IK
                    stand_ang = compute_ik_auto(
                        0, cur_w/2, walking.stand_height, 0,
                        0, -cur_w/2, walking.stand_height, 0,
                        0, 0, 0, 0
                    )
                    stand_ang = apply_ankle_compensation(stand_ang)
                    
                    # 3. 計算相對於「初始錨點 (initial_ang)」的變化量
                    if initial_ang and initial_ticks:
                        rel_cmd = calc_rel_gp_from_ang(stand_ang, initial_ang)
                        
                        new_ticks = {}
                        for mid in LEG_IDS:
                            base = initial_ticks.get(mid, DEFAULT_TICKS)
                            delta = rel_cmd.get(mid, 0)
                            
                            # if abs(delta) > 800:
                            #     print(f"[WARNING] Motor {mid} delta {delta} too large! Ignored.")
                            #     delta = 0
                            
                            new_val = int((base + delta) % 4096)
                            new_ticks[mid] = new_val
                        
                        # 發送指令
                        node.publish_command(new_ticks)
                        
                        # 更新 Baseline (為了讓下次走路是從這個新姿勢開始)
                        baseline_ticks = dict(new_ticks)
                        baseline_ang = list(stand_ang)
                        print(f"[Idle] Done. Baseline moved to W={cur_w}")
                    
                    last_applied_width = cur_w
                    last_applied_height = cur_h
                    last_applied_com = cur_com

            # 頻率鎖定
            now = time.perf_counter()
            if now < t_next: time.sleep(t_next - now); t_next += DT
            else: skipped = int((now - t_next) // DT) + 1; t_next += DT * skipped

    except KeyboardInterrupt:
        print("\n[Ctrl+C]")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[final] done.")

if __name__ == "__main__":
    main()
