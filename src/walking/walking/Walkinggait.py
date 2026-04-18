# -*- coding: utf-8 -*-
"""
Python port of WalkingGaitByLIPM::process() + math helpers implemented
(安全修正版：偶數離散 + ε 緩衝 + 開區間，避免 Tdsp=0 邊界抖動)
"""

import math
from . import Parameter as parameter

StartStep = 0
FirstStep = 1
Repeat    = 3
StopStep  = 4

class WalkingGaitByLIPM:
    def __init__(self):
        # ====== 參數（從 Parameter 讀）======
        self.period_t      = parameter.period_t           # [ms]   每一步的時間
        self.sample_time_  = parameter.sample_time        # [ms]   取樣時間
        self.T_DSP_        = parameter.Tdsp               # [0..1] 雙支撐占整步的比例
        self.lift_height_  = parameter.lift_height        # [cm]   抬腳高度
        self.width_size_   = parameter.width_size         # [cm]   步寬全寬(左右腳總距離，gait 內用 half=width/2)
        self.step_length_  = parameter.step_length        # [cm]   步長
        self.shift_length_ = parameter.shift_length       # [cm]   側移
        self.com_y_swing   = getattr(parameter, "com_y_swing", 0.0)     # [cm]   質心側擺幅度

        self.g_            = parameter.G                  # [cm/s^2] 重力加速度
        self.com_z_height  = parameter.COM_HEIGHT         # [cm]   質心高度
        self.stand_height  = parameter.STAND_HEIGHT       # [cm]   站姿高度
        self.length_pelvis = parameter.LENGTH_PELVIS      # [cm]   骨盆寬度

        self.STARTSTEPCOUNTER = getattr(parameter, "STARTSTEPCOUNTER", 2)

        # ====== 內部狀態 ======
        self.sample_point_ = 0                            # 全域 sample 計數
        self.time_point_   = 0.0                          # 全域時間 ms
        self.Tc_           = math.sqrt(self.com_z_height / self.g_)  # 自然週期
        self.TT_           = self.period_t * 0.001                   # [s] 一步時間
        self.t_            = 0.0                                     # [s] 週期相位時間

        self.now_step_ = 0
        self.pre_step_ = -1
        self.step_     = 999999
        self.ready_to_stop_ = False

        self.walking_state = StartStep
        self.if_finish_    = False
        self.plot_once_    = False

        self.swing_leg_ = "none"   # "left" / "right" / "none"

        # 下一步預計落腳點（世界）
        self.footstep_x = 0.0
        half_w = 0.5 * self.width_size_
        self.footstep_y = half_w
        
        # 下一步 COM 目標中心
        self.base_x     = 0.0
        self.base_y     = 0.0
        self.last_base_x = 0.0
        self.last_base_y = 0.0

        self.displacement_x = 0.0
        self.displacement_y = 0.0
        self.last_displacement_x = 0.0
        self.last_displacement_y = 0.0

        # 當下支撐腳 ZMP（世界）
        self.zmp_x = 0.0
        self.zmp_y = 0.0
        self.last_zmp_x = 0.0
        self.last_zmp_y = 0.0

        # 腳當前落腳（世界）
        self.now_right_x_ = 0.0
        half_w = 0.5 * self.width_size_
        self.now_right_y_ = -half_w
        self.now_left_x_  = 0.0
        self.now_left_y_  =  half_w

        # 朝向
        self.theta_      = 0.0
        self.var_theta_  = 0.0
        self.last_theta_ = 0.0
        self.abs_theta_  = 0.0
        self.last_abs_theta_ = 0.0

        self.now_width_ = 0.0
        self.width_x = 0.0
        self.width_y = 0.0

        self.is_parameter_load_ = False
        self.delay_push_ = False
        self.push_data_  = False

        # COM（可由外部感測覆寫）
        self.com_x = 0.0
        self.com_y = 0.0

        self.left_step_  = 0
        self.right_step_ = 0
        self.StartHeight_ = 0.0

        # 輸出給 IK/下游（COM 參考 + 足端軌跡）
        self.vx0_ = self.vy0_ = 0.0
        self.px_ = self.py_ = 0.0
        self.px_u = self.py_u = 0.0
        self.pz_ = self.com_z_height  # 對齊 C++ 用於足端 z 計算

        self.lpx_ = self.lpy_ = self.lpz_ = 0.0
        self.rpx_ = self.rpy_ = self.rpz_ = 0.0
        self.lpt_ = 0.0
        self.rpt_ = 0.0

        # 中間/末端點（body frame）
        self.step_point_lx_ = self.step_point_ly_ = self.step_point_lz_ = 0.0
        self.step_point_rx_ = self.step_point_ry_ = self.step_point_rz_ = 0.0
        self.step_point_lthta_ = 0.0
        self.step_point_rthta_ = 0.0

        self.end_point_lx_ = self.end_point_ly_ = self.end_point_lz_ = 0.0
        self.end_point_rx_ = self.end_point_ry_ = self.end_point_rz_ = 0.0
        self.end_point_lthta_ = 0.0
        self.end_point_rthta_ = 0.0

        # 內部離散設定（在 process() 動態更新）
        self._N = 0
        self._eps = 1e-6

    # ====== 主流程（同前一版）======
    def process(self):
        # 把外部可調參數讀進來
        self.readWalkParameter()

        # 全域取樣計數
        self.sample_point_ += 1
        # 把取樣點轉成相對時間刻度（單位 ms）
        self.time_point_ = self.sample_point_ * self.sample_time_

        # LIPM 自然週期常數
        self.Tc_ = math.sqrt(self.com_z_height / self.g_)
        # 一整步的時間
        self.TT_ = self.period_t * 0.001

        # 設定偶數離散數 N 與邊界緩衝 eps（避免 Tdsp=0 邊界抖動）
        dt = float(self.sample_time_) / 1000.0
        N  = max(2, int(round(self.TT_ / dt)))
        if N % 2 == 1:
            N += 1                     # 強制偶數，讓 0.5 不落在格點
        self._N = N
        self._eps = max(2.0 / N, 1e-6) # 至少兩格當緩衝

        # 這一步內的相位時間（秒）
        self.t_ = ((self.time_point_ - int(self.sample_time_)) % self.period_t + self.sample_time_) / 1000.0
        # 依取樣點落在第幾個步
        self.now_step_ = int((self.sample_point_ - 1) / int(self.period_t / self.sample_time_))

        # ====== 步態狀態更新 ======
        if self.now_step_ == self.step_:
            # 最後一步收回站姿
            self.walking_state = StopStep
        elif self.now_step_ < self.STARTSTEPCOUNTER:
            # 只左右換腳，不前進
            self.walking_state = StartStep
        elif self.now_step_ > self.step_:
            self.if_finish_ = True
            self.plot_once_ = True
        elif self.now_step_ == self.STARTSTEPCOUNTER:
            # 第一次真正跨出去
            self.walking_state = FirstStep
        else:
            # 穩態走路
            self.walking_state = Repeat

        # 跨步邏輯
        if self.pre_step_ != self.now_step_:
            # 計數左右步數
            if (self.now_step_ % 2) == 1 and self.now_step_ > 1:
                self.left_step_ += 1
            elif (self.now_step_ % 2) == 0 and self.now_step_ > 1:
                self.right_step_ += 1

            # 更新 now_right_* 或 now_left_* 的落地點
            if (self.pre_step_ % 2) == 1:
                self.now_right_x_ = self.footstep_x
                self.now_right_y_ = self.footstep_y
            elif (self.pre_step_ % 2) == 0:
                self.now_left_x_ = self.footstep_x
                self.now_left_y_ = self.footstep_y
            elif self.pre_step_ == -1:
                self.footstep_x = 0.0
                half_w = 0.5 * self.width_size_
                self.footstep_y = -half_w
                self.now_right_x_ = self.footstep_x
                self.now_right_y_ = self.footstep_y
                self.now_left_x_  = 0.0
                self.now_left_y_  = half_w

            # ZMP跟支撐腳落點一致
            self.last_zmp_x = self.zmp_x
            self.last_zmp_y = self.zmp_y
            self.zmp_x = self.footstep_x
            self.zmp_y = self.footstep_y
            
            # 保存上一個displacement/base/theta
            self.last_displacement_x = self.displacement_x
            self.last_displacement_y = self.displacement_y
            self.last_base_x = self.base_x
            self.last_base_y = self.base_y
            self.last_theta_ = self.var_theta_
            self.last_abs_theta_ = self.abs_theta_
            self.is_parameter_load_ = False

            self.readWalkData()
            
            # 起步，只有左右換支撐，沒有前進步長
            if self.walking_state == StartStep:
                self.theta_ = 0.0
                self.var_theta_ = 0.0
                self.now_width_ = self.width_size_ * (-pow(-1.0, self.now_step_ + 1))  # FULL width
                self.width_x = -math.sin(self.theta_) * self.now_width_
                self.width_y =  math.cos(self.theta_) * self.now_width_
                self.displacement_x = self.width_x
                self.displacement_y = self.width_y
                self.footstep_x += self.width_x
                self.footstep_y += self.width_y

            # 結束，把最後一步收回站姿寬度
            elif self.walking_state == StopStep:
                self.theta_ = self.var_theta_
                self.now_width_ = self.width_size_ * (-pow(-1.0, self.now_step_ + 1))  # FULL width
                self.width_x = -math.sin(self.theta_) * self.now_width_
                self.width_y =  math.cos(self.theta_) * self.now_width_
                self.displacement_x = self.width_x
                self.displacement_y = self.width_y
                self.footstep_x += self.width_x
                self.footstep_y += self.width_y

            # 前進與側移
            else:
                self.theta_ = self.var_theta_
                self.now_width_ = self.width_size_ * (-pow(-1.0, self.now_step_ + 1))  # FULL width
                self.width_x = -math.sin(self.theta_) * self.now_width_
                self.width_y =  math.cos(self.theta_) * self.now_width_
                self.displacement_x = (self.step_length_ * math.cos(self.theta_) - self.shift_length_ * math.sin(self.theta_)) + self.width_x
                self.displacement_y = (self.step_length_ * math.sin(self.theta_) + self.shift_length_ * math.cos(self.theta_)) + self.width_y
                self.footstep_x += self.displacement_x
                self.footstep_y += self.displacement_y

            # 更新 base（COM 的目標中心
            self.base_x = (self.footstep_x + self.zmp_x) / 2.0
            self.base_y = (self.footstep_y + self.zmp_y) / 2.0

        self.pre_step_ = self.now_step_

        if self.ready_to_stop_:
            self.step_ = self.now_step_ + 1
            self.ready_to_stop_ = False

        # === 狀態分派 ===
        t, T, Tc, Tdsp = self.t_, self.TT_, self.Tc_, self.T_DSP_
        if self.walking_state == StartStep:
            # === StartStep ===
            self.vx0_ = self.wComVelocityInit(0.0, 0.0, self.zmp_x, T, Tc)
            self.px_  = self.wComPosition(0.0, self.vx0_, self.zmp_x, t, Tc)
            self.vy0_ = self.wComVelocityInit(0.0, 0.0, self.zmp_y, T, Tc)

            # 奇偶步鏡像的 COM-y 擺動（幅值相同、符號相反）
            self.py_ = self.wComPosition(0.0, self.vy0_, self.zmp_y, t, Tc)

            if (self.now_step_ % 2) == 1:
                # 奇數步：右腳擺
                self.lpx_, self.lpy_, self.lpz_ = self.zmp_x, self.zmp_y, 0.0
                self.rpx_ = self.wFootPositionRepeat(self.now_right_x_, 0.0, t, T, Tdsp)
                self.rpy_ = self.wFootPositionRepeat(self.now_right_y_, 0.0, t, T, Tdsp)
                self.rpz_ = self.wFootPositionZ(self.lift_height_ * (2/3), t, T, Tdsp)
                self.lpt_, self.rpt_ = 0.0, 0.0
            else:
                # 偶數步：左腳擺
                self.rpx_, self.rpy_, self.rpz_ = self.zmp_x, self.zmp_y, 0.0
                self.lpx_ = self.wFootPositionRepeat(self.now_left_x_, 0.0, t, T, Tdsp)
                self.lpy_ = self.wFootPositionRepeat(self.now_left_y_, 0.0, t, T, Tdsp)
                self.lpz_ = self.wFootPositionZ(self.lift_height_ * (2/3), t, T, Tdsp)
                self.lpt_, self.rpt_ = 0.0, 0.0

        elif self.walking_state == FirstStep:
            self.vx0_ = self.wComVelocityInit(0.0, self.base_x, self.zmp_x, T, Tc)
            self.px_  = self.wComPosition(0.0, self.vx0_, self.zmp_x, t, Tc)
            self.vy0_ = self.wComVelocityInit(0.0, self.base_y, self.zmp_y, T, Tc)
            self.py_  = self.wComPosition(0.0, self.vy0_, self.zmp_y, t, Tc)
            self.lpx_ = self.wFootPosition(self.now_left_x_, self.displacement_x, t, T, Tdsp)
            self.lpy_ = self.wFootPosition(self.now_left_y_, self.displacement_y - self.now_width_, t, T, Tdsp)
            self.lpz_ = self.wFootPositionZ(self.lift_height_, t, T, Tdsp)
            self.rpx_, self.rpy_, self.rpz_ = self.zmp_x, self.zmp_y, 0.0
            self.lpt_ = self.wFootTheta(-self.last_theta_, 1, t, T, Tdsp)
            self.rpt_ = self.wFootTheta(-self.var_theta_, 0, t, T, Tdsp)

        elif self.walking_state == StopStep:
            self.vx0_ = self.wComVelocityInit(self.last_base_x, self.base_x, self.zmp_x, T, Tc)
            self.px_  = self.wComPosition(self.last_base_x, self.vx0_, self.zmp_x, t, Tc)
            self.vy0_ = self.wComVelocityInit(self.last_base_y, self.base_y, self.zmp_y, T, Tc)
            self.py_  = self.wComPosition(self.last_base_y, self.vy0_, self.zmp_y, t, Tc)
            if (self.now_step_ % 2) == 1:
                self.lpx_, self.lpy_, self.lpz_ = self.zmp_x, self.zmp_y, 0.0
                self.rpx_ = self.wFootPosition(self.now_right_x_, (self.last_displacement_x + self.displacement_x), t, T, Tdsp)
                self.rpy_ = self.wFootPosition(self.now_right_y_, (self.last_displacement_y + self.displacement_y), t, T, Tdsp)
                self.rpz_ = self.wFootPositionZ(self.lift_height_, t, T, Tdsp)
                self.lpt_, self.rpt_ = 0.0, self.wFootTheta(-self.last_theta_, 1, t, T, Tdsp)
            else:
                self.rpx_, self.rpy_, self.rpz_ = self.zmp_x, self.zmp_y, 0.0
                self.lpx_ = self.wFootPosition(self.now_left_x_, (self.last_displacement_x + self.displacement_x), t, T, Tdsp)
                self.lpy_ = self.wFootPosition(self.now_left_y_, (self.last_displacement_y + self.displacement_y), t, T, Tdsp)
                self.lpz_ = self.wFootPositionZ(self.lift_height_, t, T, Tdsp)
                self.lpt_, self.rpt_ = self.wFootTheta(-self.last_theta_, 1, t, T, Tdsp), 0.0

        else:  # Repeat
            self.vx0_ = self.wComVelocityInit(self.last_base_x, self.base_x, self.zmp_x, T, Tc)
            self.px_  = self.wComPosition(self.last_base_x, self.vx0_, self.zmp_x, t, Tc)
            self.vy0_ = self.wComVelocityInit(self.last_base_y, self.base_y, self.zmp_y, T, Tc)
            self.py_  = self.wComPosition(self.last_base_y, self.vy0_, self.zmp_y, t, Tc)
            if (self.now_step_ % 2) == 1:
                self.lpx_, self.lpy_, self.lpz_ = self.zmp_x, self.zmp_y, 0.0
                self.rpx_ = self.wFootPositionRepeat(self.now_right_x_, (self.last_displacement_x + self.displacement_x) / 2.0, t, T, Tdsp)
                self.rpy_ = self.wFootPositionRepeat(self.now_right_y_, (self.last_displacement_y + self.displacement_y) / 2.0, t, T, Tdsp)
                self.rpz_ = self.wFootPositionZ(self.lift_height_, t, T, Tdsp)
                if self.var_theta_ * self.last_theta_ >= 0:
                    self.lpt_ = self.wFootTheta(-self.var_theta_, 0, t, T, Tdsp)
                    self.rpt_ = self.wFootTheta(-self.last_theta_, 1, t, T, Tdsp)
                else:
                    self.lpt_ = 0.0
                    self.rpt_ = 0.0
            else:
                self.rpx_, self.rpy_, self.rpz_ = self.zmp_x, self.zmp_y, 0.0
                self.lpx_ = self.wFootPositionRepeat(self.now_left_x_, (self.last_displacement_x + self.displacement_x) / 2.0, t, T, Tdsp)
                self.lpy_ = self.wFootPositionRepeat(self.now_left_y_, (self.last_displacement_y + self.displacement_y) / 2.0, t, T, Tdsp)
                self.lpz_ = self.wFootPositionZ(self.lift_height_, t, T, Tdsp)
                if self.var_theta_ * self.last_theta_ >= 0:
                    self.lpt_ = self.wFootTheta(-self.last_theta_, 1, t, T, Tdsp)
                    self.rpt_ = self.wFootTheta(-self.var_theta_, 0, t, T, Tdsp)
                else:
                    self.lpt_ = 0.0
                    self.rpt_ = 0.0

        # 內回授微調
        self.py_u = self.py_
        self.px_u = self.px_
        self.pz_  = self.com_z_height

        # 保證本幀左右腳輸出欄位完整
        for name in ("lpx_","lpy_","lpz_","rpx_","rpy_","rpz_","lpt_","rpt_"):
            if getattr(self, name, None) is None:
                setattr(self, name, 0.0)

        self.coordinate_transformation()
        self.coordinate_offset()

        if self.now_step_ > self.step_:
            self.delay_push_ = True
            self.final_step()
        else:
            self.push_data_ = True

    # ====== 參數同步 ======
    def readWalkParameter(self):
        self.period_t           = parameter.period_t
        self.sample_time_       = parameter.sample_time
        self.T_DSP_             = parameter.Tdsp
        self.lift_height_       = parameter.lift_height
        self.width_size_        = parameter.width_size
        self.step_length_       = parameter.step_length
        self.shift_length_      = parameter.shift_length
        self.com_y_swing        = getattr(parameter, "com_y_swing", self.com_y_swing)
        self.g_                 = parameter.G
        self.com_z_height       = parameter.COM_HEIGHT
        self.stand_height       = parameter.STAND_HEIGHT
        self.length_pelvis      = parameter.LENGTH_PELVIS
        self.STARTSTEPCOUNTER   = getattr(parameter, "STARTSTEPCOUNTER", self.STARTSTEPCOUNTER)
        self.step_length_       = float(getattr(parameter, "step_length", 0.0))
        self.shift_length_      = float(getattr(parameter, "shift_length", 0.0))
        self.var_theta_         = float(getattr(parameter, "theta", 0.0))
        
    def readWalkData(self):
        def _set_param(name, value):
            if hasattr(parameter, name):
                setattr(parameter, name, value)

        if self.pre_step_ != self.now_step_:
            self.step_length_  = getattr(parameter, "X",    self.step_length_)
            self.shift_length_ = getattr(parameter, "Y",    self.shift_length_)
            thta_in            = getattr(parameter, "THTA", self.var_theta_)

            if (self.var_theta_ >= 0 and (self.pre_step_ % 2) == 1) or \
               (self.var_theta_ <= 0 and (self.pre_step_ % 2) == 0):
                self.var_theta_ = thta_in

            self.abs_theta_ = abs(self.var_theta_)

            stepout_x  = getattr(parameter, "Stepout_flag_X_", False)
            stepout_y  = getattr(parameter, "Stepout_flag_Y_", False)
            ctrl_x     = getattr(parameter, "Control_Step_length_X_", 0.0)
            ctrl_y     = getattr(parameter, "Control_Step_length_Y_", 0.0)
            step_count = int(getattr(parameter, "Step_Count_", 0))

            if (stepout_x or stepout_y) and step_count >= 2:
                stepout_x  = False
                stepout_y  = False
                ctrl_x     = 0.0
                ctrl_y     = 0.0
                step_count = 0
            elif (stepout_x or stepout_y) and (step_count <= 1):
                bad_dir = ((self.pre_step_ % 2) == 0 and (ctrl_y < 0)) or \
                          ((self.pre_step_ % 2) == 1 and (ctrl_y > 0))
                if not bad_dir:
                    step_count += 1

            _set_param("Stepout_flag_X_", stepout_x)
            _set_param("Stepout_flag_Y_", stepout_y)
            _set_param("Control_Step_length_X_", ctrl_x)
            _set_param("Control_Step_length_Y_", ctrl_y)
            _set_param("Step_Count_", step_count)
            self.is_parameter_load_ = True

    # ====== LIPM / 足端軌跡（加上開區間 + ε）======
    def wComVelocityInit(self, x0, xt, px, t, T):
        ct = math.cosh(t / T)
        st = math.sinh(t / T)
        return (xt - x0 * ct + px * (ct - 1.0)) / (T * st)

    def wComPosition(self, x0, vx0, px, t, T):
        ct = math.cosh(t / T)
        st = math.sinh(t / T)
        return x0 * ct + T * vx0 * st - px * (ct - 1.0)

    def wFootPositionRepeat(self, start, length, t, T, T_DSP):
        new_T = T * (1.0 - T_DSP)
        new_t = t - T * T_DSP / 2.0
        omega = 2.0 * math.pi / new_T
        a = T * T_DSP / 2.0 + self._eps             # 開區間下界
        b = T * (1.0 - T_DSP / 2.0) - self._eps     # 開區間上界
        if t < a:
            return start
        elif t < b:
            return 2.0 * length * (omega * new_t - math.sin(omega * new_t)) / (2.0 * math.pi) + start
        else:
            return 2.0 * length + start

    def wFootPosition(self, start, length, t, T, T_DSP):
        new_T = T * (1.0 - T_DSP)
        new_t = t - T * T_DSP / 2.0
        omega = 2.0 * math.pi / new_T
        a = T * T_DSP / 2.0 + self._eps
        b = T * (1.0 - T_DSP / 2.0) - self._eps
        if t < a:
            return start
        elif t < b:
            return length * (omega * new_t - math.sin(omega * new_t)) / (2.0 * math.pi) + start
        else:
            return length + start

    def wFootPositionZ(self, lift_height, t, T, Tdsp):
        """
        修改為平滑升餘弦軌跡：
        讓起始速度 (v=0) 與 終點速度 (v=0) 均平滑對接。
        """
        if T <= 0: return 0.0
        Tdsp = max(0.0, min(float(Tdsp), 1.0))

        t = float(t)
        T = float(T)
        dsp_end = Tdsp * T
        
        if t <= dsp_end: return 0.0

        ssp_T = (1.0 - Tdsp) * T
        if ssp_T <= 0.0: return 0.0

        # 歸一化相位 s ∈ [0, 1]
        s = (t - dsp_end) / ssp_T
        if s <= 0.0 or s >= 1.0: return 0.0

        # --- 修改重點：從拋物線改為餘弦平滑曲線 ---
        H = float(lift_height)
        # 此公式確保 s=0 時速度為 0，s=0.5 時達到最高點 H，s=1 時速度回到 0
        return H * 0.5 * (1.0 - math.cos(2.0 * math.pi * s))
    
    # def wFootPositionZ(self, lift_height, t, T, Tdsp):
    #     """
    #     五階多項式軌跡 (Quintic Polynomial Trajectory)
    #     實現起點與終點的速度、加速度皆為 0，達到軟著陸效果。
    #     """
    #     if T <= 0:
    #         return 0.0
    #     Tdsp = max(0.0, min(float(Tdsp), 1.0))

    #     t = float(t)
    #     T = float(T)

    #     # DSP 雙支撐期間：腳掌貼地
    #     dsp_end = Tdsp * T
    #     if t <= dsp_end:
    #         return 0.0

    #     # SSP 單支撐期間長度
    #     ssp_T = (1.0 - Tdsp) * T
    #     if ssp_T <= 0.0:
    #         return 0.0

    #     # 歸一化相位 s ∈ [0, 1]
    #     s = (t - dsp_end) / ssp_T
    #     if s <= 0.0 or s >= 1.0:
    #         return 0.0
        
    #     H = float(lift_height)
    #     if s <= 0.5:
    #         # 前半段：從 0 升到 H (將 s 映射到 0~1 區間使用五階公式)
    #         s_half = s * 2.0
    #         return H * (10 * pow(s_half, 3) - 15 * pow(s_half, 4) + 6 * pow(s_half, 5))
    #     else:
    #         # 後半段：從 H 降到 0
    #         s_half = (1.0 - s) * 2.0
    #         return H * (10 * pow(s_half, 3) - 15 * pow(s_half, 4) + 6 * pow(s_half, 5))

    def wFootTheta(self, theta, reverse, t, T, T_DSP):
        new_T = T * (1.0 - T_DSP)
        new_t = t - T * T_DSP / 2.0
        omega = 2.0 * math.pi / new_T
        a = T * T_DSP / 2.0 + self._eps
        b = T * (1.0 - T_DSP / 2.0) - self._eps
        if t < a and not reverse:
            return 0.0
        elif t < a and reverse:
            return theta
        elif t < b and not reverse:
            return 0.5 * theta * (1.0 - math.cos(0.5 * omega * (new_t)))
        elif t < b and reverse:
            return 0.5 * theta * (1.0 - math.cos(0.5 * omega * (new_t - new_T)))
        elif t >= b and not reverse:
            return theta
        elif t >= b and reverse:
            return 0.0
        else:
            return 0.0

    # ====== 座標轉換與偏移 ======
    def coordinate_transformation(self):
        # W -> B 平移（以 px_u, py_u 為 COM 參考）
        step_point_lx_W = self.lpx_ - self.px_u
        step_point_rx_W = self.rpx_ - self.px_u
        step_point_ly_W = self.lpy_ - self.py_u
        step_point_ry_W = self.rpy_ - self.py_u

        self.step_point_lz_ = self.pz_ - self.lpz_
        self.step_point_rz_ = self.pz_ - self.rpz_
        self.step_point_lthta_ = -self.lpt_
        self.step_point_rthta_ = -self.rpt_

        # W -> B 旋轉（-theta_）
        c = math.cos(-self.theta_)
        s = math.sin(-self.theta_)
        self.step_point_lx_ = step_point_lx_W * c - step_point_ly_W * s
        self.step_point_ly_ = step_point_lx_W * s + step_point_ly_W * c
        self.step_point_rx_ = step_point_rx_W * c - step_point_ry_W * s
        self.step_point_ry_ = step_point_rx_W * s + step_point_ry_W * c

    def coordinate_offset(self):
        self.end_point_lx_ = self.step_point_lx_
        self.end_point_rx_ = self.step_point_rx_
        self.end_point_ly_ = self.step_point_ly_ - self.length_pelvis / 2.0
        self.end_point_ry_ = self.step_point_ry_ + self.length_pelvis / 2.0
        self.end_point_lz_ = self.step_point_lz_ - (self.com_z_height - self.stand_height)
        self.end_point_rz_ = self.step_point_rz_ - (self.com_z_height - self.stand_height)
        self.end_point_lthta_ = self.step_point_lthta_
        self.end_point_rthta_ = self.step_point_rthta_

    def final_step(self):
        # 歸零 → 站姿
        self.step_point_lx_ = 0.0
        self.step_point_rx_ = 0.0
        self.step_point_ly_ = 0.0
        self.step_point_ry_ = 0.0
        self.step_point_lz_ = self.com_z_height
        self.step_point_rz_ = self.com_z_height
        self.step_point_lthta_ = 0.0
        self.step_point_rthta_ = 0.0

        self.end_point_lx_ = 0.0
        self.end_point_rx_ = 0.0
        half_w = 0.5 * self.width_size_
        self.end_point_ly_ = half_w - self.length_pelvis / 2.0
        self.end_point_ry_ = -half_w + self.length_pelvis / 2.0
        self.end_point_lz_ = self.step_point_lz_ - (self.com_z_height - self.stand_height)
        self.end_point_rz_ = self.step_point_rz_ - (self.com_z_height - self.stand_height)
        self.end_point_lthta_ = 0.0
        self.end_point_rthta_ = 0.0

        self.if_finish_ = True
        self.resetParameter()

    # 讓你可在 final_step() 後回到初始狀態（保留 Parameter 參數）
    def resetParameter(self):
        self.sample_point_ = 0
        self.time_point_ = 0.0
        self.now_step_ = 0
        self.pre_step_ = -1
        self.step_ = 999999
        self.ready_to_stop_ = False
        self.walking_state = StartStep
        self.delay_push_ = False
        self.push_data_ = False
        self.footstep_x = 0.0
        half_w = 0.5 * self.width_size_
        self.footstep_y = -half_w
        self.base_x = self.base_y = 0.0
        self.last_base_x = self.last_base_y = 0.0
        self.displacement_x = self.displacement_y = 0.0
        self.last_displacement_x = self.last_displacement_y = 0.0
        self.zmp_x = self.zmp_y = 0.0
        self.now_right_x_ = 0.0
        half_w = 0.5 * self.width_size_
        self.now_right_y_ = -half_w
        self.now_left_x_ = 0.0
        self.now_left_y_ = half_w
        self.theta_ = self.var_theta_ = self.last_theta_ = 0.0
        self.if_finish_ = False
