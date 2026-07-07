#!/usr/bin/env python3
# coding=utf-8
"""
path_planner.py — 決策層(只做決策,不碰視覺、不碰 ROS 發布)

吃 board_detector.detect() 的量測結果(建議先用 ipm_calibration.board_result_to_cm()
轉成公分),用 edge_fit.Kalman1D 做時序濾波,輸出:
    - 任務 A(連續步態):forward/theta/translation 修正量,每幀可以修
    - 任務 B(觸發時機判斷):是否觸發上/下板步態。用 edge_fit.ConsecutiveConfirm
      確保只有「Kalman 估計值連續 N 幀都達到門檻」才觸發,不是單幀 threshold
      就觸發(單步步態一旦觸發沒有回頭路,不能信任單幀雜訊值)。

門檻/檔位數值沿用 ros2_kid `lc.py` 原始碼裡的實際數值當預設值(注意:README
草稿轉述的數字跟 lc.py 原始碼有出入,這裡以原始碼為準)。

**重要提醒**:lc.py 那些距離門檻(GO_UP_DISTANCE 等)當時比對的是「像素距離」
(舊系統 return_real_board() 的輸出),大人型這邊改用 IPM 校準後的「公分距離」,
兩者尺度完全不同,直接沿用數字大概率不準,一定要上機重新調校。這裡先當
PlannerConfig 的預設佔位值,不是定案數字,跟 FOOT/FOOTBOARD_LINE 的狀況一樣。
"""

from dataclasses import dataclass
from typing import Optional

from strategy.strategy.sr.new_detectorAndCalculate.edge_fit import ConsecutiveConfirm, Kalman1D

# 層數結構、BOARD_COLOR、START_LAYER 這些「現場會改」的參數集中放在 sr.py
# 最上面那個參數區,不在這裡重複定義,避免兩份不同步。

LEFT_THETA = 1
RIGHT_THETA = -1
FORWARD_PARAM = 1
BACK_PARAM = -1


@dataclass
class PlannerConfig:
    # ---- 觸發門檻(沿用 KID lc.py 數值當佔位,單位待大人型 cm 尺度重新調校)----
    go_up_distance: float = 13.0
    go_down_distance: float = 8.0
    up_warning_distance: float = 3.0
    down_warning_distance: float = 0.0

    # ---- 前進速度依距離切換檔位的分界線 ----
    first_forward_change_line: float = 50.0
    second_forward_change_line: float = 100.0
    third_forward_change_line: float = 150.0

    # ---- 旋轉/平移修正檔位 ----
    theta_min: float = 4.0
    theta_normal: float = 5.0
    theta_big: float = 8.0

    translation_min: float = 700.0
    translation_normal: float = 1000.0
    translation_big: float = 1200.0

    # ---- 前進/後退檔位 ----
    forward_min: float = 600.0
    forward_normal: float = 1000.0
    forward_big: float = 1400.0
    forward_super: float = 2000.0

    back_min: float = -800.0
    back_normal: float = -1200.0

    # ---- 斜率修正檔位門檻 ----
    slope_min: float = 4.0
    slope_normal: float = 5.0
    slope_big: float = 12.0

    # ---- Kalman 濾波參數 ----
    distance_process_var: float = 2.0
    distance_measurement_var: float = 9.0
    slope_process_var: float = 0.5
    slope_measurement_var: float = 4.0

    # ---- 穩定性 ----
    trigger_confirm_count: int = 5   # 連續 N 幀都達到觸發門檻才觸發
    min_confidence: float = 0.4      # confidence 低於此值,這幀直接跳過(沿用上一幀指令)

    # ---- 板深/兩段式上板(待確認機器人腳長/步距參數,目前先留 None 不判斷)----
    min_single_step_depth_cm: Optional[float] = None


@dataclass
class PlannerOutput:
    forward: float = 0.0
    translation: float = 0.0
    theta: float = 0.0
    should_trigger_up: bool = False
    should_trigger_down: bool = False
    split_two_step: Optional[bool] = None
    distance_estimate: Optional[float] = None
    slope_estimate: Optional[float] = None
    confidence: float = 0.0
    skipped: bool = False  # confidence 太低,這幀被跳過,指令沿用上一幀


class PathPlanner:
    """決策層。每個板子/每個 approach 階段建議各自建立一個新實例
    (或呼叫 reset()),避免 Kalman 狀態/連續確認計數跨板子污染。
    """

    def __init__(self, cfg: Optional[PlannerConfig] = None):
        self.cfg = cfg or PlannerConfig()
        self.distance_kf = Kalman1D(self.cfg.distance_process_var, self.cfg.distance_measurement_var)
        self.slope_kf = Kalman1D(self.cfg.slope_process_var, self.cfg.slope_measurement_var)
        self.up_confirm = ConsecutiveConfirm(self.cfg.trigger_confirm_count)
        self.down_confirm = ConsecutiveConfirm(self.cfg.trigger_confirm_count)
        self._last_output = PlannerOutput()

    def reset(self) -> None:
        self.distance_kf.reset()
        self.slope_kf.reset()
        self.up_confirm.reset()
        self.down_confirm.reset()
        self._last_output = PlannerOutput()

    def update(self, cm_result: dict, dt: float, direction: str = 'up') -> PlannerOutput:
        """
        cm_result: ipm_calibration.board_result_to_cm() 的輸出,至少要有
            'confidence'、'near_slope'、'near_forward_cm'、'board_depth_cm'。
        direction: 'up' 或 'down',決定跟 go_up_distance 還是 go_down_distance 比較。
        """
        cfg = self.cfg
        confidence = cm_result.get('confidence', 0.0)

        if confidence < cfg.min_confidence or cm_result.get('near_forward_cm') is None:
            # 量測不可信:沿用上一幀指令,不餵給 Kalman(避免拿雜訊污染濾波狀態)
            out = PlannerOutput(
                forward=self._last_output.forward,
                translation=self._last_output.translation,
                theta=self._last_output.theta,
                distance_estimate=self._last_output.distance_estimate,
                slope_estimate=self._last_output.slope_estimate,
                confidence=confidence,
                skipped=True,
            )
            self._last_output = out
            return out

        distance_est, _ = self.distance_kf.step(cm_result['near_forward_cm'], dt)

        slope_est = None
        if cm_result.get('near_slope') is not None:
            slope_est, _ = self.slope_kf.step(cm_result['near_slope'], dt)

        theta, translation = self._slope_to_correction(slope_est)
        forward = self._distance_to_forward(distance_est)

        threshold = cfg.go_up_distance if direction == 'up' else cfg.go_down_distance
        confirm = self.up_confirm if direction == 'up' else self.down_confirm
        idle_confirm = self.down_confirm if direction == 'up' else self.up_confirm
        idle_confirm.reset()  # 不同方向的確認計數互不影響,避免切換方向時殘留誤觸發

        triggered = confirm.update(distance_est <= threshold)

        split_two_step = None
        depth_cm = cm_result.get('board_depth_cm')
        if depth_cm is not None and cfg.min_single_step_depth_cm is not None:
            split_two_step = depth_cm < cfg.min_single_step_depth_cm

        out = PlannerOutput(
            forward=forward,
            translation=translation,
            theta=theta,
            should_trigger_up=(triggered and direction == 'up'),
            should_trigger_down=(triggered and direction == 'down'),
            split_two_step=split_two_step,
            distance_estimate=distance_est,
            slope_estimate=slope_est,
            confidence=confidence,
            skipped=False,
        )
        self._last_output = out
        return out

    def _slope_to_correction(self, slope_est: Optional[float]) -> "tuple[float, float]":
        cfg = self.cfg
        if slope_est is None:
            return 0.0, 0.0

        direction_sign = LEFT_THETA if slope_est > 0 else RIGHT_THETA
        abs_slope = abs(slope_est)

        if abs_slope > cfg.slope_big:
            return cfg.theta_big * direction_sign, cfg.translation_normal * direction_sign * -1
        elif abs_slope > cfg.slope_normal:
            return cfg.theta_normal * direction_sign, cfg.translation_min * direction_sign * -1
        elif abs_slope > cfg.slope_min:
            return cfg.theta_min * direction_sign, 0.0
        return 0.0, 0.0

    def _distance_to_forward(self, distance_est: Optional[float]) -> float:
        cfg = self.cfg
        if distance_est is None:
            return cfg.forward_min * FORWARD_PARAM
        if distance_est > cfg.third_forward_change_line:
            return cfg.forward_super * FORWARD_PARAM
        elif distance_est > cfg.second_forward_change_line:
            return cfg.forward_big * FORWARD_PARAM
        elif distance_est > cfg.first_forward_change_line:
            return cfg.forward_normal * FORWARD_PARAM
        return cfg.forward_min * FORWARD_PARAM


if __name__ == '__main__':
    # 自我測試:模擬機器人接近板子(距離持續下降,疊加雜訊+一次短暫掉到門檻以下的雜訊尖峰),
    # 驗證:1) 距離檔位切換符合預期 2) 短暫雜訊尖峰不會誤觸發 3) 真正連續低於門檻才觸發
    # 4) confidence 太低的幀會被跳過、指令沿用上一幀

    import numpy as np

    rng = np.random.default_rng(1)
    cfg = PlannerConfig(trigger_confirm_count=5, min_confidence=0.4)
    planner = PathPlanner(cfg)
    dt = 0.05

    true_distances = np.linspace(200.0, 5.0, 80)
    confidences = np.full(80, 0.9)
    confidences[30:33] = 0.1  # 中間插入 3 幀低信心度,應該被跳過

    noisy_distances = true_distances + rng.normal(0, 1.5, 80)
    noisy_distances[50] = 2.0  # 單幀雜訊尖峰,提早掉到門檻(13)以下,但只有 1 幀,不該觸發

    triggered_frames = []
    forwards = []
    skipped_frames = []
    for i in range(80):
        cm_result = {
            'confidence': confidences[i],
            'near_slope': 0.01,
            'near_forward_cm': float(noisy_distances[i]),
            'board_depth_cm': 20.0,
        }
        out = planner.update(cm_result, dt, direction='up')
        forwards.append(out.forward)
        if out.skipped:
            skipped_frames.append(i)
        if out.should_trigger_up:
            triggered_frames.append(i)

    print(f"skipped frames (應該是 [30, 31, 32]): {skipped_frames}")
    print(f"first triggered frame: {triggered_frames[0] if triggered_frames else None}")
    print(f"forward gear at start (遠,應該是 forward_super={cfg.forward_super}): {forwards[0]}")
    print(f"forward gear near end (近,應該是 forward_min={cfg.forward_min}): {forwards[-1]}")

    assert skipped_frames == [30, 31, 32], "低信心度的幀應該被跳過"
    assert forwards[0] == cfg.forward_super, "距離很遠時應該用最快檔位"
    assert forwards[-1] == cfg.forward_min, "距離很近時應該用最慢檔位"
    # 單幀雜訊尖峰(index 50)不該觸發(連續確認數不足);真正觸發應該在距離持續低於門檻累積 5 幀之後
    assert 50 not in triggered_frames, "單幀雜訊尖峰不應該觸發上板步態"
    assert triggered_frames, "距離持續下降到門檻以下,最終應該要觸發"
    print("自我測試通過(門檻數值仍是 KID 佔位值,大人型需要上機依 cm 尺度重新調校)")
