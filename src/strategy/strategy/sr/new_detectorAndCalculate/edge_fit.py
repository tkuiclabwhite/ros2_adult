#!/usr/bin/env python3
# coding=utf-8
"""
edge_fit.py — 時序濾波層(只做濾波/穩定性判斷,不做決策)

取代舊系統被註解掉的 EMA 低通濾波(見 ros2_kid lc.py theta_change() 裡的
`alpha = 0.2  # 低通濾波係數`,整段被註解掉沒有真的在用)。EMA 的權重是固定
常數,不管雜不雜訊都一樣;Kalman 對量測雜訊會動態調整信任權重,量測穩就跟
得快,量測亂就多信預測,而且有「速度」狀態,對機器人等速前進的情境預測更
準、延遲更小。

state = [value, velocity]^T,一維常速度模型。可分別餵距離(distance,
distance_velocity)或斜率(slope, angular_velocity)兩種量測,各自建一個
Kalman1D 實例。

這裡也放了 ConsecutiveConfirm——「連續 N 幀都符合條件才算確認」的通用工具,
是任務 B(單步觸發判斷)穩定性的關鍵:決策層(path_planner.py,尚未開始)
應該用 Kalman 估計值餵給 ConsecutiveConfirm,連續 N 幀都小於門檻才觸發上板
步態,而不是單幀 `< threshold` 就觸發。實際門檻值/N 是決策層的事,這裡只
提供「連續確認」這個穩定性原語,不寫死任何觸發邏輯。
"""

import numpy as np


class Kalman1D:
    """一維常速度模型的 Kalman filter。state = [value, velocity]。"""

    def __init__(self, process_var: float = 1.0, measurement_var: float = 4.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2) * 100.0  # 初始不確定性給大一點,讓第一筆量測值主導
        self._initialized = False

    def reset(self, value: float = 0.0, velocity: float = 0.0) -> None:
        self.x = np.array([[value], [velocity]], dtype=np.float64)
        self.P = np.eye(2) * 100.0
        self._initialized = False

    def predict(self, dt: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.process_var
        # 離散白噪聲加速度模型(discrete white noise acceleration)
        Q = q * np.array([
            [dt ** 4 / 4.0, dt ** 3 / 2.0],
            [dt ** 3 / 2.0, dt ** 2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: float) -> None:
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.measurement_var]])
        z = np.array([[measurement]])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        self._initialized = True

    def step(self, measurement: float, dt: float) -> "tuple[float, float]":
        """預測+更新一起做,回傳 (value_estimate, velocity_estimate)。

        第一次呼叫時直接拿量測值當初始狀態,不做預測(沒有前一幀可以預測)。
        """
        if not self._initialized:
            self.x[0, 0] = measurement
            self._initialized = True
        else:
            self.predict(dt)
            self.update(measurement)
        return self.value, self.velocity

    @property
    def value(self) -> float:
        return float(self.x[0, 0])

    @property
    def velocity(self) -> float:
        return float(self.x[1, 0])


class ConsecutiveConfirm:
    """連續 N 次都符合條件才算確認,避免單幀雜訊觸發不可逆的單步動作。"""

    def __init__(self, required_count: int):
        if required_count < 1:
            raise ValueError("required_count 必須 >= 1")
        self.required_count = required_count
        self._count = 0

    def update(self, condition: bool) -> bool:
        """回傳這一幀是否已經達到「連續 required_count 次都成立」。"""
        self._count = self._count + 1 if condition else 0
        return self._count >= self.required_count

    def reset(self) -> None:
        self._count = 0

    @property
    def count(self) -> int:
        return self._count


if __name__ == '__main__':
    # 自我測試:模擬機器人以固定速度接近板子,量測值疊加雜訊,
    # 驗證 Kalman 估計值比原始雜訊量測值更接近真實值、速度估計收斂到真實速度。

    rng = np.random.default_rng(0)
    dt = 0.05  # 20Hz
    true_velocity = -8.0  # cm/s,負值代表距離持續變小(接近板子)
    n_steps = 60
    true_values = 100.0 + true_velocity * (np.arange(n_steps) * dt)
    noisy_measurements = true_values + rng.normal(0, 3.0, n_steps)

    kf = Kalman1D(process_var=2.0, measurement_var=9.0)
    estimates = []
    velocities = []
    for z in noisy_measurements:
        val, vel = kf.step(float(z), dt)
        estimates.append(val)
        velocities.append(vel)

    estimates = np.array(estimates)
    raw_rmse = float(np.sqrt(np.mean((noisy_measurements - true_values) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((estimates - true_values) ** 2)))
    final_velocity = velocities[-1]

    print(f"raw measurement RMSE      = {raw_rmse:.3f}")
    print(f"kalman estimate RMSE      = {filtered_rmse:.3f} (應該明顯小於 raw)")
    print(f"true velocity             = {true_velocity:.3f} cm/s")
    print(f"kalman final velocity est = {final_velocity:.3f} cm/s (應該接近 true velocity)")

    assert filtered_rmse < raw_rmse * 0.6, "Kalman 濾波後誤差應該明顯小於原始雜訊量測"
    assert abs(final_velocity - true_velocity) < 2.0, "速度估計應該收斂到接近真實速度"

    # ConsecutiveConfirm 測試:連續 3 次以上才算確認
    confirm = ConsecutiveConfirm(required_count=3)
    sequence = [True, True, False, True, True, True, True]
    triggered_at = [i for i, cond in enumerate(sequence) if confirm.update(cond)]
    print(f"ConsecutiveConfirm triggered at indices: {triggered_at} (expect [5, 6])")
    assert triggered_at == [5, 6], "連續確認邏輯應該在第 3 次連續成立(index 5)才開始觸發"

    print("自我測試通過")
