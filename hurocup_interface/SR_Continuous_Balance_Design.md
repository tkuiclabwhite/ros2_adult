# SR_Continuous 自適應平衡步態設計文件

## 背景與目標

SR_Continuous 是在 SR（Stair Recognition）策略下新增的第三種步態模式，定位為**自適應平衡的連續走路步態**。

場景假設：
- 地面有厚度最大 2.5 cm 的薄板（類上樓梯障礙）
- 可能同時踩到兩塊不同高度的板子（兩腳高度差最大 5 cm）
- 機器人需要持續向前走、踩上薄板再踩下來

現有步態分工：
| 模式 | statetype | 用途 |
|------|-----------|------|
| Continuous | 0 | 一般平地連續行走 |
| SR_up | 1 | 踩上薄板（一次性觸發） |
| SR_down | 2 | 踩下薄板（一次性觸發） |
| **SR_Continuous** | **3** | **踩薄板時的連續行走 + 即時平衡補償** |

---

## 控制架構

### 核心概念

以 IMU 讀值為輸入，透過 PD 控制計算補償量，修正走路時的重心偏移。

```
pitch_error = imu_pitch - target_pitch
roll_error  = imu_roll  - target_roll
yaw_error   = imu_yaw   - locked_yaw     ← 進入 SR_Continuous 時自動鎖定

pitch_correction = imukp * pitch_error + imukd * d(pitch_error)/dt
roll_correction  = imukp * roll_error  + imukd * d(roll_error)/dt
yaw_correction   = yaw_kp * yaw_error

correction = clamp(correction, -max_correction, +max_correction)
```

補償量作用在步態生成器的重心 X/Y 位置上（待後端實作時確認具體接入點）。

### 為何需要同時補 Pitch 和 Roll

- **Pitch（前後）**：踩上/踩下薄板時身體前後傾
- **Roll（側向）**：一腳在板上一腳在地面，兩腳高度差導致身體側傾

兩軸共用同一組 `imukp / imukd`，若日後發現需要不同強度再拆開。

### Yaw 鎖定邏輯

進入 SR_Continuous 模式時，程式記錄當下 `imu_yaw` 作為 `locked_yaw`。
之後持續用 `yaw_kp` 把偏移修正回去，防止走薄板時方向累積偏轉。
`target_yaw` 不在 UI 上設定，由程式自動記錄。

---

## 介面參數說明

SR_Continuous Step 面板共有 **13 個參數**，分為兩組：

### 基本步態參數（同 SR_up / SR_down）

| 參數 | 說明 |
|------|------|
| com_y_swing | 重心側向擺動量 |
| width_size | 步寬 |
| period_t | 步態週期（ms） |
| T_DSP | 雙腳支撐期比例 |
| Clearance | 抬腳高度 |
| Board_High | 薄板高度補償 |
| STAND_HEIGHT | 站立高度 |
| COM_HEIGHT | 重心高度 |

### 自適應平衡參數（SR_Continuous 專用）

| 參數 | 說明 | 初始建議 |
|------|------|---------|
| `imukp` | Pitch/Roll 誤差比例增益 | 從 0.1 開始調 |
| `imukd` | Pitch/Roll 誤差微分增益 | imukp 的 1/10 |
| `target_pitch` | 目標俯仰角（°） | 實測穩定走路時的 Pitch 值 |
| `target_roll` | 目標側傾角（°） | 通常為 0 |
| `yaw_kp` | Yaw 偏移比例增益 | 從 0.05 開始調 |
| `max_correction` | 補償量上限（安全限制） | 先設保守值（如 5） |

---

## Generate 行為

| 模式 | 第一次按 | 第二次按 |
|------|---------|---------|
| SR_up / SR_down | 發送 data=1（單發） | 同上，不 toggle |
| **SR_Continuous** | **發送 data=1（開始走）** | **發送 data=0（停止）** |
| Continuous | 同 SR_Continuous | 同 SR_Continuous |

---

## 調參流程

1. **確認 target_pitch / target_roll**
   - 切換到一般 Continuous 步態讓機器人走穩
   - 觀察 Sensor_Value 的 Pitch / Roll 讀值（可開啟 WaveformMonitor 看波形）
   - 穩定走路時的角度值即為 target 值

2. **調整 max_correction**
   - 先填一個保守值（如 5°），確認補償不過頭後慢慢調大

3. **調整 imukp**
   - 從小值開始（如 0.1），觀察波形是否有振盪
   - 振盪 → 降低 imukp；反應太慢 → 提高 imukp

4. **調整 imukd**
   - 通常為 imukp 的 1/5 ~ 1/10 作為起點
   - 用於抑制過衝和振盪

5. **調整 yaw_kp**
   - 觀察走幾步後 Yaw 漂移量
   - 從 0.05 開始，以 WaveformMonitor 的 Yaw 波形輔助觀察

---

## 波形監控（WaveformMonitor）

開啟 `WaveformMonitor.html` 可即時觀看 Roll / Pitch / Yaw 波形。

- **Ref 欄位**：填入 `target_pitch` 和 `target_roll` 值，畫面會顯示黃色虛線基準
- **Time Window**：顯示的時間長度（秒），建議調參時用 10s
- **Y Scale**：縱軸範圍（度），正常走路建議 ±30°

---

## 後續規劃

| 項目 | 狀態 | 說明 |
|------|------|------|
| 手臂 IK | 規劃中 | 補上後加入 `arm_kp` 參數，pitch 誤差對應手臂角度補償 |
| 馬達電流反饋 | 評估中 | 電流突增可偵測腳踩到薄板的時機，比 IMU 反應更快 |
| 後端 ROS service/topic | 待實作 | `LoadSRContinuousParameterClient` / `web_src_parameter_Topic` |

---

## 相關 Topic / Service（前端 Placeholder）

| 名稱 | 類型 | 用途 |
|------|------|------|
| `/web/src_parameter_Topic` | Topic `tku_msgs/SRContinuousParameter` | Save 參數 |
| `/web/LoadSRContinuousParameter` | Service `tku_msgs/LoadSRContinuousParameter` | Load 參數 |
| `/package/sensorpackage` | Topic `tku_msgs/SensorPackage` | IMU 讀值（Roll/Pitch/Yaw） |
| `/walking_params_update` | Topic `std_msgs/String` | Send 參數（JSON 格式） |

Send 的 JSON 格式：
```json
{
  "com_y_swing": 0,
  "width_size": 0,
  "period_t": 360,
  "Tdsp": 0,
  "Board_High": 0,
  "STAND_HEIGHT": 50,
  "COM_HEIGHT": 40,
  "Clearance": 3.0,
  "imukp": 0.1,
  "imukd": 0.01,
  "target_pitch": 3.0,
  "target_roll": 0.0,
  "yaw_kp": 0.05,
  "max_correction": 5.0
}
```
