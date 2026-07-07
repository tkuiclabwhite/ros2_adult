#!/usr/bin/env python3
# coding=utf-8
"""
sr.py — Spartan Race(上下板)策略節點

把 board_detector(視覺量測)+ ipm_calibration(pixel -> cm)+ path_planner
(決策層,內部含 edge_fit.Kalman1D 濾波)串起來:每幀解碼影像 -> 量測邊緣 ->
換算實際距離 -> 決策 -> 送連續步態指令,量測值連續穩定低於門檻時觸發單步
上/下板步態。

⚠️ 這份檔案取代了原本的 sr.py。原內容是一份完全不相關的「爬牆
(WallClimbing)」策略,是很久以前不明來源的程式碼,跟 README.md 描述的
上下板任務對不上,已確認可以整份刪除重寫。

所有「現場/上機測試時會調的參數」都集中放在檔案最上面那個區塊,不用去改
board_detector.py / path_planner.py 裡的預設值(那些預設值只是給模組自己
的 __main__ 自我測試用)。
"""

import time
from typing import Optional

import rclpy
from sensor_msgs.msg import Image as RosImage

from strategy.API import API
from strategy.strategy.sr.new_detectorAndCalculate.board_detector import BoardDetector, COLOR_LABEL, ScanConfig, decode_label_matrix
from strategy.strategy.sr.new_detectorAndCalculate.ipm_calibration import HomographyCalibration, board_result_to_cm
from strategy.strategy.sr.new_detectorAndCalculate.path_planner import PathPlanner, PlannerConfig

# ============================================================================
# 現場校正參數區 —— 比賽現場/上機測試時要改的數值都集中在這裡
# ============================================================================

# ---- 校正量(疊加在最終送出的連續步態指令上,沿用舊 sr.py 的 CORRECTION 慣例)----
FORWARD_CORRECTION     = 0
TRANSLATION_CORRECTION = 0
THETA_CORRECTION       = 0

# ---- 層數與板子顏色順序(跟 ros2_kid 一樣上 3 層下 3 層,顏色順序現場才知道)----
UP_LAYERS   = 3
DOWN_LAYERS = 3
START_LAYER = 1
BOARD_COLOR = ['Green', 'Yellow', 'Red', 'Yellow', 'Red', 'Blue', 'Green']  # 待現場確認

# ---- 板高(給 IPM 校準/sendLCWalkParameter 用,沿用 README 提到的 ini 數值)----
UP_BOARD_HIGH   = 3.0  # 對應 UpStair.ini board_high
DOWN_BOARD_HIGH = 1.0  # 對應 DownStair.ini board_high

# ---- 視覺量測參數(board_detector.ScanConfig)----
FOOT           = [93, 116, 136, 165, 190, 220]  # 六個固定 x 座標,大人型待現場校正
FOOTBOARD_LINE = 215                            # 掃描基準列,大人型待現場校正
SCAN_X_MIN            = 40
SCAN_X_MAX            = 280
SCAN_NUM_COLUMNS      = 24
SCAN_Y_TOP            = 30
SCAN_MIN_RUN          = 10      # 連續多少像素同色才算確認邊界
RANSAC_ITERS          = 200
RANSAC_THRESHOLD      = 2.0
MIN_CONFIDENCE_COLUMNS = 0.4    # 有效欄位比例低於此值,這幀量測直接判不可信

# ---- 決策門檻(path_planner.PlannerConfig),沿用 ros2_kid lc.py 數值當佔位 ----
# ⚠️ lc.py 這些距離門檻當年比對的是「像素距離」,大人型這邊改用 IPM 換算後的
#    「公分距離」,尺度不同,一定要上機重新調校
GO_UP_DISTANCE       = 13.0
GO_DOWN_DISTANCE     = 8.0
UP_WARNING_DISTANCE  = 3.0
DOWN_WARNING_DISTANCE = 0.0

FIRST_FORWARD_CHANGE_LINE  = 50.0
SECOND_FORWARD_CHANGE_LINE = 100.0
THIRD_FORWARD_CHANGE_LINE  = 150.0

THETA_MIN, THETA_NORMAL, THETA_BIG = 4.0, 5.0, 8.0
TRANSLATION_MIN, TRANSLATION_NORMAL, TRANSLATION_BIG = 700.0, 1000.0, 1200.0
FORWARD_MIN, FORWARD_NORMAL, FORWARD_BIG, FORWARD_SUPER = 600.0, 1000.0, 1400.0, 2000.0
BACK_MIN, BACK_NORMAL = -800.0, -1200.0
SLOPE_MIN, SLOPE_NORMAL, SLOPE_BIG = 4.0, 5.0, 12.0

TRIGGER_CONFIRM_COUNT = 5     # 連續 N 幀都達到觸發門檻才觸發上/下板步態
MIN_CONFIDENCE = 0.4          # confidence 低於此值,這幀直接跳過(沿用上一幀指令)
MIN_SINGLE_STEP_DEPTH_CM: Optional[float] = None  # 待確認機器人腳長/步距參數,None=不判斷兩段式上板

# ---- 上/下板步態參數(sendLCWalkParameter),沿用 ros2_kid 當年手調數值當起點 ----
UP_GAIT_PARAMS = dict(
    com_y_swing=-4.0, width_size=4.0, period_t=280, t_dsp=0.35,
    clearance=3.0, stand_height=23.5, com_height=29.5, hip_roll=0.0, ankle_roll=0.0,
)
DOWN_GAIT_PARAMS = dict(
    com_y_swing=-3.0, width_size=4.0, period_t=280, t_dsp=0.35,
    clearance=3.0, stand_height=23.5, com_height=29.5, hip_roll=0.0, ankle_roll=0.0,
)

# ---- 上/下板前的站姿微調(沿用 ros2_kid lc.py 用 sendBodySector 做站姿微調的
#      機制;數字是 KID 動作表的編號,大人型動作表不同,不能沿用,這裡先放 0
#      當佔位——0 代表「不送」,現場請自行改成大人型動作表對應的 sector 編號。
#      每層可能需要不同姿勢,所以用 list,index 對應「該方向的第幾層」(0-based)----
STANCE_SECTOR_UP   = [0] * UP_LAYERS
STANCE_SECTOR_DOWN = [0] * DOWN_LAYERS

# ---- 時間等待(目前 API 沒有「單步步態是否執行完畢」的回饋 topic,只能先用
#      固定秒數等待;現場請依實測調整,之後如果有回饋機制應該換掉這段邏輯)----
GAIT_PARAM_SETTLE_SEC    = 1.5   # 送完 sendLCWalkParameter 後的等待
STANCE_SETTLE_SEC        = 1.5   # 送完站姿微調 sector 後的等待
SINGLE_STEP_DURATION_SEC = 3.0   # 單步步態預估耗時

# ---- IPM 校準檔(還沒現場校準前留 None,path_planner 會退回用像素距離當近似)----
CALIBRATION_PATH: Optional[str] = None  # 例如 'Parameter/ipm_calibration.json'

# ---- 頭部瞄準(要看向板子掃描區域,也就是 FOOTBOARD_LINE 那條線附近;數值沿用
#      其他策略的置中值當佔位,現場要改成「看得到板子掃描區域」的角度)----
HEAD_LOOK_H = 2048
HEAD_LOOK_V = 2048
HEAD_MOTOR_SPEED = 50

# ---- 迴圈頻率 ----
LOOP_PERIOD_SEC = 0.05  # 20Hz

# ============================================================================


class SpartanRace(API):
    def __init__(self):
        super().__init__('spartan_race_node')

        self.detector = BoardDetector(ScanConfig(
            x_min=SCAN_X_MIN, x_max=SCAN_X_MAX, num_columns=SCAN_NUM_COLUMNS,
            outset=FOOTBOARD_LINE, y_top=SCAN_Y_TOP, min_run=SCAN_MIN_RUN,
            ransac_iters=RANSAC_ITERS, ransac_threshold=RANSAC_THRESHOLD,
            min_confidence_columns=MIN_CONFIDENCE_COLUMNS, foot_x=list(FOOT),
        ))

        self.calibration = HomographyCalibration()
        if CALIBRATION_PATH:
            self.calibration.load(CALIBRATION_PATH)

        self.planner = PathPlanner(PlannerConfig(
            go_up_distance=GO_UP_DISTANCE, go_down_distance=GO_DOWN_DISTANCE,
            up_warning_distance=UP_WARNING_DISTANCE, down_warning_distance=DOWN_WARNING_DISTANCE,
            first_forward_change_line=FIRST_FORWARD_CHANGE_LINE,
            second_forward_change_line=SECOND_FORWARD_CHANGE_LINE,
            third_forward_change_line=THIRD_FORWARD_CHANGE_LINE,
            theta_min=THETA_MIN, theta_normal=THETA_NORMAL, theta_big=THETA_BIG,
            translation_min=TRANSLATION_MIN, translation_normal=TRANSLATION_NORMAL,
            translation_big=TRANSLATION_BIG,
            forward_min=FORWARD_MIN, forward_normal=FORWARD_NORMAL,
            forward_big=FORWARD_BIG, forward_super=FORWARD_SUPER,
            back_min=BACK_MIN, back_normal=BACK_NORMAL,
            slope_min=SLOPE_MIN, slope_normal=SLOPE_NORMAL, slope_big=SLOPE_BIG,
            trigger_confirm_count=TRIGGER_CONFIRM_COUNT, min_confidence=MIN_CONFIDENCE,
            min_single_step_depth_cm=MIN_SINGLE_STEP_DEPTH_CM,
        ))

        self.layer = START_LAYER
        self.mode = 'approach'  # 'approach'(連續步態接近) / 'triggering'(單步步態執行中)
        self._was_started = False  # 上一幀 self.is_start 的狀態,用來偵測下降緣(送 stop 指令)
        self._initialized = False  # 是否已經做過一次性初始化(感測器重置/轉頭)

        self._label_matrix = None

        self.processed_image_sub = self.create_subscription(
            RosImage, 'processed_image', self._processed_image_cb,
            self.qos_fast, callback_group=self.image_cbg,
        )

        self._last_time = time.time()
        self.timer = self.create_timer(LOOP_PERIOD_SEC, self.strategy_loop)
        self.get_logger().info('Spartan Race(上下板)策略節點啟動')

    def _processed_image_cb(self, msg: RosImage) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._label_matrix = decode_label_matrix(bgr)
        except Exception as e:
            self.get_logger().error(f'[processed_image] 解碼失敗: {e}')

    @property
    def direction(self) -> str:
        return 'up' if self.layer <= UP_LAYERS else 'down'

    @property
    def board_high(self) -> float:
        return UP_BOARD_HIGH if self.direction == 'up' else DOWN_BOARD_HIGH

    def _layer_index_within_direction(self) -> int:
        """回傳「該方向的第幾層」(0-based),給 STANCE_SECTOR_UP/DOWN 這種
        per-layer 設定用。"""
        if self.direction == 'up':
            return self.layer - 1
        return self.layer - UP_LAYERS - 1

    def current_board_color_id(self) -> int:
        idx = min(self.layer - 1, len(BOARD_COLOR) - 1)
        color_name = BOARD_COLOR[idx].lower()
        return COLOR_LABEL[color_name]

    def _on_start(self) -> None:
        """is_start 第一次變成 True 時執行一次的初始化(沿用其他策略
        initial()/status=='First' 的慣例:重置感測器、把頭轉去看板子掃描區域)。

        只做一次——如果 is_start 之後又關閉再開啟(例如現場安全暫停),不會
        再重跑這段、也不會重置 layer/planner 進度,避免暫停一次就前功盡棄。
        """
        self.get_logger().info('[is_start] 硬體開關開啟,執行初始化\033[K')
        self.sendSensorReset(True)
        self.sendHeadMotor(1, HEAD_LOOK_H, HEAD_MOTOR_SPEED)
        self.sendHeadMotor(2, HEAD_LOOK_V, HEAD_MOTOR_SPEED)

    def _on_stop(self) -> None:
        """is_start 變成 False 時執行:立刻停止步態(每次下降緣都做,是安全機制)。"""
        self.get_logger().info('[is_start] 硬體開關關閉,停止步態\033[K')
        self.sendbodyAuto(0)
        self.sendContinuousValue(0, 0, 0)

    def strategy_loop(self) -> None:
        now = time.time()
        dt = max(now - self._last_time, 1e-3)
        self._last_time = now

        # is_start 是硬體實體開關狀態(見 API._dio_callback),所有策略程式
        # 執行迴圈的入口條件都要先檢查這個,開關沒開就不能送任何移動指令。
        if not self.is_start:
            if self._was_started:
                self._on_stop()
            self._was_started = False
            return

        self._was_started = True
        if not self._initialized:
            self._on_start()
            self._initialized = True

        if self.mode != 'approach':
            return  # 單步步態執行中,不要送連續步態指令

        if self.layer > len(BOARD_COLOR):
            self.get_logger().info('所有層都完成了,停止連續步態\033[K')
            self.sendbodyAuto(0)
            return

        if self._label_matrix is None:
            return  # 還沒收到任何影像

        board_color = self.current_board_color_id()
        detector_result = self.detector.detect(self._label_matrix, board_color)

        if self.calibration.homographies:
            cm_result = board_result_to_cm(detector_result, self.calibration, self.board_high)
        else:
            # 還沒做 IPM 校準:退回用像素距離當近似,門檻/檔位數字會更不準,
            # 只是先求整條 pipeline 能動,細節等校準完成再調
            cm_result = dict(detector_result)
            cm_result['near_forward_cm'] = detector_result.get('near_distance_px')
            cm_result['board_depth_cm'] = detector_result.get('board_depth_px')

        out = self.planner.update(cm_result, dt, direction=self.direction)

        self.get_logger().info(
            f'layer={self.layer} dir={self.direction} conf={out.confidence:.2f} '
            f'dist={out.distance_estimate} slope={out.slope_estimate} '
            f'skipped={out.skipped}\033[K'
        )

        if out.skipped:
            return

        self.sendbodyAuto(1)
        self.sendContinuousValue(
            int(out.forward + FORWARD_CORRECTION),
            int(out.translation + TRANSLATION_CORRECTION),
            int(out.theta + THETA_CORRECTION),
        )

        if out.should_trigger_up:
            self._trigger_board_gait(walking_mode=1)
        elif out.should_trigger_down:
            self._trigger_board_gait(walking_mode=2)

    def _trigger_board_gait(self, walking_mode: int) -> None:
        self.mode = 'triggering'
        is_up = walking_mode == 1
        gait_params = UP_GAIT_PARAMS if is_up else DOWN_GAIT_PARAMS
        stance_sectors = STANCE_SECTOR_UP if is_up else STANCE_SECTOR_DOWN
        stance_idx = min(max(self._layer_index_within_direction(), 0), len(stance_sectors) - 1)
        stance_sector = stance_sectors[stance_idx]

        self.get_logger().info(f'觸發單步步態 walking_mode={walking_mode}(layer={self.layer})\033[K')

        self.sendbodyAuto(0)
        self.sendContinuousValue(0, 0, 0)

        self.sendLCWalkParameter(board_high=self.board_high, **gait_params)
        time.sleep(GAIT_PARAM_SETTLE_SEC)

        # 上/下板前的站姿微調(沿用 KID 用 sendBodySector 做站姿微調的機制)。
        # sector == 0 代表這層還沒設定站姿微調,直接跳過不送。
        if stance_sector:
            self.get_logger().info(f'站姿微調 sector={stance_sector}\033[K')
            self.sendBodySector(stance_sector)
            time.sleep(STANCE_SETTLE_SEC)

        self.sendBodyAutoCmd(x=20000, walking_mode=walking_mode)
        time.sleep(SINGLE_STEP_DURATION_SEC)

        self.layer += 1
        self.planner.reset()
        self.mode = 'approach'
        self.get_logger().info(f'單步步態完成,切回連續步態(layer={self.layer})\033[K')


def main(args=None):
    rclpy.init(args=args)
    node = SpartanRace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
