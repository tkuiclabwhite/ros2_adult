#!/usr/bin/env python3
# coding=utf-8
import sys
import time
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from strategy.API import API

# ── 頭部角度參數 ───────────────────────────────────────────────────────────────
X_RATIO             = 64.5   # 水平視角（度）
Y_RATIO             = 40.0   # 垂直視角（度）

HEAD_HORIZONTAL     = 2048   # 水平初始位置（馬達絕對值）
HEAD_VERTICAL       = 2900   # 垂直初始位置（往下看）

MAX_HEAD_HORIZONTAL = 2648   # 左邊界（HORIZONTAL 較大）
MIN_HEAD_HORIZONTAL = 1448   # 右邊界（HORIZONTAL 較小）
MAX_HEAD_VERTICAL   = 2900   # 下邊界（VERTICAL 較大，> 2048 = 往下看）
MIN_HEAD_VERTICAL   = 2700   # 上邊界（VERTICAL 較小，< 2048 = 往上看）

# ── 守門判定參數 ───────────────────────────────────────────────────────────────
BALL_INCOMING_SIZE = 1000   # 球飛來的像素面積門檻
BALL_COUNT_DEFEND  = 5      # 觸發守門動作的累積幀數

# ── 測試用起始狀態 ─────────────────────────────────────────────────────────────
START_STATE = 0   # 從第幾個狀態開始（0~4）

BALL_COLOR        = 'Yellow'                    # 球的顏色'Orange', 'Yellow', 'Blue',
                                                        #'Green', 'Black', 'Red', 'White',
# ── 狀態名稱、描述與顯示顏色 ────────────────────────────────────────────────────
_STATE_NAMES = [
    'SEARCHING',    # 0 掃頭找球
    'TRACKING',     # 1 找到球，追蹤中
    'CONFIRMING',   # 2 球飛來，累積幀數
    'DEFENDING',    # 3 執行撲球
    'RESETTING',    # 4 動作後等待復位
    'FINISH',       # 5 動作完成，等待比賽結束
]
_STATE_DESCS = [
    '掃頭尋找橘色球',
    '找到球，頭部追蹤，等待球靠近',
    '球飛來中，累積確認幀數準備撲球',
    '執行守門撲球動作（blocking）',
    '動作結束，等球離開後回到搜尋',
    '撲球完成，等待比賽結束',
]
_STATE_COLORS = [
    '\033[96m',   # 0 SEARCHING  — 亮青色
    '\033[92m',   # 1 TRACKING   — 亮綠色
    '\033[93m',   # 2 CONFIRMING — 亮黃色
    '\033[91m',   # 3 DEFENDING  — 亮紅色
    '\033[95m',   # 4 RESETTING  — 亮洋紅
    '\033[90m',   # 5 FINISH     — 亮黑色
]


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ObjectInfo:
    """橘色球偵測：從 API 原始感測陣列中找到最符合條件的圓形橘色物件。"""
    COLOR_IDX = {
        'Orange': 0, 'Yellow': 1, 'Blue': 2,
        'Green':  3, 'Black':  4, 'Red':  5, 'White': 6,
    }

    def __init__(self, color, object_type, node):
        self.node        = node
        self.color_idx   = self.COLOR_IDX[color]
        self.edge_max    = Coordinate(0, 0)
        self.edge_min    = Coordinate(0, 0)
        self.center      = Coordinate(0, 0)
        self.get_target  = False
        self.target_size = 0
        dispatch = {'Ball': self._get_ball}
        self.find_object = dispatch[object_type]

    def _get_ball(self):
        idx    = self.color_idx
        result = None
        for i in range(self.node.color_counts[idx]):
            if self.node.object_sizes[idx][i] > 500:
                result = i
        return result

    def update(self):
        idx = self.color_idx
        obj = self.find_object()
        if obj is not None:
            self.get_target  = True
            self.edge_max.x  = self.node.object_x_max[idx][obj]
            self.edge_min.x  = self.node.object_x_min[idx][obj]
            self.edge_max.y  = self.node.object_y_max[idx][obj]
            self.edge_min.y  = self.node.object_y_min[idx][obj]
            self.center.x    = (self.edge_max.x + self.edge_min.x) // 2
            self.center.y    = (self.edge_max.y + self.edge_min.y) // 2
            self.target_size = self.node.object_sizes[idx][obj]
        else:
            self.edge_max.x = self.edge_min.x = 0
            self.edge_max.y = self.edge_min.y = 0
            self.center.x   = self.center.y   = 0
            self.target_size = 0
            self.get_target  = False


class PenaltyKickDef(API):

    def __init__(self):
        super().__init__('pk_def_node')

        self.ball = ObjectInfo(BALL_COLOR, 'Ball', self)

        # 頭部當前位置
        self.head_horizon  = HEAD_HORIZONTAL
        self.head_vertical = HEAD_VERTICAL

        # 掃描方向
        self._search_dir = 'right'

        # 狀態機
        self._state       = _STATE_NAMES[START_STATE]
        self._check_count = 0   # CONFIRMING 狀態的累積幀數

        # 顯示用快取
        self._disp_ball_size  = 0
        self._disp_ball_cx    = 0
        self._disp_last_event = ''

        self.create_timer(0.1, self._tick)
        self._init_state()

        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True
        )
        self._display_thread.start()

    # ──────────────────────────────────────────────────────────────── 初始化 ──

    def _init_state(self):
        self._search_dir   = 'right'
        self.head_horizon  = HEAD_HORIZONTAL
        self.head_vertical = HEAD_VERTICAL
        self._state        = _STATE_NAMES[START_STATE]
        self._check_count  = 0

        self.sendSensorReset(True)
        self.sendHeadMotor(1, self.head_horizon,  30)
        self.sendHeadMotor(2, self.head_vertical, 30)

    # ──────────────────────────────────────────────────────────────── 頭部控制 ──

    def _control_head(self, motor_id, position, speed):
        """送出頭部馬達指令，同時將位置限制在安全範圍內並更新快取值。"""
        if motor_id == 1:
            self.head_horizon = max(MIN_HEAD_HORIZONTAL, min(MAX_HEAD_HORIZONTAL, position))
            self.sendHeadMotor(1, self.head_horizon, speed)
        elif motor_id == 2:
            self.head_vertical = max(MIN_HEAD_VERTICAL, min(MAX_HEAD_VERTICAL, position))
            self.sendHeadMotor(2, self.head_vertical, speed)

    def _search_ball(self, scale=30):
        """頭部依 右→下→左→上 順序掃描尋球。"""
        if self._search_dir == 'right':
            self._control_head(1, self.head_horizon - scale, scale)
            if self.head_horizon <= MIN_HEAD_HORIZONTAL:
                self._search_dir = 'down'
        elif self._search_dir == 'down':
            self._control_head(2, self.head_vertical + scale, scale)
            if self.head_vertical >= MAX_HEAD_VERTICAL:
                self._search_dir = 'left'
        elif self._search_dir == 'left':
            self._control_head(1, self.head_horizon + scale, scale)
            if self.head_horizon >= MAX_HEAD_HORIZONTAL:
                self._search_dir = 'up'
        elif self._search_dir == 'up':
            self._control_head(2, self.head_vertical - scale, scale)
            if self.head_vertical <= MIN_HEAD_VERTICAL:
                self._search_dir = 'right'

    def _track_ball(self):
        """頭部追蹤：依球的畫面位置計算誤差並調整馬達。"""
        cx, cy = self.ball.center.x, self.ball.center.y
        if cx == 0 and cy == 0:
            return
        x_err = cx - 160
        y_err = cy - 120
        x_deg = x_err * (X_RATIO / 320)
        y_deg = y_err * (Y_RATIO / 240)
        new_h = self.head_horizon  - round(x_deg * 4096 / 360 * 0.15)
        new_v = self.head_vertical + round(y_deg * 4096 / 360 * 0.15)
        self._control_head(1, new_h, 25)
        self._control_head(2, new_v, 25)

    # ──────────────────────────────────────────────────────── 各狀態處理函式 ──

    def _handle_searching(self):
        """SEARCHING — 掃頭尋找橘色球，找到後進入 TRACKING。"""
        self.ball.update()
        self._disp_ball_size = self.ball.target_size
        self._disp_ball_cx   = self.ball.center.x

        if self.ball.get_target:
            self._disp_last_event = '找到球 → TRACKING'
            self._state = 'TRACKING'
        else:
            self._search_ball()

    def _handle_tracking(self):
        """TRACKING — 追蹤球，等待球面積超過門檻後進入 CONFIRMING。"""
        self.ball.update()
        self._disp_ball_size = self.ball.target_size
        self._disp_ball_cx   = self.ball.center.x

        if not self.ball.get_target:
            self._disp_last_event = '失去球 → SEARCHING'
            self._state = 'SEARCHING'
            return

        self._track_ball()

        if self.ball.target_size > BALL_INCOMING_SIZE:
            self._disp_last_event = '球開始飛來 → CONFIRMING'
            self._check_count = 1
            self._state = 'CONFIRMING'

    def _handle_confirming(self):
        """CONFIRMING — 球持續在門檻以上，累積幀數確認後執行撲球。"""
        self.ball.update()
        self._disp_ball_size = self.ball.target_size
        self._disp_ball_cx   = self.ball.center.x

        if not self.ball.get_target or self.ball.target_size <= BALL_INCOMING_SIZE:
            self._disp_last_event = '球消失或縮小 → TRACKING'
            self._check_count = 0
            self._state = 'TRACKING'
            return

        self._track_ball()
        self._check_count += 1

        if self._check_count >= BALL_COUNT_DEFEND:
            self._disp_last_event = f'確認球進入！幀數={self._check_count} → DEFENDING'
            self._state = 'DEFENDING'

    def _handle_defending(self):
        """DEFENDING — 執行守門撲球動作（blocking），完成後停止。"""
        self._disp_last_event = '執行撲球動作'
        self.sendBodySector(67)
        time.sleep(26)
        self._check_count = 0
        self._disp_last_event = '撲球完成 → FINISH'
        self._state = 'FINISH'

    def _handle_resetting(self):
        """RESETTING — 保留備用，目前流程不會進入此狀態。"""
        self.ball.update()
        self._disp_ball_size = self.ball.target_size

        if not self.ball.get_target or self.ball.target_size <= BALL_INCOMING_SIZE:
            self._init_state()
            self._disp_last_event = '復位完成 → SEARCHING'
            self._state = 'SEARCHING'

    def _handle_stopped(self):
        """比賽未開始或中斷：重置所有狀態並站直。"""
        self.ball.update()
        self._disp_ball_size = self.ball.target_size
        self._disp_ball_cx   = self.ball.center.x
        self.sendBodySector(29)
        self._init_state()

    # ──────────────────────────────────────────────────────────────── 顯示 ──

    def _print_status(self):
        """將目前所有狀態資訊一次印到終端機（清除後重寫）。"""
        R = '\033[0m'
        B = '\033[1m'
        D = '\033[2m'

        state_idx  = _STATE_NAMES.index(self._state) if self._state in _STATE_NAMES else -1
        state_name = _STATE_NAMES[state_idx] if state_idx >= 0 else self._state
        state_desc = _STATE_DESCS[state_idx] if state_idx >= 0 else ''
        state_col  = _STATE_COLORS[state_idx] if 0 <= state_idx < len(_STATE_COLORS) else '\033[97m'
        state_idx_str = str(state_idx) if state_idx >= 0 else '?'

        is_start_str = (
            f'{B}\033[92mTrue{R}' if self.is_start
            else f'{B}\033[91mFalse (未開始){R}'
        )

        # 球資訊：飛來時顯示紅色，追蹤中顯示黃色
        if self._disp_ball_size > 0:
            ball_col = '\033[91m' if self._disp_ball_size > BALL_INCOMING_SIZE else '\033[93m'
            ball_str = f'{ball_col}cx={self._disp_ball_cx}  area={self._disp_ball_size}{R}'
        else:
            ball_str = f'{D}\033[33m未偵測到{R}'

        # 進度條
        def _bar(val, total, col='\033[93m'):
            filled = min(int(val / total * 8), 8) if total > 0 else 0
            return f'{col}{"█" * filled}{R}\033[90m{"░" * (8 - filled)}{R} {val}/{total}'

        confirm_bar = _bar(self._check_count, BALL_COUNT_DEFEND)

        SEP = f'\033[2;33m{"─" * 40}{R}'
        def sec(title):
            return f'{B}\033[33m▌ {title}{R}'

        sys.stdout.write('\033[H\033[J')
        sys.stdout.write(
            f'\033[1;97;44m  PK 守門狀態                       {R}\n'
            f'  is_start   : {is_start_str}\n'
            f'{SEP}\n'
            f'{sec("目前狀態")}\n'
            f'  {state_col}{B}[{state_idx_str}] {state_name}{R}\n'
            f'  {D}{state_desc}{R}\n'
            f'  \033[96m最後事件 : {self._disp_last_event}{R}\n'
            f'{SEP}\n'
            f'{sec("視覺資訊")}\n'
            f'  球 (黃色)  : {ball_str}\n'
            f'  門檻面積   : \033[90m{BALL_INCOMING_SIZE}{R}\n'
            f'{SEP}\n'
            f'{sec("頭部位置")}\n'
            f'  Horizontal : \033[36m{self.head_horizon}{R}  '
            f'\033[90m[{MIN_HEAD_HORIZONTAL} ~ {MAX_HEAD_HORIZONTAL}]{R}\n'
            f'  Vertical   : \033[36m{self.head_vertical}{R}  '
            f'\033[90m[{MIN_HEAD_VERTICAL} ~ {MAX_HEAD_VERTICAL}]{R}\n'
            f'  掃描方向   : \033[97m{self._search_dir}{R}\n'
            f'{SEP}\n'
            f'{sec("計數器")}\n'
            f'  confirm [{confirm_bar}]\n'
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

    # ──────────────────────────────────────────────────────────────── 主計時器 ──

    def _tick(self):
        """每 0.1 秒觸發一次，根據目前狀態呼叫對應的處理函式。"""
        if not self.is_start:
            self._handle_stopped()
            return

        match self._state:
            case 'SEARCHING':
                self._handle_searching()
            case 'TRACKING':
                self._handle_tracking()
            case 'CONFIRMING':
                self._handle_confirming()
            case 'DEFENDING':
                self._handle_defending()
            case 'RESETTING':
                self._handle_resetting()
            case 'FINISH':
                pass


def main(args=None):
    rclpy.init(args=args)
    node = PenaltyKickDef()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('pk_def 節點停止中...')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
