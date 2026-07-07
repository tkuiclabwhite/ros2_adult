#!/usr/bin/env python3
# coding=utf-8
"""
board_detector.py — 上下板視覺幾何量測層(只做量測,不做決策、不碰 ROS 訂閱)

背景與現況(詳見 README.md):
    - `API.py` 訂閱的 `/label_matrix`(存進 `self.label_matrix`)在目前的
      `imageprocess/image.py` 多色模式(build_all_hsv_table)下,是把每個顏色的
      0/255 mask 用 `cv2.bitwise_or` 疊出來的 `total_mask`,每個像素只剩「是不是
      任一已設定顏色」,已經遺失「是哪個顏色」的資訊,不能直接拿來做逐色板子偵測。
    - `image.py` 同時會把各顏色以固定 BGR 值畫成偽彩色圖,發布在 `processed_image`
      (bgr8)。這跟 ros2_kid `lc/calculate_edge.py` 的 `deep_calculate.
      generate_custom_label_matrix()` 是同一招:把偽彩色圖的 BGR 值反查回顏色代碼,
      重建一份「每個像素是哪個顏色」的離散標籤矩陣。
    - 所以本模組不吃 `API.label_matrix`,而是吃 `processed_image` 的 BGR 影像,
      用 `decode_label_matrix()` 自己解碼出 label_matrix(值為 1~8 的顏色代碼,
      0 是沒對到任何顏色的背景),再做多欄掃描+RANSAC。
    - 把 `processed_image` 訂閱起來、每幀呼叫 `decode_label_matrix()`,是策略節點
      (sr.py)要做的事,不在這個檔案的範圍內。

FOOT / FOOTBOARD_LINE 目前沿用 ros2_kid 的數值當佔位符(大人型還沒實機量測),
之後校正時只需要改這幾個常數,不影響其他邏輯。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 顏色代碼表:必須跟 imageprocess/image.py 的 self.color_labels 保持一致
# (label 值 1~8,BGR 是該顏色在 processed_image 偽彩色圖裡的畫法)
# ---------------------------------------------------------------------------
COLOR_LABEL = {
    'black': 1, 'blue': 2, 'green': 3, 'orange': 4,
    'red': 5, 'yellow': 6, 'white': 7, 'others': 8,
}

COLOR_BGR = {
    COLOR_LABEL['black']:  (255,   0, 255),
    COLOR_LABEL['blue']:   (128,   0, 128),
    COLOR_LABEL['green']:  (  0,   0, 128),
    COLOR_LABEL['orange']: (128,   0,   0),
    COLOR_LABEL['red']:    (255, 255,   0),
    COLOR_LABEL['yellow']: (128, 128,   0),
    COLOR_LABEL['white']:  (  0, 255, 255),
    COLOR_LABEL['others']: (255,   0, 128),
}


def decode_label_matrix(bgr_image: np.ndarray) -> np.ndarray:
    """把 processed_image(bgr8 偽彩色圖)解碼成離散顏色標籤矩陣。

    回傳的矩陣跟輸入同樣的 (H, W) 大小,dtype=uint8,值為 0(背景/未匹配任何
    設定顏色)或 COLOR_LABEL 裡的 1~8。
    """
    h, w = bgr_image.shape[:2]
    label_matrix = np.zeros((h, w), dtype=np.uint8)
    for label_id, bgr in COLOR_BGR.items():
        mask = np.all(bgr_image == bgr, axis=-1)
        label_matrix[mask] = label_id
    return label_matrix


# ---------------------------------------------------------------------------
# 舊系統(ros2_kid lc.py)可重用常數 —— 大人型還沒現場校正前的佔位值
# ---------------------------------------------------------------------------
FOOTBOARD_LINE = 215  # 掃描基準列(舊系統稱 outset),大人型待現場校正
#                  右腳            左腳
#              左 , 中,  右 |  左,  中,   右
FOOT = [93, 116, 136, 165, 190, 220]  # 六個固定 x 座標,大人型待現場校正


@dataclass
class ScanConfig:
    x_min: int = 40
    x_max: int = 280
    num_columns: int = 24          # 多欄掃描的欄數(README 建議 20~30)
    outset: int = FOOTBOARD_LINE   # 掃描基準列,往上(y 變小)找板子
    y_top: int = 30                # 掃描上限,避免掃出畫面
    min_run: int = 10              # 連續多少像素同色才算確認邊界(沿用 KID)
    ransac_iters: int = 200
    ransac_threshold: float = 2.0
    min_confidence_columns: float = 0.4  # 有效欄位比例低於此值,confidence 直接判不可信
    foot_x: List[int] = field(default_factory=lambda: list(FOOT))  # query_foot() 用的六個 x 座標


def _scan_column(
    label_matrix: np.ndarray,
    x: int,
    board_color: int,
    outset: int,
    y_top: int,
    min_run: int,
) -> Tuple[Optional[int], Optional[int], bool]:
    """從 outset 往 y_top 方向(y 遞減)掃描單一欄,找近邊/遠邊。

    近邊:第一次出現連續 min_run 個像素都符合 board_color 的位置(該連續區間
    最靠近 outset 的那一列),對應「板子開始的位置」。
    遠邊:近邊之後,第一次出現連續 min_run 個像素都不符合 board_color 的位置,
    對應「板子顏色結束的位置」。若一路掃到 y_top 都還是 board_color,
    遠邊視為無效(far_valid=False),代表板子超出掃描範圍。

    回傳 (near_y, far_y, far_valid);找不到近邊時回傳 (None, None, False)。
    """
    h = label_matrix.shape[0]
    y0 = min(outset, h - 1)
    y1 = max(y_top, 0)
    if y0 <= y1 or y0 - y1 + 1 < min_run:
        return None, None, False

    # col[k] 對應實際列 y = y0 - k,k 由 0(outset)往上遞增
    col = label_matrix[y1:y0 + 1, x][::-1]
    match = (col == board_color)

    near_k = None
    run = 0
    for k, m in enumerate(match):
        run = run + 1 if m else 0
        if run >= min_run:
            near_k = k - min_run + 1
            break
    if near_k is None:
        return None, None, False
    near_y = y0 - near_k

    far_k = None
    run = 0
    for k in range(near_k, len(match)):
        run = run + 1 if not match[k] else 0
        if run >= min_run:
            far_k = k - min_run + 1
            break

    if far_k is None:
        far_y = y0 - (len(match) - 1)
        return near_y, far_y, False

    far_y = y0 - far_k
    return near_y, far_y, True


def ransac_fit_line(
    points: List[Tuple[float, float]],
    n_iters: int = 200,
    threshold: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> Optional[dict]:
    """純 numpy 的簡易 RANSAC 直線擬合(y = slope*x + intercept)。

    points: [(x, y), ...]。回傳 None 代表點數不足以擬合。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return None

    rng = rng or np.random.default_rng()
    best_inliers = None
    best_count = -1
    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        residuals = np.abs(pts[:, 1] - (slope * pts[:, 0] + intercept))
        inliers = residuals <= threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 2:
        return None

    inlier_pts = pts[best_inliers]
    A = np.vstack([inlier_pts[:, 0], np.ones(len(inlier_pts))]).T
    slope, intercept = np.linalg.lstsq(A, inlier_pts[:, 1], rcond=None)[0]
    residuals = inlier_pts[:, 1] - (slope * inlier_pts[:, 0] + intercept)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'inlier_mask': best_inliers,
        'rmse': rmse,
        'inlier_ratio': best_count / n,
    }


class BoardDetector:
    """板子邊緣幾何量測。吃 decode_label_matrix() 產生的 label_matrix,
    board_color 由外部(決策層)傳入,不寫死順序。
    """

    def __init__(self, cfg: Optional[ScanConfig] = None):
        self.cfg = cfg or ScanConfig()

    def detect(self, label_matrix: np.ndarray, board_color: int) -> dict:
        cfg = self.cfg
        xs = np.linspace(cfg.x_min, cfg.x_max, cfg.num_columns).astype(int)

        near_points: List[Tuple[int, int]] = []
        far_points: List[Tuple[int, int]] = []
        valid_cols = 0

        for x in xs:
            near_y, far_y, far_valid = _scan_column(
                label_matrix, int(x), board_color, cfg.outset, cfg.y_top, cfg.min_run
            )
            if near_y is None:
                continue
            valid_cols += 1
            near_points.append((int(x), near_y))
            if far_valid:
                far_points.append((int(x), far_y))

        confidence = valid_cols / len(xs)

        near_distance_px = (
            float(np.mean([cfg.outset - y for _, y in near_points])) if near_points else None
        )

        result = {
            'confidence': confidence,
            'near_slope': None,
            'near_intercept': None,
            'near_fit_valid': False,
            'flatter_side': 'unknown',
            'board_depth_px': None,
            'near_distance_px': near_distance_px,
            'near_points': near_points,
            'far_points': far_points,
        }

        if confidence < cfg.min_confidence_columns or len(near_points) < 2:
            return result

        fit = ransac_fit_line(near_points, cfg.ransac_iters, cfg.ransac_threshold)
        if fit is None:
            return result

        result['near_slope'] = fit['slope']
        result['near_intercept'] = fit['intercept']
        result['near_fit_valid'] = True

        median_x = float(np.median([p[0] for p in near_points]))
        left_pts = [p for p in near_points if p[0] <= median_x]
        right_pts = [p for p in near_points if p[0] > median_x]
        left_fit = ransac_fit_line(left_pts, cfg.ransac_iters, cfg.ransac_threshold) if len(left_pts) >= 2 else None
        right_fit = ransac_fit_line(right_pts, cfg.ransac_iters, cfg.ransac_threshold) if len(right_pts) >= 2 else None

        if left_fit and right_fit:
            result['flatter_side'] = 'left' if left_fit['rmse'] <= right_fit['rmse'] else 'right'
        elif left_fit:
            result['flatter_side'] = 'left'
        elif right_fit:
            result['flatter_side'] = 'right'

        if far_points:
            near_dist_by_x = {x: cfg.outset - y for x, y in near_points}
            far_dist_vals = [cfg.outset - y for x, y in far_points if x in near_dist_by_x]
            near_dist_vals = [near_dist_by_x[x] for x, _ in far_points if x in near_dist_by_x]
            if far_dist_vals:
                result['board_depth_px'] = float(np.mean(far_dist_vals) - np.mean(near_dist_vals))

        return result

    def query_foot(self, label_matrix: np.ndarray, board_color: int, foot_index: int) -> dict:
        """查詢單一腳位(cfg.foot_x[foot_index])的近邊/遠邊距離,供決策層逐腳判斷用。"""
        cfg = self.cfg
        x = cfg.foot_x[foot_index]
        near_y, far_y, far_valid = _scan_column(
            label_matrix, x, board_color, cfg.outset, cfg.y_top, cfg.min_run
        )
        if near_y is None:
            return {'valid': False, 'near_dist': None, 'far_dist': None, 'depth_px': None}

        near_dist = cfg.outset - near_y
        far_dist = cfg.outset - far_y
        return {
            'valid': True,
            'near_dist': near_dist,
            'far_dist': far_dist if far_valid else None,
            'far_valid': far_valid,
            'depth_px': (far_dist - near_dist) if far_valid else None,
        }


# ---------------------------------------------------------------------------
# 自我測試:用合成資料驗證斜率/深度數值是否符合預期(不代表大人型實機常數已校正)
# ---------------------------------------------------------------------------
def _make_synthetic_label_matrix(
    width=320, height=240, board_color=COLOR_LABEL['red'],
    near_slope=0.05, near_intercept=190.0, depth_px=30,
):
    label_matrix = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        near_y = int(near_intercept + near_slope * x)
        far_y = near_y - depth_px
        near_y = max(0, min(height - 1, near_y))
        far_y = max(0, min(height - 1, far_y))
        if far_y < near_y:
            label_matrix[far_y:near_y + 1, x] = board_color
    return label_matrix


if __name__ == '__main__':
    true_slope = 0.05
    true_intercept = 190.0
    true_depth = 30
    board_color = COLOR_LABEL['red']

    lm = _make_synthetic_label_matrix(
        board_color=board_color,
        near_slope=true_slope,
        near_intercept=true_intercept,
        depth_px=true_depth,
    )

    detector = BoardDetector(ScanConfig(outset=215, y_top=30, min_run=10, num_columns=24))
    result = detector.detect(lm, board_color)

    print(f"confidence      = {result['confidence']:.2f}")
    print(f"near_slope      = {result['near_slope']:.4f} (expect ~{true_slope})")
    print(f"near_intercept  = {result['near_intercept']:.2f} (expect ~{true_intercept})")
    print(f"board_depth_px  = {result['board_depth_px']:.2f} (expect ~{true_depth})")
    print(f"flatter_side    = {result['flatter_side']}")

    foot_result = detector.query_foot(lm, board_color, foot_index=0)
    print(f"query_foot(0)   = {foot_result}")

    assert abs(result['near_slope'] - true_slope) < 0.01, "斜率擬合誤差過大"
    assert abs(result['near_intercept'] - true_intercept) < 3.0, "截距擬合誤差過大"
    assert abs(result['board_depth_px'] - true_depth) < 3.0, "板深計算誤差過大"
    print("自我測試通過(僅驗證演算法邏輯,常數/介面仍待大人型實機確認)")
