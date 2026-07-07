#!/usr/bin/env python3
# coding=utf-8
"""
ipm_calibration.py — Homography IPM(Inverse Perspective Mapping)校準模組

把 board_detector.py 量出來的 pixel 座標,換算成地面實際公分距離。

背景(詳見 README.md):
    舊系統的「距離」本質上是簡化版 IPM——利用相機角度/高度固定,影像列(y)理論上
    對應固定的地面實際距離,靠「經驗式的像素數硬編碼」(FOOTBOARD_LINE 為基準,
    像素差當距離)。這次改用 cv2.findHomography 實地量測校準,不再是經驗值。

    板子墊高後地平面被墊高,pixel-row -> 實際距離的對應關係會跟著改變(這是舊
    系統距離漂移的真正原因)。目前已知的 board_high 只有兩個值:UpStair.ini
    board_high=3.0、DownStair.ini board_high=1.0(見 Parameter 目錄),所以校準
    採「每個 board_high 各自存一份 homography」,兩個值各自現場校正一次即可,
    不需要精確的相機外部參數(高度/傾角)去解析推導修正公式。

    若查詢到還沒校正過的 board_high,退而求其次用最接近的兩個已校正高度線性
    內插(不是嚴謹的物理模型,只是合理近似;若只有一個已校正高度就直接沿用它;
    超出已校正高度範圍則用最近的兩個值外插)。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class HomographyCalibration:
    homographies: Dict[float, np.ndarray] = field(default_factory=dict)

    def calibrate(
        self,
        board_high: float,
        image_points: Sequence[Tuple[float, float]],
        world_points: Sequence[Tuple[float, float]],
    ) -> float:
        """用實地量測的對應點校準某個 board_high 的 homography。

        image_points: [(pixel_x, pixel_y), ...],至少 4 組
        world_points: [(forward_cm, lateral_cm), ...],跟 image_points 一一對應

        回傳重投影誤差(RMSE,單位公分),用來判斷這次校準點量得準不準。
        """
        if len(image_points) < 4 or len(image_points) != len(world_points):
            raise ValueError("至少需要 4 組對應點,且 image_points/world_points 數量要相同")

        img_pts = np.asarray(image_points, dtype=np.float64)
        world_pts = np.asarray(world_points, dtype=np.float64)

        H, _ = cv2.findHomography(img_pts, world_pts, method=0)
        if H is None:
            raise ValueError("cv2.findHomography 求解失敗,檢查對應點是否共線/退化")

        self.homographies[float(board_high)] = H

        reprojected = self._apply(H, img_pts)
        rmse = float(np.sqrt(np.mean(np.sum((reprojected - world_pts) ** 2, axis=1))))
        return rmse

    @staticmethod
    def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        n = len(pts)
        homog = np.hstack([pts, np.ones((n, 1))])
        mapped = (H @ homog.T).T
        return mapped[:, :2] / mapped[:, 2:3]

    def pixel_to_world(self, x: float, y: float, board_high: float) -> Tuple[float, float]:
        """把 pixel 座標換算成 (forward_cm, lateral_cm)。"""
        if not self.homographies:
            raise RuntimeError("尚未校準任何 board_high,無法轉換")

        pt = np.array([[x, y]], dtype=np.float64)

        if board_high in self.homographies:
            world = self._apply(self.homographies[board_high], pt)[0]
            return float(world[0]), float(world[1])

        heights = sorted(self.homographies.keys())
        if len(heights) == 1:
            world = self._apply(self.homographies[heights[0]], pt)[0]
            return float(world[0]), float(world[1])

        lower_candidates = [h for h in heights if h <= board_high]
        upper_candidates = [h for h in heights if h >= board_high]
        lower = max(lower_candidates) if lower_candidates else None
        upper = min(upper_candidates) if upper_candidates else None

        if lower is None:
            lower, upper = heights[0], heights[1]
        elif upper is None:
            lower, upper = heights[-2], heights[-1]

        if lower == upper:
            world = self._apply(self.homographies[lower], pt)[0]
            return float(world[0]), float(world[1])

        world_lower = self._apply(self.homographies[lower], pt)[0]
        world_upper = self._apply(self.homographies[upper], pt)[0]
        t = (board_high - lower) / (upper - lower)
        world = world_lower + t * (world_upper - world_lower)
        return float(world[0]), float(world[1])

    def save(self, path: str) -> None:
        data = {str(h): H.tolist() for h, H in self.homographies.items()}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self.homographies = {float(h): np.array(H, dtype=np.float64) for h, H in data.items()}


def board_result_to_cm(
    detector_result: dict,
    calibration: HomographyCalibration,
    board_high: float,
) -> dict:
    """把 BoardDetector.detect() 的 pixel 結果換算成公分,回傳附加欄位後的新 dict。

    逐點換算 near_points/far_points 後取平均前進距離,再相減得到板深(cm),
    比直接換算 near_slope/board_depth_px 更直接、不受斜率符號影響。
    """
    out = dict(detector_result)
    near_points = detector_result.get('near_points') or []
    far_points = detector_result.get('far_points') or []

    if near_points:
        near_world = [calibration.pixel_to_world(x, y, board_high) for x, y in near_points]
        out['near_forward_cm'] = float(np.mean([w[0] for w in near_world]))
        out['near_lateral_cm'] = float(np.mean([w[1] for w in near_world]))
    else:
        out['near_forward_cm'] = None
        out['near_lateral_cm'] = None

    if far_points:
        far_world = [calibration.pixel_to_world(x, y, board_high) for x, y in far_points]
        far_forward_cm = float(np.mean([w[0] for w in far_world]))
    else:
        far_forward_cm = None

    if out['near_forward_cm'] is not None and far_forward_cm is not None:
        out['board_depth_cm'] = float(far_forward_cm - out['near_forward_cm'])
    else:
        out['board_depth_cm'] = None

    return out


if __name__ == '__main__':
    # 自我測試:用已知 ground-truth homography 產生合成對應點,驗證擬合/查詢/內插/存讀檔邏輯
    # (不代表大人型實際相機外參已知,現場還是要用真實量測點校準)

    rng = np.random.default_rng(0)

    def make_ground_truth_H(scale=1.0):
        # 簡單但非平凡的 3x3 homography(帶透視項),模擬「板高改變」對映射的影響
        return np.array([
            [0.0,           -0.30 * scale, 60.0],
            [0.25 * scale,   0.0,          -5.0],
            [0.0,            0.0008 * scale, 1.0],
        ])

    def gen_points(H, n=8):
        xs = rng.uniform(40, 280, n)
        ys = rng.uniform(30, 215, n)
        img_pts = np.stack([xs, ys], axis=1)
        homog = np.hstack([img_pts, np.ones((n, 1))])
        mapped = (H @ homog.T).T
        world_pts = mapped[:, :2] / mapped[:, 2:3]
        return img_pts, world_pts

    calib = HomographyCalibration()

    H1 = make_ground_truth_H(scale=1.0)
    img1, world1 = gen_points(H1)
    rmse1 = calib.calibrate(board_high=1.0, image_points=img1, world_points=world1)
    print(f"board_high=1.0 校準 RMSE = {rmse1:.6f} cm(期望接近 0)")

    H3 = make_ground_truth_H(scale=1.3)
    img3, world3 = gen_points(H3)
    rmse3 = calib.calibrate(board_high=3.0, image_points=img3, world_points=world3)
    print(f"board_high=3.0 校準 RMSE = {rmse3:.6f} cm(期望接近 0)")

    test_x, test_y = 160.0, 150.0
    expect1 = HomographyCalibration._apply(H1, np.array([[test_x, test_y]]))[0]
    got1 = calib.pixel_to_world(test_x, test_y, board_high=1.0)
    print(f"exact match board_high=1.0: got={got1}, expect={tuple(expect1)}")

    got_interp = calib.pixel_to_world(test_x, test_y, board_high=2.0)
    print(f"interpolated board_high=2.0: {got_interp}")

    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, 'ipm_calibration.json')
        calib.save(save_path)
        calib2 = HomographyCalibration()
        calib2.load(save_path)
        got_after_load = calib2.pixel_to_world(test_x, test_y, board_high=1.0)
        assert np.allclose(got_after_load, got1), "存檔/讀檔後結果不一致"
        print("存檔/讀檔驗證通過")

    # detect() 風格結果換算成 cm 的整合測試
    fake_detector_result = {
        'near_points': [(100, 190), (160, 185), (220, 180)],
        'far_points': [(100, 160), (160, 155), (220, 150)],
    }
    cm_result = board_result_to_cm(fake_detector_result, calib, board_high=1.0)
    print(f"board_result_to_cm: near_forward_cm={cm_result['near_forward_cm']:.2f}, "
          f"board_depth_cm={cm_result['board_depth_cm']:.2f}")

    assert abs(rmse1) < 1e-4, "board_high=1.0 校準誤差應該趨近 0(合成資料無雜訊)"
    assert abs(rmse3) < 1e-4, "board_high=3.0 校準誤差應該趨近 0(合成資料無雜訊)"
    assert np.allclose(got1, expect1, atol=1e-3), "exact match 查詢結果不準"
    print("自我測試通過(僅驗證擬合/查詢/內插/存讀檔邏輯,不代表大人型相機外參已知)")
