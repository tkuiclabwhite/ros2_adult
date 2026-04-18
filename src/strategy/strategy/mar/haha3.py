import os
import time
import cv2
import numpy as np
from collections import deque
from pupil_apriltags import Detector
from ultralytics import YOLO
import signal
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── AprilTag 設定 ─────────────────────────────────────────
april_detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=0.5,
    refine_edges=True,
    decode_sharpening=0.25,
)
TAG_ACTION = {1: "FORWARD", 2: "RIGHT", 3: "LEFT"}
TAG_ANGLE  = {1: 0.0, 2: 90.0, 3: -90.0}

# ── 預設參數 ──────────────────────────────────────────────
CONF_THRESH    = 0.85
CONFIRM_FRAMES = 10


def find_arrow_angle(roi_bgr):
    """
    箭頭方向偵測 v2：
      1. 黑卡遮罩 → 白色箭頭輪廓
      2. 凸包頂點裡找離質心最遠的 = 三角形 tip
      3. convexityDefects 找所有凹點
      4. 凹點按離 tip 距離排序，取最近兩個 = 三角形兩肩
      5. 方向 = 兩肩中點 → tip
    座標系：FORWARD=0°, RIGHT=+90°, LEFT=-90°
    回傳: (action, angle_deg) or (None, None)
    """
    MIN_SIDE = 200
    h0, w0 = roi_bgr.shape[:2]
    short_side = min(h0, w0)
    if short_side < MIN_SIDE and short_side > 0:
        scale = MIN_SIDE / short_side
        roi_bgr = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    MIN_AREA = 800

    # === Step 1: 黑卡遮罩 ===
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    card_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    dark_contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if dark_contours:
        card_cnt = max(dark_contours, key=cv2.contourArea)
        card_mask_final = np.zeros_like(gray)
        cv2.drawContours(card_mask_final, [card_cnt], -1, 255, -1)
    else:
        card_mask_final = np.ones_like(gray) * 255

    # === Step 2: 卡片內白色箭頭 ===
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray_e = clahe.apply(gray)

    card_pixels = gray_e[card_mask_final > 0]
    if len(card_pixels) < 100:
        return None, None

    top_b = np.percentile(card_pixels, 95)
    bot_d = np.percentile(card_pixels, 5)
    thresh = (top_b + bot_d) / 2

    _, binary_raw = cv2.threshold(gray_e, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary_raw, card_mask_final)

    ks = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ks, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ks, iterations=1)

    if cv2.countNonZero(binary) < MIN_AREA:
        return None, None

    # === Step 3: 最大白色輪廓 ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None, None

    contour = largest.reshape(-1, 2).astype(np.float32)
    if len(contour) < 5:
        return None, None

    # === Step 4: 凸包頂點 → 離質心最遠的 = tip ===
    M = cv2.moments(largest)
    if M["m00"] < 1e-6:
        return None, None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    centroid = np.array([cx, cy], dtype=np.float32)

    hull_pts = cv2.convexHull(largest, returnPoints=True).reshape(-1, 2).astype(np.float32)
    if len(hull_pts) < 3:
        return None, None

    dists_to_centroid = np.linalg.norm(hull_pts - centroid, axis=1)
    tip_idx_hull = int(np.argmax(dists_to_centroid))
    tip = hull_pts[tip_idx_hull]

    # === Step 5: 凸包缺陷 → 凹點按離 tip 距離排序 → 取最近兩個 = 兩肩 ===
    hull_idx = cv2.convexHull(largest, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None, None
    try:
        defects = cv2.convexityDefects(largest, hull_idx)
    except cv2.error:
        return None, None
    if defects is None or len(defects) < 2:
        return None, None

    min_depth = max(8, int(np.sqrt(cv2.contourArea(largest)) * 0.08)) * 256
    deep_defects = [d for d in defects[:, 0, :] if d[3] >= min_depth]
    if len(deep_defects) < 2:
        return None, None

    defect_points = []
    for d in deep_defects:
        pt = contour[int(d[2])]
        dist_to_tip = np.linalg.norm(pt - tip)
        defect_points.append((pt, dist_to_tip))

    defect_points.sort(key=lambda x: x[1])
    shoulder1 = defect_points[0][0]
    shoulder2 = defect_points[1][0]
    shoulder_mid = (shoulder1 + shoulder2) / 2.0

    # === Step 6: 方向 = 兩肩中點 → tip ===
    arrow_vec = tip - shoulder_mid
    vec_len = np.linalg.norm(arrow_vec)
    if vec_len < 1e-6:
        return None, None
    arrow_dir = arrow_vec / vec_len

    angle_deg = np.degrees(np.arctan2(arrow_dir[0], -arrow_dir[1]))

    # === Step 7: 角度 → 動作 ===
    if abs(angle_deg) < 45:
        action = "FORWARD"
    elif angle_deg > 0:
        action = "RIGHT"
    else:
        action = "LEFT"

    return action, angle_deg


class NavigationHeadless:
    def __init__(self):
        self.running = True

        self.action_history        = deque(maxlen=CONFIRM_FRAMES)
        self.last_confirmed_action = None

        print("載入 YOLO 模型...")
        self.yolo_model       = YOLO(os.path.join(BASE_DIR, 'best1.onnx'))
        self.yolo_class_names = {0: "left", 1: "right", 2: "straight"}
        self.yolo_to_action   = {"left": "LEFT", "right": "RIGHT", "straight": "FORWARD"}
        print("✅ 模型載入完成")

    def run(self):
        cap = cv2.VideoCapture("/dev/video0")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        t_prev = time.time()
        ema_fps = 0.0
        alpha   = 0.1

        print("開始偵測... (Ctrl+C 結束)")
        print("-" * 70)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # FPS
            now = time.time()
            dt  = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                ema_fps = inst if ema_fps == 0.0 else (1 - alpha) * ema_fps + alpha * inst

            result = self._process(frame)

            # 只在有新確認動作時印出
            if result["new_confirmed"] and result["confirmed"]:
                ang_str = f"{result['arrow_angle']:.1f}°" if result["arrow_angle"] is not None else "—"
                print(
                    f"[CONFIRMED] {result['confirmed']:8s}  "
                    f"src: TAG={result['april_action']}  GEO={result['geo_action']}  YOLO={result['yolo_action']}  "
                    f"angle={ang_str}  "
                    f"center={result['yolo_center']}  area={result['yolo_area']}  "
                    f"FPS={ema_fps:.1f}"
                )

        cap.release()
        print("\n結束。")

    def _process(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── AprilTag ──
        april_results = april_detector.detect(gray)
        april_action  = None
        april_angle   = None
        for r in april_results:
            action = TAG_ACTION.get(r.tag_id)
            if action:
                april_action = action
                april_angle  = TAG_ANGLE.get(r.tag_id)

        yolo_action   = None
        geo_action    = None
        final_action  = None
        yolo_center   = None
        yolo_area     = None
        yolo_bbox_bot = None
        arrow_angle   = None

        if not april_action:
            # ── YOLO ──
            results    = self.yolo_model(frame, imgsz=640, verbose=False)
            candidates = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                if cls not in self.yolo_class_names or conf < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                candidates.append({
                    'name':  self.yolo_class_names[cls],
                    'conf':  conf,
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'area':  area,
                    'bbox_bottom': y2,
                    'bbox':  (x1, y1, x2, y2),
                })

            if candidates:
                best = max(candidates, key=lambda x: x['area'])
                yolo_action   = self.yolo_to_action.get(best['name'])
                yolo_center   = best['center']
                yolo_area     = best['area']
                yolo_bbox_bot = best['bbox_bottom']
                x1, y1, x2, y2 = best['bbox']

                pad = 20
                ax1 = max(0, x1 - pad)
                ay1 = max(0, y1 - pad)
                ax2 = min(w, x2 + pad)
                ay2 = min(h, y2 + pad)
                roi = frame[ay1:ay2, ax1:ax2]

                if roi.size > 0:
                    geo_action, arrow_angle = find_arrow_angle(roi)

                final_action = geo_action or yolo_action

        # ── 確認邏輯 ──
        current_action = april_action or final_action
        if april_action:
            arrow_angle = april_angle
        self.action_history.append(current_action)

        accum = 0
        if current_action is not None:
            for a in reversed(self.action_history):
                if a == current_action:
                    accum += 1
                else:
                    break

        confirmed     = None
        new_confirmed = False
        if (len(self.action_history) == CONFIRM_FRAMES and
                len(set(self.action_history)) == 1):
            confirmed = self.action_history[0]
            if confirmed != self.last_confirmed_action:
                new_confirmed = True
                self.last_confirmed_action = confirmed

        return {
            "confirmed":     confirmed,
            "new_confirmed": new_confirmed,
            "current":       current_action,
            "accum":         accum,
            "april_action":  april_action,
            "geo_action":    geo_action,
            "yolo_action":   yolo_action,
            "arrow_angle":   arrow_angle,
            "yolo_center":   yolo_center,
            "yolo_area":     yolo_area,
            "yolo_bbox_bot": yolo_bbox_bot,
        }

    def stop(self):
        self.running = False


if __name__ == "__main__":
    nav = NavigationHeadless()

    def _sig_handler(sig, frame):
        nav.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    nav.run()