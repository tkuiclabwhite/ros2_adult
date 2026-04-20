"""
Navigation Node (ROS + TUI 版, v4 - YOLO only)
純 YOLO 偵測,不再使用 AprilTag 偵測器 (模型本身的 F/L/R 類別取代)

影像來源切換:
  USE_ROS_IMAGE = True  → 訂閱 /camera1/image_raw
  USE_ROS_IMAGE = False → 從 /dev/video0 擷取

模型類別對應 (best.engine / best(6).onnx):
  0: F              → STRAIGHT  (AprilTag 代號,不算幾何角度)
  1: L              → LEFT      (AprilTag 代號,不算幾何角度)
  2: R              → RIGHT     (AprilTag 代號,不算幾何角度)
  3: forward arrow  → STRAIGHT  (箭頭,要算幾何角度)
  4: left arrow     → LEFT      (箭頭,要算幾何角度)
  5: right arrow    → RIGHT     (箭頭,要算幾何角度)

發布說明:
  /class_id_topic (String)  : 投票確認後才發,格式 "action,x,y,area,angle"
  /sign_coordinates (Point) : 每 frame 發;無偵測時 x=y=NaN
"""
import os

os.environ['YOLO_VERBOSE'] = 'False'

import time
import threading
import curses
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

# ── 影像來源設定 ────────────────────────────────────────────────────────────
USE_ROS_IMAGE   = True
CAMERA_DEVICE   = "/dev/video0"
ROS_IMAGE_TOPIC = "/camera1/image_raw"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 投票參數 ─────────────────────────────────────────────────────────────────
WINDOW_SECONDS     = 1.0
VOTE_THRESHOLD     = 0.8
MIN_VOTES          = 3
REPUBLISH_INTERVAL = 0.3
YOLO_CONF_THRESH   = 0.5

# ── 幾何驗證角度合理範圍 (含±30°旋轉裕度) ────────────────────────────────────
GEO_ANGLE_RANGE = {
    "STRAIGHT": (-35.0,  35.0),
    "RIGHT":    ( 55.0, 125.0),
    "LEFT":     (-125.0, -55.0),
}

# ── 不做幾何角度計算的類別 (AprilTag 類別沒有箭頭尖點) ───────────────────────
NO_GEO_CLASSES = {"F", "L", "R"}


# ════════════════════════════════════════════════════════════════════════════════
# 幾何箭頭方向驗證
# ════════════════════════════════════════════════════════════════════════════════

def _extract_arrow_contour(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  k)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    cts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cts:
        return None, None

    h_roi, w_roi = gray.shape
    valid = []
    for c in cts:
        x, y, w, hh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x+w >= w_roi-1 or y+hh >= h_roi-1:
            continue
        valid.append(c)

    candidates = valid if valid else cts
    if not candidates:
        return None, None

    arrow = max(candidates, key=cv2.contourArea)
    if cv2.contourArea(arrow) < 500:
        return None, None

    M = cv2.moments(arrow)
    if M["m00"] < 1:
        return None, None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return arrow, (cx, cy)


def _find_key_line(contour, arrow_type):
    if arrow_type == "STRAIGHT":
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        e1 = box[1] - box[0]
        e2 = box[2] - box[1]
        long_vec = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        if long_vec[1] > 0:
            long_vec = -long_vec
        return float(np.degrees(np.arctan2(long_vec[0], -long_vec[1])))

    contour_pts = contour.reshape(-1, 2).astype(np.float32)

    if arrow_type == "RIGHT":
        tip = contour_pts[np.argmax(contour_pts[:, 0])]
    else:
        tip = contour_pts[np.argmin(contour_pts[:, 0])]

    hi = cv2.convexHull(contour, returnPoints=False)
    if hi is None or len(hi) < 3:
        return None
    try:
        defects = cv2.convexityDefects(contour, hi)
    except cv2.error:
        return None
    if defects is None or len(defects) < 2:
        return None

    area = cv2.contourArea(contour)
    min_depth = max(4, int(np.sqrt(area) * 0.04)) * 256
    deep_pts = [contour_pts[int(d[0, 2])]
                for d in defects if d[0, 3] >= min_depth]
    if len(deep_pts) < 2:
        return None

    deep_pts.sort(key=lambda p: np.linalg.norm(p - tip))
    n0, n1 = deep_pts[0], deep_pts[1]
    sm = (n0 + n1) / 2.0

    av = tip - sm
    if np.linalg.norm(av) < 1e-6:
        return None

    return float(np.degrees(np.arctan2(av[0], -av[1])))


def find_arrow_direction(roi_bgr, yolo_action):
    if yolo_action not in GEO_ANGLE_RANGE:
        return None, None

    MIN_SIDE = 200
    h0, w0 = roi_bgr.shape[:2]
    short = min(h0, w0)
    if 0 < short < MIN_SIDE:
        scale = MIN_SIDE / short
        roi_bgr = cv2.resize(roi_bgr, None, fx=scale, fy=scale)

    contour, _ = _extract_arrow_contour(roi_bgr)
    if contour is None:
        return None, None

    angle = _find_key_line(contour, yolo_action)
    if angle is None:
        return None, None

    lo, hi = GEO_ANGLE_RANGE[yolo_action]
    confirmed = yolo_action if lo <= angle <= hi else None
    return confirmed, round(angle, 1)


# ════════════════════════════════════════════════════════════════════════════════
# TUI 狀態
# ════════════════════════════════════════════════════════════════════════════════

class TUIState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_count = 0; self.fps = 0.0; self.frame_size = (0, 0)
        self.yolo_action = None; self.yolo_name = None; self.yolo_conf = 0.0
        self.yolo_center = None; self.yolo_area = 0; self.yolo_num_candidates = 0
        self.geo_action = None; self.geo_angle = None; self.final_action = None
        self.current_action = None; self.last_confirmed_action = None
        self.vote_counts = {}; self.vote_leader = None
        self.vote_leader_ratio = 0.0; self.window_votes = 0
        self.published_count = 0; self.last_publish_time = 0.0
        self.last_source = "-"; self.publish_log = deque(maxlen=5)
        self.coord_pub_count = 0
        self.last_error = ""

    def snapshot(self):
        with self.lock:
            snap = {k: v for k, v in self.__dict__.items() if k != "lock"}
            snap['publish_log'] = list(self.publish_log)
            return snap


# ════════════════════════════════════════════════════════════════════════════════
# TUI 顯示
# ════════════════════════════════════════════════════════════════════════════════

def tui_loop(stdscr, state, stop_event):
    curses.curs_set(0); stdscr.nodelay(True)
    curses.start_color(); curses.use_default_colors()
    for i, c in enumerate([curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_CYAN,
                           curses.COLOR_RED, curses.COLOR_MAGENTA], 1):
        curses.init_pair(i, c, -1)
    GREEN, YELLOW, CYAN, RED, MAGENTA = (curses.color_pair(i) for i in range(1, 6))

    while not stop_event.is_set():
        s = state.snapshot()
        stdscr.erase()
        try:
            r = 0
            stdscr.addstr(r, 2, "=== Navigation Node Monitor (YOLO-only) ===",
                          CYAN | curses.A_BOLD); r += 2
            stdscr.addstr(r, 2, f"Frames: {s['frame_count']:>6}   FPS: {s['fps']:>5.1f}   "
                                f"Size: {s['frame_size'][0]}x{s['frame_size'][1]}"); r += 2

            stdscr.addstr(r, 2, "-- YOLO + GEO --", YELLOW | curses.A_BOLD); r += 1
            stdscr.addstr(r, 4, f"YOLO  : {s['yolo_action'] or '-'}  "
                                f"({s['yolo_name'] or '-'} conf={s['yolo_conf']:.2f})",
                          GREEN if s['yolo_action'] else curses.A_DIM); r += 1
            angle_str = f"{s['geo_angle']:+.1f}deg" if s['geo_angle'] is not None else '-'
            geo_color = GREEN if s['geo_action'] else (YELLOW if s['geo_angle'] is not None else curses.A_DIM)
            stdscr.addstr(r, 4, f"GEO   : {s['geo_action'] or 'SKIP/FAIL':<10} angle={angle_str}",
                          geo_color); r += 1
            stdscr.addstr(r, 4, f"Final : {s['final_action'] or '-'}",
                          MAGENTA if s['final_action'] else curses.A_DIM); r += 1
            stdscr.addstr(r, 4, f"Center: {s['yolo_center'] or '-'}  area={s['yolo_area']}  "
                                f"candi={s['yolo_num_candidates']}"); r += 2

            stdscr.addstr(r, 2, "-- Decision --", YELLOW | curses.A_BOLD); r += 1
            stdscr.addstr(r, 4, f"Current  : {s['current_action'] or 'NONE'}",
                          GREEN if s['current_action'] else RED); r += 1
            ratio = s['vote_leader_ratio']
            bar = "#" * int(20 * ratio) + "." * (20 - int(20 * ratio))
            rc = GREEN if ratio >= VOTE_THRESHOLD else YELLOW
            stdscr.addstr(r, 4, f"Window   : {s['window_votes']} votes / {WINDOW_SECONDS:.1f}s"); r += 1
            stdscr.addstr(r, 4, f"Leader   : {s['vote_leader'] or '-':<10} [{bar}] {ratio*100:5.1f}%", rc); r += 1
            cs = "  ".join(f"{k}:{v}" for k, v in s['vote_counts'].items()) or "-"
            stdscr.addstr(r, 4, f"Counts   : {cs}"); r += 1
            stdscr.addstr(r, 4, f"Confirmed: {s['last_confirmed_action'] or '-'}",
                          MAGENTA | curses.A_BOLD); r += 2

            stdscr.addstr(r, 2, "-- Publish --", YELLOW | curses.A_BOLD); r += 1
            stdscr.addstr(r, 4, f"Cmd   : {s['published_count']}   Source: {s['last_source']}"); r += 1
            stdscr.addstr(r, 4, f"Coord : {s['coord_pub_count']} (live, inc NaN)"); r += 1
            if s['last_publish_time']:
                stdscr.addstr(r, 4, f"Last  : {time.time() - s['last_publish_time']:5.1f}s ago"); r += 1
            r += 1

            stdscr.addstr(r, 2, "-- Recent Publishes (last 5) --", YELLOW | curses.A_BOLD); r += 1
            if not s['publish_log']:
                stdscr.addstr(r, 4, "(no publishes yet)", curses.A_DIM); r += 1
            else:
                stdscr.addstr(r, 4, f"{'#':<3}{'time':<11}{'action':<10}{'src':<6}data",
                              curses.A_UNDERLINE); r += 1
                for e in reversed(s['publish_log']):
                    color = CYAN
                    line = (f"{e['idx']:<3}{time.time() - e['t']:>5.1f}s ago "
                            f"{e['action']:<10}{e['src']:<6}{e['data']}")
                    stdscr.addstr(r, 4, line[:curses.COLS - 6], color); r += 1
            r += 1

            if s['last_error']:
                stdscr.addstr(r, 2, f"ERROR: {s['last_error']}", RED); r += 1
            stdscr.addstr(r + 1, 2, "Press Ctrl+C to quit", curses.A_DIM)
        except curses.error:
            pass
        stdscr.refresh()
        time.sleep(0.1)


# ════════════════════════════════════════════════════════════════════════════════
# Navigation Node
# ════════════════════════════════════════════════════════════════════════════════

class NavigationNode(Node):
    def __init__(self, state):
        super().__init__('navigation_node')
        self.state = state
        self.bridge = CvBridge()
        self.vote_window = deque()
        self.last_confirmed_action = None
        self.last_publish_time = 0.0
        self._fps_t0 = time.time()
        self._fps_count = 0

        model_path = os.path.join(BASE_DIR, 'best.engine')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"[Nav] Loading YOLO model: {model_path}")
        self.yolo_model = YOLO(model_path, task='detect')

        # trtexec 轉的 engine 沒有 metadata,直接用 class id 寫死對應
        # best(6).onnx 類別順序:
        #   0:F  1:L  2:R  3:forward arrow  4:left arrow  5:right arrow
        self.yolo_class_names = {
            0: "F",
            1: "L",
            2: "R",
            3: "forward arrow",
            4: "left arrow",
            5: "right arrow",
        }
        self.yolo_to_action = {
            "F":             "STRAIGHT",
            "L":             "LEFT",
            "R":             "RIGHT",
            "forward arrow": "STRAIGHT",
            "left arrow":    "LEFT",
            "right arrow":   "RIGHT",
        }

        # Warmup (讓 engine 一次初始化完,避免後續延遲)
        print("[Nav] Warming up engine...")
        _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.yolo_model(_dummy, imgsz=320, verbose=False)
        print("[Nav] YOLO ready.")

        self.class_id_pub = self.create_publisher(String, 'class_id_topic', 1)
        self.coord_pub    = self.create_publisher(Point,  'sign_coordinates', 1)

        self.cap = None
        if USE_ROS_IMAGE:
            self.image_sub = self.create_subscription(
                Image, ROS_IMAGE_TOPIC, self._ros_image_callback, 10)
            print(f"[Nav] Subscribing: {ROS_IMAGE_TOPIC}")
        else:
            self.cap = cv2.VideoCapture(CAMERA_DEVICE)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            threading.Thread(target=self._camera_loop, daemon=True).start()
            print(f"[Nav] Capturing from {CAMERA_DEVICE}")

    def _ros_image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame(frame)
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f'cv_bridge: {e}'

    def _camera_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.process_frame(frame)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2

        self._fps_count += 1
        now = time.time()
        fps = None
        if now - self._fps_t0 >= 1.0:
            fps = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now

        # ── YOLO 偵測 ───────────────────────────────────────────────────────
        yolo_action = yolo_name = yolo_center = None
        yolo_conf   = 0.0
        yolo_area   = 0
        yolo_bbox_bottom = None
        num_candidates   = 0
        geo_action  = None
        geo_angle   = None
        final_action = None

        results = self.yolo_model(frame, imgsz=320, verbose=False)
        candidates = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            if cls not in self.yolo_class_names or conf < YOLO_CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            candidates.append({
                'name':        self.yolo_class_names[cls],
                'conf':        conf,
                'center':      ((x1+x2)//2, (y1+y2)//2),
                'area':        (x2-x1) * (y2-y1),
                'bbox_bottom': y2,
                'bbox':        (x1, y1, x2, y2),
            })
        num_candidates = len(candidates)

        if candidates:
            best = max(candidates, key=lambda x: x['area'])
            yolo_name        = best['name']
            yolo_conf        = best['conf']
            yolo_action      = self.yolo_to_action.get(best['name'])
            yolo_center      = best['center']
            yolo_area        = best['area']
            yolo_bbox_bottom = best['bbox_bottom']

            # 幾何角度計算 (只對箭頭類別做,AprilTag 類別 F/L/R 跳過)
            if yolo_action is not None and best['name'] not in NO_GEO_CLASSES:
                x1, y1, x2, y2 = best['bbox']
                pad = 20
                ax1 = max(0, x1 - pad); ay1 = max(0, y1 - pad)
                ax2 = min(w, x2 + pad); ay2 = min(h, y2 + pad)
                roi = frame[ay1:ay2, ax1:ax2]

                if roi.size > 0:
                    geo_action, geo_angle = find_arrow_direction(roi, yolo_action)

            final_action = geo_action if geo_action is not None else yolo_action

        current_action = final_action

        # ── 時間視窗投票 ────────────────────────────────────────────────────
        now_t = time.time()
        self.vote_window.append((now_t, current_action))
        cutoff = now_t - WINDOW_SECONDS
        while self.vote_window and self.vote_window[0][0] < cutoff:
            self.vote_window.popleft()

        vote_counts = {}
        for _, a in self.vote_window:
            if a is None:
                continue
            vote_counts[a] = vote_counts.get(a, 0) + 1

        total_votes  = len(self.vote_window)
        leader       = None
        leader_ratio = 0.0
        if vote_counts:
            leader       = max(vote_counts, key=vote_counts.get)
            leader_ratio = vote_counts[leader] / total_votes if total_votes else 0.0
        elif self.last_confirmed_action is not None:
            self.last_confirmed_action = None

        # ── 發布 class_id_topic ─────────────────────────────────────────────
        published_now = False
        source = None
        data   = None
        is_new_action = False
        time_since_pub = now_t - self.last_publish_time

        if (total_votes >= MIN_VOTES
                and leader is not None
                and leader_ratio >= VOTE_THRESHOLD
                and (leader != self.last_confirmed_action
                     or time_since_pub >= REPUBLISH_INTERVAL)):

            confirmed    = leader
            is_new_action = (leader != self.last_confirmed_action)
            self.last_confirmed_action = confirmed
            self.last_publish_time     = now_t

            if geo_action == confirmed:
                source = "GEO"
            else:
                source = "YOLO"

            pub_angle = geo_angle if geo_angle is not None else 0.0

            if yolo_center and yolo_area:
                data = f"{confirmed.lower()},{yolo_center[0]},{yolo_bbox_bottom},{yolo_area},{pub_angle:.1f}"
            else:
                data = f"{confirmed.lower()},{cx_frame},{cy_frame},0,{pub_angle:.1f}"

            ms = String(); ms.data = data
            self.class_id_pub.publish(ms)
            published_now = True

        # ── 發布 sign_coordinates (每 frame) ────────────────────────────────
        mp_live = Point()
        if yolo_center is not None:
            mp_live.x = float(yolo_center[0])
            mp_live.y = float(yolo_center[1])
            mp_live.z = float(geo_angle) if geo_angle is not None else 0.0
        else:
            mp_live.x = float('nan')
            mp_live.y = float('nan')
            mp_live.z = 0.0
        self.coord_pub.publish(mp_live)

        # ── TUI 狀態更新 ────────────────────────────────────────────────────
        with self.state.lock:
            self.state.frame_count += 1
            if fps is not None:
                self.state.fps = fps
            self.state.frame_size            = (w, h)
            self.state.yolo_action           = yolo_action
            self.state.yolo_name             = yolo_name
            self.state.yolo_conf             = yolo_conf
            self.state.yolo_center           = yolo_center
            self.state.yolo_area             = yolo_area
            self.state.yolo_num_candidates   = num_candidates
            self.state.geo_action            = geo_action
            self.state.geo_angle             = geo_angle
            self.state.final_action          = final_action
            self.state.current_action        = current_action
            self.state.vote_counts           = vote_counts
            self.state.vote_leader           = leader
            self.state.vote_leader_ratio     = leader_ratio
            self.state.window_votes          = total_votes
            self.state.last_confirmed_action = self.last_confirmed_action
            self.state.coord_pub_count      += 1
            if published_now:
                self.state.published_count   += 1
                self.state.last_publish_time  = time.time()
                self.state.last_source        = source
                if is_new_action:
                    self.state.publish_log.append({
                        'idx':    self.state.published_count,
                        't':      time.time(),
                        'action': self.last_confirmed_action,
                        'src':    source,
                        'data':   data,
                    })


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    state = TUIState()
    stop_event = threading.Event()
    node = NavigationNode(state)

    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    try:
        curses.wrapper(lambda scr: tui_loop(scr, state, stop_event))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if node.cap is not None:
            node.cap.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
