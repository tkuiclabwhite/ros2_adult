import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque
from pupil_apriltags import Detector
import threading
import queue

april_detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,    # 預設2.0，改1.0精度更高（俯角變形時更重要）
    quad_sigma=0.5,       # 輕微模糊去除雜訊
    refine_edges=True,    # 邊緣精修，提升變形時的辨識率
    decode_sharpening=0.25,
)
TAG_ACTION = {1: "FORWARD", 2: "RIGHT", 3: "LEFT"}

DEFAULT_MIN_ARROW_AREA = 2000
DEFAULT_MIN_DARK_AREA  = 1500
DEFAULT_BLACK_THRESH   = 80
DEFAULT_WHITE_THRESH   = 150
DEFAULT_CONFIRM_FRAMES = 3
DEFAULT_OFFSET_THRESH  = 25


def find_arrow_boxes(frame, masked, black_thresh, min_dark_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_f, w_f = frame.shape[:2]
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_dark_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > w_f * 0.9 or h > h_f * 0.9:
            continue
        # 面積上限（避免大區塊誤判）
        if w * h > w_f * h_f * 0.4:
            continue
        if not (0.4 < w / h < 2.5):
            continue
        roi_gray = gray[y:y+h, x:x+w]
        _, roi_dark = cv2.threshold(roi_gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
        if cv2.countNonZero(roi_dark) / (w * h) < 0.30:
            continue

        # 排除 AprilTag 遮蔽區域（灰色 120~135）
        roi_bgr_check = masked[y:y+h, x:x+w]
        gray_mean = cv2.mean(cv2.cvtColor(roi_bgr_check, cv2.COLOR_BGR2GRAY))[0]
        if 115 < gray_mean < 140:
            continue
        boxes.append((x, y, w, h))
    return boxes


def detect_arrow(roi_bgr, white_thresh, min_arrow_area, offset_thresh):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE 自適應對比度增強
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # 百分位自適應閾值
    top_bright = np.percentile(gray, 95)
    bottom_dark = np.percentile(gray, 5)
    threshold = (top_bright + bottom_dark) / 2
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(binary) < min_arrow_area:
        return None, binary

    # 找白色區域輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, binary

    # 取最大白色輪廓（箭頭本體）
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_arrow_area:
        return None, binary

    # 計算白色區域重心
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, binary
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 以重心為分界，分析白色和黑色象限分布
    h, w = binary.shape
    cx_i, cy_i = int(cx), int(cy)

    # 白色上下差異判斷直走
    w_top    = cv2.countNonZero(binary[:cy_i, :])
    w_bottom = cv2.countNonZero(binary[cy_i:, :])
    w_left   = cv2.countNonZero(binary[:, :cx_i])
    w_right  = cv2.countNonZero(binary[:, cx_i:])

    # 黑色下半左右比較（最關鍵的區域）
    inv = cv2.bitwise_not(binary)
    b_bottom_left  = cv2.countNonZero(inv[cy_i:, :cx_i])
    b_bottom_right = cv2.countNonZero(inv[cy_i:, cx_i:])

    tb_diff = w_top - w_bottom
    lr_diff = w_right - w_left
    total   = w_top + w_bottom

    # 直走判斷：中間垂直帶白色較多（箭桿在中間）
    center_band = binary[:, w//3 : 2*w//3]
    center_white = cv2.countNonZero(center_band)
    center_ratio = center_white / center_band.size

    if center_ratio > 0.25 and tb_diff > 0:
        return "FORWARD", binary

    # 左右轉：比較下半左右角落的黑色數量
    # 右下角黑色多 → 右轉
    # 左下角黑色多 → 左轉
    if b_bottom_right > b_bottom_left:
        return "RIGHT", binary
    else:
        return "LEFT", binary


class NavigationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Navigation Vision — Debug")
        self.root.configure(bg="#0d0d0d")
        self.root.resizable(True, True)

        # 執行緒間共享
        self.frame_queue  = queue.Queue(maxsize=1)  # 相機 → 處理
        self.result_queue = queue.Queue(maxsize=1)  # 處理 → GUI

        self.params = {}
        self.running = True

        self.action_history        = deque(maxlen=DEFAULT_CONFIRM_FRAMES)
        self.last_confirmed_action = None

        self._build_ui()

        # 相機執行緒
        self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self.cam_thread.start()

        # 處理執行緒
        self.proc_thread = threading.Thread(target=self._proc_loop, daemon=True)
        self.proc_thread.start()

        # GUI 更新
        self._gui_update()

    # ── UI ───────────────────────────────────────────────
    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0d0d0d")
        header.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(header, text="NAVIGATION VISION", font=("Courier", 14, "bold"),
                 fg="#00ff99", bg="#0d0d0d").pack(side="left")
        self.lbl_action = tk.Label(header, text="ACTION: —",
                                   font=("Courier", 14, "bold"),
                                   fg="#ffdd00", bg="#0d0d0d")
        self.lbl_action.pack(side="right")

        cam_frame = tk.Frame(self.root, bg="#0d0d0d")
        cam_frame.pack(padx=16, pady=4)

        def cam_panel(parent, title):
            f = tk.Frame(parent, bg="#1a1a1a")
            f.pack(side="left", padx=6)
            tk.Label(f, text=title, font=("Courier", 9),
                     fg="#555555", bg="#1a1a1a").pack(pady=(4, 0))
            lbl = tk.Label(f, bg="#1a1a1a")
            lbl.pack(padx=4, pady=4)
            return lbl

        self.lbl_orig   = cam_panel(cam_frame, "ORIGINAL")
        self.lbl_masked = cam_panel(cam_frame, "MASKED + BOX")
        self.lbl_binary = cam_panel(cam_frame, "BINARY (ROI)")

        tk.Frame(self.root, bg="#222222", height=1).pack(fill="x", padx=16, pady=6)

        ctrl = tk.Frame(self.root, bg="#0d0d0d")
        ctrl.pack(fill="x", padx=16, pady=(0, 12))

        sliders = [
            ("min_arrow_area", "白色最小面積",    500, 20000, DEFAULT_MIN_ARROW_AREA),
            ("min_dark_area",  "黑框最小面積",    200, 10000, DEFAULT_MIN_DARK_AREA),
            ("black_thresh",   "黑色閾值",         20,   150, DEFAULT_BLACK_THRESH),
            ("white_thresh",   "白色閾值",        100,   254, DEFAULT_WHITE_THRESH),
            ("confirm_frames", "確認幀數",          1,    10, DEFAULT_CONFIRM_FRAMES),
            ("offset_thresh",  "箭頭角度門檻(度)", 10,   80, 45),
        ]

        for i, (key, label, lo, hi, default) in enumerate(sliders):
            row = tk.Frame(ctrl, bg="#0d0d0d")
            row.grid(row=i // 2, column=i % 2, padx=12, pady=4, sticky="ew")
            ctrl.columnconfigure(i % 2, weight=1)
            tk.Label(row, text=label, font=("Courier", 9),
                     fg="#aaaaaa", bg="#0d0d0d", width=18, anchor="w").pack(side="left")
            var = tk.IntVar(value=default)
            self.params[key] = var
            val_lbl = tk.Label(row, text=str(default), font=("Courier", 9),
                               fg="#00ff99", bg="#0d0d0d", width=6)
            val_lbl.pack(side="right")

            def make_cb(v, lbl):
                def cb(_): lbl.config(text=str(v.get()))
                return cb

            ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                      command=make_cb(var, val_lbl)).pack(side="left", fill="x", expand=True, padx=6)

        tk.Frame(self.root, bg="#222222", height=1).pack(fill="x", padx=16, pady=4)
        log_frame = tk.Frame(self.root, bg="#0d0d0d")
        log_frame.pack(fill="x", padx=16, pady=(0, 4))
        tk.Label(log_frame, text="LOG", font=("Courier", 9),
                 fg="#555555", bg="#0d0d0d").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=3, font=("Courier", 9),
                                bg="#111111", fg="#00ff99", bd=0,
                                insertbackground="#00ff99", state="disabled")
        self.log_text.pack(fill="x")

        # 偵測數值顯示區
        stats_frame = tk.Frame(self.root, bg="#0d0d0d")
        stats_frame.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(stats_frame, text="DETECTED BOXES", font=("Courier", 9),
                 fg="#555555", bg="#0d0d0d").pack(anchor="w")
        self.lbl_stats = tk.Label(stats_frame, text="—",
                                  font=("Courier", 9), fg="#aaaaaa",
                                  bg="#111111", anchor="w", justify="left")
        self.lbl_stats.pack(fill="x")

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        if int(self.log_text.index("end-1c").split(".")[0]) > 100:
            self.log_text.delete("1.0", "2.0")
        self.log_text.config(state="disabled")

    # ── 相機執行緒 ────────────────────────────────────────
    def _cam_loop(self):
        cap = cv2.VideoCapture("/dev/video1")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            # 丟棄舊幀，只保留最新
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        cap.release()

    # ── 處理執行緒 ────────────────────────────────────────
    def _proc_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            p = {k: v.get() for k, v in self.params.items()}
            confirm_frames = p["confirm_frames"]

            if self.action_history.maxlen != confirm_frames:
                self.action_history = deque(self.action_history, maxlen=confirm_frames)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # AprilTag
            april_results = april_detector.detect(gray)
            april_action = None
            april_corners_list = []
            orig_vis = frame.copy()
            for r in april_results:
                action = TAG_ACTION.get(r.tag_id)
                if action:
                    april_action = action
                    corners = r.corners.astype(int)
                    april_corners_list.append(corners)
                    cv2.polylines(orig_vis, [corners], True, (0, 255, 80), 2)
                    cv2.putText(orig_vis, f"TAG:{action}", tuple(corners[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)

            # 遮蔽
            masked = frame.copy()
            for corners in april_corners_list:
                cv2.fillPoly(masked, [corners], (127, 127, 127))

            # 找黑框（AprilTag 已偵測到則跳過箭頭辨識）
            boxes = [] if april_action else find_arrow_boxes(masked, masked, p["black_thresh"], p["min_dark_area"])
            arrow_action = None
            arrow_pos = None
            masked_vis = masked.copy()
            binary_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            for (x, y, w, h) in boxes:
                roi = masked[y:y+h, x:x+w]
                action, binary_roi = detect_arrow(roi, p["white_thresh"],
                                                  p["min_arrow_area"], p["offset_thresh"])
                binary_canvas[y:y+h, x:x+w] = binary_roi
                color = (0, 255, 100) if action else (80, 80, 200)
                cv2.rectangle(masked_vis, (x, y), (x+w, y+h), color, 2)
                area = w * h
                # 計算黑白比例供調參
                roi_gray_vis = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, rd = cv2.threshold(roi_gray_vis, p["black_thresh"], 255, cv2.THRESH_BINARY_INV)
                _, rw = cv2.threshold(roi_gray_vis, 180, 255, cv2.THRESH_BINARY)
                dark_r  = cv2.countNonZero(rd) / (w * h)
                white_r = cv2.countNonZero(rw) / (w * h)
                label = f"{action or '?'} a={area} d={dark_r:.2f} w={white_r:.2f}"
                cv2.putText(masked_vis, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if action and not arrow_action:
                    arrow_action = action
                    arrow_pos = (x, y, w, h)

            # stats 文字
            if boxes:
                stats_lines = []
                for (x, y, w, h) in boxes:
                    roi = masked[y:y+h, x:x+w]
                    roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, rd = cv2.threshold(roi_g, p["black_thresh"], 255, cv2.THRESH_BINARY_INV)
                    _, rw = cv2.threshold(roi_g, 180, 255, cv2.THRESH_BINARY)
                    dr = cv2.countNonZero(rd) / (w * h)
                    wr = cv2.countNonZero(rw) / (w * h)
                    stats_lines.append(f"area={w*h}  dark={dr:.2f}  white={wr:.2f}  pos=({x},{y})")
                stats_text = "\n".join(stats_lines)
            else:
                stats_text = "（無偵測到框）"

            # 決策
            current_action = april_action or arrow_action

            # 連續幀確認
            self.action_history.append(current_action)
            confirmed = None
            new_confirmed = False
            if (len(self.action_history) == confirm_frames and
                    len(set(self.action_history)) == 1):
                confirmed = self.action_history[0]
                if confirmed != self.last_confirmed_action:
                    new_confirmed = True
                    self.last_confirmed_action = confirmed

            # 送結果給 GUI
            result = {
                "orig":    orig_vis,
                "masked":  masked_vis,
                "binary":  binary_canvas,
                "action":  self.last_confirmed_action,
                "log":     f"ACTION: {confirmed}  (TAG:{april_action} ARROW:{arrow_action})"
                           if confirmed and confirmed != (self.last_confirmed_action
                               if confirmed == self.last_confirmed_action else None)
                           else None,
                "new_confirmed": new_confirmed,
                "confirmed": confirmed,
                "april": april_action,
                "arrow": arrow_action,
                "arrow_pos": arrow_pos,
                "stats": stats_text,
            }
            if not self.result_queue.full():
                self.result_queue.put(result)

    # ── GUI 更新（主執行緒）──────────────────────────────
    def _cv2tk(self, img, w=320, h=240):
        img = cv2.resize(img, (w, h))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(img))

    def _gui_update(self):
        try:
            result = self.result_queue.get_nowait()

            tk_orig   = self._cv2tk(result["orig"])
            tk_masked = self._cv2tk(result["masked"])
            tk_binary = self._cv2tk(result["binary"])

            self.lbl_orig.config(image=tk_orig);     self.lbl_orig.image   = tk_orig
            self.lbl_masked.config(image=tk_masked); self.lbl_masked.image = tk_masked
            self.lbl_binary.config(image=tk_binary); self.lbl_binary.image = tk_binary

            display = result["action"] or "—"
            self.lbl_action.config(text=f"ACTION: {display}")

            if result["new_confirmed"] and result["confirmed"]:
                self._log(f"ACTION: {result['confirmed']}  "
                          f"(TAG:{result['april']} ARROW:{result['arrow']})  "
                          f"pos={result['arrow_pos']}")

            self.lbl_stats.config(text=result.get("stats", "—"))

        except queue.Empty:
            pass

        self.root.after(33, self._gui_update)

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NavigationGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()