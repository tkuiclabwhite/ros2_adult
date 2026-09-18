#!/usr/bin/env python3
"""
zed_pt_test.py — 在 AGX 上，用 ZED 相機讀取的即時畫面，
                  搭配訓練好的 best.pt 權重做即時辨識測試。

跟你們專案原本的 YoloRosNode（讀 .engine）是同樣的風格，
差別只在於這支讀 .pt（PyTorch 權重），方便你在還沒轉出
.engine 之前，先驗證模型本身準不準、conf 門檻對不對。

用法：
  1. 開一個視窗先啟動 ZED：
       cd ~/ros2_adult && source install/setup.bash
       ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedxm

  2. 開另一個視窗（或 byobu 分割）執行本檔：
       source ~/ros2_adult/install/setup.bash
       python3 zed_pt_test.py

  3. 顯示方式三選一：
     a) --no-show（預設不開任何顯示，純終端機文字模式，SSH 無螢幕時最穩）
       python3 zed_pt_test.py --no-show
     b) --show-window（用 cv2.imshow 直接彈窗，需要本機螢幕或 ssh -X）
       python3 zed_pt_test.py --show-window
     c) --publish（不彈窗，改成發布成 ROS2 影像 topic，
        另外用 rqt_image_view 或 ros2 run image_view 訂閱來看，
        畫面顯示交給官方工具處理，通常比自己開視窗更穩、更不卡）
       python3 zed_pt_test.py --publish
       # 另開一個視窗：
       ros2 run rqt_image_view rqt_image_view
       # 下拉選單選 /zed_pt_test/annotated

  4. 按 Ctrl+C 結束（或有彈窗時按 q），會印出統計摘要。

常用參數：
  --model    .pt 權重路徑（預設見下方 MODEL_PATH，可用 --model 覆蓋）
  --topic    影像 topic（預設 ZED 左眼校正後畫面）
  --imgsz    推論尺寸，要跟訓練時一致（預設 960）
  --conf     信心度門檻（預設 0.5）
  --label    這次手上拿的牌子類別（F / L / R），用來自動算正確率
  --show-window 彈出 cv2 視窗顯示（預設不開）
  --publish  發布標註後影像成 topic，用 rqt_image_view 等工具訂閱看
  --pub-topic 發布的 topic 名稱（預設 /zed_pt_test/annotated）
  --save     把每一幀原始畫面存檔，方便事後離線分析
"""
import argparse
import os
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# ── 預設設定，依實際路徑調整 ─────────────────────────────────────────────
MODEL_PATH = "/home/iclab/ros2_adult/best_zed.pt"
IMAGE_TOPIC = "/zed/zed_node/left/image_rect_color"

SAVE_DIR = os.path.expanduser("~/zed_pt_debug_frames")

# 三類的類別對照（若之後補齊 6 類，記得同步 navigation_node.py 的對照表）
CLASS_NAMES_HINT = "F / L / R"


def parse_args():
    p = argparse.ArgumentParser(description="AGX 上用 ZED + best.pt 做即時辨識測試")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--topic", default=IMAGE_TOPIC)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--label", default=None, choices=[None, "F", "L", "R"])
    p.add_argument("--show-window", action="store_true",
                    help="彈出 cv2 視窗顯示（預設不開，SSH 環境容易卡/需要 X11）")
    p.add_argument("--publish", action="store_true",
                    help="不彈窗，改成發布標註後影像成 ROS2 topic，用 rqt_image_view 訂閱看")
    p.add_argument("--pub-topic", default="/zed_pt_test/annotated",
                    help="發布的 topic 名稱（配合 --publish 使用）")
    p.add_argument("--save", action="store_true",
                    help="把每一幀原始畫面存檔到 SAVE_DIR，方便事後離線分析")
    return p.parse_args()


class ZedPtTest(Node):
    def __init__(self, args):
        super().__init__("zed_pt_test")
        self.args = args
        self.bridge = CvBridge()
        # 預設不開視窗、不發布，純終端機文字輸出最穩；用參數才打開額外顯示方式
        self.show = args.show_window
        self.publish = args.publish

        self.frame_count = 0
        self.save_count = 0
        self.first_frame = True
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._fps = 0.0

        # 滾動統計
        self.history = deque(maxlen=99999999)
        self.conf_by_class = {"F": [], "L": [], "R": []}

        if args.save:
            os.makedirs(SAVE_DIR, exist_ok=True)

        self.get_logger().info(f"載入模型：{args.model}")
        if not os.path.exists(args.model):
            raise FileNotFoundError(
                f"找不到模型檔案：{args.model}\n"
                f"請確認 best.pt 已經傳到這台機器，或用 --model 指定正確路徑"
            )
        self.model = YOLO(args.model)
        self.get_logger().info(f"類別對照（模型內建）：{self.model.names}")
        self.get_logger().info(f"預期類別（{CLASS_NAMES_HINT}）")

        # Warmup，讓第一幀不要因為初始化拖慢、干擾判讀
        self.model(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                   imgsz=args.imgsz, rect=False, verbose=False)
        self.get_logger().info("模型就緒")

        if self.show:
            cv2.namedWindow("ZED + best.pt Detection", cv2.WINDOW_NORMAL)

        if self.publish:
            self.image_pub = self.create_publisher(Image, args.pub_topic, 1)
            self.get_logger().info(f"標註後影像將發布到：{args.pub_topic}")
            self.get_logger().info(
                "另開視窗訂閱查看： ros2 run rqt_image_view rqt_image_view")

        self.create_subscription(Image, args.topic, self.image_callback, 10)

        label_msg = (f"（標記手上拿的是 {args.label}，會自動統計正確率）"
                     if args.label else "（未標記類別，只顯示原始偵測結果）")
        display_mode = ("彈窗" if self.show else
                         ("發布 topic" if self.publish else "純終端機文字"))
        print()
        print(f"訂閱 topic：{args.topic}")
        print(f"conf 門檻：{args.conf}   imgsz：{args.imgsz}   顯示方式：{display_mode}")
        print(label_msg)
        print("結束方式：" + ("視窗按 q，或終端機 Ctrl+C" if self.show else "終端機 Ctrl+C"))
        print("-" * 70)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = frame.shape[:2]
            self.frame_count += 1
            self._fps_n += 1

            if self.first_frame:
                self.get_logger().info(f"實際畫面尺寸：{w}x{h}")
                self.first_frame = False

            now = time.time()
            if now - self._fps_t0 >= 1.0:
                self._fps = self._fps_n / (now - self._fps_t0)
                self._fps_n = 0
                self._fps_t0 = now

            # ── 推論：imgsz + rect=False 跟訓練時一致 ─────────────────────
            results = self.model(frame, imgsz=self.args.imgsz, rect=False,
                                 conf=self.args.conf, verbose=False)
            boxes = results[0].boxes

            dets = []
            for b in boxes:
                cls_id = int(b.cls[0])
                name = self.model.names[cls_id]
                conf = float(b.conf[0])
                area = float((b.xyxy[0][2] - b.xyxy[0][0])
                              * (b.xyxy[0][3] - b.xyxy[0][1]))
                dets.append({"name": name, "conf": conf, "area": area})

            # 取面積最大的當這幀結論（跟 navigation_node.py 篩選邏輯一致）
            top = max(dets, key=lambda d: d["area"]) if dets else None
            top_name = top["name"] if top else None
            if top_name in self.conf_by_class:
                self.conf_by_class[top_name].append(top["conf"])
            self.history.append(top_name)

            # ── 終端機輸出 ───────────────────────────────────────────────
            counter = Counter(x for x in self.history if x is not None)
            n_none = sum(1 for x in self.history if x is None)
            total = len(self.history)

            mark = ""
            if self.args.label and top_name is not None:
                mark = " ✔" if top_name == self.args.label else " ✘"
            elif self.args.label and top_name is None:
                mark = " (漏判)"

            det_str = " | ".join(f'{d["name"]}:{d["conf"]:.3f}' for d in dets) \
                      if dets else "(無偵測)"
            stat_str = "  ".join(f"{k}:{v}" for k, v in counter.most_common())
            if n_none:
                stat_str += f"  無:{n_none}"

            acc_str = ""
            if self.args.label and total > 0:
                n_correct = sum(1 for x in self.history if x == self.args.label)
                acc_str = f"  acc={n_correct}/{total}({n_correct/total*100:.0f}%)"

            print(f"[{self.frame_count:>5}] fps={self._fps:4.1f}  "
                  f"{det_str:<28} -> {top_name or '-':<3}{mark:<8} "
                  f"累計[{stat_str}]{acc_str}")

            if self.args.save:
                path = os.path.join(
                    SAVE_DIR, f"{self.args.label or 'x'}_{self.save_count:05d}.jpg")
                cv2.imwrite(path, frame)
                self.save_count += 1

            # ── 顯示（三選一：彈窗 / 發布 topic / 都不開，純文字）──────────────
            annotated = None
            if self.show or self.publish:
                annotated = results[0].plot()   # 自動畫框 + 類別 + 信心度
                info_lines = [
                    f"frame:{w}x{h}  imgsz:{self.args.imgsz}  conf_thresh:{self.args.conf}",
                    f"fps:{self._fps:.1f}   label:{self.args.label or '-'}   saved:{self.save_count}",
                ]
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated, line, (5, 20 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if self.show:
                # 畫面太大的話（960 以上）稍微縮小顯示，不影響送進模型的原始畫面
                disp = annotated
                if max(h, w) > 960:
                    scale = 960 / max(h, w)
                    disp = cv2.resize(annotated, None, fx=scale, fy=scale)

                cv2.imshow("ZED + best.pt Detection (q=quit, s=save)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.print_summary()
                    rclpy.shutdown()
                elif key == ord("s"):
                    # 手動存檔（跟 live_test.py 的空白鍵/s 邏輯一致），存原始未縮放畫面
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    path = os.path.join(
                        SAVE_DIR, f"{self.args.label or 'x'}_manual_{self.save_count:05d}.jpg")
                    cv2.imwrite(path, frame)
                    print(f"  >>> 手動存檔 {path}")
                    self.save_count += 1

            if self.publish:
                # 發布標註後畫面到 topic，不彈窗，用 rqt_image_view 等工具訂閱查看
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out_msg.header = msg.header
                self.image_pub.publish(out_msg)

        finally:
            # 跟你們原本 YoloRosNode 一樣，每幀清一次 CUDA 快取，避免記憶體堆積
            # 用 locals() 檢查，避免 try 區塊還沒跑到 results 賦值就出錯時，
            # 這裡又因為 results 不存在而報 UnboundLocalError
            if 'results' in locals():
                del results
            torch.cuda.empty_cache()

    def print_summary(self):
        print()
        print("=" * 70)
        print(f"結束。共處理 {self.frame_count} 幀")
        if self.args.save:
            print(f"已存檔 {self.save_count} 張到 {SAVE_DIR}")

        if self.args.label:
            counter = Counter(x for x in self.history if x is not None)
            n_none = sum(1 for x in self.history if x is None)
            total = len(self.history)
            n_correct = sum(1 for x in self.history if x == self.args.label)
            print(f"標記類別：{self.args.label}")
            if total:
                print(f"正確率：{n_correct}/{total} ({n_correct/total*100:.1f}%)")
            print(f"判斷分布：{dict(counter)}   漏判(無偵測)：{n_none}")

        print()
        print("各類別信心度統計：")
        for name in ("F", "L", "R"):
            vals = self.conf_by_class[name]
            if vals:
                print(f"  {name:<3} n={len(vals):<5} "
                      f"min={min(vals):.3f}  avg={sum(vals)/len(vals):.3f}  max={max(vals):.3f}")
            else:
                print(f"  {name:<3} (這次沒有偵測到)")
        print("=" * 70)

    def destroy_node(self):
        if self.show:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    parsed = parse_args()
    rclpy.init(args=args)
    node = ZedPtTest(parsed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()