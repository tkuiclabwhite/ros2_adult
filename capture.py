#!/usr/bin/env python3
"""
capture_photos.py — 用機器人上的鏡頭拍照，收集 YOLO 訓練素材。

啟動後會開一個拍照網頁（預設 port 8888），在筆電瀏覽器打開就能：
  - 看相機的大畫面（原始解析度，不是控制網頁上縮小的那個）
  - 按空白鍵 / S 拍照
  - 右邊看已經拍的照片縮圖，點開放大檢查，沒拍好可以刪掉

每次執行會在 ros2_adult/capture/ 底下建一個以「日期_時間」命名的資料夾，
例如 ros2_adult/capture/20261001_153012/，這次拍的照片全部存在裡面：
    capture/20261001_153012/20261001_153012_0001.jpg
    capture/20261001_153012/20261001_153012_0002.jpg
    ...
檔名前面帶資料夾名稱，之後把好幾次拍的照片合併成一個資料集時不會撞名。
刪掉的照片編號不會重複使用。

用法：
  1. 開一個 SSH 視窗先啟動相機（二選一）：
       # ZED
       ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedxm
       # 或整套系統（USB / ZED 都可以）
       ros2 launch usb_cam camera.launch.py

  2. 開另一個 SSH 視窗執行本檔：
       source ~/ros2_adult/install/setup.bash
       python3 ~/ros2_adult/capture_photos.py              # 預設 ZED
       python3 ~/ros2_adult/capture_photos.py --source usb # USB 相機

  3. 照終端機印出的網址，用筆電瀏覽器打開 http://<機器人IP>:8888

網頁按鍵：
  空白鍵 / S        拍一張
  點縮圖            放大檢查；放大時 ← → 切換、Delete 刪除、Esc 關閉
終端機按鍵（沒開網頁也能拍）：
  空白鍵 / s        拍一張
  q                 結束（Ctrl+C 也可以）

常用參數：
  --source     zed / usb，決定訂閱哪個 topic（預設 zed）
  --topic      直接指定影像 topic，會蓋過 --source
  --out-root   照片資料夾要建在哪裡（預設 ros2_adult/capture/）
  --port       拍照網頁的 port（預設 8888）
"""
import argparse
import json
import os
import re
import select
import socket
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── 預設設定 ───────────────────────────────────────────────────────────────
# 相機來源 -> 原始影像 topic（跟 photo_capture_node.py 一致）
RAW_TOPICS = {
    'zed': '/zed/zed_node/left/image_rect_color',
    'usb': '/camera1/image_raw',
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, 'capture')            # 照片資料夾預設建在 ros2_adult/capture/
PAGE_PATH = os.path.join(HERE, 'capture_photos.html')

# 8080 web_video_server、9090 rosbridge、9999 控制網頁、5050 熱點 API 都被用掉了
PORT = 8888

JPEG_QUALITY = 95          # 存檔品質
STREAM_QUALITY = 80        # 網頁即時畫面品質（只影響預覽，不影響存檔）
STREAM_FPS = 15            # 網頁即時畫面最高幀率，避免 WiFi 塞爆
THUMB_WIDTH = 320
STALE_SEC = 1.0            # 最新一幀超過這麼久沒更新，就當作相機斷線，不存檔

PHOTO_NAME_RE = re.compile(r'^\d{8}_\d{6}_\d{4}\.jpg$')


def parse_args():
    p = argparse.ArgumentParser(description='用機器人鏡頭拍照，收集 YOLO 訓練素材')
    p.add_argument('--source', default='zed', choices=list(RAW_TOPICS))
    p.add_argument('--topic', default='',
                   help='直接指定影像 topic，非空時蓋過 --source')
    p.add_argument('--out-root', default=OUT_ROOT,
                   help=f'照片資料夾要建在哪裡（預設 {OUT_ROOT}）')
    p.add_argument('--port', type=int, default=PORT)
    return p.parse_args()


class PhotoCapture(Node):
    def __init__(self, args):
        super().__init__('capture_photos')
        self.bridge = CvBridge()
        self.topic = args.topic or RAW_TOPICS[args.source]

        # 資料夾名稱 = 程式啟動時的日期＋時間
        self.session = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(os.path.expanduser(args.out_root), self.session)
        os.makedirs(self.save_dir, exist_ok=True)
        self.count = 0             # 已用掉的編號（刪掉的不回收）

        # 影像 callback 跑在 spin 執行緒，網頁 / 終端機在別的執行緒，用 lock 保護
        self._lock = threading.Lock()
        self._frame = None
        self._frame_t = 0.0
        self._frame_seq = 0
        self._jpeg = None          # 網頁串流用：最新一幀編好的 jpg，多個分頁共用
        self._jpeg_seq = -1
        self._thumbs = {}          # 檔名 -> 縮圖 jpg bytes

        # BEST_EFFORT + depth=1：只要最新一張，而且 RELIABLE / BEST_EFFORT
        # 的 publisher 都接得到
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.topic, self.image_callback, qos)

    def image_callback(self, msg: Image):
        # ZED 是 bgra8、USB 是 bgr8，統一轉成 bgr8 存 jpg
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock:
            self._frame = frame
            self._frame_t = time.time()
            self._frame_seq += 1

    def signal_age(self):
        """距離上一幀幾秒；還沒收到任何畫面時回傳 None。"""
        with self._lock:
            return None if self._frame is None else time.time() - self._frame_t

    def stream_jpeg(self):
        """回傳 (seq, 最新一幀的串流用 jpg)，同一幀只編碼一次。"""
        with self._lock:
            if self._frame is None:
                return -1, None
            if self._jpeg_seq != self._frame_seq:
                ok, buf = cv2.imencode('.jpg', self._frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
                if ok:
                    self._jpeg, self._jpeg_seq = buf.tobytes(), self._frame_seq
            return self._jpeg_seq, self._jpeg

    # ── 照片操作（網頁和終端機共用）──────────────────────────────────────

    def capture(self):
        """存一張最新畫面，回傳 (檔名, None) 或 (None, 錯誤訊息)。"""
        with self._lock:
            frame, t = self._frame, self._frame_t
            if frame is None:
                return None, f'還沒收到任何畫面，請確認相機有開、topic 是 {self.topic}'
            age = time.time() - t
            if age > STALE_SEC:
                return None, f'畫面已經 {age:.1f} 秒沒更新，相機可能斷線，這張不存'
            self.count += 1
            name = f'{self.session}_{self.count:04d}.jpg'

        cv2.imwrite(os.path.join(self.save_dir, name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        h, w = frame.shape[:2]
        print(f'  📷 {name}  ({w}x{h})')
        return name, None

    def list_photos(self):
        """這次拍的照片檔名，新的在前面。"""
        return sorted((f for f in os.listdir(self.save_dir) if PHOTO_NAME_RE.match(f)),
                      reverse=True)

    def photo_path(self, name):
        """檔名合法且存在才回傳完整路徑，擋掉 ../ 之類的路徑。"""
        if not PHOTO_NAME_RE.match(name):
            return None
        path = os.path.join(self.save_dir, name)
        return path if os.path.isfile(path) else None

    def thumbnail(self, name):
        with self._lock:
            if name in self._thumbs:
                return self._thumbs[name]
        path = self.photo_path(name)
        img = cv2.imread(path) if path else None
        if img is None:
            return None
        h, w = img.shape[:2]
        img = cv2.resize(img, (THUMB_WIDTH, round(h * THUMB_WIDTH / w)),
                         interpolation=cv2.INTER_AREA)
        data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        with self._lock:
            self._thumbs[name] = data
        return data

    def delete(self, name):
        path = self.photo_path(name)
        if not path:
            return False
        os.remove(path)
        with self._lock:
            self._thumbs.pop(name, None)
        print(f'  🗑  已刪除 {name}')
        return True


# ── 拍照網頁 ──────────────────────────────────────────────────────────────

def make_handler(node, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass                    # 不要每個請求都印一行，洗掉拍照訊息

        def send_body(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code=200):
            self.send_body(code, json.dumps(obj).encode(), 'application/json')

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                self.send_body(200, page, 'text/html; charset=utf-8')
            elif path == '/stream':
                self.stream()
            elif path == '/api/state':
                age = node.signal_age()
                self.send_json({
                    'session': node.session,
                    'save_dir': node.save_dir,
                    'topic': node.topic,
                    'signal_age': age,
                    'stale_sec': STALE_SEC,
                    'photos': node.list_photos(),
                })
            elif path.startswith('/thumb/'):
                data = node.thumbnail(unquote(path[len('/thumb/'):]))
                if data is None:
                    self.send_error(404)
                else:
                    self.send_body(200, data, 'image/jpeg')
            elif path.startswith('/photo/'):
                p = node.photo_path(unquote(path[len('/photo/'):]))
                if not p:
                    self.send_error(404)
                    return
                with open(p, 'rb') as f:
                    self.send_body(200, f.read(), 'image/jpeg')
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path == '/api/capture':
                name, err = node.capture()
                if err:
                    print(f'  ✘ {err}')
                    self.send_json({'error': err}, 409)
                else:
                    self.send_json({'name': name})
            else:
                self.send_error(404)

        def do_DELETE(self):
            path = urlparse(self.path).path
            if path.startswith('/photo/') and node.delete(unquote(path[len('/photo/'):])):
                self.send_json({'ok': True})
            else:
                self.send_error(404)

        def stream(self):
            """MJPEG 串流：瀏覽器用 <img src="/stream"> 就能直接顯示即時畫面。"""
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            last_seq = -1
            try:
                while rclpy.ok():
                    seq, jpg = node.stream_jpeg()
                    if jpg is None or seq == last_seq:
                        time.sleep(0.02)
                        continue
                    last_seq = seq
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                     + f'Content-Length: {len(jpg)}\r\n\r\n'.encode()
                                     + jpg + b'\r\n')
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass                # 瀏覽器分頁關掉了

    return Handler


def robot_ips():
    """列出本機的 IP，方便印出可以直接點的網址。"""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    try:
        # 不會真的送封包，只是讓系統挑出對外的那張網卡
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith('127.')) or ['<機器人IP>']


def run_terminal(node):
    """終端機按鍵：空白鍵 / s 拍照，q 結束。沒有終端機時就只等 Ctrl+C。"""
    if not sys.stdin.isatty():
        while rclpy.ok():
            time.sleep(0.2)
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key in (' ', 's'):
                _, err = node.capture()
                if err:
                    print(f'  ✘ {err}')
            elif key == 'q':
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    args = parse_args()
    with open(PAGE_PATH, 'rb') as f:
        page = f.read()

    rclpy.init()
    node = PhotoCapture(args)

    # ROS spin 放背景執行緒，主執行緒負責終端機按鍵
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        server = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(node, page))
    except OSError as e:
        print(f'port {args.port} 開不起來（{e}），可能已經有一支 capture_photos.py 在跑，'
              f'或用 --port 換一個')
        os.rmdir(node.save_dir)
        rclpy.shutdown()
        return
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print()
    print(f'訂閱 topic：{node.topic}')
    print(f'存檔資料夾：{node.save_dir}')
    print('拍照網頁（用筆電瀏覽器打開）：')
    for ip in robot_ips():
        print(f'  http://{ip}:{args.port}')
    print('終端機按鍵：空白鍵 / s = 拍照     q = 結束')
    print('-' * 60)

    try:
        run_terminal(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print('-' * 60)
        n = len(node.list_photos())
        if n:
            print(f'結束，共留下 {n} 張，存在 {node.save_dir}')
        else:
            # 一張都沒留就把空資料夾刪掉，避免 capture/ 底下留一堆空資料夾
            try:
                os.rmdir(node.save_dir)
            except OSError:
                pass
            print('結束，這次沒有留下任何照片')
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == '__main__':
    main()