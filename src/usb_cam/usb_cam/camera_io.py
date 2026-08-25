"""USB 相機 I/O 包裝層。

負責直接與 V4L2 / 影像 codec 打交道，把抓 frame、解碼、控制項設定等實作細
節集中在這裡，讓節點層 (usb_cam_node.py) 可以專注處理 ROS 介面與業務邏輯。

實作策略：
  - **影像抓取**：使用 OpenCV ``cv2.VideoCapture`` 走 V4L2 後端。OpenCV
    內部透過 libjpeg / libavcodec 處理 MJPEG、YUYV 等格式，並一律輸出
    BGR numpy 陣列。對 320x240@30fps 等實際工作負載性能完全足夠。
  - **控制項**（亮度 / 對比 / 白平衡 / 自動曝光等）：透過 ``v4l2-ctl``
    命令列工具設定。
  - **MJPEG 壓縮輸出**：OpenCV API 不暴露 driver 給的 raw JPEG buffer，
    因此 pixel_format=='mjpeg' 時採取「解碼後再用 cv2.imencode 重新編碼」
    產生等效 JPEG。對下游視覺等效，僅多一次 JPEG 編碼成本。
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 接受的 io_method 字串。OpenCV V4L2 後端內部只走 mmap / read 兩種，這裡仍
# 接受三種字串純粹是為了相容既有的 params YAML。
# ---------------------------------------------------------------------------
VALID_IO_METHODS = {"mmap", "read", "userptr"}

# pixel_format 名稱 → V4L2 fourcc 四字元代碼。
# cv2 設定 fourcc 後若硬體支援，OpenCV 會請 driver 用該格式輸出。
_PIXEL_FORMAT_TO_FOURCC = {
    "yuyv": "YUYV",
    "uyvy": "UYVY",
    "mjpeg": "MJPG",
    "mjpeg2rgb": "MJPG",  # 由 cv2 解 MJPEG → BGR
    "m420": "M420",
    "mono8": "GREY",
    "mono16": "Y16 ",
    "rgb": "RGB3",
}


def resolve_device_path(path: str) -> str:
    """解析裝置路徑：若是 symlink 解析到真實裝置。"""
    p = Path(path)
    try:
        if p.is_symlink():
            return str(p.resolve())
    except OSError:
        # symlink 解析失敗就維持原字串，後續開檔錯誤再交由上層處理
        pass
    return str(p)


def set_v4l_parameter(device: str, name: str, value) -> bool:
    """呼叫 v4l2-ctl 設定單一控制項。

    任何非空輸出都視為錯誤；回傳 True=失敗、False=成功。
    """
    cmd = ["v4l2-ctl", f"--device={device}", "-c", f"{name}={value}"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=2.0
        )
        output = (result.stdout or "") + (result.stderr or "")
        if output.strip():
            print(output, flush=True)
            return True
        return False
    except FileNotFoundError:
        print("[camera_io] v4l2-ctl not found in PATH", flush=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"[camera_io] v4l2-ctl timeout: {' '.join(cmd)}", flush=True)
        return True


# ---------------------------------------------------------------------------
# 參數結構：欄位名稱與 ROS2 參數宣告對齊
# ---------------------------------------------------------------------------
@dataclass
class CameraParameters:
    """所有可設定的相機參數，欄位名稱與 ROS2 參數宣告對齊。"""
    # 基本資訊
    camera_name: str = "default_cam"
    camera_info_url: str = ""
    frame_id: str = "default_cam"
    framerate: float = 30.0
    image_width: int = 320
    image_height: int = 240
    io_method_name: str = "mmap"
    pixel_format_name: str = "yuyv"
    av_device_format: str = "YUV422P"
    device_name: str = "/dev/video0"
    # 控制項：-1 / 小於 -64 等代表 "保留不動"，由 _set_v4l2_params 判斷
    brightness: int = 140
    contrast: int = 200
    saturation: int = 100
    sharpness: int = -1
    gain: int = -1
    auto_white_balance: bool = False
    white_balance: int = -1
    autoexposure: bool = False
    exposure: int = -1
    autofocus: bool = False
    focus: int = -1


@dataclass
class CapturedFrame:
    """一次擷取結果：解碼後的 BGR ndarray + 來源時間戳。"""
    image: np.ndarray
    stamp_sec: int = 0
    stamp_nsec: int = 0


# ---------------------------------------------------------------------------
# UsbCam：相機開啟 / 抓取 / 關閉
# ---------------------------------------------------------------------------
class UsbCam:
    """OpenCV V4L2 後端的相機抓取包裝。

    提供：
      - configure(): 開啟裝置、設定 fourcc、解析度、framerate
      - start() / start_capturing() / stop_capturing(): 串流啟停
      - is_capturing(): 狀態查詢
      - get_image(): 抓一張 BGR frame
      - shutdown(): 釋放資源
    """

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._params: Optional[CameraParameters] = None
        self._is_capturing: bool = False
        self._device_name: str = ""

        # 讀取執行緒：持續把 V4L2 佇列排空，只保留最新一張。
        # 見 _reader_loop() 的說明，這是低延遲的關鍵。
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._latest_seq: int = 0      # 每讀到一張 +1
        self._taken_seq: int = 0       # 上次交給節點層的序號

    # ------------------------------------------------------------------
    # 設定 / 啟停
    # ------------------------------------------------------------------
    def configure(self, params: CameraParameters, io_method: str) -> None:
        """開啟裝置並套用基本格式 / 解析度 / framerate。"""
        if io_method not in VALID_IO_METHODS:
            raise ValueError(f"Unknown IO method: {io_method!r}")

        self._params = params
        self._device_name = params.device_name

        # 1) 確認裝置節點存在（character device check）
        if not os.path.exists(self._device_name):
            raise RuntimeError(
                f"Device path does not exist: {self._device_name}"
            )

        # 2) 開啟 OpenCV VideoCapture（CAP_V4L2 強制走 V4L2 後端）
        cap = cv2.VideoCapture(self._device_name, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open device: {self._device_name}")

        # 3) 設定 fourcc：依 pixel_format 對應到 V4L2 格式碼
        fourcc_str = _PIXEL_FORMAT_TO_FOURCC.get(
            params.pixel_format_name.lower()
        )
        if fourcc_str is None:
            cap.release()
            raise ValueError(
                f"Unsupported pixel_format: {params.pixel_format_name!r}"
            )
        # cv2.VideoWriter_fourcc 只吃單一字元；用 4-tuple 解開
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # 4) 解析度與 framerate（V4L2 driver 可能會調整實際值）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(params.image_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(params.image_height))
        cap.set(cv2.CAP_PROP_FPS, float(params.framerate))
        # 縮短內部 buffer，降低延遲（並非所有 driver 都支援）
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._is_capturing = False

    def start(self) -> None:
        """啟動串流。"""
        self.start_capturing()

    def start_capturing(self) -> None:
        if self._cap is None:
            raise RuntimeError("Camera not configured")
        # OpenCV 在 cv2.VideoCapture 開啟後即進入 streaming，這裡只需設旗標
        self._is_capturing = True
        if self._reader_thread is None or not self._reader_thread.is_alive():
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._reader_loop, name="usb_cam_reader", daemon=True)
            self._reader_thread.start()

    def stop_capturing(self) -> None:
        self._is_capturing = False
        self._reader_stop.set()
        t = self._reader_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._reader_thread = None

    def is_capturing(self) -> bool:
        return self._is_capturing

    def shutdown(self) -> None:
        """釋放相機資源。"""
        self.stop_capturing()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # 抓取
    # ------------------------------------------------------------------
    def _reader_loop(self) -> None:
        """獨立執行緒：不停地讀，只保留最新一張。

        為什麼需要這條執行緒 ——
        ``cap.read()`` 回傳的是 V4L2 driver 佇列裡**最舊**的那一張，而且會阻塞
        等到有 frame 為止。原本的作法是讓 ROS timer 以 1/framerate 的週期呼叫
        read()，也就是「讀取速度 == 相機產出速度」；這種情況下佇列裡一旦因為
        任何抖動堆了幾張，就**永遠排不完** —— 吞吐量看起來完全正常（``topic hz``
        是滿的），但每一張都是好幾張之前的畫面，延遲固定累積且不會自己好。
        ``CAP_PROP_BUFFERSIZE`` 在 OpenCV 的 V4L2 後端是被忽略的，設了沒用。

        改成獨立執行緒全速讀，讀取速度就由相機決定而非由 timer 決定，佇列永遠
        是空的，延遲收斂到一張。發佈端跟不上時丟掉的是**舊**畫面，而不是把延遲
        累積下來。cv2 在 read() 期間會釋放 GIL，不會卡住 ROS executor。
        """
        while not self._reader_stop.is_set():
            cap = self._cap
            if cap is None or not self._is_capturing:
                time.sleep(0.01)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            # OpenCV 沒有暴露 V4L2 的 monotonic timestamp，這裡直接用 wall clock；
            # 若要嚴格同步可改用 linuxpy 取得 v4l2_buffer.timestamp。
            now_ns = time.time_ns()
            with self._lock:
                self._latest = CapturedFrame(
                    image=frame,
                    stamp_sec=now_ns // 1_000_000_000,
                    stamp_nsec=now_ns % 1_000_000_000,
                )
                self._latest_seq += 1

    def get_image(self) -> Optional[CapturedFrame]:
        """取回最新一張影像 + 時間戳。

        沒有新畫面（自上次取用後相機還沒送新的）時回傳 None，上層視為這次
        timer tick 跳過，避免重複發佈同一張。
        """
        if self._cap is None or not self._is_capturing:
            return None
        with self._lock:
            if self._latest is None or self._latest_seq == self._taken_seq:
                return None
            self._taken_seq = self._latest_seq
            return self._latest

    # ------------------------------------------------------------------
    # 控制項：透過 v4l2-ctl 設定單一控制項
    # ------------------------------------------------------------------
    def set_v4l_parameter(self, name: str, value) -> bool:
        return set_v4l_parameter(self._device_name, name, value)

    # 取得實際解析度（driver 可能調整過），給節點層在組訊息時參考。
    def get_image_width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_image_height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
