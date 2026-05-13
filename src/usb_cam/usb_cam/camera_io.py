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

    def stop_capturing(self) -> None:
        self._is_capturing = False

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
    def get_image(self) -> Optional[CapturedFrame]:
        """讀一張影像並回傳 BGR 陣列 + 時間戳。

        讀取失敗時回傳 None；上層 (節點層) 應視為這次 timer tick 跳過。
        """
        if self._cap is None or not self._is_capturing:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        # OpenCV 沒有暴露 V4L2 的 monotonic timestamp，這裡直接用 wall clock；
        # 若要嚴格同步可改用 linuxpy 取得 v4l2_buffer.timestamp。
        now_ns = time.time_ns()
        return CapturedFrame(
            image=frame,
            stamp_sec=now_ns // 1_000_000_000,
            stamp_nsec=now_ns % 1_000_000_000,
        )

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
