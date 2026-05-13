"""Camera intrinsics 載入器：解析 ``camera_info_url`` 並回傳 CameraInfo msg。

支援兩種 URL：
  - ``package://<pkg>/<rel/path.yaml>``：透過 ament index 找到對應 share 目錄
  - ``file:///abs/path.yaml``：絕對路徑直接使用
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from sensor_msgs.msg import CameraInfo


def _resolve_url(url: str) -> Optional[Path]:
    """把 ``camera_info_url`` 轉成檔案系統路徑。

    支援格式：
      - ``package://<pkg>/<rel/path.yaml>`` — 透過 ament index 找 share 目錄
      - ``file:///abs/path.yaml``           — 直接使用絕對路徑
      - 空字串                              — 回傳 None
    """
    if not url:
        return None
    if url.startswith("package://"):
        rest = url[len("package://"):]
        pkg, _, rel = rest.partition("/")
        # 為了避免 hard-import 時與 ament_index_python 失效，這裡延遲載入
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory(pkg)
        return Path(share) / rel
    if url.startswith("file://"):
        return Path(url[len("file://"):])
    # fallback: 視為純粹的絕對 / 相對路徑
    return Path(url)


def load_camera_info(url: str, frame_id: str, width: int, height: int) -> CameraInfo:
    """讀取 camera_info YAML 並組成 CameraInfo msg。

    若 URL 為空或檔案讀不到，回傳一個 width/height/frame_id 已填入但 K/D/R/P
    皆為空陣列的 CameraInfo（代表「未校正」）。
    """
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = int(width)
    info.height = int(height)

    path = _resolve_url(url)
    if path is None or not path.exists():
        return info

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return info

    # YAML 內若有 image_width / image_height，優先使用它們
    if "image_width" in data:
        info.width = int(data["image_width"])
    if "image_height" in data:
        info.height = int(data["image_height"])

    info.distortion_model = str(data.get("distortion_model", "plumb_bob"))

    def _matrix_data(key: str) -> list:
        node = data.get(key) or {}
        return [float(x) for x in (node.get("data") or [])]

    info.k = _matrix_data("camera_matrix")
    info.d = _matrix_data("distortion_coefficients")
    info.r = _matrix_data("rectification_matrix")
    info.p = _matrix_data("projection_matrix")
    return info
