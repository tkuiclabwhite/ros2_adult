"""啟動設定資料模型：定義 CameraConfig 並自動產生 topic 重映射。"""
from pathlib import Path
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory
from pydantic import BaseModel, root_validator, validator

# usb_cam 自己的 share 目錄；params_1.yaml 等都安裝在這裡
USB_CAM_DIR = get_package_share_directory('usb_cam')


class CameraConfig(BaseModel):
    """單一相機的啟動設定。

    name        節點名稱，會被自動拿去組 remap（image_raw → <name>/image_raw 等）
    param_path  ROS2 參數檔（YAML）路徑
    remappings  topic 重映射；若沒指定會由 root_validator 自動生成
    namespace   節點 namespace；通常留空
    """
    name: str = 'camera1'
    param_path: Path = Path(USB_CAM_DIR, 'config', 'params_1.yaml')
    remappings: Optional[List]
    namespace: Optional[str]

    @validator('param_path')
    def validate_param_path(cls, value):
        # 嚴格檢查檔案存在，避免 launch 起來才報錯
        if value and not value.exists():
            raise FileNotFoundError(f'Could not find parameter file: {value}')
        return value

    @root_validator
    def validate_root(cls, values):
        name = values.get('name')
        remappings = values.get('remappings')
        if name and not remappings:
            remappings = [
                ('image_raw', f'{name}/image_raw'),
                ('image_raw/compressed', f'{name}/image_compressed'),
                ('camera_info', f'{name}/camera_info'),
            ]
        values['remappings'] = remappings
        return values
