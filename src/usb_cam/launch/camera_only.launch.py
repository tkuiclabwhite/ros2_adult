# Copyright 2026 usb_cam contributors
"""只啟動相機節點的精簡 launch，方便單獨測試 usb_cam 是否正常。"""
import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from camera_config import CameraConfig, USB_CAM_DIR  # noqa: E402


def generate_launch_description():
    camera = CameraConfig(
        name='camera1',
        param_path=Path(USB_CAM_DIR, 'config', 'params_1.yaml'),
    )
    cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node',
        output='screen',
        name=camera.name,
        namespace=(camera.namespace or ''),
        parameters=[
            str(camera.param_path),
            {'save_dir': '/home/iclab/ros2_adult/src/usb_cam/config'},
        ],
        remappings=(camera.remappings or []),
    )
    return LaunchDescription([cam_node])
