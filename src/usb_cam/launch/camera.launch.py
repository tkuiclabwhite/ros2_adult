import os
import sys
from pathlib import Path
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

# 保留你原本的攝影機路徑設定邏輯
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# 嘗試匯入 CameraConfig，如果檔案移動了可能會報錯，請確保 camera_config.py 在同目錄
try:
    from camera_config import CameraConfig, USB_CAM_DIR
    CAMERAS = [
        CameraConfig(
            name='camera1',
            param_path=Path(USB_CAM_DIR, 'config', 'params_1.yaml')
        )
    ]
except ImportError:
    # 防呆：如果移動到 walking package 後找不到 config，給個預設值以免報錯
    print("Warning: camera_config not found, using default params.")
    CAMERAS = [] 

def generate_launch_description():
    # ==========================================
    # 1. 攝影機節點 (保留你原本的邏輯)
    # ==========================================
    camera_nodes = [
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            output='screen',
            name=camera.name,
            namespace=(camera.namespace or ''),
            # 參數檔轉成字串路徑
            parameters=[str(camera.param_path), {'save_dir': '/home/iclab/ros2_adult/src/usb_cam/config'}],
            remappings=(camera.remappings or []),
        )
        for camera in CAMERAS
    ]

    # ==========================================
    # 2. 走路系統核心 (Driver + Walking + Motion)
    # ==========================================
    # 注意：這裡假設你已經把所有 python 檔都整合進 'walking' package 了
    
    driver_node = Node(
        package='motor_control',
        executable='driver_node',
        name='dynamixel_driver',
        output='screen',
        parameters=[{'baudrate': 1000000}]
    )

    walking_node = Node(
        package='walking',
        executable='walking_node',
        name='walking_strategy',
        output='screen'
    )

    motion_node = Node(
        package='motionpackage',
        executable='motionpackage',
        name='motion_strategy',
        output='screen',
        parameters=[{'location': 'ar'}]
    )
    
    switch_node = Node(
        package='motionpackage',
        executable='switch',
        name='switch_node',
        output='screen'
    )
    
    web_bridge_node = Node(
        package='walking', executable='walking_web_bridge', name='walking_web_bridge',
        output='screen'
    )

    imu_node = Node(
        package='walking', executable='imu_node', name='imu_node',
        output='screen',
        parameters=[{'port': '/dev/ttyTHS1'}, {'baud': 115200}]
    )

    # ==========================================
    # 3. 影像處理與網頁介面
    # ==========================================
    
    image_node = Node(
        package='imageprocess', executable='image', name='image_node',
        output='screen',
    )

    # 負責影像串流到網頁
    web_video = Node(
        package='web_video_server', executable='web_video_server',
        name='web_video_server',
    )

    # 負責網頁按鈕通訊 (Port 9090) - 這是全系統唯一的通訊口
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{'port': 9090, 'address': '0.0.0.0'}],
        output='screen'
    )

    # ==========================================
    # 4. 組合所有動作
    # ==========================================
    # 把所有清單串起來
    actions =  camera_nodes + \
              [driver_node, walking_node, motion_node, web_bridge_node, imu_node] + \
              [image_node, web_video, rosbridge_node]

    ld = LaunchDescription()
    ld.add_action(GroupAction(actions=actions))
    return ld