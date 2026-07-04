"""完整系統 launch：啟動 usb_cam 相機節點及其他系統節點。"""
import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction
from launch_ros.actions import Node

# 把本目錄加進 sys.path 以匯入 camera_config.py
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

try:
    from camera_config import CameraConfig, USB_CAM_DIR
    CAMERAS = [
        CameraConfig(
            name='camera1',
            param_path=Path(USB_CAM_DIR, 'config', 'params_1.yaml'),
        )
    ]
except ImportError:
    # 防呆：找不到 camera_config 時不要直接 crash，給空清單以便其他節點仍可啟動
    print("Warning: camera_config not found, using default params.")
    CAMERAS = []


def generate_launch_description():
    # ================================================================
    # 1) 相機節點（usb_cam）
    # ================================================================
    # save_dir 指向 usb_cam/config，作為 CameraSet.ini 的 fallback 寫入位置。
    camera_nodes = [
        Node(
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
        for camera in CAMERAS
    ]

    # ================================================================
    # 2) 走路系統核心：與原 launch 相同
    # ================================================================
    driver_node = Node(
        package='motor_control',
        executable='driver_node',
        name='dynamixel_driver',
        output='screen',
        parameters=[{'baudrate': 1000000}],
    )

    walking_node = Node(
        package='walking',
        executable='walking_node',
        name='walking_strategy',
        output='screen',
    )

    motion_node = Node(
        package='motionpackage',
        executable='motionpackage',
        name='motion_strategy',
        output='screen',
        parameters=[{'location': 'ar'}],
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
    # ================================================================
    # 3) 影像處理 + 網頁
    # ================================================================
    image_node = Node(
        package='imageprocess',
        executable='image',
        name='image_node',
        output='screen',
    )

    web_video = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
    )

    # 網頁按鈕通訊（Port 9090）— 全系統的單一通訊口
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{'port': 9090, 'address': '0.0.0.0'}],
        output='screen',
    )

    # # ================================================================
    # # 4) 手臂 IK
    # # ================================================================
    # arm_ik_node = Node(
    #     package='arm',
    #     executable='arm_ik',
    #     name='arm_ik_node',
    #     output='screen',
    #     parameters=[{'arm_speed': 100}],
    # )

    # ================================================================
    # 5) 熱點裝置管理：網頁伺服器 + API
    # ================================================================
    # 用自訂的 http_server.py 而不是 python3 -m http.server：
    # /strategy.js 要即時讀 src/strategy/strategy/strategy.js 的最新內容
    # （那個檔案會一直變動，複製或 symlink 都會過期/傳輸失真）。
    http_server = ExecuteProcess(
        cmd=['python3', '/home/iclab/ros2_adult/hurocup_interface/http_server.py'],
        cwd='/home/iclab/ros2_adult/hurocup_interface',
        output='screen',
    )

    # sudo：讀 NetworkManager 租約檔、執行 iptables 都需要 root。
    # 對應的 NOPASSWD 規則設定在機器人的 /etc/sudoers.d/hotspot_api。
    hotspot_api = ExecuteProcess(
        cmd=['sudo', 'python3', os.path.join(dir_path, 'hotspot_api.py')],
        output='screen',
    )

    # 拍照節點：訂閱 zoom_in/processed_image，收到 /capture_photo 觸發後存檔
    photo_capture_node = ExecuteProcess(
        cmd=['python3', os.path.join(dir_path, 'photo_capture_node.py')],
        output='screen',
    )

    actions =  camera_nodes + \
              [driver_node, walking_node, motion_node, web_bridge_node, imu_node, switch_node] + \
              [image_node, web_video, rosbridge_node, http_server, hotspot_api, photo_capture_node]



    ld = LaunchDescription()
    ld.add_action(GroupAction(actions=actions))
    return ld
