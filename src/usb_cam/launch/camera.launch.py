"""完整系統 launch：啟動 usb_cam 或 ZED 相機節點及其他系統節點。"""
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
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
    # 0) 相機來源選擇參數：'usb'（預設，原本的相機）或 'zed'
    # ================================================================
    camera_source_arg = DeclareLaunchArgument(
        'camera_source',
        default_value='zed',
        description="要啟動的相機來源：'usb' 或 'zed'"
    )
    camera_source = LaunchConfiguration('camera_source')
    is_usb = IfCondition(PythonExpression(["'", camera_source, "' == 'usb'"]))
    is_zed = IfCondition(PythonExpression(["'", camera_source, "' == 'zed'"]))

    # ================================================================
    # 1) 相機節點
    # ================================================================
    # 1a) usb_cam（原本的相機，camera_source:=usb 時啟動，也是預設值）
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
            condition=is_usb,
        )
        for camera in CAMERAS
    ]

    # 1b) ZED（camera_source:=zed 時啟動）
    zed_wrapper_dir = get_package_share_directory('zed_wrapper')
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zed_wrapper_dir, 'launch', 'zed_camera.launch.py')
        ),
        launch_arguments={'camera_model': 'zedxm'}.items(),
        condition=is_zed,
    )

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

    # ZED 內建 IMU：usb 模式沒有這顆感測器，不啟動
    zed_imu_node = Node(
        package='walking', executable='zed_imu_node', name='zed_imu_node',
        output='screen',
        condition=is_zed,
    )
    # ================================================================
    # 3) 影像處理 + 網頁
    # ================================================================
    image_node = Node(
        package='imageprocess',
        executable='image',
        name='image_node',
        output='screen',
        # camera_source 決定 zoomin 讀 CameraSet.ini（usb）或 ZedCameraSet.ini（zed）
        parameters=[{'camera_source': camera_source}],
    )

    # 深度處理：只有 ZED 提供深度圖，usb 模式不啟動
    depth_process_node = Node(
        package='imageprocess',
        executable='depth_process_node',
        name='depth_process_node',
        output='screen',
        condition=is_zed,
    )

    # 疊合：依賴深度標籤圖，同樣只在 zed 模式啟動
    overlap_node = Node(
        package='imageprocess',
        executable='overlap_node',
        name='overlap_node',
        output='screen',
        condition=is_zed,
    )

    # ZED 相機參數橋接：usb 模式沿用 usb_cam 既有的 /Camera_Topic 等介面，不啟動本節點
    camera_param_bridge_node = Node(
        package='imageprocess',
        executable='camera_param_bridge_node',
        name='camera_param_bridge_node',
        output='screen',
        condition=is_zed,
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

    # ================================================================
    # 5) 熱點裝置管理：網頁伺服器 + API
    # ================================================================
    http_server = ExecuteProcess(
        cmd=['python3', '/home/iclab/ros2_adult/hurocup_interface/http_server.py'],
        cwd='/home/iclab/ros2_adult/hurocup_interface',
        output='screen',
    )

    hotspot_api = ExecuteProcess(
        cmd=['sudo', 'python3', os.path.join(dir_path, 'hotspot_api.py')],
        output='screen',
    )

    # 拍照：原始畫面的 topic 名稱在 usb / zed 兩種來源下不同，把 camera_source
    # 傳進去讓節點自己決定要訂閱哪一路
    photo_capture_node = ExecuteProcess(
        cmd=['python3', os.path.join(dir_path, 'photo_capture_node.py'),
             '--ros-args', '-p', ['camera_source:=', camera_source]],
        output='screen',
    )
    
    actions = camera_nodes + [zed_launch] + \
              [driver_node, walking_node, motion_node, web_bridge_node, imu_node,
               zed_imu_node, switch_node] + \
              [image_node, depth_process_node, overlap_node, camera_param_bridge_node,
               web_video, rosbridge_node, http_server, hotspot_api, photo_capture_node]

    ld = LaunchDescription()
    ld.add_action(camera_source_arg)
    ld.add_action(GroupAction(actions=actions))
    return ld