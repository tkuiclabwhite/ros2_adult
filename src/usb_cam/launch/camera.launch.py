"""完整系統 launch：啟動 usb_cam 或 ZED 相機節點及其他系統節點。"""
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction, DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
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

    # 機身 IMU 來源：'xsens'（MTi-620，預設）或 'arduino'（舊的 /dev/ttyTHS1）
    imu_source_arg = DeclareLaunchArgument(
        'imu_source',
        default_value='xsens',
        description="機身 IMU 來源：'xsens' 或 'arduino'"
    )
    imu_source = LaunchConfiguration('imu_source')
    is_xsens = IfCondition(PythonExpression(["'", imu_source, "' == 'xsens'"]))
    is_arduino = IfCondition(PythonExpression(["'", imu_source, "' == 'arduino'"]))

    # Xsens 序列埠：接 AGX 40-pin 排針的 UART（pin 8 TX / pin 10 RX）= /dev/ttyTHS1。
    # 走 SoC 內建 UART，沒有 FTDI latency timer 的緩衝問題，也不會像 ttyUSB 那樣
    # 因插拔順序和三顆 U2D2 搶編號，所以直接寫死裝置路徑即可。
    # 與舊的 Arduino IMU 是同一個埠 —— 兩者本來就由 imu_source 二擇一，不會同時開。
    # 暫時要改回 USB 轉接板測試：
    #   xsens_port:=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A198O4CI-if00-port0
    # （序號 A198O4CI 的第 5 個字是英文字母 O 不是數字 0）
    xsens_port_arg = DeclareLaunchArgument(
        'xsens_port',
        default_value='/dev/ttyTHS1',
        description='Xsens MTi 的序列埠（/dev/ttyTHSx、by-id 或 /dev/ttyUSBx 皆可，symlink 會自動解析成實際裝置）'
    )

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

    # ----------------------------------------------------------------
    # 機身 IMU：xsens_bridge 與 imu_node 都發 /package/sensorpackage，
    # 兩個同時跑會互相蓋掉，所以用 imu_source 二擇一
    # ----------------------------------------------------------------
    # Xsens 官方 driver（~/ros2_ws）。沒裝這個 package 時不要讓整包 launch 掛掉 ——
    # 常見情境是在沒有 IMU 的機器上只跑影像與策略
    try:
        xsens_driver_dir = get_package_share_directory('xsens_mti_ros2_driver')
    except Exception:
        xsens_driver_dir = None
        print("Warning: 找不到 xsens_mti_ros2_driver（沒有 source ~/ros2_ws/install/setup.bash？），不啟動 IMU driver。"
              "xsens_bridge 仍會啟動，但收不到資料會持續發 watchdog 警告。")

    # 不 include 官方的 xsens_mti_node.launch.py：那支沒有 launch argument，
    # 要改 port 只能去動 ros2_ws 裡的 yaml。這裡直接起 Node，以官方 yaml 為底，
    # 只蓋掉連線相關的參數 —— parameters 清單後面的 dict 會覆寫前面 yaml 的同名值
    def _xsens_driver(context):
        # GPIO UART（/dev/ttyTHS1）不是 symlink，這段對它是 no-op。
        # 保留給改回 USB 轉接板時用：driver 直接吃 /dev/serial/by-id/... 的 symlink 會握手失敗
        # （實機 by-id 0/78 次成功，改傳 /dev/ttyUSB0 就連上，原因在 Xsens 函式庫內，未查明），
        # 所以先把 symlink 解析成實際裝置再交給 driver。只在 launch 啟動時解析一次
        requested = LaunchConfiguration('xsens_port').perform(context)
        port = os.path.realpath(requested)
        if not os.path.exists(port):
            print(f"Warning: 找不到 Xsens 序列埠 {requested}，driver 會持續重試；"
                  "確認 UART 已在 jetson-io 啟用（或 USB 已插上）後請重開 launch")
            port = requested
        elif port != requested:
            print(f"[Xsens] {requested} -> {port}")

        return [Node(
            package='xsens_mti_ros2_driver',
            executable='xsens_mti_node',
            name='xsens_mti_node',
            output='screen',
            parameters=[
                os.path.join(xsens_driver_dir, 'param', 'xsens_mti_node.yaml'),
                {
                    # scan_for_devices 必須維持 false：全域掃描會對每個 ttyUSB
                    # 輪流送 Xsens 握手封包，包括接著 Dynamixel 匯流排的 U2D2
                    'scan_for_devices': False,
                    'port': port,
                    # MTi-620 出廠鮑率，實機驗證過（Device: MTi-620-8A1G6, FW 1.6.0）。
                    # 改 GPIO UART 後沿用同一個鮑率，裝置端設定不用動
                    'baudrate': 115200,

                    # 裝置本身的輸出設定（要送哪些項目、幾 Hz）不在這裡，
                    # 在 ~/ros2_ws/src/xsens_mti_ros2_driver/param/xsens_mti_node.yaml：
                    # enable_deviceConfig: true + 精簡的 pub_* + output_data_rate: 50。
                    # 那些設定會被 driver 寫進 IMU，斷電後保留
                },
            ],
            # 握手失敗（exit 255）後 2 秒自動重開：偶發的握手逾時、走路時杜邦線震鬆後
            # 重新接上都會自己接回（ttyTHS1 編號固定，不會像 ttyUSB 重插後跑掉）。
            # 代價：IMU 真的沒接時會每 2 秒印一次 No MTi device found
            respawn=True,
            respawn_delay=2.0,
        )]

    xsens_driver_launch = []
    if xsens_driver_dir is not None:
        xsens_driver_launch = [OpaqueFunction(function=_xsens_driver, condition=is_xsens)]

    # /filter/euler -> /package/sensorpackage（介面與舊的 imu_node 相同）
    #                + /xsens_imu/data（絕對角、角速度、free_acceleration、頻率）
    # sign_* / swap_roll_pitch 是座標對位，第一次上機務必驗證正負號
    xsens_bridge_node = Node(
        package='walking', executable='xsens_bridge', name='xsens_bridge',
        output='screen',
        condition=is_xsens,
        parameters=[{
            'pub_hz': 50.0,
            'swap_roll_pitch': False,
            'sign_roll': 1.0,
            'sign_pitch': 1.0,
            'sign_yaw': 1.0,
            'yaw_mount_offset': 0.0,
        }],
    )

    # 舊的 Arduino IMU，保留給 Xsens 出問題時回退：imu_source:=arduino
    imu_node = Node(
        package='walking', executable='imu_node', name='imu_node',
        output='screen',
        condition=is_arduino,
        parameters=[{'port': '/dev/ttyTHS1'}, {'baud': 115200}]
    )

    # ZED 內建 IMU：與機身 IMU 並存，各自發各自的 topic，不受 imu_source 影響。
    # usb 模式沒有這顆感測器，不啟動
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
    
    actions = camera_nodes + [zed_launch] + xsens_driver_launch + \
              [driver_node, walking_node, motion_node, web_bridge_node,
               xsens_bridge_node, imu_node, zed_imu_node, switch_node] + \
              [image_node, depth_process_node, overlap_node, camera_param_bridge_node,
               web_video, rosbridge_node, http_server, hotspot_api, photo_capture_node]

    ld = LaunchDescription()
    ld.add_action(camera_source_arg)
    ld.add_action(imu_source_arg)
    ld.add_action(xsens_port_arg)
    ld.add_action(GroupAction(actions=actions))
    return ld