import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():

    # ==========================================
    # 0. 設定啟動參數 (Launch Arguments)
    # ==========================================
    # 這裡定義一個開關 'use_imu'，預設值是 'false'
    # 當你之後接上 IMU 時，只需要在終端機輸入:
    # ros2 launch walking walking.launch.py use_imu:=true
    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='false',
        description='Whether to start the IMU node'
    )

    # 取得參數值
    use_imu_config = LaunchConfiguration('use_imu')

    # ==========================================
    # 1. 啟動 IMU Node (受開關控制)
    # ==========================================
    imu_node = Node(
        package='walking',             # 請確認你的 imu_web 放在哪個 package
        executable='imu_web',
        name='imu_node',
        output='screen',
        parameters=[
            {'port': '/dev/ttyACM0'},  # 你的 IMU Port
            {'baud': 115200},
            {'pub_hz': 100.0},
        ],
        # ★ 關鍵：只有當 use_imu 為 true 時，這個節點才會被啟動
        condition=IfCondition(use_imu_config)
    )

    # ==========================================
    # 2. 啟動 Motor Driver (司機)
    # ==========================================
    driver_node = Node(
        package='motor_control',       # 如果還沒分包，請改回 'walking'
        executable='driver_node',
        name='dynamixel_driver',
        output='screen',
        parameters=[
            {'baudrate': 1000000},
        ]
    )

    # ==========================================
    # 3. 啟動 Walking Node (大腦)
    # ==========================================
    walking_node = Node(
        package='walking_strategy',    # 如果還沒分包，請改回 'walking'
        executable='walking_node',
        name='walking_strategy',
        output='screen',
    )

    # ==========================================
    # 4. 啟動 Web Bridge (參數橋接)
    # ==========================================
    web_bridge_node = Node(
        package='walking',
        executable='walking_web_bridge',
        name='walking_web_bridge',
        output='screen',
    )

    # ==========================================
    # 5. 啟動網頁伺服器
    # ==========================================
    web_server = ExecuteProcess(
        cmd=['python3', '-m', 'http.server', '8000'],
        cwd='/home/aa/Desktop/ros2_adult/hurocup_interface', # 確保路徑正確
        output='screen',
    )

    return LaunchDescription([
        use_imu_arg,      # 註冊參數
        imu_node,         # 加入 IMU (會檢查參數)
        driver_node,
        walking_node,
        web_bridge_node,
        web_server,
    ])