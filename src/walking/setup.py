from setuptools import setup

package_name = 'walking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],   # 這裡是對的，保持不動
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/walking.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iclab',
    maintainer_email='iclab@todo.todo',
    description='Humanoid walking launcher',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # 格式: '執行檔名 = 資料夾名.程式檔名:main'
            'walking_node = walking.walking_node:main', 
            
            # 其他的也要確認有沒有在裡面：
            'walking_web_bridge = walking.walking_web_bridge:main',
            'imu_node = walking.imu_node:main', 
        ],
    },
)
