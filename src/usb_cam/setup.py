"""ament_python setup for usb_cam."""
import os
from glob import glob

from setuptools import setup

package_name = 'usb_cam'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # ament index 必備
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch 檔（含 camera_config.py 工具模組）
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # config（params_*.yaml、camera_info.yaml、CameraSet.ini）
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.ini')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iclab',
    maintainer_email='5454sky@gmail.com',
    description='Pure-Python USB camera driver (ROS2).',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'usb_cam_node = usb_cam.usb_cam_node:main',
            'show_image = usb_cam.show_image:main',
        ],
    },
)
