from setuptools import find_packages, setup

package_name = 'strategy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iclab',
    maintainer_email='keninhuang920517@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'api = strategy.API:main',
            'ar  = strategy.ar.ar:main',
            'bb  = strategy.bb.bb:main',
            'sp  = strategy.sp.sp:main',
            'obs = strategy.obs.obs:main',
            'sr  = strategy.sr.sr:main',
            'wl  = strategy.wl.wl:main',
            'mar = strategy.mar.mar:main',
            'mar1 = strategy.mar.mar1:main',
            'yolo = strategy.mar.ros2_engine_inference:main',
            'yolo2 = strategy.mar.visualization:main',
            'yolo3 = strategy.mar.viz_optimized:main',
            'mar0919 = strategy.mar.strategy_MAR0919:main',
            'testyolo = strategy.mar.test:main',
            'navigation_node = strategy.mar.navigation_node:main',
            'nav_debug = strategy.mar.nav_debug:main',
            'haha = strategy.mar.haha3:main'
            ],
    },
)
