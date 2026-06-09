from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mirte_workshop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/launch', glob('launch/*.xml')),
        ('share/' + package_name + '/params', glob('params/*.yaml')),
        ('share/' + package_name + '/trees', glob('trees/*.xml')),
        ('share/' + package_name + '/config', glob('config/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mirte',
    maintainer_email='m.wisse@tudelft.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_arm_simple_script.py = mirte_workshop.test_arm_simple_script:main",
            "arm_server.py = mirte_workshop.arm_server:main",
            "arm_task_server.py = mirte_workshop.arm_task_server:main",
            "gripper_server.py = mirte_workshop.gripper_server:main",
            "mirte_keyboard.py = mirte_workshop.mirte_keyboard:main",
            "arm_joint_controller.py = mirte_workshop.arm_joint_controller:main",
            "exploration_manager.py = mirte_workshop.exploration_manager:main",
            "odom_to_tf.py = mirte_workshop.odom_to_tf:main",
            "scan_filter.py = mirte_workshop.scan_filter:main",
            "zone_detector.py = mirte_workshop.zone_detector:main",
            "box_perception.py = mirte_workshop.box_perception:main",
            "point_shuttle.py = mirte_workshop.point_shuttle:main",
            "strafe_shuttle.py = mirte_workshop.strafe_shuttle:main",
            "shuttle_manager.py = mirte_workshop.shuttle_manager:main",
            # Precision-team node (its .py lives in this package on the robot).
            "marker_navigator.py = mirte_workshop.marker_navigator:main",
        ],
    },
)

# Note: the entry_points here have a .py extension. This is unusual.
# I have done this to make running a node consistent with nodes
# from ament_cmake packages.  