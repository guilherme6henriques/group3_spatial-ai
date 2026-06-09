"""
mission.launch.py — ONE launch for the full mission on the REAL robot.

Brings up the shuttle stack (SLAM + Nav2 + shuttle_manager) with all the
real-robot args baked in (DICT_4X4_250, A=104, B=101/102, 8 cm markers,
dock_at_b, the real cmd_vel).  The precise B dock is handled by shuttle_manager
itself: when it reaches Zone B it LAUNCHES the precision team's
marker_navigator.py as a subprocess, waits for the dock-done signal
(/robot_positioned), kills it, and drives back to A.  Nothing in the friend's
marker_navigator is changed, and the box/gripper step is skipped for now
(dock_wait_for_box defaults False in shuttle_manager).

    # detection ON THE ROBOT (default):
    ros2 launch mirte_workshop mission.launch.py

    # detection OFFLOADED TO THE LAPTOP (run detector.launch.py there):
    ros2 launch mirte_workshop mission.launch.py run_zone_detector:=false
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    run_zone_detector = LaunchConfiguration('run_zone_detector')

    shuttle = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'launch', 'shuttle.launch.py'])

    return LaunchDescription([
        # true  = detect A/B on the robot (camera read locally).
        # false = OFFLOAD detection to the laptop (run detector.launch.py there);
        #         the shuttle then uses the laptop's /zone_a_pose + /zone_b_pose.
        DeclareLaunchArgument('run_zone_detector', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([shuttle]),
            launch_arguments={
                'use_sim_time':      'false',
                'provide_sim_tf':    'false',
                'run_zone_detector': run_zone_detector,
                'use_compressed':    'false',
                'aruco_dict':        'DICT_4X4_250',
                'zone_a_id':         '104',
                'zone_b_left_id':    '101',
                'zone_b_right_id':   '102',
                'zone_marker_size':  '0.08',
                'dock_at_b':         'true',
                'image_topic':       '/camera/color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'cmd_vel_topic':     '/mirte_base_controller/cmd_vel',
            }.items()),
    ])
