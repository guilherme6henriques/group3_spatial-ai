"""
detector.launch.py — run zone_detector on the LAPTOP (offloaded from the robot).

Pairs with the robot's `shuttle.launch.py run_zone_detector:=false`.  The laptop
subscribes to the robot's camera (compressed) + /tf over wifi, detects the ArUco
A/B markers, and publishes /zone_a_pose, /zone_b_pose back to the robot's
shuttle_manager.  Keeps the heavy cv2 work + a DDS participant + the camera
stream OFF the SBC so SLAM's /tf delivery stops being starved.

On the LAPTOP (same wifi + domain as the robot, MIRTE_FASTDDS=false):
    export ROS_DOMAIN_ID=5
    source /opt/ros/humble/setup.bash
    source ~/.../ws/install/setup.bash
    ros2 launch mirte_workshop detector.launch.py
Defaults are the real-robot values (DICT_4X4_250, ids 104/100, 8 cm, compressed,
/camera/color/...); override per the arena if needed.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    aruco_dict        = LaunchConfiguration('aruco_dict')
    zone_a_id         = LaunchConfiguration('zone_a_id')
    zone_b_id         = LaunchConfiguration('zone_b_id')
    zone_marker_size  = LaunchConfiguration('zone_marker_size')
    image_topic       = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    use_compressed    = LaunchConfiguration('use_compressed')

    return LaunchDescription([
        DeclareLaunchArgument('aruco_dict',        default_value='DICT_4X4_250'),
        DeclareLaunchArgument('zone_a_id',         default_value='104'),
        DeclareLaunchArgument('zone_b_id',         default_value='100'),
        DeclareLaunchArgument('zone_marker_size',  default_value='0.08'),
        DeclareLaunchArgument('image_topic',       default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/color/camera_info'),
        DeclareLaunchArgument('use_compressed',    default_value='true'),

        Node(package='mirte_workshop', executable='zone_detector.py',
             name='zone_detector', output='screen',
             parameters=[{'use_sim_time': False,
                          'aruco_dict': aruco_dict,
                          'zone_a_id': zone_a_id,
                          'zone_b_id': zone_b_id,
                          'zone_marker_size': zone_marker_size,
                          'use_compressed': use_compressed}],
             remappings=[('/camera/image_raw', image_topic),
                         ('/camera/image_raw/compressed',
                          [image_topic, TextSubstitution(text='/compressed')]),
                         ('/camera/camera_info', camera_info_topic)]),
    ])
