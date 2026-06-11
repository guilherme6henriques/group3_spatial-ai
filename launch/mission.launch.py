"""
mission.launch.py — ONE launch for the full mission on the REAL robot.

Brings up the shuttle stack (SLAM + Nav2 + shuttle_manager) with all the
real-robot args baked in (DICT_4X4_250, A=100, B=101/102, 8 cm printed markers,
dock_at_b, the real cmd_vel, compressed camera).  At Zone B shuttle_manager
SPAWNS the precision team's marker_navigator.py (precise dock between 101/102)
and box_placer.py (lay-down → walk-back → return-home; no box is actually
grabbed), then resumes to A on /robot_backed_up.  Their scripts are unchanged —
configured purely via params/remaps.

    # detection ON THE ROBOT (default):
    ros2 launch mirte_workshop mission.launch.py

    # detection OFFLOADED TO THE LAPTOP (run detector.launch.py there):
    ros2 launch mirte_workshop mission.launch.py run_zone_detector:=false

    # if the base ALREADY broadcasts odom→base_link (check with
    # `ros2 run tf2_ros tf2_echo odom base_link` BEFORE launching):
    ros2 launch mirte_workshop mission.launch.py publish_odom_tf:=false
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    run_zone_detector = LaunchConfiguration('run_zone_detector')
    publish_odom_tf   = LaunchConfiguration('publish_odom_tf')
    dock_wait_for_box = LaunchConfiguration('dock_wait_for_box')
    use_depth_scan    = LaunchConfiguration('use_depth_scan')

    shuttle = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'launch', 'shuttle.launch.py'])

    return LaunchDescription([
        # true  = detect A/B on the robot (camera read locally).
        # false = OFFLOAD detection to the laptop (run detector.launch.py there);
        #         the shuttle then uses the laptop's /zone_a_pose + /zone_b_pose.
        DeclareLaunchArgument('run_zone_detector', default_value='true'),
        # This unit's base doesn't broadcast odom→base_link → run odom_to_tf.
        # Set false on a robot whose base already publishes that TF.
        DeclareLaunchArgument('publish_odom_tf', default_value='true'),
        # Full place cycle at B (precise dock → lay-down → walk-back → home),
        # then back to A.  Set false for just the precise adjust then back to A.
        DeclareLaunchArgument('dock_wait_for_box', default_value='true'),
        # Real lidar is back → use it. Set true ONLY on a unit with no lidar
        # (synthesizes /scan from the depth camera; would double-publish otherwise).
        DeclareLaunchArgument('use_depth_scan', default_value='false'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([shuttle]),
            launch_arguments={
                'use_sim_time':      'false',
                'provide_sim_tf':    'false',
                'run_zone_detector': run_zone_detector,
                'publish_odom_tf':   publish_odom_tf,
                'dock_wait_for_box': dock_wait_for_box,
                'use_depth_scan':    use_depth_scan,
                # The real camera's raw color stream is lazy/unreliable; the
                # compressed (JPEG) stream is always there and ~20x lighter on
                # the SBC — robot-side zone_detector must use it.
                'use_compressed':    'true',
                'aruco_dict':        'DICT_4X4_250',
                'zone_a_id':         '100',
                'zone_b_left_id':    '101',
                'zone_b_right_id':   '102',
                'zone_marker_size':  '0.08',
                'dock_at_b':         'true',
                'image_topic':       '/camera/color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'cmd_vel_topic':     '/mirte_base_controller/cmd_vel',
            }.items()),
    ])
