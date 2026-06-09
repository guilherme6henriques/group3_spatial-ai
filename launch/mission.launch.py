"""
mission.launch.py — ONE launch for the full composed mission on the REAL robot.

Brings up, in a single command:
  • the shuttle stack  (SLAM + Nav2 + on-robot zone_detector + shuttle_manager),
    with all the real-robot args baked in (DICT_4X4_250, A=104, B=101/102,
    8 cm markers, dock_at_b, the /camera/color/* topics, the real cmd_vel);
  • marker_navigator (precision dock between the two B markers) — gated to idle
    until the shuttle publishes /start_docking, so it never fights for /cmd_vel;
  • box_placer (precision box placement) — OFF by default; run_box_placer:=true.

    ros2 launch mirte_workshop mission.launch.py

If the precision nodes live in a different package / have different executable
names, override them:

    ros2 launch mirte_workshop mission.launch.py \
        precision_pkg:=<pkg> marker_navigator_exec:=marker_navigator.py \
        box_placer_exec:=box_placer.py run_box_placer:=true
"""
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    precision_pkg     = LaunchConfiguration('precision_pkg')
    nav_exec          = LaunchConfiguration('marker_navigator_exec')
    placer_exec       = LaunchConfiguration('box_placer_exec')
    run_box_placer    = LaunchConfiguration('run_box_placer')
    run_zone_detector = LaunchConfiguration('run_zone_detector')

    shuttle = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'launch', 'shuttle.launch.py'])

    return LaunchDescription([
        # ── precision team nodes (their .py live in THIS package on the robot) ──
        DeclareLaunchArgument('precision_pkg',         default_value='mirte_workshop'),
        DeclareLaunchArgument('marker_navigator_exec', default_value='marker_navigator.py'),
        DeclareLaunchArgument('box_placer_exec',       default_value='box_placer.py'),
        DeclareLaunchArgument('run_box_placer',        default_value='false'),
        # true  = detect A/B on the robot (camera read locally).
        # false = OFFLOAD detection to the laptop (run detector.launch.py there);
        #         the shuttle then uses the laptop's /zone_a_pose + /zone_b_pose.
        DeclareLaunchArgument('run_zone_detector',     default_value='true'),

        # 1) the whole shuttle stack, real-robot args baked in.
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

        # 2) precision dock — start after Nav2 is up (t=35 in shuttle.launch.py).
        #    It detects 101/102 continuously but stays IDLE (no driving) until the
        #    shuttle publishes /start_docking at Zone B.
        TimerAction(period=40.0, actions=[
            Node(package=precision_pkg, executable=nav_exec,
                 name='marker_navigator', output='screen',
                 parameters=[{'use_sim_time': False,
                              'marker_id_left':  101,
                              'marker_id_right': 102,
                              'marker_size':     0.08}]),
        ]),

        # 3) box_placer — enable when ready:  run_box_placer:=true
        TimerAction(period=40.0, actions=[
            Node(package=precision_pkg, executable=placer_exec,
                 name='box_placer', output='screen',
                 condition=IfCondition(run_box_placer)),
        ]),
    ])
