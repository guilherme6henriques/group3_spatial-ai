"""
shuttle.launch.py — the autonomous stack with the boxes removed.

Brings up scan_filter + zone_detector + SLAM + Nav2 + shuttle_manager, which
finds the Zone A / Zone B markers and Nav2-navigates A↔B for `round_trips`
round trips, avoiding obstacles via the lidar costmap.  No survey, no
box_perception, no delivery.

SIM (default):   ros2 launch mirte_workshop shuttle.launch.py
REAL ROBOT:      run the robot's own bringup first (camera, lidar /scan, base
                 odom + odom→base_link TF, cmd_vel), then:
    ros2 launch mirte_workshop shuttle.launch.py \
        use_sim_time:=false provide_sim_tf:=false \
        aruco_dict:=DICT_4X4_250 zone_a_id:=104 zone_b_id:=100 \
        image_topic:=/camera/color/image_raw camera_info_topic:=/camera/color/camera_info \
        cmd_vel_topic:=/mirte_base_controller/cmd_vel
  provide_sim_tf:=false is REQUIRED on hardware: it skips the sim-only odom relay
  and base_footprint static, and instead publishes the base_link→laser /
  base_link→camera_link mounts the robot's bringup doesn't (see below).
  (check the real topic names with `ros2 topic list`).
"""

from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    aruco_dict   = LaunchConfiguration('aruco_dict')
    zone_a_id    = LaunchConfiguration('zone_a_id')
    zone_b_id    = LaunchConfiguration('zone_b_id')
    zone_marker_size = LaunchConfiguration('zone_marker_size')
    round_trips  = LaunchConfiguration('round_trips')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    image_topic   = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    provide_sim_tf = LaunchConfiguration('provide_sim_tf')

    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('aruco_dict',   default_value='DICT_4X4_50'),   # real: DICT_4X4_250
        DeclareLaunchArgument('zone_a_id',    default_value='0'),             # real: 104
        DeclareLaunchArgument('zone_b_id',    default_value='1'),             # real: 100
        DeclareLaunchArgument('zone_marker_size', default_value='0.20'),     # ← your PRINTED marker side, metres
        DeclareLaunchArgument('round_trips',  default_value='3'),
        DeclareLaunchArgument('cmd_vel_topic',
                              default_value='/mirte_base_controller/cmd_vel_unstamped'),
        DeclareLaunchArgument('image_topic',       default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        # SIM provides odom relay + base_footprint/base_frame static TF.  The
        # REAL robot's own bringup already publishes these (and would conflict),
        # so set provide_sim_tf:=false on hardware.
        DeclareLaunchArgument('provide_sim_tf',    default_value='true'),
    ]

    nav_params = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'params', 'exploration_nav2_params.yaml'])
    slam_params = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'params', 'slam_params.yaml'])
    bt_xml = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'trees', 'nav2_minimal_tree.xml'])

    sim = {'use_sim_time': use_sim_time}

    return LaunchDescription(args + [

        Node(package='mirte_workshop', executable='scan_filter.py',
             name='scan_filter', output='screen', parameters=[sim]),

        # Zone detector — marker IDs/dict are params so the same node works in
        # sim (0/1, 4x4_50) and on the robot (104/100, 4x4_250).
        Node(package='mirte_workshop', executable='zone_detector.py',
             name='zone_detector', output='screen',
             parameters=[sim, {'aruco_dict': aruco_dict,
                               'zone_a_id': zone_a_id,
                               'zone_b_id': zone_b_id,
                               'zone_marker_size': zone_marker_size}],
             remappings=[('/camera/image_raw', image_topic),
                         ('/camera/camera_info', camera_info_topic)]),

        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
             output='screen', parameters=[sim],
             condition=IfCondition(provide_sim_tf)),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_frame'],
             output='screen', parameters=[sim],
             condition=IfCondition(provide_sim_tf)),

        # Sim publishes odom on /mirte_base_controller/odom; the real base
        # already publishes /odom + odom→base_link TF, so this is sim-only.
        Node(package='topic_tools', executable='relay',
             arguments=['/mirte_base_controller/odom', '/odom'],
             output='screen', parameters=[sim],
             condition=IfCondition(provide_sim_tf)),

        # REAL ROBOT ONLY (provide_sim_tf:=false): the robot's minimal bringup
        # publishes only odom→base_link — it does NOT run robot_state_publisher
        # with the URDF, so the lidar (`laser`) and camera (`camera_link`) mount
        # transforms are missing.  Without base_link→laser, SLAM drops every scan
        # ("Message Filter dropping … queue full") and never maps; without
        # base_link→camera_link, zone_detector can't transform marker poses to map.
        # These two statics are composed from mirte_master_description/urdf
        # (frame_base_joint ∘ lidar chain, and ∘ camera_rgb mount), so they match
        # what robot_state_publisher would have produced.  In sim Gazebo already
        # publishes them, hence UnlessCondition(provide_sim_tf).
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_laser_tf',
             arguments=['--x', '0.1005', '--y', '0.0', '--z', '0.10721',
                        '--roll', '3.14159', '--pitch', '0.0', '--yaw', '-1.5708',
                        '--frame-id', 'base_link', '--child-frame-id', 'laser'],
             output='screen', condition=UnlessCondition(provide_sim_tf)),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_camera_tf',
             arguments=['--x', '0.146659', '--y', '-0.0025', '--z', '0.1277',
                        '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
                        '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
             output='screen', condition=UnlessCondition(provide_sim_tf)),

        TimerAction(period=5.0, actions=[
            Node(package='slam_toolbox', executable='sync_slam_toolbox_node',
                 name='slam_toolbox', output='screen', parameters=[slam_params, sim]),
        ]),

        TimerAction(period=35.0, actions=[
            Node(package='nav2_planner', executable='planner_server',
                 name='planner_server', output='screen', parameters=[nav_params, sim]),
            Node(package='nav2_controller', executable='controller_server',
                 name='controller_server', output='screen', parameters=[nav_params, sim],
                 remappings=[('cmd_vel', cmd_vel_topic)]),
            Node(package='nav2_bt_navigator', executable='bt_navigator',
                 name='bt_navigator', output='screen',
                 parameters=[nav_params, sim,
                             {'default_nav_to_pose_bt_xml': bt_xml,
                              'default_nav_through_poses_bt_xml': bt_xml}]),
            Node(package='nav2_behaviors', executable='behavior_server',
                 name='behavior_server', output='screen', parameters=[nav_params, sim],
                 remappings=[('cmd_vel', cmd_vel_topic)]),
            Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                 name='lifecycle_manager_navigation', output='screen',
                 parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                              'bond_timeout': 0.0,
                              'node_names': ['planner_server', 'controller_server',
                                             'behavior_server', 'bt_navigator']}]),
        ]),

        TimerAction(period=50.0, actions=[
            Node(package='mirte_workshop', executable='shuttle_manager.py',
                 name='shuttle_manager', output='screen',
                 parameters=[sim, {'round_trips': round_trips,
                                   'cmd_vel_topic': cmd_vel_topic}]),
        ]),
    ])
