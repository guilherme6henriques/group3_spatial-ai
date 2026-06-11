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
from launch.actions import TimerAction, DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    aruco_dict   = LaunchConfiguration('aruco_dict')
    zone_a_id        = LaunchConfiguration('zone_a_id')
    zone_b_left_id   = LaunchConfiguration('zone_b_left_id')
    zone_b_right_id  = LaunchConfiguration('zone_b_right_id')
    zone_marker_size = LaunchConfiguration('zone_marker_size')
    round_trips  = LaunchConfiguration('round_trips')
    approach_dist = LaunchConfiguration('approach_dist')
    dock_at_b    = LaunchConfiguration('dock_at_b')
    dock_approach_dist = LaunchConfiguration('dock_approach_dist')
    dock_wait_for_box = LaunchConfiguration('dock_wait_for_box')
    dock_marker_size   = LaunchConfiguration('dock_marker_size')
    dock_image_topic   = LaunchConfiguration('dock_image_topic')
    dock_info_topic    = LaunchConfiguration('dock_info_topic')
    dock_cmd_vel_topic = LaunchConfiguration('dock_cmd_vel_topic')
    dock_approach_m    = LaunchConfiguration('dock_approach_m')
    dock_seek_dist     = LaunchConfiguration('dock_seek_dist')
    grasp_at_a           = LaunchConfiguration('grasp_at_a')
    grasp_detect_timeout = LaunchConfiguration('grasp_detect_timeout')
    grasp_timeout        = LaunchConfiguration('grasp_timeout')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    image_topic   = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    provide_sim_tf = LaunchConfiguration('provide_sim_tf')
    use_compressed = LaunchConfiguration('use_compressed')
    run_zone_detector = LaunchConfiguration('run_zone_detector')
    publish_odom_tf = LaunchConfiguration('publish_odom_tf')
    use_depth_scan = LaunchConfiguration('use_depth_scan')

    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        # Sim arena now matches the REAL marker scheme: DICT_4X4_250, Zone A
        # pole = id 100, Zone B = ids 101/102 glued on the east wall.  Only the
        # marker size differs (sim panels 0.15 m vs 0.08 m printed).
        DeclareLaunchArgument('aruco_dict',   default_value='DICT_4X4_250'),
        DeclareLaunchArgument('zone_a_id',       default_value='100'),
        # Zone B is the precision pair's TWO markers; /zone_b_pose = midpoint.
        DeclareLaunchArgument('zone_b_left_id',  default_value='101'),
        DeclareLaunchArgument('zone_b_right_id', default_value='102'),
        DeclareLaunchArgument('zone_marker_size', default_value='0.15'),     # real: 0.08 (printed size)
        DeclareLaunchArgument('round_trips',  default_value='3'),
        # Hand off the precise B docking to the precision team (marker_navigator +
        # box_placer): at B the shuttle stops, publishes /start_docking, and waits
        # for /robot_backed_up before the B→A leg.  Set false for the stand-alone
        # single-target shuttle with the arm-carry mimic.
        DeclareLaunchArgument('dock_at_b',          default_value='true'),
        # Standoff (m) for the B leg when docking — stop further back so BOTH B
        # markers stay in the camera FOV for marker_navigator's precise dock.
        DeclareLaunchArgument('dock_approach_dist', default_value='0.5'),
        # true = run the FULL place cycle at B (spawn box_placer too; bridge
        # /robot_positioned→/start_placing; resume on /robot_backed_up — i.e. the
        # lay-down + walk-back).  false = just the precise adjust, then back to A.
        DeclareLaunchArgument('dock_wait_for_box', default_value='false'),
        # Settings handed to the SPAWNED marker_navigator/box_placer (the friend's
        # scripts, configured via params/remaps only).  Defaults = real robot;
        # the sim mission overrides camera/cmd_vel/sizes.
        DeclareLaunchArgument('dock_marker_size',   default_value='0.08'),
        DeclareLaunchArgument('dock_image_topic',   default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('dock_info_topic',    default_value='/camera/color/camera_info'),
        DeclareLaunchArgument('dock_cmd_vel_topic', default_value='/mirte_base_controller/cmd_vel'),
        # < 0 → keep marker_navigator's own defaults (approach 0.40 / seek 0.22).
        DeclareLaunchArgument('dock_approach_m',    default_value='-1.0'),
        DeclareLaunchArgument('dock_seek_dist',     default_value='-1.0'),
        # Handle grasp at A.  The mirte_perception stack (YOLO) runs ON THE
        # LAPTOP (`ros2 launch mirte_perception grasp.launch.py` there); the
        # shuttle only consumes /perception/object_markers + /grasp_handle over
        # the network: on reaching A it waits for a handle detection, calls the
        # service once, and proceeds to B when it returns (success or not).
        DeclareLaunchArgument('grasp_at_a',           default_value='false'),
        DeclareLaunchArgument('grasp_detect_timeout', default_value='30.0'),
        DeclareLaunchArgument('grasp_timeout',        default_value='180.0'),
        # Distance (m) from the marker to the robot CENTRE at the approach
        # standoff.  Front bumper is ~0.20 m ahead of base_link, so 0.1 m puts
        # the robot's front right up against the marker.  Override here instead of
        # editing the source (editing source on the robot blocks `git pull`).
        DeclareLaunchArgument('approach_dist', default_value='0.3'),
        # SIM: the robot body is moved by the URDF's gazebo_planar_move plugin,
        # which listens on /cmd_vel (the ros2_control wheel chain accepts commands
        # but does not actuate the body in gazebo).  REAL robot: the mission
        # launch overrides this with /mirte_base_controller/cmd_vel.
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('image_topic',       default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        # SIM provides odom relay + base_footprint/base_frame static TF.  The
        # REAL robot's own bringup already publishes these (and would conflict),
        # so set provide_sim_tf:=false on hardware.
        DeclareLaunchArgument('provide_sim_tf',    default_value='true'),
        # Real robot: subscribe to the camera's compressed (JPEG) stream instead
        # of raw — ~20x less data to deserialize on the SBC.  Sim publishes raw.
        DeclareLaunchArgument('use_compressed',    default_value='false'),
        # Set false to OFFLOAD zone_detector to the laptop (run detector.launch.py
        # there).  Frees the SBC's DDS bus for SLAM's /tf so localization stops
        # drifting under full mission load.
        DeclareLaunchArgument('run_zone_detector', default_value='true'),
        # Some MIRTE units' base publishes the odom→base_link TF itself; others
        # only publish the /mirte_base_controller/odom TOPIC (no TF, or broken
        # frame ids).  Set true on a unit that lacks the TF → run odom_to_tf to
        # broadcast odom→base_link (without it SLAM can't anchor).  Set false if
        # the base already publishes it (else you'd get a duplicate publisher).
        DeclareLaunchArgument('publish_odom_tf', default_value='false'),
        # On a unit with NO lidar, synthesize /scan from the depth camera
        # (pointcloud_to_laserscan off /camera/depth/points).  Needs
        # ros-humble-pointcloud-to-laserscan.  Forward cone only — weaker than a
        # 360° lidar, but a real scan SLAM/costmaps can use.
        DeclareLaunchArgument('use_depth_scan', default_value='false'),
    ]

    nav_params = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'params', 'exploration_nav2_params.yaml'])
    slam_params = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'params', 'slam_params.yaml'])
    bt_xml = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'trees', 'nav2_minimal_tree.xml'])

    # NOTE: DDS transport/discovery is NOT configured here.  On the real robot we
    # use the MIRTE Fast-DDS *discovery server* (set MIRTE_FASTDDS=true in
    # ~/.mirte_settings.sh, restart mirte-ros; export MIRTE_FASTDDS=true before
    # sourcing in this shell).  That routes every participant through one server
    # on :11811 instead of multicast — the multicast storm (MIRTE_FASTDDS=false,
    # ~24 participants) was starving /scan and map->odom, so SLAM fell seconds
    # behind and the costmaps broke.  The discovery-server env hook manages
    # ROS_DISCOVERY_SERVER + FASTRTPS_DEFAULT_PROFILES_FILE, so we must NOT set a
    # competing Fast-DDS profile here (an earlier UDP-only band-aid was removed).

    sim = {'use_sim_time': use_sim_time}

    return LaunchDescription(args + [

        # No-lidar units: build /scan from the depth camera's point cloud.  Output
        # in base_link as a horizontal slice; scan_filter then makes /scan_filtered.
        Node(package='pointcloud_to_laserscan',
             executable='pointcloud_to_laserscan_node',
             name='pointcloud_to_laserscan', output='screen',
             condition=IfCondition(use_depth_scan),
             remappings=[('cloud_in', '/camera/depth/points'), ('scan', '/scan')],
             parameters=[sim, {'target_frame': 'base_link',
                               'transform_tolerance': 0.1,
                               'min_height': 0.08, 'max_height': 0.50,
                               'angle_min': -1.0, 'angle_max': 1.0,
                               'angle_increment': 0.0087, 'scan_time': 0.1,
                               'range_min': 0.2, 'range_max': 5.0, 'use_inf': True}]),

        Node(package='mirte_workshop', executable='scan_filter.py',
             name='scan_filter', output='screen', parameters=[sim]),

        # Zone detector — marker IDs/dict are params so the same node works in
        # sim (A=0, B=1/2, 4x4_50) and on the robot (A=104, B=101/102, 4x4_250).
        Node(package='mirte_workshop', executable='zone_detector.py',
             name='zone_detector', output='screen',
             condition=IfCondition(run_zone_detector),   # false → run it on the laptop
             parameters=[sim, {'aruco_dict': aruco_dict,
                               'zone_a_id': zone_a_id,
                               'zone_b_left_id': zone_b_left_id,
                               'zone_b_right_id': zone_b_right_id,
                               'zone_marker_size': zone_marker_size,
                               'use_compressed': use_compressed}],
             remappings=[('/camera/image_raw', image_topic),
                         ('/camera/image_raw/compressed',
                          [image_topic, TextSubstitution(text='/compressed')]),
                         ('/camera/camera_info', camera_info_topic)]),

        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
             output='screen', parameters=[sim],
             condition=IfCondition(provide_sim_tf)),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_frame'],
             output='screen', parameters=[sim],
             condition=IfCondition(provide_sim_tf)),

        # NOTE: no odom relay in sim.  The URDF's gazebo_planar_move plugin
        # publishes /odom AND the odom→base_link TF directly; the ros2_control
        # /mirte_base_controller/odom in sim reads all-zeros (the wheel chain
        # doesn't actuate the body), so relaying it into /odom would inject
        # frozen zeros next to planar_move's live odometry.

        # On a unit whose base does NOT broadcast odom→base_link (only the
        # /mirte_base_controller/odom topic), publish that TF so SLAM can anchor.
        # publish_odom_tf:=true enables it.
        Node(package='mirte_workshop', executable='odom_to_tf.py',
             name='odom_to_tf', output='screen', parameters=[sim],
             condition=IfCondition(publish_odom_tf)),

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
            # async (not sync): processes scans in a background thread and drops
            # gracefully under CPU pressure instead of stalling its callback and
            # overflowing the scan message-filter queue ("queue is full"), which
            # was corrupting the map/localisation and making Nav2 goals time out.
            Node(package='slam_toolbox', executable='async_slam_toolbox_node',
                 name='slam_toolbox', output='screen', parameters=[slam_params, sim]),
        ]),

        # COMPOSED Nav2: all five servers run in ONE container, sharing a single
        # DDS participant.  On this robot MIRTE_FASTDDS=false, so ~24 separate
        # participants (mirte-ros + our stack) flood multicast discovery — it took
        # 25 s just to discover planner_server/get_state, and the lifecycle
        # manager's service calls then timed out ("async_send_request failed"),
        # aborting bringup.  In one container the lifecycle get_state/change_state
        # calls are intra-process (no DDS discovery), so they can't time out, and
        # Nav2 contributes 1 participant instead of 5.
        TimerAction(period=35.0, actions=[
            ComposableNodeContainer(
                name='nav2_container', namespace='',
                package='rclcpp_components', executable='component_container_isolated',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
                composable_node_descriptions=[
                    ComposableNode(
                        package='nav2_planner', plugin='nav2_planner::PlannerServer',
                        name='planner_server', parameters=[nav_params, sim]),
                    ComposableNode(
                        package='nav2_controller', plugin='nav2_controller::ControllerServer',
                        name='controller_server', parameters=[nav_params, sim],
                        remappings=[('cmd_vel', cmd_vel_topic)]),
                    ComposableNode(
                        package='nav2_behaviors', plugin='behavior_server::BehaviorServer',
                        name='behavior_server', parameters=[nav_params, sim],
                        remappings=[('cmd_vel', cmd_vel_topic)]),
                    ComposableNode(
                        package='nav2_bt_navigator', plugin='nav2_bt_navigator::BtNavigator',
                        name='bt_navigator', parameters=[nav_params, sim,
                            {'default_nav_to_pose_bt_xml': bt_xml,
                             'default_nav_through_poses_bt_xml': bt_xml}]),
                    ComposableNode(
                        package='nav2_lifecycle_manager',
                        plugin='nav2_lifecycle_manager::LifecycleManager',
                        name='lifecycle_manager_navigation',
                        parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                                     'bond_timeout': 0.0,
                                     'node_names': ['planner_server', 'controller_server',
                                                    'behavior_server', 'bt_navigator']}]),
                ]),
        ]),

        TimerAction(period=50.0, actions=[
            Node(package='mirte_workshop', executable='shuttle_manager.py',
                 name='shuttle_manager', output='screen',
                 parameters=[sim, {'round_trips': round_trips,
                                   'approach_dist': approach_dist,
                                   'dock_at_b': dock_at_b,
                                   'dock_approach_dist': dock_approach_dist,
                                   'dock_wait_for_box': dock_wait_for_box,
                                   'dock_marker_size': dock_marker_size,
                                   'dock_image_topic': dock_image_topic,
                                   'dock_info_topic': dock_info_topic,
                                   'dock_cmd_vel_topic': dock_cmd_vel_topic,
                                   'dock_approach_m': dock_approach_m,
                                   'dock_seek_dist': dock_seek_dist,
                                   'grasp_at_a': grasp_at_a,
                                   'grasp_detect_timeout': grasp_detect_timeout,
                                   'grasp_timeout': grasp_timeout,
                                   'cmd_vel_topic': cmd_vel_topic}]),
        ]),
    ])
