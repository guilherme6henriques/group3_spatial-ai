"""
point_shuttle.launch.py — SLAM + Nav2 + a fixed-point A<->B shuttle.

No camera, no zone_detector, no marker search.  This is the isolation test:
does SLAM localisation + Nav2 + the base drive cleanly between two fixed map
points?  Get THIS solid before adding marker detection (shuttle.launch.py).

REAL ROBOT:
    ros2 launch mirte_workshop point_shuttle.launch.py \
        use_sim_time:=false provide_sim_tf:=false \
        cmd_vel_topic:=/mirte_base_controller/cmd_vel \
        forward_a:=1.0 forward_b:=0.0 round_trips:=3
    (forward_a/forward_b are metres AHEAD of the start pose along the robot's
     heading — not absolute map coords — because the SLAM map origin isn't the
     robot's start.)
"""
from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time   = LaunchConfiguration('use_sim_time')
    cmd_vel_topic  = LaunchConfiguration('cmd_vel_topic')
    provide_sim_tf = LaunchConfiguration('provide_sim_tf')

    args = [
        DeclareLaunchArgument('use_sim_time',  default_value='true'),
        DeclareLaunchArgument('cmd_vel_topic',
                              default_value='/mirte_base_controller/cmd_vel_unstamped'),
        DeclareLaunchArgument('provide_sim_tf', default_value='true'),
        DeclareLaunchArgument('forward_a', default_value='1.0'),   # m ahead of start
        DeclareLaunchArgument('forward_b', default_value='0.0'),   # m ahead of start (0 = start)
        DeclareLaunchArgument('round_trips', default_value='3'),
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

        # Sim-only TF/odom (real robot's bringup provides these).
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
             output='screen', parameters=[sim], condition=IfCondition(provide_sim_tf)),
        Node(package='topic_tools', executable='relay',
             arguments=['/mirte_base_controller/odom', '/odom'],
             output='screen', parameters=[sim], condition=IfCondition(provide_sim_tf)),

        # Real-robot ONLY: lidar mount (no camera mount needed — no zone_detector).
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_laser_tf',
             arguments=['--x', '0.1005', '--y', '0.0', '--z', '0.10721',
                        '--roll', '3.14159', '--pitch', '0.0', '--yaw', '-1.5708',
                        '--frame-id', 'base_link', '--child-frame-id', 'laser'],
             output='screen', condition=UnlessCondition(provide_sim_tf)),

        TimerAction(period=5.0, actions=[
            Node(package='slam_toolbox', executable='async_slam_toolbox_node',
                 name='slam_toolbox', output='screen', parameters=[slam_params, sim]),
        ]),

        # Composed Nav2 (one DDS participant; reliable intra-process lifecycle).
        TimerAction(period=30.0, actions=[
            ComposableNodeContainer(
                name='nav2_container', namespace='',
                package='rclcpp_components', executable='component_container_isolated',
                output='screen', parameters=[{'use_sim_time': use_sim_time}],
                composable_node_descriptions=[
                    ComposableNode(package='nav2_planner', plugin='nav2_planner::PlannerServer',
                                   name='planner_server', parameters=[nav_params, sim]),
                    ComposableNode(package='nav2_controller', plugin='nav2_controller::ControllerServer',
                                   name='controller_server', parameters=[nav_params, sim],
                                   remappings=[('cmd_vel', cmd_vel_topic)]),
                    ComposableNode(package='nav2_behaviors', plugin='behavior_server::BehaviorServer',
                                   name='behavior_server', parameters=[nav_params, sim],
                                   remappings=[('cmd_vel', cmd_vel_topic)]),
                    ComposableNode(package='nav2_bt_navigator', plugin='nav2_bt_navigator::BtNavigator',
                                   name='bt_navigator', parameters=[nav_params, sim,
                                       {'default_nav_to_pose_bt_xml': bt_xml,
                                        'default_nav_through_poses_bt_xml': bt_xml}]),
                    ComposableNode(package='nav2_lifecycle_manager',
                                   plugin='nav2_lifecycle_manager::LifecycleManager',
                                   name='lifecycle_manager_navigation',
                                   parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                                                'bond_timeout': 0.0,
                                                'node_names': ['planner_server', 'controller_server',
                                                               'behavior_server', 'bt_navigator']}]),
                ]),
        ]),

        TimerAction(period=45.0, actions=[
            Node(package='mirte_workshop', executable='point_shuttle.py',
                 name='point_shuttle', output='screen',
                 parameters=[sim, {'cmd_vel_topic': cmd_vel_topic,
                                   'forward_a': LaunchConfiguration('forward_a'),
                                   'forward_b': LaunchConfiguration('forward_b'),
                                   'round_trips': LaunchConfiguration('round_trips')}]),
        ]),
    ])
