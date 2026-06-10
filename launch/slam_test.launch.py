"""
slam_test.launch.py — SLAM-only bring-up to test mapping by hand (teleop + rviz).

For a unit with NO lidar: it synthesizes /scan from the depth camera, then runs
the same SLAM pipeline as the mission — but WITHOUT Nav2 or shuttle_manager, so
you can drive with teleop and watch the map build in rviz.

  pointcloud_to_laserscan  (/camera/depth/points → /scan, in base_link)
  scan_filter              (/scan → /scan_filtered)
  odom_to_tf               (odom → base_link)
  async_slam_toolbox       (builds /map, publishes map→odom)

    # on the robot (needs: sudo apt install ros-humble-pointcloud-to-laserscan):
    ros2 launch mirte_workshop slam_test.launch.py

    # second terminal — drive by hand (sudo apt install ros-humble-teleop-twist-keyboard):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \
        --ros-args -r /cmd_vel:=/mirte_base_controller/cmd_vel

    # rviz: Fixed Frame = map ; add a Map display on /map.  As you drive, the
    # "Frame [map] does not exist" error clears and the map fills in.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam_params = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'params', 'slam_params.yaml'])
    sim = {'use_sim_time': False}     # real robot — wall clock

    return LaunchDescription([
        # /scan from the depth camera (no lidar on this unit).
        Node(package='pointcloud_to_laserscan',
             executable='pointcloud_to_laserscan_node',
             name='pointcloud_to_laserscan', output='screen',
             remappings=[('cloud_in', '/camera/depth/points'), ('scan', '/scan')],
             parameters=[sim, {'target_frame': 'base_link',
                               'transform_tolerance': 0.1,
                               'min_height': 0.08, 'max_height': 0.50,
                               'angle_min': -1.0, 'angle_max': 1.0,
                               'angle_increment': 0.0087, 'scan_time': 0.1,
                               'range_min': 0.2, 'range_max': 5.0, 'use_inf': True}]),

        Node(package='mirte_workshop', executable='scan_filter.py',
             name='scan_filter', output='screen', parameters=[sim]),

        Node(package='mirte_workshop', executable='odom_to_tf.py',
             name='odom_to_tf', output='screen', parameters=[sim]),

        Node(package='slam_toolbox', executable='async_slam_toolbox_node',
             name='slam_toolbox', output='screen', parameters=[slam_params, sim]),
    ])
