"""
sim_mission.launch.py — ONE command for the full mission in GAZEBO.

Brings up, in a single command:
  • Gazebo with the exploration arena (random robot spawn) — Zone A = ArUco
    pole id 100, Zone B = ids 101/102 glued on the EAST WALL (DICT_4X4_250,
    0.15 m panels, midpoint (4.9, 2.0));
  • the full shuttle stack (SLAM + Nav2 + on-board zone_detector +
    shuttle_manager), started 15 s later so Gazebo + controllers settle;
  • the precision dock at B: shuttle_manager spawns the team's UNCHANGED
    marker_navigator.py + box_placer.py (full lay-down → walk-back cycle, no
    actual box), adapted to the sim purely via params/remaps:
      - camera:   /camera/image_raw + /camera/camera_info  (sim topics, raw)
      - cmd_vel:  remapped to /mirte_base_controller/cmd_vel_unstamped
      - markers:  0.15 m panels (vs 0.08 printed)
      - approach 0.45 m / walk-back to 0.60 m (the wall is solid — his real
        0.22 m seek would push the chassis into it)

    ros2 launch mirte_workshop sim_mission.launch.py

Cycle per round trip: find A+B → A (arm grab pose) → B standoff 0.8 m →
marker_navigator precise dock between 101/102 → box_placer lay-down →
/arm_placed → walk-back → /robot_backed_up → box_placer return-home →
arm to zero → back to A.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gazebo = PathJoinSubstitution([
        FindPackageShare('mirte_gazebo'), 'launch',
        'gazebo_exploration_arena.launch.py'])
    shuttle = PathJoinSubstitution([
        FindPackageShare('mirte_workshop'), 'launch', 'shuttle.launch.py'])

    return LaunchDescription([
        # 1) Gazebo + arena + robot (random spawn) + ros2_control controllers.
        IncludeLaunchDescription(PythonLaunchDescriptionSource([gazebo])),

        # 2) The shuttle stack, 15 s later (Gazebo + controller spawners settle).
        #    shuttle.launch.py defaults are already the sim values (use_sim_time,
        #    provide_sim_tf, raw /camera/image_raw, cmd_vel_unstamped,
        #    DICT_4X4_250 ids 100/101/102 @ 0.15 m); here we add what differs.
        TimerAction(period=15.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([shuttle]),
                launch_arguments={
                    # sim base controller's odom TF has broken frame ids
                    # ("$(var frame_prefix '')base_link") → publish a clean one.
                    'publish_odom_tf':   'true',
                    # full place cycle at B (dock → lay-down → walk-back → home)
                    'dock_at_b':         'true',
                    'dock_wait_for_box': 'true',
                    # Nav2 standoff before the dock: 0.8 m back so BOTH wall
                    # markers (0.40 m apart + 0.15 m panels) fit the 60° camera.
                    'dock_approach_dist': '0.8',
                    # spawned marker_navigator/box_placer — sim adaptation:
                    'dock_marker_size':   '0.15',
                    'dock_image_topic':   '/camera/image_raw',
                    'dock_info_topic':    '/camera/camera_info',
                    'dock_cmd_vel_topic': '/mirte_base_controller/cmd_vel_unstamped',
                    'dock_approach_m':    '0.45',
                    'dock_seek_dist':     '0.60',
                    # Zone A pole is a physical cylinder — don't park inside it.
                    'approach_dist':      '0.4',
                }.items()),
        ]),
    ])
