#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from builtin_interfaces.msg import Duration
import time


class ArmTaskNode(Node):
    def __init__(self):
        super().__init__('arm_task_server')

        cb_group = ReentrantCallbackGroup()

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/mirte_master_arm_controller/joint_trajectory',
            10)

        self.gripper_client = ActionClient(
            self, GripperCommand,
            '/mirte_master_gripper_controller/gripper_cmd',
            callback_group=cb_group)

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_joint'
        ]

        self.positions = {
            'home':      [0.0,   0.0,   0.0,  0.0],
            'front':     [0.0,  -1.57, -1.57,  0.0],
            'package_1': [0.5,  -0.5,   0.5,  0.0],  # tune these angles in Gazebo
            'package_2': [-0.5, -0.5,   0.5,  0.0],  # tune these angles in Gazebo
        }

        self.create_service(Trigger, '/deliver_package_1', self.handle_deliver_package_1, callback_group=cb_group)
        self.create_service(Trigger, '/deliver_package_2', self.handle_deliver_package_2, callback_group=cb_group)

        self.get_logger().info('Arm task services /deliver_package_1 and /deliver_package_2 are ready.')

    def move_arm(self, position_name):
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.positions[position_name]
        point.time_from_start = Duration(sec=3, nanosec=0)

        trajectory.points.append(point)
        self.arm_pub.publish(trajectory)
        self.get_logger().info(f'Moving arm to {position_name}.')
        time.sleep(3)

    def control_gripper(self, open_gripper):
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available.')
            return

        goal = GripperCommand.Goal()
        goal.command.position = -0.6 if open_gripper else 0.5
        goal.command.max_effort = 10.0

        self.get_logger().info('Opening gripper.' if open_gripper else 'Closing gripper.')
        self.gripper_client.send_goal_async(goal)
        time.sleep(2)

    def _deliver(self, package_name):
        self.get_logger().info(f'Starting {package_name} delivery sequence.')
        self.control_gripper(open_gripper=True)
        self.move_arm(package_name)
        self.control_gripper(open_gripper=False)
        self.move_arm('front')
        self.control_gripper(open_gripper=True)
        self.move_arm('home')
        self.get_logger().info(f'{package_name} delivery complete.')

    def handle_deliver_package_1(self, request, response):
        self._deliver('package_1')
        response.success = True
        response.message = 'Package 1 delivered.'
        return response

    def handle_deliver_package_2(self, request, response):
        self._deliver('package_2')
        response.success = True
        response.message = 'Package 2 delivered.'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ArmTaskNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
