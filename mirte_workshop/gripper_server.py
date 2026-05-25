#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from control_msgs.action import GripperCommand


class GripperServiceNode(Node):
    def __init__(self):
        super().__init__('gripper_service_node')

        # Action client to control the gripper
        self._action_client = ActionClient(self, GripperCommand, '/mirte_master_gripper_controller/gripper_cmd')

        # Services to trigger open/close
        self.create_service(Trigger, '/gripper_open', self.handle_gripper_open)
        self.create_service(Trigger, '/gripper_close', self.handle_gripper_close)

        self.get_logger().info('Gripper service node is ready.')

    def send_gripper_goal(self, position):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available.')
            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 10.0

        self.get_logger().info(f'Sending gripper goal: position = {position}, max_effort = {goal_msg.command.max_effort}')
        goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, goal_future)

        if not goal_future.result():
            self.get_logger().error('Gripper goal rejected by action server.')
            return False

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        if not result_future.result():
            self.get_logger().error('Failed to get gripper result.')
            return False

        result = result_future.result().result
        self.get_logger().info(f'Gripper action completed: {result}')
        return True


    def handle_gripper_open(self, request, response):
        self.get_logger().info('Received /gripper_open service request')
        success = self.send_gripper_goal(-0.6)
        response.success = success
        response.message = 'Gripper open command completed.' if success else 'Failed to open gripper.'
        return response

    def handle_gripper_close(self, request, response):
        self.get_logger().info('Received /gripper_close service request')
        success = self.send_gripper_goal(0.5)
        response.success = success
        response.message = 'Gripper close command completed.' if success else 'Failed to close gripper.'
        return response



def main(args=None):
    rclpy.init(args=args)
    node = GripperServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
