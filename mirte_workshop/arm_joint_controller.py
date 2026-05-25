#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
import math

class ArmJointController(Node):
    def __init__(self):
        super().__init__('arm_joint_controller')
        
        # Publisher for joint trajectory commands (only 4 joints for the arm)
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/mirte_master_arm_controller/joint_trajectory',
            10)
        
        # Create subscription for movement commands
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/arm_joint_angles',
            self.joint_callback,
            10)
        
        # Only 4 joints controlled by the arm controller
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_joint']
        
        self.get_logger().info("Arm Joint Controller Ready!")
        self.get_logger().info("Available joints: shoulder_pan, shoulder_lift, elbow, wrist")
        self.get_logger().info("Subscribe to /arm_joint_angles with 4 float values (in radians)")
        self.get_logger().info("Note: Gripper is controlled separately via /set_arm_home, /set_arm_front, etc.")

    def joint_callback(self, msg):
        """Receive joint angles and move the arm"""
        if len(msg.data) < 4:
            self.get_logger().warn(f"Expected 4 joint angles, got {len(msg.data)}")
            return
        
        # Only use first 4 values for arm joints
        angles = list(msg.data[:4])
        angles[0] = self.clamp(angles[0], -math.pi/2, math.pi/2)  # shoulder_pan
        angles[1] = self.clamp(angles[1], -math.pi/2, math.pi/2)  # shoulder_lift
        angles[2] = self.clamp(angles[2], -math.pi/2, math.pi/2)  # elbow
        angles[3] = self.clamp(angles[3], -math.pi/2, math.pi/2)  # wrist
        
        self.get_logger().info(
            f"Moving to angles - "
            f"Pan:{angles[0]:.2f}, Lift:{angles[1]:.2f}, "
            f"Elbow:{angles[2]:.2f}, Wrist:{angles[3]:.2f}")
        
        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start.sec = 2  # 2 second movement
        
        trajectory.points.append(point)
        
        self.joint_pub.publish(trajectory)

    def clamp(self, value, min_val, max_val):
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))


def main(args=None):
    rclpy.init(args=args)
    node = ArmJointController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
