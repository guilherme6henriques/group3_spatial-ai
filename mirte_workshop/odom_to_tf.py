#!/usr/bin/env python3
"""
Converts the /mirte_base_controller/odom Odometry message to a TF transform.

The MecanumDriveController publishes correct position data but broken frame IDs
(unresolved xacro expressions like "$(var frame_prefix '')base_link").
This node hardcodes the correct frame names so SLAM toolbox can look up
odom → base_link.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Odometry, '/mirte_base_controller/odom', self._odom_cb, 10
        )
        self.get_logger().info('odom_to_tf ready: broadcasting odom → base_link')

    def _odom_cb(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self._tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
