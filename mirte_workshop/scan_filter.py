#!/usr/bin/env python3
"""
Filters LaserScan readings below MIN_RANGE (robot chassis self-returns).
Publishes /scan_filtered so SLAM and Nav2 never see near-zero readings
that would mark the robot's own position as a lethal obstacle.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

MIN_RANGE = 0.25  # metres — LIDAR sits at base_link (x=+0.10, y=0), 10 cm
                  # forward of robot centre.  Self-return distances:
                  #   chassis rear : 0.10 + 0.14 = 0.24 m  ← worst case
                  #   left wheels  : 0.18 m
                  #   right wheels : 0.13 m
                  #   chassis front: 0.04 m
                  # 0.25 m clears all self-returns in every direction.


class ScanFilter(Node):
    def __init__(self):
        super().__init__('scan_filter')
        self._pub = self.create_publisher(LaserScan, '/scan_filtered', 10)
        self.create_subscription(LaserScan, '/scan', self._cb, 10)
        self.get_logger().info(f'Filtering scan readings below {MIN_RANGE} m')

    def _cb(self, msg):
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = MIN_RANGE
        out.range_max = msg.range_max
        out.ranges = tuple(
            r if r >= MIN_RANGE else float('inf') for r in msg.ranges
        )
        out.intensities = msg.intensities
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
