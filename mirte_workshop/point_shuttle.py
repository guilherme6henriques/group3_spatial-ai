#!/usr/bin/env python3
"""
point_shuttle.py — the simplest possible A<->B test.

Drive between two FIXED points in the `map` frame, back and forth, using Nav2.
No camera, no ArUco, no search/spin — just navigation.  Use this to validate
SLAM localisation + Nav2 + the base in isolation, BEFORE adding marker detection.

The `map` frame origin is the robot's pose when SLAM starts, so the defaults
A=(1.0, 0.0), B=(0.0, 0.0) mean "drive 1 m straight ahead, then back to start",
repeated `round_trips` times.  Override with params, e.g.:
    ax:=1.5 ay:=0.0 bx:=0.0 by:=0.0 round_trips:=3
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import rclpy.time

import tf2_ros
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

TICK_HZ = 2.0


class PointShuttle(Node):
    def __init__(self):
        super().__init__('point_shuttle')
        self._ax = float(self.declare_parameter('ax', 1.0).value)
        self._ay = float(self.declare_parameter('ay', 0.0).value)
        self._bx = float(self.declare_parameter('bx', 0.0).value)
        self._by = float(self.declare_parameter('by', 0.0).value)
        self._round_trips  = int(self.declare_parameter('round_trips', 3).value)
        self._goal_timeout = float(self.declare_parameter('goal_timeout', 60.0).value)
        self._slam_wait    = float(self.declare_parameter('slam_wait_timeout', 60.0).value)
        cmd_topic = self.declare_parameter(
            'cmd_vel_topic', '/mirte_base_controller/cmd_vel').value

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        self._cmd = self.create_publisher(Twist, cmd_topic, 10)
        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # A, B, A, B, …  (one round trip = go to A, then back to B)
        self._legs = [('A', self._ax, self._ay),
                      ('B', self._bx, self._by)] * self._round_trips
        self._leg = 0
        self._state = 'WAIT_SLAM'
        self._navigating = False
        self._goal_handle = None
        self._goal_sent_ns = 0
        self._start_ns = None

        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(
            f'point_shuttle up — A=({self._ax:.2f},{self._ay:.2f}) '
            f'B=({self._bx:.2f},{self._by:.2f}), {self._round_trips} round trips.')

    def _robot_ready(self) -> bool:
        try:
            self._tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time(),
                                          timeout=Duration(seconds=0.1))
            return True
        except Exception:
            return False

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if self._start_ns is None:        # anchor once the clock is valid
            self._start_ns = now
            return

        if self._state == 'WAIT_SLAM':
            if self._robot_ready():
                self.get_logger().info('SLAM/TF ready — starting point shuttle.')
                self._state = 'RUN'
            elif (now - self._start_ns) / 1e9 > self._slam_wait:
                self.get_logger().error('No map→base_link TF — is SLAM running? Aborting.')
                self._state = 'DONE'
            else:
                self.get_logger().info('Waiting for SLAM (map→base_link)…',
                                       throttle_duration_sec=5.0)
            return

        if self._state == 'RUN':
            if self._navigating:
                if (now - self._goal_sent_ns) / 1e9 > self._goal_timeout:
                    self.get_logger().warn('Goal timeout — cancelling & retrying.')
                    self._cancel()
                    self._navigating = False    # re-send next tick
                return
            if self._leg >= len(self._legs):
                self.get_logger().info('Point shuttle complete — all legs done.')
                self._state = 'DONE'
                return
            name, x, y = self._legs[self._leg]
            self.get_logger().info(
                f'Leg {self._leg + 1}/{len(self._legs)} → {name} ({x:.2f}, {y:.2f})')
            self._send_goal(x, y)
            return
        # DONE → idle

    def _send_goal(self, x, y):
        if not self._nav.server_is_ready():
            self._nav.wait_for_server(timeout_sec=2.0)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = rclpy.time.Time().to_msg()   # 0 = use latest transform
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.w = 1.0
        self._navigating = True
        self._goal_sent_ns = self.get_clock().now().nanoseconds
        self._nav.send_goal_async(goal).add_done_callback(self._goal_accepted)

    def _goal_accepted(self, fut):
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().warn('Goal rejected — retrying.', throttle_duration_sec=2.0)
            self._navigating = False
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._goal_done)

    def _goal_done(self, fut):
        status = fut.result().status if fut.result() else GoalStatus.STATUS_UNKNOWN
        self._navigating = False
        self._goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'✓ Reached {self._legs[self._leg][0]}.')
            self._leg += 1
        else:
            self.get_logger().warn('Goal did not succeed — retrying same leg.')

    def _cancel(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()


def main(args=None):
    rclpy.init(args=args)
    node = PointShuttle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._cmd.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
