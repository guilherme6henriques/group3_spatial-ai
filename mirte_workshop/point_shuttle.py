#!/usr/bin/env python3
"""
point_shuttle.py — the simplest possible A<->B test, RELATIVE to the start pose.

Drive forward/back along the robot's OWN heading using Nav2.  No camera, no
ArUco, no search.  Validates SLAM localisation + Nav2 + the base in isolation.

IMPORTANT: the SLAM map origin is NOT the robot's start (wheel odom isn't zeroed
at launch, so the robot can start at e.g. map (1.2, 0.66, 43 deg)).  So we do NOT
use absolute map coords — we capture the robot's actual start pose, then place
the goals `forward_a` / `forward_b` metres ahead ALONG ITS HEADING.  Defaults:
forward_a=1.0 (1 m straight ahead), forward_b=0.0 (back to the start point).
That way "forward" is really forward and the robot barely has to rotate on the
outbound leg (rotation is where mecanum localisation drifts).
"""
import math
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
        self._fwd_a = float(self.declare_parameter('forward_a', 1.0).value)
        self._fwd_b = float(self.declare_parameter('forward_b', 0.0).value)
        self._round_trips  = int(self.declare_parameter('round_trips', 3).value)
        self._goal_timeout = float(self.declare_parameter('goal_timeout', 60.0).value)
        self._slam_wait    = float(self.declare_parameter('slam_wait_timeout', 60.0).value)
        cmd_topic = self.declare_parameter(
            'cmd_vel_topic', '/mirte_base_controller/cmd_vel').value

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        self._cmd = self.create_publisher(Twist, cmd_topic, 10)
        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self._legs = []           # built once we know the start pose
        self._leg = 0
        self._state = 'WAIT_SLAM'
        self._navigating = False
        self._goal_handle = None
        self._goal_sent_ns = 0
        self._start_ns = None

        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(
            f'point_shuttle up — forward_a={self._fwd_a} m, forward_b={self._fwd_b} m '
            f'(relative to start heading), {self._round_trips} round trips.')

    def _robot_pose(self):
        """(x, y, yaw) of base_link in map, or None."""
        try:
            tf = self._tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time(),
                                               timeout=Duration(seconds=0.1))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if self._start_ns is None:        # anchor once the clock is valid
            self._start_ns = now
            return

        if self._state == 'WAIT_SLAM':
            p = self._robot_pose()
            if p is not None:
                x0, y0, yaw0 = p
                ca, sa = math.cos(yaw0), math.sin(yaw0)
                ax, ay = x0 + self._fwd_a * ca, y0 + self._fwd_a * sa
                bx, by = x0 + self._fwd_b * ca, y0 + self._fwd_b * sa
                # Orientation along the travel direction so the robot doesn't do
                # an extra spin on arrival: A faces forward (yaw0), B faces back.
                self._legs = [('A', ax, ay, yaw0),
                              ('B', bx, by, yaw0 + math.pi)] * self._round_trips
                self.get_logger().info(
                    f'Start pose map=({x0:.2f}, {y0:.2f}, {math.degrees(yaw0):.0f}°). '
                    f'A={self._fwd_a} m fwd=({ax:.2f}, {ay:.2f}), '
                    f'B={self._fwd_b} m fwd=({bx:.2f}, {by:.2f}). Starting.')
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
                    self._navigating = False
                return
            if self._leg >= len(self._legs):
                self.get_logger().info('Point shuttle complete — all legs done.')
                self._state = 'DONE'
                return
            name, x, y, yaw = self._legs[self._leg]
            self.get_logger().info(
                f'Leg {self._leg + 1}/{len(self._legs)} → {name} ({x:.2f}, {y:.2f})')
            self._send_goal(x, y, yaw)
            return
        # DONE → idle

    def _send_goal(self, x, y, yaw):
        if not self._nav.server_is_ready():
            self._nav.wait_for_server(timeout_sec=2.0)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = rclpy.time.Time().to_msg()   # 0 = use latest transform
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
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
