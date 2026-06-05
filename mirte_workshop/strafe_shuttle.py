#!/usr/bin/env python3
"""
strafe_shuttle.py — HOLONOMIC test: reach a point by STRAFING, never rotating.

Diagnosis from the point_shuttle runs: this mecanum robot's localization jumps
hard whenever it ROTATES (the scan-matcher mis-corrects during turns), but pure
translation is fine.  Nav2's pure-pursuit controller rotates constantly, so it
thrashes.  A mecanum base can move sideways, so we drive to the goal with
vx/vy and hold heading (wz=0) — no rotation, no localization jumps.

This minimal test uses ONLY the robot's odom->base_link TF (published by the
base) — no SLAM, no Nav2, no costmap.  It just proves the base can reach a
relative point by strafing with a stable pose.  Goals are forward_a / forward_b
metres AHEAD of the start pose along the start heading (so "1 m forward" is
really forward); it strafes there and back, round_trips times.

    ros2 run mirte_workshop strafe_shuttle.py --ros-args \
        -p forward_a:=1.0 -p forward_b:=0.0 -p round_trips:=2 \
        -p cmd_vel_topic:=/mirte_base_controller/cmd_vel

NOTE: open space only — there is NO obstacle avoidance here.  This is a driving-
primitive test; if it works cleanly we layer SLAM + obstacle checking on top.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time

import tf2_ros
from geometry_msgs.msg import Twist

CTRL_HZ = 10.0


class StrafeShuttle(Node):
    def __init__(self):
        super().__init__('strafe_shuttle')
        self._fwd_a = float(self.declare_parameter('forward_a', 1.0).value)
        self._fwd_b = float(self.declare_parameter('forward_b', 0.0).value)
        self._round_trips = int(self.declare_parameter('round_trips', 2).value)
        self._speed = float(self.declare_parameter('speed', 0.12).value)        # m/s strafe
        self._tol   = float(self.declare_parameter('tolerance', 0.12).value)    # m arrival
        self._odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        cmd_topic = self.declare_parameter(
            'cmd_vel_topic', '/mirte_base_controller/cmd_vel').value

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        self._cmd = self.create_publisher(Twist, cmd_topic, 10)

        self._legs = []
        self._leg = 0
        self._state = 'WAIT_TF'
        self._start_ns = None
        self.create_timer(1.0 / CTRL_HZ, self._tick)
        self.get_logger().info(
            f'strafe_shuttle up — forward_a={self._fwd_a} m, forward_b={self._fwd_b} m, '
            f'speed={self._speed} m/s, {self._round_trips} round trips. NO obstacle avoidance.')

    def _pose(self):
        """(x, y, yaw) of base in odom, or None."""
        try:
            tf = self._tf_buf.lookup_transform(self._odom_frame, self._base_frame,
                                               rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if self._start_ns is None:
            self._start_ns = now
            return

        p = self._pose()
        if p is None:
            self.get_logger().warn(f'No {self._odom_frame}→{self._base_frame} TF yet…',
                                   throttle_duration_sec=3.0)
            return

        if self._state == 'WAIT_TF':
            x0, y0, yaw0 = p
            ca, sa = math.cos(yaw0), math.sin(yaw0)
            ax, ay = x0 + self._fwd_a * ca, y0 + self._fwd_a * sa
            bx, by = x0 + self._fwd_b * ca, y0 + self._fwd_b * sa
            self._legs = [('A', ax, ay), ('B', bx, by)] * self._round_trips
            self.get_logger().info(
                f'Start odom=({x0:.2f}, {y0:.2f}, {math.degrees(yaw0):.0f}°). '
                f'A=({ax:.2f}, {ay:.2f}), B=({bx:.2f}, {by:.2f}). Strafing (no rotation).')
            self._state = 'RUN'
            return

        if self._state == 'RUN':
            if self._leg >= len(self._legs):
                self._cmd.publish(Twist())
                self.get_logger().info('Strafe shuttle complete — all legs done.')
                self._state = 'DONE'
                return
            name, gx, gy = self._legs[self._leg]
            x, y, yaw = p
            dx, dy = gx - x, gy - y                 # goal vector in odom
            dist = math.hypot(dx, dy)
            if dist < self._tol:
                self._cmd.publish(Twist())
                self.get_logger().info(f'✓ Reached {name} (err {dist:.2f} m).')
                self._leg += 1
                return
            # Rotate the odom-frame goal vector into the robot/base frame, so we
            # command body-frame strafe velocities and DON'T rotate.
            c, s = math.cos(-yaw), math.sin(-yaw)
            vx_b = c * dx - s * dy
            vy_b = s * dx + c * dy
            n = math.hypot(vx_b, vy_b) or 1.0
            scale = self._speed * min(1.0, dist / 0.4)   # ease in near the goal
            tw = Twist()
            tw.linear.x = scale * vx_b / n
            tw.linear.y = scale * vy_b / n
            tw.angular.z = 0.0                            # hold heading — never rotate
            self._cmd.publish(tw)
            self.get_logger().info(f'→ {name}: dist {dist:.2f} m', throttle_duration_sec=1.0)
            return
        # DONE → idle


def main(args=None):
    rclpy.init(args=args)
    node = StrafeShuttle()
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
