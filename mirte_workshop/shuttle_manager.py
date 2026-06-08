#!/usr/bin/env python3
"""
shuttle_manager.py — the autonomous stack with the boxes forgotten.

A minimal mission: spin to find the Zone A and Zone B ArUco markers (published
as /zone_a_pose, /zone_b_pose by zone_detector), then NAV2-navigate
A → B → A → B … for `round_trips` round trips.  Nav2 + the lidar costmap do the
obstacle avoidance — this node only sequences goals.

States:  WAIT_SLAM → SEARCH (spin until both zones seen) → SHUTTLE → DONE

It reuses the same primitives as exploration_manager (spin via cmd_vel, a
straight-line standoff approach, NavigateToPose goals) but with none of the
survey / box / delivery logic.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import rclpy.time

import tf2_ros
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import GripperCommand

OCCUPIED = 65   # occupancy-grid cost above which a cell counts as a (tall) obstacle

TICK_HZ        = 2.0
CMD_HZ         = 10.0


class ShuttleManager(Node):
    def __init__(self):
        super().__init__('shuttle_manager')

        self._round_trips   = int(self.declare_parameter('round_trips', 3).value)
        # Distance from the marker to the robot CENTRE (base_link).  The front
        # bumper is ~0.20 m ahead of base_link, so 0.25 m leaves a ~5 cm gap
        # between the robot's front and the marker (the precision team's hand-off
        # point).  NOTE: if the marker sits on a lidar-visible stand it's an
        # obstacle in the costmap, and inflation_radius (0.30) may stop the
        # planner short of 0.25 m — drop inflation if the leg won't plan that close.
        self._approach_dist = float(self.declare_parameter('approach_dist', 0.25).value)
        self._search_w      = float(self.declare_parameter('search_angular', 0.4).value)
        self._goal_timeout  = float(self.declare_parameter('goal_timeout', 60.0).value)
        self._slam_wait     = float(self.declare_parameter('slam_wait_timeout', 60.0).value)
        # Search: spin ~one revolution looking for the markers; if not all found,
        # drive to a fresh vantage and look again (don't spin forever in place).
        self._spin_time     = float(self.declare_parameter('search_spin_time', 17.0).value)
        self._relocate_dist = float(self.declare_parameter('relocate_dist', 1.5).value)
        # Shorter than goal_timeout: a relocate drive that stalls should give up
        # quickly and resume spinning, not sit frozen for the full minute.
        self._relocate_timeout = float(self.declare_parameter('relocate_timeout', 20.0).value)
        cmd_topic           = self.declare_parameter(
            'cmd_vel_topic', '/mirte_base_controller/cmd_vel_unstamped').value

        # Arm choreography (mimics carrying a box A→B).  Angles are
        # [shoulder_pan, shoulder_lift, elbow, wrist] in rad — params so they can
        # be tuned in the field without a rebuild.  Defaults: "up" = arm straight
        # up (compact footprint, empty), "box" = the /set_arm_front reach pose
        # (curled forward as if cradling a box → slightly larger front footprint).
        self._arm_up_angles  = [float(v) for v in self.declare_parameter(
            'arm_up_angles',  [0.0,  1.5,  0.0, 0.0]).value]
        self._arm_box_angles = [float(v) for v in self.declare_parameter(
            'arm_box_angles', [0.0, -1.2, -1.5, 1.4]).value]
        self._grip_open_pos  = float(self.declare_parameter('gripper_open_pos',  -0.6).value)
        self._grip_close_pos = float(self.declare_parameter('gripper_close_pos',  0.5).value)

        self._zone_a: PoseStamped | None = None
        self._zone_b: PoseStamped | None = None

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        self._map_data = None       # SLAM occupancy grid (has walls/pillars, not the short boxes)

        self._cmd = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(PoseStamped, '/zone_a_pose', self._a_cb, 10)
        self.create_subscription(PoseStamped, '/zone_b_pose', self._b_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 10)

        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Arm + gripper: the real robot's controllers consume these directly
        # (the arm_server / gripper_server wrappers publish to the same topics).
        self._arm_pub = self.create_publisher(
            JointTrajectory, '/mirte_master_arm_controller/joint_trajectory', 10)
        self._grip = ActionClient(
            self, GripperCommand, '/mirte_master_gripper_controller/gripper_cmd')

        # Visit sequence: A, B, A, B, …
        self._legs = ['A', 'B'] * self._round_trips
        self._leg = 0

        self._state = 'WAIT_SLAM'
        self._searching = False
        self._navigating = False
        self._relocating = False        # driving to a new search vantage
        self._spin_start_ns = 0
        self._relocate_k = 0
        self._goal_handle = None
        self._goal_sent_ns = 0
        # Anchored on the FIRST tick, not here: with use_sim_time the clock may
        # still read 0 in the constructor (no /clock yet), and a 0 start makes
        # the slam-wait timeout fire instantly once sim time jumps.
        self._start_ns = None

        self.create_timer(1.0 / CMD_HZ, self._cmd_cb)
        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(
            f'shuttle_manager up — {self._round_trips} round trips '
            f'(legs={self._legs}), cmd_vel="{cmd_topic}".')

    # ── callbacks ──────────────────────────────────────────────────────────
    def _a_cb(self, msg): self._zone_a = msg
    def _b_cb(self, msg): self._zone_b = msg

    def _map_cb(self, msg: OccupancyGrid):
        self._map_res = msg.info.resolution
        self._map_ox = msg.info.origin.position.x
        self._map_oy = msg.info.origin.position.y
        self._map_w = msg.info.width
        self._map_h = msg.info.height
        self._map_data = np.array(msg.data, dtype=np.int16).reshape(self._map_h, self._map_w)

    def _has_clearance(self, wx, wy, clearance):
        """True if no obstacle cell within `clearance` m of (wx, wy)."""
        if self._map_data is None:
            return True
        col0 = int((wx - self._map_ox) / self._map_res)
        row0 = int((wy - self._map_oy) / self._map_res)
        rc = int(math.ceil(clearance / self._map_res))
        r2 = (clearance / self._map_res) ** 2
        for dr in range(-rc, rc + 1):
            for dc in range(-rc, rc + 1):
                if dr * dr + dc * dc > r2:
                    continue
                row, col = row0 + dr, col0 + dc
                if 0 <= row < self._map_h and 0 <= col < self._map_w \
                        and int(self._map_data[row, col]) > OCCUPIED:
                    return False
        return True

    def _has_los(self, x1, y1, x2, y2):
        """True if the straight segment (x1,y1)->(x2,y2) crosses no obstacle —
        i.e. nothing stands between the standoff and the tag."""
        if self._map_data is None:
            return True
        n = max(int(math.hypot(x2 - x1, y2 - y1) / (self._map_res * 0.7)), 2)
        for i in range(n + 1):
            t = i / n
            col = int((x1 + t * (x2 - x1) - self._map_ox) / self._map_res)
            row = int((y1 + t * (y2 - y1) - self._map_oy) / self._map_res)
            if 0 <= row < self._map_h and 0 <= col < self._map_w \
                    and int(self._map_data[row, col]) > OCCUPIED:
                return False
        return True

    def _cmd_cb(self):
        if self._searching:
            tw = Twist()
            tw.angular.z = self._search_w
            self._cmd.publish(tw)

    def _robot_xy(self):
        p = self._robot_pose()
        return None if p is None else (p[0], p[1])

    def _robot_pose(self):
        try:
            tf = self._tf_buf.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.1))
            t, q = tf.transform.translation, tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (t.x, t.y, yaw)
        except Exception:
            return None

    def _relocate_target(self):
        """A new search vantage that is actually reachable: fan out across
        headings and accept the first one whose target cell is clear AND has
        clear line-of-sight from the robot — so we never fling a blind goal into
        a wall.  Prefer the full `relocate_dist`, but take a shorter clear hop if
        that's all that's open.  Returns None if boxed in (→ caller keeps
        spinning)."""
        p = self._robot_pose()
        if p is None:
            return None
        x, y, yaw = p
        offsets = [0, 70, -70, 35, -35, 140, -140, 110, -110, 180]
        start = self._relocate_k
        self._relocate_k += 1
        for oi in range(len(offsets)):
            h = yaw + math.radians(offsets[(start + oi) % len(offsets)])
            for d in (self._relocate_dist, 1.0, 0.6):
                tx, ty = x + d * math.cos(h), y + d * math.sin(h)
                if self._has_clearance(tx, ty, 0.35) and self._has_los(x, y, tx, ty):
                    return (tx, ty, h)
        return None

    # ── main FSM ───────────────────────────────────────────────────────────
    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if self._start_ns is None:        # anchor once the clock is valid
            self._start_ns = now
            self.get_logger().info(f'First tick — clock anchored at {now / 1e9:.1f}s.')
            return

        if self._state == 'WAIT_SLAM':
            if self._robot_xy() is not None:
                self.get_logger().info('SLAM/TF ready — searching for zone markers.')
                self._state = 'SEARCH'
                self._searching = True
                self._spin_start_ns = now
            elif (now - self._start_ns) / 1e9 > self._slam_wait:
                self.get_logger().error('No map→base_link TF — is SLAM running? Aborting.')
                self._state = 'DONE'
            else:
                self.get_logger().info('Waiting for SLAM (map→base_link)…',
                                       throttle_duration_sec=5.0)
            return

        if self._state == 'SEARCH':
            # Spin one full revolution looking for BOTH markers; if a sweep ends
            # without both, WANDER to a fresh, reachable vantage and sweep again
            # (a marker the camera can't see from here won't be found by spinning
            # in the same spot forever).  We only leave SEARCH for SHUTTLE when
            # NOT mid-wander — cancelling a Nav2 drive right before leg 1 used to
            # leave Nav2 not ready; letting the short wander settle avoids that.
            have = [z for z, v in (('A', self._zone_a), ('B', self._zone_b)) if v is not None]
            if self._zone_a is not None and self._zone_b is not None and not self._relocating:
                self._searching = False
                self._cmd.publish(Twist())          # stop spinning
                self.get_logger().info('Both zones found — arm up, starting shuttle.')
                self._arm_up()                      # arm straight up as the first leg begins
                self._state = 'SHUTTLE'
                return

            if self._relocating:
                # Driving to a new vantage; _goal_done resumes the spin.  Guard a
                # stalled drive so we don't sit frozen — give up and spin again.
                if (now - self._goal_sent_ns) / 1e9 > self._relocate_timeout:
                    self.get_logger().warn('Wander drive stalled — cancelling & spinning.')
                    self._cancel()
                    self._relocating = False
                    self._searching = True
                    self._spin_start_ns = now
                return

            if (now - self._spin_start_ns) / 1e9 < self._spin_time:
                self._searching = True              # _cmd_cb spins us in place
                self.get_logger().info(f'Spinning to find zones (have {have})…',
                                       throttle_duration_sec=3.0)
                return

            # A full sweep finished without both markers → wander to a new vantage.
            self._searching = False
            self._cmd.publish(Twist())              # stop spinning before driving
            tgt = self._relocate_target()
            if tgt is None:                         # boxed in → just sweep again
                self.get_logger().warn('No clear vantage to wander to — spinning again.',
                                       throttle_duration_sec=3.0)
                self._searching = True
                self._spin_start_ns = now
                return
            self.get_logger().info(
                f'Sweep done (have {have}) — wandering to a new vantage '
                f'({tgt[0]:.2f}, {tgt[1]:.2f}) to look again.')
            self._relocating = True
            self._send_goal(*tgt)
            return

        if self._state == 'SHUTTLE':
            if self._navigating:
                if (now - self._goal_sent_ns) / 1e9 > self._goal_timeout:
                    self.get_logger().warn('Goal timeout — cancelling & retrying.')
                    self._cancel()
                    self._navigating = False   # re-send next tick (don't spin on a wedged cancel)
                return
            if self._leg >= len(self._legs):
                self.get_logger().info('Shuttle complete — all legs done.')
                self._state = 'DONE'
                return
            tgt = self._zone_a if self._legs[self._leg] == 'A' else self._zone_b
            wp = self._approach(tgt)
            if wp is None:
                self.get_logger().warn('No clear line-of-sight standoff yet — retrying.',
                                       throttle_duration_sec=2.0)
                return
            self.get_logger().info(
                f'Leg {self._leg + 1}/{len(self._legs)} → Zone {self._legs[self._leg]} '
                f'approach ({wp[0]:.2f}, {wp[1]:.2f})')
            self._send_goal(*wp)
            return

        # DONE → idle.

    # ── navigation helpers ───────────────────────────────────────────────────
    def _approach(self, target: PoseStamped):
        """A standoff `approach_dist` from the target, facing it.  ALWAYS returns
        a waypoint once the marker is known (only None if the robot pose is
        unknown), so a leg always starts — Nav2's planner does the obstacle
        avoidance to it.  If the map is available we PREFER a standoff that is
        clear and has line-of-sight to the marker (so a pillar isn't between
        robot and tag), but if none is found we fall back to the plain
        straight-line standoff rather than stalling."""
        r = self._robot_pose()
        if r is None or target is None:
            return None
        rx, ry = r[0], r[1]
        tx, ty = target.pose.position.x, target.pose.position.y
        d = self._approach_dist
        base = math.atan2(ry - ty, rx - tx)        # target → robot (dead-front)

        # Always-valid default: straight-line standoff `d` from the marker toward
        # the robot, facing the marker.
        dr = math.hypot(tx - rx, ty - ry)
        if dr <= d + 0.05:
            fallback = (rx, ry, math.atan2(ty - ry, tx - rx))
        else:
            ratio = (dr - d) / dr
            fallback = (rx + ratio * (tx - rx), ry + ratio * (ty - ry),
                        math.atan2(ty - ry, tx - rx))

        if self._map_data is not None:
            # The marker (pole/stand) is itself a lidar obstacle, so check LOS
            # only up to a point `los_margin` short of it — otherwise the target
            # cell is "occupied" and every angle fails.
            los_margin = 0.30
            for da in (0, 20, -20, 40, -40, 60, -60, 80, -80,
                       100, -100, 130, -130, 160, -160, 180):
                ang = base + math.radians(da)
                ax = tx + d * math.cos(ang)
                ay = ty + d * math.sin(ang)
                ex = tx + (ax - tx) * (los_margin / d)     # stop short of the marker
                ey = ty + (ay - ty) * (los_margin / d)
                if self._has_clearance(ax, ay, 0.40) and self._has_los(ax, ay, ex, ey):
                    return (ax, ay, math.atan2(ty - ay, tx - ax))   # face the tag

        return fallback   # no clear-LOS standoff found → go straight at it anyway

    def _send_goal(self, x, y, yaw):
        if not self._nav.server_is_ready():
            self._nav.wait_for_server(timeout_sec=2.0)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        # Stamp 0 = "use the latest transform".  If we stamp with now(), Nav2
        # looks up the robot pose (base_link->map) at that fixed time; while the
        # robot sits on a goal it can't finish, the stamp ages out of the ~26 s
        # TF buffer -> "extrapolation into the past" on every planning attempt ->
        # the leg can never plan.  The goal is a static point in the map, so the
        # exact stamp is meaningless; 0 avoids the aging entirely.
        goal.pose.header.stamp = rclpy.time.Time().to_msg()
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
            # Nav2 not ready yet (common right after activation).  Reset so we
            # actually retry — a rejected RELOCATE goal must clear _relocating
            # and resume the spin, or SEARCH gets stuck waiting on it.
            self.get_logger().warn('Goal rejected — retrying.', throttle_duration_sec=2.0)
            self._navigating = False
            if self._relocating:
                self._relocating = False
                self._searching = True
                self._spin_start_ns = self.get_clock().now().nanoseconds
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._goal_done)

    def _goal_done(self, fut):
        status = fut.result().status if fut.result() else GoalStatus.STATUS_UNKNOWN
        self._navigating = False
        self._goal_handle = None

        if self._relocating:
            # Reached (or gave up on) a new search vantage → spin and look again.
            self._relocating = False
            self._searching = True
            self._spin_start_ns = self.get_clock().now().nanoseconds
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            zone = self._legs[self._leg]
            self.get_logger().info(f'✓ Reached Zone {zone}.')
            # At A: curl the arm + close the gripper (mimic picking up the box),
            # held through the A→B leg.  At B: arm straight up + open (dropped),
            # held through the B→A leg.
            if zone == 'A':
                self._arm_box()
            else:
                self._arm_up()
            self._leg += 1
        else:
            self.get_logger().warn('Goal did not succeed — retrying same leg.')

    # ── arm choreography ──────────────────────────────────────────────────────
    def _arm_traj(self, angles):
        """Send a 4-joint arm pose to the robot's arm controller (2 s move)."""
        t = JointTrajectory()
        t.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                         'elbow_joint', 'wrist_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(a) for a in angles]
        pt.time_from_start = DurationMsg(sec=2)
        t.points.append(pt)
        self._arm_pub.publish(t)

    def _gripper(self, pos):
        """Fire-and-forget gripper command (don't block the FSM waiting)."""
        if self._grip.server_is_ready() or self._grip.wait_for_server(timeout_sec=0.5):
            g = GripperCommand.Goal()
            g.command.position = float(pos)
            g.command.max_effort = 10.0
            self._grip.send_goal_async(g)
        else:
            self.get_logger().warn('Gripper action server not available.',
                                   throttle_duration_sec=5.0)

    def _arm_up(self):
        """Arm straight up + gripper open — empty / box released."""
        self.get_logger().info('Arm → straight up (box released).')
        self._arm_traj(self._arm_up_angles)
        self._gripper(self._grip_open_pos)

    def _arm_box(self):
        """Arm curled forward + gripper closed — mimic holding the box."""
        self.get_logger().info('Arm → box-holding pose (box picked up).')
        self._arm_traj(self._arm_box_angles)
        self._gripper(self._grip_close_pos)

    def _cancel(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()


def main(args=None):
    rclpy.init(args=args)
    node = ShuttleManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
