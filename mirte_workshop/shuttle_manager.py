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
import os
import signal
import subprocess
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
from std_msgs.msg import Bool
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

        # Precision dock at Zone B.  When dock_at_b is True: on reaching B the
        # shuttle STOPS and LAUNCHES the precision team's marker_navigator.py as a
        # subprocess (so we never touch their code, and it only ever owns /cmd_vel
        # while docking — no tug-of-war during transit).  It waits for the dock-
        # done signal, KILLS the subprocess, and drives back to A.
        #   dock_wait_for_box=False (default): resume on /robot_positioned — the
        #     precise adjust is enough; the box/gripper step is skipped entirely
        #     (their arm/gripper code isn't ready, so we forget it for now).
        #   dock_wait_for_box=True: resume on /robot_backed_up (full box cycle).
        # dock_approach_dist: stop ~0.5 m back (not the 0.1 m close approach) so
        # BOTH B markers stay in the camera FOV for marker_navigator to dock.
        self._dock_at_b        = bool(self.declare_parameter('dock_at_b', True).value)
        self._dock_approach    = float(self.declare_parameter('dock_approach_dist', 0.5).value)
        self._dock_wait_for_box = bool(self.declare_parameter('dock_wait_for_box', False).value)
        # marker_navigator.py lives next to this file in the package (the friend
        # drops it straight into the package dir, run via python3 — not colcon).
        # realpath() resolves the symlink-install link back to the SOURCE dir,
        # where his (untracked) marker_navigator.py actually is.
        _default_nav = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    'marker_navigator.py')
        self._marker_nav_path = str(self.declare_parameter(
            'marker_navigator_path', _default_nav).value)
        self._dock_left   = int(self.declare_parameter('dock_marker_left',  101).value)
        self._dock_right  = int(self.declare_parameter('dock_marker_right', 102).value)
        self._dock_size   = float(self.declare_parameter('dock_marker_size', 0.08).value)
        self._dock_timeout = float(self.declare_parameter('dock_timeout', 90.0).value)
        self._dock_proc = None        # the spawned marker_navigator process
        self._dock_start_ns = 0

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

        # Dock-done signals from the (spawned) marker_navigator:
        #   /robot_positioned → precise adjust done (we resume here when no box)
        #   /robot_backed_up  → full box cycle done (only if dock_wait_for_box)
        self.create_subscription(Bool, '/robot_positioned', self._positioned_cb, 10)
        self.create_subscription(Bool, '/robot_backed_up',  self._backed_up_cb,  10)

        # Visit sequence: A, B, A, B, …
        self._legs = ['A', 'B'] * self._round_trips
        self._leg = 0

        self._state = 'WAIT_SLAM'
        self._searching = False
        self._navigating = False
        self._relocating = False        # driving to a new search vantage
        self._docking = False           # at B, yielded to precision (marker_navigator)
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
            f'shuttle_manager up [build: wander+arm+approach_arg] — '
            f'{self._round_trips} round trips (legs={self._legs}), '
            f'approach_dist={self._approach_dist:.2f} m, cmd_vel="{cmd_topic}".')

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

    def _in_map(self, wx, wy, margin=2):
        """True if (wx,wy) is inside the current /map grid (with a cell `margin`
        from the edge).  Nav2's global costmap is sized to this SLAM map, so a
        goal outside it makes the planner fail ('off the global costmap' /
        worldToMap), which is exactly what stalled the wander — the targets fell
        off the edge of the small mapped area."""
        if self._map_data is None:
            return False
        col = int((wx - self._map_ox) / self._map_res)
        row = int((wy - self._map_oy) / self._map_res)
        return (margin <= col < self._map_w - margin and
                margin <= row < self._map_h - margin)

    def _clear_distance(self, x, y, h, maxd):
        """How far (m) a ray from (x,y) along heading `h` stays clear, up to
        maxd.  Unknown/free cells count as clear; stops at the first obstacle OR
        at the map edge (beyond the map is off the costmap → unreachable)."""
        if self._map_data is None:
            return maxd
        d = 0.0
        while d < maxd:
            d += self._map_res
            col = int((x + d * math.cos(h) - self._map_ox) / self._map_res)
            row = int((y + d * math.sin(h) - self._map_oy) / self._map_res)
            if not (0 <= row < self._map_h and 0 <= col < self._map_w):
                return max(d - self._map_res, 0.0)      # reached the map edge
            if int(self._map_data[row, col]) > OCCUPIED:
                return max(d - self._map_res, 0.0)      # hit an obstacle
        return maxd

    def _relocate_target(self):
        """A new search vantage to wander to.  We only require the TARGET cell to
        be open — Nav2's planner does the actual obstacle avoidance along the way,
        so we do NOT demand a clear straight line-of-sight (that strict check made
        the robot decide it was 'boxed in' and spin in place forever in a
        cluttered arena).  Fan out across headings/distances; if nothing passes,
        fall back to a short hop the MOST-OPEN way we can see.  Returns None only
        when genuinely walled in on all sides."""
        p = self._robot_pose()
        if p is None:
            return None
        x, y, yaw = p
        offsets = [0, 45, -45, 90, -90, 135, -135, 180, 70, -70, 110, -110]
        start = self._relocate_k
        self._relocate_k += 1
        for clearance in (0.35, 0.28):
            for oi in range(len(offsets)):
                h = yaw + math.radians(offsets[(start + oi) % len(offsets)])
                for d in (self._relocate_dist, 1.2, 0.9, 0.6):
                    tx, ty = x + d * math.cos(h), y + d * math.sin(h)
                    # MUST stay on the costmap (the small SLAM map) AND be clear,
                    # or the planner aborts with 'goal off the global costmap'.
                    if self._in_map(tx, ty) and self._has_clearance(tx, ty, clearance):
                        return (tx, ty, h)
        # Last resort: aim the most-open direction and take a short hop (stays on
        # the map by construction — _clear_distance stops at the map edge).  Better
        # to move a little (Nav2 still avoids obstacles) than spin forever.
        best_h, best_clear = None, 0.0
        for deg in range(0, 360, 20):
            h = yaw + math.radians(deg)
            clear = self._clear_distance(x, y, h, 1.5)
            if clear > best_clear:
                best_h, best_clear = h, clear
        if best_h is not None and best_clear >= 0.5:
            d = min(best_clear - 0.3, 1.0)
            tx, ty = x + d * math.cos(best_h), y + d * math.sin(best_h)
            if self._in_map(tx, ty):
                return (tx, ty, best_h)
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
                self.get_logger().info('Both zones found — starting shuttle.')
                if not self._dock_at_b:
                    self._arm_up()                  # arm mimic only when NOT handing
                                                    # the arm to the precision team
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
            if self._docking:
                # marker_navigator (spawned subprocess) owns /cmd_vel and is doing
                # the precise dock.  We just wait for its done-signal (handled in
                # _positioned_cb / _backed_up_cb).  Guard a stall: if the dock
                # process died or never signals, give up and move on.
                done = self._dock_proc is None or self._dock_proc.poll() is not None
                if done:
                    self.get_logger().warn('Dock process exited before signalling — moving on.')
                    self._finish_dock()
                elif (now - self._dock_start_ns) / 1e9 > self._dock_timeout:
                    self.get_logger().warn('Dock timeout — killing dock and moving on.')
                    self._finish_dock()
                else:
                    self.get_logger().info('Docking — marker_navigator adjusting…',
                                           throttle_duration_sec=5.0)
                return
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
            zone = self._legs[self._leg]
            tgt = self._zone_a if zone == 'A' else self._zone_b
            # Stop further back at B (dock_approach) so both B markers stay in the
            # camera FOV for the precise dock; close approach (approach_dist) at A.
            dist = self._dock_approach if (zone == 'B' and self._dock_at_b) else None
            wp = self._approach(tgt, dist)
            if wp is None:
                self.get_logger().warn('No clear line-of-sight standoff yet — retrying.',
                                       throttle_duration_sec=2.0)
                return
            self.get_logger().info(
                f'Leg {self._leg + 1}/{len(self._legs)} → Zone {zone} '
                f'approach ({wp[0]:.2f}, {wp[1]:.2f})')
            self._send_goal(*wp)
            return

        # DONE → idle.

    # ── navigation helpers ───────────────────────────────────────────────────
    def _approach(self, target: PoseStamped, dist=None):
        """A standoff `dist` (default approach_dist) from the target, facing it.
        ALWAYS returns a waypoint once the marker is known (only None if the robot
        pose is unknown), so a leg always starts — Nav2's planner does the obstacle
        avoidance to it.  If the map is available we PREFER a standoff that is
        clear and has line-of-sight to the marker (so a pillar isn't between
        robot and tag), but if none is found we fall back to the plain
        straight-line standoff rather than stalling."""
        r = self._robot_pose()
        if r is None or target is None:
            return None
        rx, ry = r[0], r[1]
        tx, ty = target.pose.position.x, target.pose.position.y
        d = self._approach_dist if dist is None else dist
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
            if zone == 'B' and self._dock_at_b:
                # PRECISION DOCK: stop here and SPAWN the friend's marker_navigator
                # (we don't modify or pre-run it, so it only owns /cmd_vel now, not
                # during transit).  Wait for its done-signal, then kill it and head
                # back to A.  The box/gripper step is skipped (see dock_wait_for_box).
                self.get_logger().info('Zone B reached — launching precise dock.')
                self._docking = True
                self._dock_start_ns = self.get_clock().now().nanoseconds
                self._spawn_dock()
                return
            # Stand-alone arm mimic (only when NOT docking at B):
            # at A curl + close (pick up); at B arm up + open (drop).
            if not self._dock_at_b:
                if zone == 'A':
                    self._arm_box()
                else:
                    self._arm_up()
            self._leg += 1
        else:
            self.get_logger().warn('Goal did not succeed — retrying same leg.')

    # ── precise dock: spawn / kill the friend's marker_navigator ───────────────
    def _spawn_dock(self):
        """Launch marker_navigator.py as a subprocess for the precise B dock."""
        if not os.path.exists(self._marker_nav_path):
            self.get_logger().error(
                f'marker_navigator not found at {self._marker_nav_path} — '
                'skipping dock, returning to A.')
            self._finish_dock()
            return
        cmd = ['python3', self._marker_nav_path, '--ros-args',
               '-p', f'marker_id_left:={self._dock_left}',
               '-p', f'marker_id_right:={self._dock_right}',
               '-p', f'marker_size:={self._dock_size}']
        self.get_logger().info('Spawning precise dock: ' + ' '.join(cmd))
        self._dock_proc = subprocess.Popen(cmd)

    def _kill_dock(self):
        """Stop the marker_navigator subprocess so it releases /cmd_vel."""
        if self._dock_proc is not None and self._dock_proc.poll() is None:
            self._dock_proc.send_signal(signal.SIGINT)
            try:
                self._dock_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._dock_proc.kill()
        self._dock_proc = None
        self._cmd.publish(Twist())     # make sure the base is stopped after handoff

    def _finish_dock(self):
        """Dock done (or aborted): kill marker_navigator, advance past Zone B."""
        self._kill_dock()
        self._docking = False
        self._leg += 1                 # Zone B leg complete → next leg is A

    def _positioned_cb(self, msg: Bool):
        """Precise adjust reached.  No box step → resume straight back to A."""
        if self._docking and not self._dock_wait_for_box and msg.data:
            self.get_logger().info(
                'Precise dock reached (/robot_positioned) — box step skipped, '
                'returning to A.')
            self._finish_dock()

    def _backed_up_cb(self, msg: Bool):
        """Full box cycle done (only used when dock_wait_for_box:=true)."""
        if self._docking and self._dock_wait_for_box and msg.data:
            self.get_logger().info(
                'Precision finished (/robot_backed_up) — resuming shuttle.')
            self._finish_dock()

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
        node._kill_dock()          # don't leave a marker_navigator subprocess running
        node._cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
