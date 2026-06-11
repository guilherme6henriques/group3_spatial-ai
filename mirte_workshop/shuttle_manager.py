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
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
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
        # Covers the FULL new cycle: servo dock + lay-down + walk-back + 180°
        # turn-around (box_placer's own wait-for-back failsafe alone is 120 s).
        self._dock_timeout = float(self.declare_parameter('dock_timeout', 240.0).value)
        # Environment overrides for the SPAWNED marker_navigator — passed as ROS
        # params/remaps so the friend's code is never edited.  Defaults = the real
        # robot (matching his hardcoded values); the SIM launch overrides them
        # (raw /camera/image_raw, cmd_vel_unstamped, gentler approach/seek since
        # the sim markers are on a solid wall).  approach/seek < 0 = "don't pass,
        # keep his defaults".
        self._dock_image_topic = str(self.declare_parameter(
            'dock_image_topic', '/camera/color/image_raw').value)
        self._dock_info_topic = str(self.declare_parameter(
            'dock_info_topic', '/camera/color/camera_info').value)
        # marker_navigator PUBLISHES /mirte_base_controller/cmd_vel (hardcoded);
        # if the base listens elsewhere (sim: cmd_vel_unstamped) we remap it.
        self._dock_cmd_vel_topic = str(self.declare_parameter(
            'dock_cmd_vel_topic', '/mirte_base_controller/cmd_vel').value)
        self._dock_approach_m = float(self.declare_parameter('dock_approach_m', -1.0).value)
        self._dock_seek_dist  = float(self.declare_parameter('dock_seek_dist', -1.0).value)
        self._dock_proc = None        # the spawned marker_navigator process
        self._dock_start_ns = 0
        # Full box-place cycle (dock_wait_for_box=True): also spawn the friend's
        # box_placer.py (arm-only, safe to run), and bridge /robot_positioned →
        # /start_placing so it auto-runs its lay-down + (wait) + return-home while
        # marker_navigator drives back.  We DON'T grab a box — it's just the
        # place/walk-back motion to test the merge.  box_placer lives next to this
        # file too (untracked, run via python3).
        _default_box = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    'box_placer.py')
        self._box_placer_path = str(self.declare_parameter(
            'box_placer_path', _default_box).value)
        self._auto_start_placing = bool(
            self.declare_parameter('auto_start_placing', True).value)
        self._box_proc = None         # the spawned box_placer process
        self._start_placing_sent = False

        # Handle grasp at Zone A.  The mirte_perception stack (YOLO
        # perception_node + grasp_node) runs ON THE LAPTOP — started there by
        # hand, like the detector — and exchanges everything over the network:
        # it subscribes the robot's cameras, publishes handle detections on
        # /perception/object_markers, and serves /grasp_handle.  When grasp_at_a
        # is True: on reaching A the shuttle waits for a handle marker, calls
        # /grasp_handle once (the service visual-servos the BASE and runs the
        # arm itself, 30–120 s, so the shuttle stays fully idle meanwhile), and
        # proceeds to B when it returns — success OR failure.  No handle within
        # grasp_detect_timeout → skip and proceed.
        self._grasp_at_a = bool(self.declare_parameter('grasp_at_a', False).value)
        self._grasp_detect_timeout = float(
            self.declare_parameter('grasp_detect_timeout', 30.0).value)
        self._grasp_timeout = float(self.declare_parameter('grasp_timeout', 180.0).value)
        self._grasping = False        # at A, yielded to mirte_perception's grasp
        self._grasp_called = False    # /grasp_handle already fired this visit
        self._grasp_start_ns = 0

        # Arm choreography around the dock (only in dock_at_b mode):
        #  - at A: move the arm to the box-grabbing pose (default = box_placer's
        #    carry pose; set arm_grab_angles to your real grab pose).
        #  - after the lay-down + return (box_placer signals /box_placed): move the
        #    arm to zero/home.  [shoulder_pan, shoulder_lift, elbow, wrist] rad.
        self._arm_grab_angles = [float(v) for v in self.declare_parameter(
            'arm_grab_angles', [0.0, -0.4329, -0.8916, -0.3]).value]
        self._arm_zero_angles = [float(v) for v in self.declare_parameter(
            'arm_zero_angles', [0.0, 0.0, 0.0, 0.0]).value]
        # Per-leg inflation: bigger while carrying (A→B, larger footprint), smaller
        # when empty (B→A).  Applied to BOTH costmaps via their param service.
        self._dynamic_inflation = bool(
            self.declare_parameter('dynamic_inflation', True).value)
        self._inflation_carry = float(self.declare_parameter('inflation_carry', 0.35).value)
        self._inflation_empty = float(self.declare_parameter('inflation_empty', 0.20).value)

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
        #   /robot_positioned → precise adjust done (resume here when no box;
        #                        else bridge to /start_placing to run box_placer)
        #   /robot_backed_up  → full box cycle done (only if dock_wait_for_box)
        self.create_subscription(Bool, '/robot_positioned', self._positioned_cb, 10)
        self.create_subscription(Bool, '/robot_backed_up',  self._backed_up_cb,  10)
        # box_placer's "fully done" signal → arm to zero + clear box_placer.
        self.create_subscription(String, '/box_placed', self._box_placed_cb, 10)
        # NEW marker_navigator contract: after /box_placed it turns the robot
        # 180° and publishes /robot_turned_around — THAT is the full-cycle end
        # (resuming on /robot_backed_up would kill it mid turn-around).  On its
        # timeout failsafe it publishes /navigation_failed → skip & proceed.
        self.create_subscription(Bool, '/robot_turned_around', self._turned_cb, 10)
        self.create_subscription(Bool, '/navigation_failed', self._nav_failed_cb, 10)
        # Auto-trigger box_placer once the dock is reached (its manual trigger).
        self._start_placing_pub = self.create_publisher(Bool, '/start_placing', 10)

        # Handle grasp at A: perception_node publishes handle detections as
        # MarkerArray (ns 'model1_sphere'); the first fresh one triggers ONE
        # async /grasp_handle call (the service itself requires a <3 s-old
        # detection, so call-on-detection is exactly its contract).
        self.create_subscription(MarkerArray, '/perception/object_markers',
                                 self._markers_cb, 10)
        self._grasp_cli = self.create_client(Trigger, '/grasp_handle')

        # Per-leg inflation: raw SetParameters service clients to the two costmaps.
        self._infl_clients = []
        if self._dynamic_inflation:
            for ns in ('global_costmap/global_costmap', 'local_costmap/local_costmap'):
                self._infl_clients.append(
                    self.create_client(SetParameters, f'/{ns}/set_parameters'))

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
            if self._grasping:
                # The laptop's mirte_perception owns the base + arm during the
                # handle grasp.  We wait for the /grasp_handle response (handled
                # in _grasp_done_cb) — or bail out on the timeouts below.
                elapsed = (now - self._grasp_start_ns) / 1e9
                if not self._grasp_called and elapsed > self._grasp_detect_timeout:
                    self.get_logger().warn(
                        f'No handle detected in {self._grasp_detect_timeout:.0f} s — '
                        'skipping grasp, proceeding to B.')
                    self._finish_grasp()
                elif elapsed > self._grasp_timeout:
                    self.get_logger().warn('Grasp timeout — killing grasp stack, proceeding.')
                    self._finish_grasp()
                else:
                    self.get_logger().info(
                        'Grasping at A — ' +
                        ('waiting for /grasp_handle to finish…' if self._grasp_called
                         else 'waiting for a handle detection…'),
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
            # Per-leg inflation: carrying to B (bigger) vs empty to A (smaller).
            if self._dynamic_inflation:
                self._set_inflation(self._inflation_carry if zone == 'B'
                                    else self._inflation_empty)
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
            if zone == 'A' and self._grasp_at_a:
                # HANDLE GRASP: the laptop's mirte_perception stack does the
                # work — we wait for its first handle detection, fire
                # /grasp_handle, and proceed on its response (success or not).
                # grasp_node owns the arm AND base meanwhile, so no canned arm
                # pose here and no leg advance until _finish_grasp().
                self.get_logger().info(
                    'Zone A reached — waiting for a handle detection from the '
                    'laptop perception stack (/perception/object_markers).')
                self._grasping = True
                self._grasp_called = False
                self._grasp_start_ns = self.get_clock().now().nanoseconds
                return
            if self._dock_at_b:
                # Reached A (B is handled above): move the arm to the box-grabbing
                # pose for the carry to B.  box_placer isn't running here (cleared
                # after the previous B), so no arm contention.
                if zone == 'A':
                    self.get_logger().info('At A — arm → box-grab pose.')
                    self._arm_traj(self._arm_grab_angles)
            else:
                # Stand-alone arm mimic: at A curl+close (pick up); at B up+open.
                if zone == 'A':
                    self._arm_box()
                else:
                    self._arm_up()
            self._leg += 1
        else:
            self.get_logger().warn('Goal did not succeed — retrying same leg.')

    def _box_placed_cb(self, msg: String):
        """box_placer finished its full place + return-home → arm to zero/home,
        and clear the box_placer process (its arm work is done)."""
        if not self._dock_wait_for_box:
            return
        self.get_logger().info(
            f'box_placer done ({msg.data}) — arm → zero, clearing box_placer.')
        self._kill_proc(self._box_proc)
        self._box_proc = None
        self._arm_traj(self._arm_zero_angles)

    def _set_inflation(self, radius):
        """Set inflation_layer.inflation_radius on both costmaps (fire-and-forget)."""
        if not self._infl_clients:
            return
        req = SetParameters.Request()
        req.parameters = [Parameter('inflation_layer.inflation_radius',
                                    Parameter.Type.DOUBLE, float(radius)).to_parameter_msg()]
        for c in self._infl_clients:
            if c.service_is_ready():
                c.call_async(req)
        self.get_logger().info(f'Inflation → {radius:.2f} m '
                               f'({"carry" if radius >= self._inflation_carry else "empty"}).')

    # ── precise dock: spawn / kill the friend's marker_navigator + box_placer ──
    def _popen(self, path, extra_args=None):
        """Run a friend's script via python3 (it isn't a colcon executable)."""
        cmd = ['python3', path]
        if extra_args:
            cmd += extra_args
        self.get_logger().info('Spawning: ' + ' '.join(cmd))
        return subprocess.Popen(cmd)

    def _spawn_dock(self):
        """Launch marker_navigator (precise dock) and, for the full cycle, the
        box_placer (lay-down + walk-back).  Both are the friend's UNCHANGED
        scripts, started fresh each B so they begin in their idle state.  The
        environment (sim vs real) is adapted purely via params/remaps."""
        self._kill_dock()                     # clear any leftovers first
        self._start_placing_sent = False
        if not os.path.exists(self._marker_nav_path):
            self.get_logger().error(
                f'marker_navigator not found at {self._marker_nav_path} — '
                'skipping dock, returning to A.')
            self._finish_dock()
            return
        use_sim = bool(self.get_parameter('use_sim_time').value)
        nav_args = ['--ros-args',
                    '-p', f'use_sim_time:={str(use_sim).lower()}',
                    '-p', f'marker_id_left:={self._dock_left}',
                    '-p', f'marker_id_right:={self._dock_right}',
                    '-p', f'marker_size:={self._dock_size}',
                    '-p', f'image_topic:={self._dock_image_topic}',
                    '-p', f'info_topic:={self._dock_info_topic}']
        if self._dock_approach_m > 0.0:
            nav_args += ['-p', f'approach_m:={self._dock_approach_m}']
        if self._dock_seek_dist > 0.0:
            nav_args += ['-p', f'seek_dist_m:={self._dock_seek_dist}']
        if self._dock_cmd_vel_topic != '/mirte_base_controller/cmd_vel':
            # his publisher topic is hardcoded — remap it to where the base listens
            nav_args += ['-r', f'/mirte_base_controller/cmd_vel:={self._dock_cmd_vel_topic}']
        self._dock_proc = self._popen(self._marker_nav_path, nav_args)
        # Full cycle: also run box_placer (arm-only; safe to run alongside).
        if self._dock_wait_for_box:
            if os.path.exists(self._box_placer_path):
                self._box_proc = self._popen(self._box_placer_path, [
                    '--ros-args', '-p', f'use_sim_time:={str(use_sim).lower()}'])
            else:
                self.get_logger().warn(
                    f'box_placer not found at {self._box_placer_path} — '
                    'dock will reach /robot_positioned but never place/back-up.')

    def _kill_proc(self, proc):
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _kill_dock(self):
        """Stop BOTH dock procs (used to clear leftovers before a fresh dock and
        on shutdown)."""
        self._kill_proc(self._dock_proc)
        self._kill_proc(self._box_proc)
        self._dock_proc = None
        self._box_proc = None
        try:
            self._cmd.publish(Twist())  # stop the base after handoff
        except Exception:
            pass                        # context may already be torn down (shutdown)

    def _finish_dock(self):
        """Dock done (or aborted): kill marker_navigator to free the base and
        resume to A.  box_placer (arm-only) is LEFT running so it can finish its
        return-home; it's cleared at the next dock / on shutdown."""
        self._kill_proc(self._dock_proc)
        self._dock_proc = None
        self._cmd.publish(Twist())
        self._docking = False
        self._leg += 1                 # Zone B leg complete → next leg is A

    def _positioned_cb(self, msg: Bool):
        """Precise adjust reached."""
        if not (self._docking and msg.data):
            return
        if not self._dock_wait_for_box:
            # Adjust-only: no box step → straight back to A.
            self.get_logger().info(
                'Precise dock reached (/robot_positioned) — box step skipped, '
                'returning to A.')
            self._finish_dock()
        elif self._auto_start_placing and not self._start_placing_sent:
            # Full cycle: trigger box_placer's lay-down (its manual /start_placing).
            self.get_logger().info(
                'Precise dock reached — triggering box_placer (/start_placing); '
                'will resume on /robot_backed_up.')
            self._start_placing_pub.publish(Bool(data=True))
            self._start_placing_sent = True

    def _backed_up_cb(self, msg: Bool):
        """Walk-back done — but in the NEW marker_navigator flow this fires
        mid-sequence (box_placer still opens/returns home, then the 180°
        turn-around follows).  Just log; /robot_turned_around ends the cycle."""
        if self._docking and self._dock_wait_for_box and msg.data:
            self.get_logger().info(
                'Walk-back done (/robot_backed_up) — waiting for the 180° '
                'turn-around (/robot_turned_around).')

    def _turned_cb(self, msg: Bool):
        """Full cycle done: dock → place → walk-back → 180° turn.  The robot is
        already facing away from B — resume the shuttle to A."""
        if self._docking and self._dock_wait_for_box and msg.data:
            self.get_logger().info(
                'Turn-around done (/robot_turned_around) — resuming shuttle to A.')
            self._finish_dock()

    def _nav_failed_cb(self, msg: Bool):
        """marker_navigator's timeout failsafe — skip the dock and proceed."""
        if self._docking and msg.data:
            self.get_logger().warn(
                'Precision dock failed (/navigation_failed) — skipping, '
                'proceeding with the mission.')
            self._finish_dock()

    # ── handle grasp at A (mirte_perception runs on the LAPTOP) ────────────────
    def _finish_grasp(self):
        """Grasp done / skipped / timed out: proceed with the mission
        (A leg complete → B)."""
        self._grasping = False
        self._grasp_called = False
        try:
            self._cmd.publish(Twist())     # make sure the base is stopped
        except Exception:
            pass
        self._leg += 1                     # Zone A leg complete → next leg is B

    def _markers_cb(self, msg: MarkerArray):
        """First handle detection while grasping → fire /grasp_handle ONCE.
        (grasp_node requires a <3 s-old detection, so call-on-detection is its
        intended trigger.)"""
        if not (self._grasping and not self._grasp_called):
            return
        if not any(m.ns == 'model1_sphere' for m in msg.markers):
            return
        if not self._grasp_cli.service_is_ready():
            return                          # grasp_node still starting — next marker retries
        self.get_logger().info('Handle detected — calling /grasp_handle.')
        self._grasp_called = True
        self._grasp_cli.call_async(Trigger.Request()).add_done_callback(
            self._grasp_done_cb)

    def _grasp_done_cb(self, fut):
        """/grasp_handle returned — proceed with the mission either way."""
        if not self._grasping:
            return                          # already timed out / finished
        try:
            res = fut.result()
            self.get_logger().info(
                f'Grasp finished: success={res.success} ({res.message}) — '
                'proceeding to B.')
        except Exception as e:
            self.get_logger().warn(f'/grasp_handle call failed: {e} — proceeding.')
        self._finish_grasp()

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
