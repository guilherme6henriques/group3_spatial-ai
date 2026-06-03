#!/usr/bin/env python3
"""

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Box-survey mission manager for MIRTE pick-and-place validation.

No world coordinates are hardcoded.  Everything derives from:
  • Detected ArUco poses (Zone A pole, Zone B stand, box markers)
  • The live SLAM occupancy map (free-space planning, box-size ranking)

Mission flow
─────────────
INIT_SPIN   Spin until Zone A pole (ID=0) detected, then proceed.
            Zone B (ID=1) is optional here; zone_detector keeps running.

GOTO_POLE   Navigate to a straight-line approach waypoint a fixed standoff
            before the Zone A pole (computed by _direct_approach).

SURVEY      Execute a set of scan positions computed at runtime from the
            pole position and the current SLAM map — no absolute coordinates.
            At each scan position the robot does a 360° spin so that the depth
            camera sweeps the zone and box_perception.py detects the floor
            boxes (short clusters) from all directions.  Survey visits every
            planned scan position (or ends early once all expected boxes are
            seen).  Boxes missed here are still picked up during delivery.

            Box poses + sizes come from box_perception (depth clustering),
            published on /box_pose/id<k> and /box_size/id<k>; the manager just
            ranks by size, largest first.  No box ArUco markers.

GOTO_BOX    Pick the largest not-yet-delivered detected box (re-ranked live, so
            late-detected boxes are included) and navigate to a straight-line
            approach waypoint a fixed standoff before it (_direct_approach —
            shortest route).  If no undelivered box is known yet, scan in place
            until one appears.

GOTO_B      Deliver the current box to the *detected* Zone B stand (Nav2 plans
            around the pillar field).  If Zone B isn't detected yet, explore
            toward the SLAM free-space estimate / scan until its marker is seen.
            Each successful arrival increments the Zone B delivery count.

DONE        Reached once the Zone B delivery count equals the number of boxes
            (one delivery each).  Save SLAM map.
"""

import math
import os
import subprocess
import threading
from enum import Enum

import numpy as np
import rclpy
import rclpy.executors
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

try:
    import tf2_ros
    _TF2 = True
except ImportError:
    _TF2 = False

try:
    from scipy.ndimage import label as ndlabel
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Paths ─────────────────────────────────────────────────────────────────────
_DEFAULT_MAP_SAVE = os.path.expanduser('~/mirte_maps/default')

# ── Timing ────────────────────────────────────────────────────────────────────
SLAM_WAIT_TIMEOUT = 30.0
GLOBAL_TIMEOUT    = 900.0
TICK_HZ           = 2.0
CMD_VEL_HZ        = 10.0

# ── Spin / search ─────────────────────────────────────────────────────────────
SPIN_RATE_RAD_S   = 0.4
SPIN_DURATION     = 2.0 * math.pi / SPIN_RATE_RAD_S
SPIN_MOVE_DIST    = 0.5
SPIN_MOVE_SPEED   = 0.25
SPIN_MOVE_DUR     = SPIN_MOVE_DIST / SPIN_MOVE_SPEED
# When the path straight ahead is blocked, turn by this much to pick a new
# heading and keep searching (instead of marching into the obstacle).
REORIENT_ANGLE    = math.radians(75.0)
REORIENT_DUR      = REORIENT_ANGLE / SPIN_RATE_RAD_S
# Zone A is guaranteed to exist, so INIT_SPIN never gives up on a cycle count —
# it keeps scanning + relocating until the pole is seen (bounded only by the
# mission-wide GLOBAL_TIMEOUT).  This constant only gates how many consecutive
# blocked headings we tolerate before falling back to scanning in place.
MAX_CONSEC_REORIENTS = 5

# ── Navigation ────────────────────────────────────────────────────────────────
APPROACH_DIST_MIN  = 1.00   # pole / Zone B standoff
BOX_APPROACH_DIST  = 0.55   # closer standoff for boxes (precision team works here)
DWELL_SEC          = 3.0    # hold, facing the box's tag-A front, for the precision team
MAX_CONSEC_ABORTS  = 3
GOAL_TIMEOUT       = 60.0   # pillar-field legs take ~20-45 s; 35 s cut them off mid-traversal

# ── Survey — all params are relative, no world coordinates ────────────────────
# Scan positions are computed from the detected Zone A pole position.
# SURVEY_DEPTHS: distances into the zone (along the pole-facing direction) at
#   which scan strips are generated.  Multiple depths handle zones of varying
#   size without knowing the zone extent in advance.
SURVEY_DEPTHS    = [0.8, 1.6, 2.4, 3.2]   # m into zone from pole (reach the back row)
# SURVEY_LAT_STEP / _MAX: lateral scan grid relative to the pole's y position.
SURVEY_LAT_STEP  = 1.0               # m between adjacent lateral scan positions
SURVEY_LAT_MAX   = 4.0               # m max lateral offset tried in each direction
# SURVEY_MAX_STALE: consecutive spin-positions with no new marker.  Logged for
# visibility only — the survey now always visits every planned scan position
# (boxes missed here are still picked up dynamically during delivery).
SURVEY_MAX_STALE = 2
# MIN_SEP: minimum distance between planned scan positions (avoids clustering).
SURVEY_MIN_SEP   = 1.2               # m

# Box sizes/poses now come from box_perception.py (depth clustering), published
# on /box_size/id<k> and /box_pose/id<k>; the manager just ranks by them.

# ── Zone B ────────────────────────────────────────────────────────────────────
MIN_ZONE_SEP   = 3.0
MIN_FREE_CELLS = 50



def _quat_rotate(qx, qy, qz, qw, v):
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


class MissionState(Enum):
    INIT_SPIN = 'init_spin'
    GOTO_POLE = 'goto_pole'
    SURVEY    = 'survey'
    GOTO_BOX  = 'goto_box'
    GOTO_B    = 'goto_b'
    DONE      = 'done'


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        self.declare_parameter('map_save_path', _DEFAULT_MAP_SAVE)
        # IDs that zone_detector publishes on /box_pose/id<id>
        # Change via ROS param if boxes use different ArUco IDs.
        self.declare_parameter('box_marker_ids', [2, 3, 4, 5, 6])

        self._start_ns: int | None = None

        # ── State ─────────────────────────────────────────────────────────────
        self._state            = MissionState.INIT_SPIN
        self._state_entered_ns: int | None = None
        self._map_saved        = False

        # ── Zone poses ────────────────────────────────────────────────────────
        self._zone_a_pose: PoseStamped | None = None
        self._zone_b_pose: PoseStamped | None = None

        # ── Box poses + sizes (ID → …), published by box_perception ───────────
        self._box_poses: dict[int, PoseStamped] = {}
        self._box_sizes: dict[int, float]       = {}   # footprint extent (m)

        # ── Survey ────────────────────────────────────────────────────────────
        self._survey_wps:           list[tuple] = []   # (x, y, yaw)
        self._survey_wp_idx:        int          = 0
        self._survey_spinning:      bool         = False
        self._survey_spin_start_ns: int | None   = None
        self._survey_prev_found:    frozenset    = frozenset()
        self._survey_stale:         int          = 0

        # ── Box delivery (dynamic; re-ranked as new markers appear) ───────────
        self._box_order:     list[int]  = []     # last computed ranking (logs)
        self._last_ranked:   frozenset  = frozenset()  # set _box_order was ranked from
        self._delivered_ids: set[int]   = set()  # boxes dropped at Zone B
        self._current_box:   int | None = None   # box being handled now
        self._dwelling           = False         # holding in front of a box
        self._dwell_start_ns:    int | None = None
        self._zone_b_visits: int        = 0      # successful Zone B deliveries
        self._delivery_target           = 0      # set once box IDs are known
        self._b_goal_is_delivery        = False  # True = goal is a real B drop
        self._box_search_goal           = False  # True = goal is a rescan move

        # ── SLAM map ──────────────────────────────────────────────────────────
        self._map_data   = None
        self._map_width  = 0
        self._map_height = 0
        self._map_res    = 0.05
        self._map_ox     = 0.0
        self._map_oy     = 0.0

        # ── Spin / search (INIT_SPIN) ─────────────────────────────────────────
        self._spinning           = False
        self._moving_forward     = False
        self._reorienting        = False
        self._spin_start_ns:     int | None = None
        self._move_start_ns:     int | None = None
        self._reorient_start_ns: int | None = None
        self._reorient_count     = 0
        self._search_cycles      = 0

        # ── Navigation ────────────────────────────────────────────────────────
        self._navigating           = False
        self._goal_handle          = None
        self._goal_sent_ns: int    = 0
        self._consecutive_aborts   = 0
        self._nav_backoff_until_ns: int = 0   # no new goals until this time

        cb = ReentrantCallbackGroup()

        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                 callback_group=cb)

        if _TF2:
            self._tf_buf      = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        else:
            self._tf_buf = None

        # ── Publishers / Subscribers ──────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(
            Twist, '/mirte_base_controller/cmd_vel_unstamped', 10)

        self.create_subscription(PoseStamped, '/zone_a_pose', self._zone_a_cb, 10)
        self.create_subscription(PoseStamped, '/zone_b_pose', self._zone_b_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 10)

        box_ids = self.get_parameter('box_marker_ids').value
        self._expected_box_ids = frozenset(box_ids)
        # Mission completes only after one Zone B delivery per expected box.
        self._delivery_target = len(self._expected_box_ids)
        for mid in box_ids:
            self.create_subscription(
                PoseStamped, f'/box_pose/id{mid}',
                lambda msg, m=mid: self._box_cb(msg, m), 10)
            self.create_subscription(
                Float32, f'/box_size/id{mid}',
                lambda msg, m=mid: self._box_size_cb(msg, m), 10)
        self.get_logger().info(f'Listening for box markers: {list(box_ids)}')

        self.create_timer(1.0 / TICK_HZ,    self._tick)
        self.create_timer(1.0 / CMD_VEL_HZ, self._spin_cb)

        self.get_logger().info('Exploration manager started — INIT_SPIN')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _zone_a_cb(self, msg: PoseStamped):
        if self._zone_a_pose is None:
            self.get_logger().info(
                f'Zone A pole detected at '
                f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
        self._zone_a_pose = msg   # approach computed fresh in _tick_goto_pole

    def _zone_b_cb(self, msg: PoseStamped):
        newly_found = self._zone_b_pose is None
        self._zone_b_pose = msg   # approach computed fresh in _tick_goto_b
        if newly_found:
            self.get_logger().info(
                f'Zone B stand detected at '
                f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def _box_cb(self, msg: PoseStamped, marker_id: int):
        newly_found = marker_id not in self._box_poses
        self._box_poses[marker_id] = msg
        if newly_found:
            self.get_logger().info(
                f'Box ID={marker_id} first fix → '
                f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def _box_size_cb(self, msg: Float32, marker_id: int):
        # Footprint extent (m) from box_perception; used to rank largest→smallest.
        self._box_sizes[marker_id] = float(msg.data)

    def _map_cb(self, msg: OccupancyGrid):
        self._map_data   = np.array(msg.data, dtype=np.int8).reshape(
                               msg.info.height, msg.info.width)
        self._map_width  = msg.info.width
        self._map_height = msg.info.height
        self._map_res    = msg.info.resolution
        self._map_ox     = msg.info.origin.position.x
        self._map_oy     = msg.info.origin.position.y

    # ── Spin ──────────────────────────────────────────────────────────────────

    def _spin_cb(self):
        tw = Twist()
        if self._spinning or self._reorienting:
            tw.angular.z = SPIN_RATE_RAD_S
        elif self._moving_forward:
            tw.linear.x = SPIN_MOVE_SPEED
        else:
            return
        self._cmd_vel_pub.publish(tw)

    def _stop_spin(self):
        self._spinning       = False
        self._moving_forward = False
        self._reorienting    = False
        self._cmd_vel_pub.publish(Twist())

    # ── Map helpers ───────────────────────────────────────────────────────────

    def _has_clearance(self, x, y, clearance: float = 0.45) -> bool:
        if self._map_data is None:
            return True
        col0 = int((x - self._map_ox) / self._map_res)
        row0 = int((y - self._map_oy) / self._map_res)
        r_cells = int(math.ceil(clearance / self._map_res))
        r2 = (clearance / self._map_res) ** 2
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc > r2:
                    continue
                row, col = row0 + dr, col0 + dc
                if not (0 <= row < self._map_height and 0 <= col < self._map_width):
                    continue
                if int(self._map_data[row, col]) > 65:
                    return False
        return True

    def _robot_xy(self) -> tuple | None:
        """Return (x, y) of the robot in map frame, or None on TF failure."""
        if not _TF2 or self._tf_buf is None:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

    def _robot_pose(self) -> tuple | None:
        """Return (x, y, yaw) of the robot in map frame, or None on TF failure."""
        if not _TF2 or self._tf_buf is None:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (t.x, t.y, yaw)
        except Exception:
            return None

    def _direct_approach(self, target_pose: PoseStamped,
                          dist: float = APPROACH_DIST_MIN) -> tuple | None:
        """Approach waypoint on the current-robot → target straight line.

        Stops 'dist' metres before the target and faces it.  No orientation
        inference from the ArUco quaternion — uses the actual robot position
        so the path is always the most direct possible route.

        Tries the straight-line standoff first, then standoff points at other
        angles AROUND the target — so a target tucked next to a pillar (whose
        straight-line approach is blocked) can still be reached from a clear
        side, with Nav2 planning the path there."""
        tx = target_pose.pose.position.x
        ty = target_pose.pose.position.y
        robot = self._robot_xy()
        if robot is None:
            return None
        rx, ry = robot
        d = math.hypot(tx - rx, ty - ry)

        # Already within standoff → just face the target from here.
        if d <= dist + 0.05:
            return (rx, ry, math.atan2(ty - ry, tx - rx))

        # Preferred direction = from the target back toward the robot (the
        # straight-line approach).  Fall back to other angles around the target.
        base_ang = math.atan2(ry - ty, rx - tx)
        angles = [0, 25, -25, 50, -50, 75, -75, 100, -100,
                  130, -130, 155, -155, 180]
        for r in (dist, dist * 0.8, dist * 1.2):
            for da in angles:
                ang = base_ang + math.radians(da)
                ax = tx + r * math.cos(ang)
                ay = ty + r * math.sin(ang)
                if self._has_clearance(ax, ay, clearance=0.45):
                    return (ax, ay, math.atan2(ty - ay, tx - ax))
        return None

    def _estimate_zone_b_from_slam(self):
        if self._map_data is None or self._zone_a_pose is None or not _SCIPY:
            return None
        free    = (self._map_data == 0).astype(np.uint8)
        labeled, n = ndlabel(free)
        if n == 0:
            return None
        ax = self._zone_a_pose.pose.position.x
        ay = self._zone_a_pose.pose.position.y
        best_dist, best_pos = 0.0, None
        for lbl in range(1, n + 1):
            cells = np.argwhere(labeled == lbl)
            if len(cells) < MIN_FREE_CELLS:
                continue
            r_mean, c_mean = cells.mean(axis=0)
            wx = self._map_ox + c_mean * self._map_res
            wy = self._map_oy + r_mean * self._map_res
            d  = math.hypot(wx - ax, wy - ay)
            if d >= MIN_ZONE_SEP and d > best_dist:
                best_dist, best_pos = d, (wx, wy)
        return best_pos

    def _slam_ready(self):
        if self._tf_buf is None:
            return False
        try:
            self._tf_buf.lookup_transform('map', 'odom',
                                          rclpy.time.Time(),
                                          timeout=Duration(seconds=0.05))
            return True
        except Exception:
            return False

    # ── Survey planning ───────────────────────────────────────────────────────

    def _plan_survey(self) -> list[tuple]:
        """
        Compute scan positions from the detected Zone A pole — no hardcoded
        world coordinates.

        The pole's ArUco orientation tells us which direction is "into the
        zone" (the opposite of the marker's facing direction, i.e. behind the
        pole).  We plant a strip of scan positions SURVEY_DEPTH metres into
        the zone from the pole, at lateral steps of SURVEY_LAT_STEP, filtered
        to free/clear cells in the current SLAM map.

        Yaw at every position = face back toward the pole.  This aligns the
        camera with the scan strip so markers between the robot and the pole
        are visible, and a 360° spin at each position covers the rest.
        """
        if self._zone_a_pose is None:
            return []

        pole_x = self._zone_a_pose.pose.position.x
        pole_y = self._zone_a_pose.pose.position.y

        # Derive "into zone" direction from the pole's detected orientation.
        # The pole marker faces outward (+Z of its local frame).  The zone
        # interior is behind the pole (opposite to the facing direction).
        q = self._zone_a_pose.pose.orientation
        facing  = _quat_rotate(q.x, q.y, q.z, q.w, np.array([0.0, 0.0, 1.0]))
        face_a  = math.atan2(facing[1], facing[0])
        into_a  = face_a + math.pi          # opposite = into zone
        perp_a  = into_a + math.pi / 2      # lateral axis

        candidates = []
        lat = 0.0
        while lat <= SURVEY_LAT_MAX + 1e-6:
            for sign in ([0] if lat == 0.0 else [+1, -1]):
                # Lateral offset: perpendicular to the into-zone axis.
                # perp_a IS the lateral axis — do NOT add another π/2.
                cx_lat = pole_x + sign * lat * math.cos(perp_a)
                cy_lat = pole_y + sign * lat * math.sin(perp_a)
                for depth in SURVEY_DEPTHS:
                    cx = cx_lat + depth * math.cos(into_a)
                    cy = cy_lat + depth * math.sin(into_a)
                    if self._has_clearance(cx, cy, 0.45):
                        candidates.append((cx, cy))
            lat += SURVEY_LAT_STEP

        # Deduplicate positions that are too close together
        kept: list[tuple] = []
        for pos in candidates:
            if all(math.hypot(pos[0] - k[0], pos[1] - k[1]) >= SURVEY_MIN_SEP
                   for k in kept):
                kept.append(pos)

        # Convert to (x, y, yaw) — face back toward pole so camera covers
        # the approach corridor (detection also happens during the 360° spin)
        waypoints = []
        for cx, cy in kept:
            yaw = math.atan2(pole_y - cy, pole_x - cx)
            waypoints.append((cx, cy, yaw))

        self.get_logger().info(
            f'Survey plan: {len(waypoints)} scan positions '
            f'(depths={SURVEY_DEPTHS} m, lat_step={SURVEY_LAT_STEP:.1f} m)')
        return waypoints

    # ── Navigation helpers ────────────────────────────────────────────────────

    def _send_nav_goal(self, x, y, yaw):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self._nav_backoff_until_ns:
            self.get_logger().warn(
                'Nav2 goal on backoff after rejection — waiting.',
                throttle_duration_sec=3.0)
            return
        self._stop_spin()   # never fight Nav2 for cmd_vel
        if not self._nav.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn('Nav2 not ready — goal deferred.',
                                   throttle_duration_sec=5.0)
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._navigating   = True
        self._goal_handle  = None
        self._goal_sent_ns = self.get_clock().now().nanoseconds
        future = self._nav.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted_cb)
        self.get_logger().info(
            f'Nav goal → ({x:.2f}, {y:.2f})  yaw={math.degrees(yaw):.0f}°')

    def _goal_accepted_cb(self, future):
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error(
                'Nav goal rejected — Nav2 not ready yet. Backing off 3 s.')
            self._nav_backoff_until_ns = (self.get_clock().now().nanoseconds
                                          + int(3.0 * 1e9))
            self._navigating = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_done_cb)

    def _goal_done_cb(self, future):
        result = future.result()
        status = result.status if result else GoalStatus.STATUS_UNKNOWN
        self._navigating  = False
        self._goal_handle = None
        labels = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
                  GoalStatus.STATUS_ABORTED:   'ABORTED',
                  GoalStatus.STATUS_CANCELED:  'CANCELED'}
        self.get_logger().info(
            f'Nav {labels.get(status, str(status))} — state={self._state.value}')

        if self._state == MissionState.SURVEY:
            # Arrived (or aborted) at scan position → start 360° spin
            self._survey_spinning      = True
            self._survey_spin_start_ns = self.get_clock().now().nanoseconds
            self._spinning             = True
            self._spin_start_ns        = self._survey_spin_start_ns
            self._consecutive_aborts   = 0
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._consecutive_aborts = 0
            self._on_goal_succeeded()
        elif status == GoalStatus.STATUS_ABORTED:
            self._consecutive_aborts += 1
            if self._consecutive_aborts >= MAX_CONSEC_ABORTS:
                self._on_nav_failed()

    def _on_goal_succeeded(self):
        state = self._state
        if state == MissionState.GOTO_POLE:
            self._transition(MissionState.SURVEY)
        elif state == MissionState.GOTO_BOX:
            # Reached the box standoff (facing its tag-A front) → hold for the
            # precision team, then go to Zone B.  Stay in GOTO_BOX; the tick
            # runs the dwell timer and transitions when it elapses.
            self._dwelling      = True
            self._dwell_start_ns = self.get_clock().now().nanoseconds
            self.get_logger().info(
                f'Reached box ID={self._current_box} — holding {DWELL_SEC:.0f}s.')
        elif state == MissionState.GOTO_B:
            # Completed one box→B cycle.  Advance the fixed plan.
            self._zone_b_visits += 1
            self.get_logger().info(
                f'✓ Box ID={self._current_box} done '
                f'({self._zone_b_visits}/{len(self._box_order)}).')
            self._current_box = None
            if self._zone_b_visits >= len(self._box_order):
                self._transition(MissionState.DONE)
            else:
                self._transition(MissionState.GOTO_BOX)

    def _on_nav_failed(self):
        state = self._state
        # GOTO_BOX / GOTO_B must keep trying — every box has to be delivered to
        # reach the target count.  Clear the abort streak and let the next tick
        # re-plan (re-detected poses / cleared costmaps give a fresh path).
        if state in (MissionState.GOTO_BOX, MissionState.GOTO_B):
            self.get_logger().warn(
                f'{MAX_CONSEC_ABORTS} consecutive aborts in {state.value} — '
                f'clearing streak and retrying.')
            self._consecutive_aborts = 0
            self._b_goal_is_delivery = False
            self._box_search_goal    = False
            return
        # GOTO_POLE: fall back to surveying from the current spot.
        self.get_logger().error(
            f'{MAX_CONSEC_ABORTS} consecutive aborts in {state.value}.')
        if state == MissionState.GOTO_POLE:
            self._transition(MissionState.SURVEY)
        else:
            self._transition(MissionState.DONE)

    def _cancel_current_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    # ── State transitions ─────────────────────────────────────────────────────

    _PHASE_NUM = {
        MissionState.INIT_SPIN: 1,
        MissionState.GOTO_POLE: 2,
        MissionState.SURVEY:    3,
        MissionState.GOTO_BOX:  4,
        MissionState.GOTO_B:    5,
        MissionState.DONE:      6,
    }
    _PHASE_NAME = {
        MissionState.INIT_SPIN: 'INIT SPIN — searching for Zone A',
        MissionState.GOTO_POLE: 'GOTO POLE — navigating to Zone A',
        MissionState.SURVEY:    'SURVEY — scanning for all box markers',
        MissionState.GOTO_BOX:  'GOTO BOX',
        MissionState.GOTO_B:    'GOTO ZONE B — delivering box',
        MissionState.DONE:      'DONE — saving map',
    }

    def _transition(self, new_state: MissionState):
        self.get_logger().info(
            f'State: {self._state.value} → {new_state.value}')
        self._state              = new_state
        self._state_entered_ns   = self.get_clock().now().nanoseconds
        self._consecutive_aborts = 0
        # Goal-type flags only matter within their own state; clear on entry so
        # an in-flight flag can never be mistaken across a transition.
        self._b_goal_is_delivery = False
        self._box_search_goal    = False
        self._dwelling           = False

        # ── Phase banner (yellow, always visible) ─────────────────────────────
        n    = self._PHASE_NUM.get(new_state, '?')
        name = self._PHASE_NAME.get(new_state, new_state.value)
        if new_state == MissionState.GOTO_BOX:
            name = f'GOTO BOX — delivery {self._zone_b_visits + 1}/{self._delivery_target}'
        elif new_state == MissionState.GOTO_B:
            name = f'GOTO ZONE B — delivery {self._zone_b_visits + 1}/{self._delivery_target}'
        print(f'\033[1;33m▶▶ PHASE {n}: {name} ◀◀\033[0m', flush=True)

        if new_state == MissionState.SURVEY:
            # Plan scan positions now using detected pole + current SLAM map
            self._survey_wps        = self._plan_survey()
            self._survey_wp_idx     = 0
            self._survey_spinning   = False
            self._survey_prev_found = frozenset(self._box_poses.keys())
            self._survey_stale      = 0
            # Kick off first spin immediately at the current position
            self._survey_spinning      = True
            self._survey_spin_start_ns = self.get_clock().now().nanoseconds
            self._spinning             = True
            self._spin_start_ns        = self._survey_spin_start_ns

        if new_state == MissionState.DONE:
            self._stop_spin()
            self._cancel_current_goal()
            self._save_map()

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._start_ns is None:
            self._start_ns         = now_ns
            self._state_entered_ns = now_ns
            self.get_logger().info(
                f'First tick — sim clock anchored at {now_ns / 1e9:.1f} s')
        elapsed = (now_ns - self._start_ns) / 1e9
        if elapsed > GLOBAL_TIMEOUT and self._state != MissionState.DONE:
            self.get_logger().error('Global timeout — saving map.')
            self._transition(MissionState.DONE)
            return
        {
            MissionState.INIT_SPIN: self._tick_init_spin,
            MissionState.GOTO_POLE: self._tick_goto_pole,
            MissionState.SURVEY:    self._tick_survey,
            MissionState.GOTO_BOX:  self._tick_goto_box,
            MissionState.GOTO_B:    self._tick_goto_b,
            MissionState.DONE:      lambda: None,
        }[self._state]()

    # ── INIT_SPIN ─────────────────────────────────────────────────────────────

    def _tick_init_spin(self):
        now_ns = self.get_clock().now().nanoseconds

        if not self._slam_ready():
            elapsed_wait = (now_ns - self._start_ns) / 1e9
            if elapsed_wait > SLAM_WAIT_TIMEOUT:
                self.get_logger().error('SLAM did not start — aborting.')
                self._transition(MissionState.DONE)
            else:
                self.get_logger().info('Waiting for SLAM…',
                                       throttle_duration_sec=5.0)
            return

        if self._zone_a_pose is not None:
            self._stop_spin()
            b_status = '✓' if self._zone_b_pose is not None else '?'
            self.get_logger().info(
                f'Zone A detected — proceeding to pole.  Zone B: {b_status}')
            self._transition(MissionState.GOTO_POLE)
            return

        if self._spinning:
            elapsed_spin = (now_ns - self._spin_start_ns) / 1e9
            if elapsed_spin < SPIN_DURATION:
                self.get_logger().info(
                    f'Spinning… {math.degrees(elapsed_spin * SPIN_RATE_RAD_S):.0f}°/360°',
                    throttle_duration_sec=3.0)
                return
            self._spinning      = False
            self._spin_start_ns = None
            self._cmd_vel_pub.publish(Twist())
            self._search_cycles += 1
            # Zone A is guaranteed to exist — never abort on a cycle count.
            # Relocate to a fresh vantage point and scan again.  The mission
            # is bounded only by GLOBAL_TIMEOUT.
            self.get_logger().info(
                f'360° scan complete (search cycle {self._search_cycles}) — relocating.')
            self._begin_relocate(now_ns)
            return

        if self._reorienting:
            elapsed_turn = (now_ns - self._reorient_start_ns) / 1e9
            if elapsed_turn < REORIENT_DUR:
                return
            self._reorienting        = False
            self._reorient_start_ns  = None
            self._cmd_vel_pub.publish(Twist())
            # New heading chosen — try to advance again (no further reorient on
            # this attempt so we can't loop turning forever in a tight corner;
            # if still blocked, _begin_relocate falls back to scanning in place).
            self._begin_relocate(now_ns, allow_reorient=False)
            return

        if self._moving_forward:
            elapsed_move = (now_ns - self._move_start_ns) / 1e9
            if elapsed_move < SPIN_MOVE_DUR:
                return
            self._moving_forward = False
            self._cmd_vel_pub.publish(Twist())
            self._spinning      = True
            self._spin_start_ns = now_ns
            return

        self._spinning      = True
        self._spin_start_ns = now_ns
        self.get_logger().info('SLAM ready — starting 360° scan.')

    def _begin_relocate(self, now_ns: int, allow_reorient: bool = True):
        """Move to a fresh search vantage point.

        Advances SPIN_MOVE_DIST along the current heading when the path is
        clear; otherwise turns to a new heading (REORIENT_ANGLE) and retries.
        After MAX_CONSEC_REORIENTS blocked headings (e.g. wedged in a corner)
        it falls back to scanning in place so detection can still happen and
        the global timeout remains the only hard stop."""
        pose = self._robot_pose()
        clear_ahead = True
        if pose is not None:
            rx, ry, ryaw = pose
            ax = rx + SPIN_MOVE_DIST * math.cos(ryaw)
            ay = ry + SPIN_MOVE_DIST * math.sin(ryaw)
            clear_ahead = self._has_clearance(ax, ay, clearance=0.45)

        if clear_ahead:
            self._reorient_count = 0
            self._moving_forward = True
            self._move_start_ns  = now_ns
            return

        if allow_reorient and self._reorient_count < MAX_CONSEC_REORIENTS:
            self._reorient_count += 1
            self._reorienting       = True
            self._reorient_start_ns = now_ns
            self.get_logger().info(
                f'Path ahead blocked — reorienting {math.degrees(REORIENT_ANGLE):.0f}° '
                f'(attempt {self._reorient_count}/{MAX_CONSEC_REORIENTS}).')
            return

        # Cornered: scan in place and try again next cycle.
        self._reorient_count = 0
        self._spinning      = True
        self._spin_start_ns = now_ns
        self.get_logger().warn(
            'No clear heading to advance — scanning in place and retrying.')

    # ── GOTO_POLE ─────────────────────────────────────────────────────────────

    def _tick_goto_pole(self):
        if self._navigating:
            elapsed = (self.get_clock().now().nanoseconds - self._goal_sent_ns) / 1e9
            if elapsed > GOAL_TIMEOUT:
                self.get_logger().warn('Pole nav timeout — cancelling.')
                self._cancel_current_goal()
            return

        if self._zone_a_pose is None:
            self.get_logger().warn('No Zone A pose — waiting.',
                                   throttle_duration_sec=5.0)
            return

        # Direct approach: stop APPROACH_DIST_MIN before Zone A on the straight
        # line from the robot's current position.  No marker-orientation inference
        # — avoids wrong-direction approaches caused by noisy early ArUco readings.
        wp = self._direct_approach(self._zone_a_pose)
        if wp is None:
            self.get_logger().warn('No clear approach to Zone A — retrying.',
                                   throttle_duration_sec=5.0)
            return
        self._send_nav_goal(*wp)

    # ── SURVEY ────────────────────────────────────────────────────────────────

    def _tick_survey(self):
        now_ns = self.get_clock().now().nanoseconds

        # Finish as soon as we have enough CONFIRMED boxes AND Zone B — then the
        # plan is fully known and execution is a fixed go-here/go-there sequence.
        if (len(self._box_poses) >= self._delivery_target
                and self._zone_b_pose is not None):
            self.get_logger().info(
                f'{len(self._box_poses)} boxes + Zone B known — ending survey.')
            self._stop_spin()
            self._survey_spinning = False
            if self._navigating:
                self._cancel_current_goal()
            self._finish_survey()
            return

        # ── sub-phase: spinning at current scan position ───────────────────
        if self._survey_spinning:
            elapsed_spin = (now_ns - self._survey_spin_start_ns) / 1e9
            if elapsed_spin < SPIN_DURATION:
                self.get_logger().info(
                    f'Survey spin {math.degrees(elapsed_spin * SPIN_RATE_RAD_S):.0f}°/360°  '
                    f'boxes: {sorted(self._box_poses.keys())}',
                    throttle_duration_sec=3.0)
                return

            # Spin complete — assess new markers
            self._stop_spin()
            self._survey_spinning = False
            now_found = frozenset(self._box_poses.keys())
            if now_found == self._survey_prev_found:
                self._survey_stale += 1
                self.get_logger().info(
                    f'Survey spin: no new markers (stale {self._survey_stale}'
                    f'/{SURVEY_MAX_STALE}).  Found: {sorted(now_found)}')
            else:
                new = now_found - self._survey_prev_found
                self.get_logger().info(
                    f'Survey spin: found new marker(s): {sorted(new)}  '
                    f'total: {sorted(now_found)}')
                self._survey_stale = 0
            self._survey_prev_found = now_found

            # Visit every planned scan position so all box markers get a chance
            # to be seen (the stale counter is kept for logging only — boxes
            # missed here are still picked up dynamically during delivery).
            if self._survey_wp_idx >= len(self._survey_wps):
                self._finish_survey()
            # else: let _tick_survey drive navigation to next position
            return

        # ── sub-phase: navigating to next scan position ────────────────────
        if self._navigating:
            elapsed = (now_ns - self._goal_sent_ns) / 1e9
            if elapsed > GOAL_TIMEOUT:
                self.get_logger().warn('Survey nav timeout — skipping position.')
                self._cancel_current_goal()
            return

        if self._survey_wp_idx >= len(self._survey_wps):
            # No more planned positions — one last finish check
            if not self._survey_spinning:
                self._finish_survey()
            return

        wx, wy, wyaw = self._survey_wps[self._survey_wp_idx]
        n = len(self._survey_wps)
        self.get_logger().info(
            f'Survey → scan position {self._survey_wp_idx + 1}/{n}: '
            f'({wx:.2f}, {wy:.2f})')
        self._survey_wp_idx += 1
        self._send_nav_goal(wx, wy, wyaw)

    def _finish_survey(self):
        found = sorted(self._box_poses.keys())
        self._rank_boxes(found)          # sets self._box_order, largest→smallest
        if not self._box_order:
            self.get_logger().warn('Survey found no boxes — saving map.')
            self._transition(MissionState.DONE)
            return
        # FREEZE the plan: a fixed list of boxes to visit in size order, plus the
        # known Zone B pose.  From here it is pure execution — no re-ranking, no
        # exploration, no searching.  Index into the plan = _zone_b_visits.
        self._delivery_target = len(self._box_order)
        self._zone_b_visits   = 0
        self.get_logger().info(
            f'Survey complete.  Plan (largest→smallest): {self._box_order}.  '
            f'Zone B {"known" if self._zone_b_pose else "UNKNOWN"}.  Executing.')
        self._transition(MissionState.GOTO_BOX)

    def _rank_boxes(self, marker_ids: list[int]):
        """Sort boxes largest→smallest by the footprint size published by
        box_perception (metres, from depth clustering).  Boxes whose size hasn't
        arrived yet rank last (0) until their /box_size topic updates."""
        sizes = {mid: self._box_sizes.get(mid, 0.0) for mid in marker_ids}
        self._box_order = sorted(sizes.keys(), key=lambda k: sizes[k], reverse=True)
        order_str = ' > '.join(
            f'ID{m}({sizes[m]*100:.0f}cm)' for m in self._box_order)
        self.get_logger().info(f'Visit order (largest→smallest): {order_str}')

    # ── Delivery helpers ──────────────────────────────────────────────────────

    def _next_box(self) -> int | None:
        """Highest-ranked (largest) detected box not yet delivered, or None.

        Re-ranks the live set every call so markers discovered after the survey
        are included automatically."""
        undelivered = frozenset(m for m in self._box_poses
                                if m not in self._delivered_ids)
        if not undelivered:
            return None
        # Re-rank (and log) only when the undelivered set changes, so repeated
        # ticks don't spam the ranking output.
        if undelivered != self._last_ranked:
            self._rank_boxes(sorted(undelivered))
            self._last_ranked = undelivered
        for mid in self._box_order:
            if mid in undelivered:
                return mid
        return None

    def _search_spin(self, now_ns: int, msg: str):
        """Rotate in place to bring new markers into view (fallback when the
        thing we need next hasn't been detected yet)."""
        if not self._spinning:
            self._spinning      = True
            self._spin_start_ns = now_ns
        self.get_logger().info(msg, throttle_duration_sec=5.0)

    def _step_toward(self, tx, ty, max_step: float = 3.0):
        """A reachable (x, y, yaw) waypoint stepping from the robot toward a
        target, shrinking the step until a clear cell is found.  Used to explore
        toward Zone B without committing to a possibly-unreachable estimate."""
        robot = self._robot_xy()
        if robot is None:
            return None
        rx, ry = robot
        d = math.hypot(tx - rx, ty - ry)
        if d < 0.05:
            return None
        yaw = math.atan2(ty - ry, tx - rx)
        for step in (max_step, max_step * 0.6, max_step * 0.4, max_step * 0.25):
            s = min(step, d)
            ax = rx + (s / d) * (tx - rx)
            ay = ry + (s / d) * (ty - ry)
            if self._has_clearance(ax, ay, clearance=0.5):
                return (ax, ay, yaw)
        return None

    # ── GOTO_BOX ──────────────────────────────────────────────────────────────

    def _tick_goto_box(self):
        now_ns = self.get_clock().now().nanoseconds

        # Dwell: hold in front of the box (facing its tag-A front) for the
        # precision team, then proceed to Zone B.
        if self._dwelling:
            elapsed = (now_ns - self._dwell_start_ns) / 1e9
            if elapsed < DWELL_SEC:
                self.get_logger().info(
                    f'Holding in front of box ID={self._current_box} '
                    f'({elapsed:.1f}/{DWELL_SEC:.0f}s)…', throttle_duration_sec=1.0)
                return
            self._dwelling = False
            self._transition(MissionState.GOTO_B)
            return

        if self._navigating:
            elapsed = (now_ns - self._goal_sent_ns) / 1e9
            if elapsed > GOAL_TIMEOUT:
                self.get_logger().warn('Box nav timeout — cancelling.')
                self._cancel_current_goal()
            return

        # Fixed plan: visit boxes in the frozen size order.  Index = visits done.
        if self._zone_b_visits >= len(self._box_order):
            self._transition(MissionState.DONE)
            return

        mid  = self._box_order[self._zone_b_visits]
        self._current_box = mid
        pose = self._box_poses.get(mid)
        if pose is None:
            self.get_logger().warn(
                f'Box ID={mid} pose missing — skipping.', throttle_duration_sec=5.0)
            self._zone_b_visits += 1
            return

        # Approach from the Zone-A (pole) side so the camera faces the box's
        # tag-A-facing front, at a close standoff.
        approach = self._approach_from_pole(pose, BOX_APPROACH_DIST)
        if approach is None:
            self.get_logger().warn(
                f'No clear approach for box ID={mid} — retrying.',
                throttle_duration_sec=5.0)
            return

        ax, ay, ayaw = approach
        self.get_logger().info(
            f'→ Box {self._zone_b_visits + 1}/{len(self._box_order)} (ID={mid}) '
            f'front approach ({ax:.2f}, {ay:.2f})')
        self._send_nav_goal(ax, ay, ayaw)

    def _approach_from_pole(self, box_pose: PoseStamped, dist: float) -> tuple | None:
        """Standoff point between the box and the Zone A pole, facing the box —
        so the camera looks at the box's tag-A-facing 'front'.  Tries that
        direction first, then small angular offsets if it's blocked.  Falls back
        to the generic robot→box approach if the pole isn't known."""
        if self._zone_a_pose is None:
            return self._direct_approach(box_pose, dist)
        bx = box_pose.pose.position.x
        by = box_pose.pose.position.y
        px = self._zone_a_pose.pose.position.x
        py = self._zone_a_pose.pose.position.y
        base = math.atan2(py - by, px - bx)        # box → pole direction
        for da in (0, 20, -20, 40, -40, 60, -60, 90, -90):
            ang = base + math.radians(da)
            ax = bx + dist * math.cos(ang)
            ay = by + dist * math.sin(ang)
            if self._has_clearance(ax, ay, clearance=0.35):
                return (ax, ay, math.atan2(by - ay, bx - ax))   # face the box
        return None

    # ── GOTO_B ────────────────────────────────────────────────────────────────

    def _tick_goto_b(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._navigating:
            elapsed = (now_ns - self._goal_sent_ns) / 1e9
            if elapsed > GOAL_TIMEOUT:
                self.get_logger().warn('Zone B nav timeout — cancelling.')
                self._cancel_current_goal()
            return

        mid = self._current_box if self._current_box is not None else '?'

        # Zone B was fixed during the survey — just go to it.  No exploration.
        if self._zone_b_pose is None:
            self.get_logger().warn('Zone B pose not available — waiting.',
                                   throttle_duration_sec=5.0)
            return
        wp_b = self._direct_approach(self._zone_b_pose)
        if wp_b is None:
            self.get_logger().warn('No clear Zone B approach — retrying.',
                                   throttle_duration_sec=5.0)
            return
        bx, by, byaw = wp_b
        self.get_logger().info(
            f'→ Zone B ({bx:.2f},{by:.2f})  [box ID={mid}, '
            f'{self._zone_b_visits}/{len(self._box_order)} done]')
        self._send_nav_goal(bx, by, byaw)

    # ── Map saving ────────────────────────────────────────────────────────────

    def _save_map(self):
        if self._map_saved:
            return
        self._map_saved = True
        save_path = self.get_parameter('map_save_path').get_parameter_value().string_value
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.get_logger().info(f'Saving map → {save_path}')

        def _do_save():
            try:
                result = subprocess.run(
                    ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                     '-f', save_path, '--ros-args', '-p', 'use_sim_time:=true'],
                    timeout=30.0, capture_output=True)
                if result.returncode == 0:
                    self.get_logger().info(f'Map saved → {save_path}.pgm')
                    pgm = save_path + '.pgm'
                    if os.path.exists(pgm):
                        for viewer in ('eog', 'xdg-open', 'feh', 'display'):
                            try:
                                subprocess.Popen([viewer, pgm],
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL)
                                self.get_logger().info(f'Map displayed ({viewer})')
                                break
                            except FileNotFoundError:
                                continue
                else:
                    self.get_logger().error(
                        f'Map save failed (rc={result.returncode}): '
                        f'{result.stderr.decode().strip()}')
            except subprocess.TimeoutExpired:
                self.get_logger().error('Map save timed out after 30 s.')
            except Exception as exc:
                self.get_logger().error(f'Map save failed: {exc}')

        threading.Thread(target=_do_save, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    node     = ExplorationManager()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
