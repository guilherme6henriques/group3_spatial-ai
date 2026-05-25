#!/usr/bin/env python3
"""
Two-zone mission manager for MIRTE pick-and-place.

Zone A (pickup, red tape) and Zone B (stacking, blue tape) are located at runtime
by the zone_detector node, which publishes their map-frame centres on:
  /zone_a_pose  — red tape zone centre
  /zone_b_pose  — blue tape zone centre

Mission phases:
  MAPPING  — explore_lite drives; manager waits for frontiers to go quiet AND
              both zone poses to be received from zone_detector.
  GOTO_A   — navigate to Zone A centre (tape has no collision, freely driveable).
  SCAN_A   — four waypoints at SCAN_RADIUS from Zone A centre; SLAM maps box bodies.
  GOTO_B   — navigate to Zone B centre through obstacle corridor (Nav2 plans route).
  DONE     — publish /box_poses (largest-first) and wait for gripper commands.
"""

import math
import subprocess
from enum import Enum

import numpy as np
import rclpy
import rclpy.duration
import rclpy.executors
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal as CancelGoalSrv

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


# ── File paths ────────────────────────────────────────────────────────────────
MAP_SAVE_PATH = '/home/gui/spatial-ai/ws/src/mirte_navigation/maps/default'

# ── Timing ────────────────────────────────────────────────────────────────────
MIN_EXPLORATION_SECONDS = 30.0
NO_FRONTIER_SECONDS     = 10.0
GLOBAL_TIMEOUT          = 600.0
CHECK_INTERVAL          = 2.0

# ── SLAM cluster thresholds (calibrated for 0.02 m/cell) ─────────────────────
#   box_80  (80×120 mm)  → ~24 cells
#   box_160 (160×200 mm) → ~80 cells
#   obstacle pillar      → ~225 cells
MIN_POLYGON_CELLS    = 20
MAX_POLYGON_CELLS    = 600
ARENA_MIN_FREE_CELLS = 2500     # ~1 m² at 0.02 m/cell
_MERGE_DIST_SQ       = 1.2 ** 2

# Anything with ≤ MAX_BOX_CELLS AND inside Zone A is a box cluster.
# Gap between box_160 (80 cells) and pillar (225 cells) is wide.
MAX_BOX_CELLS = 120

# ── Zone A scan geometry ──────────────────────────────────────────────────────
# Four waypoints at SCAN_RADIUS from the detected Zone A centre, one in each
# cardinal direction, each facing inward toward the centre.
SCAN_RADIUS = 0.7   # metres — keeps robot inside the 2 m × 2 m zone

# Zone A boundary half-width (used to stay inside zone during scan)
ZONE_HALF = 0.9     # 1.0 m half-width minus 0.1 m margin

# ── Nav2 abort handling ───────────────────────────────────────────────────────
MAX_CONSEC_ABORTS = 3
ABORT_BACKOFF_SEC = 30.0

# ── Zone B SLAM fallback ──────────────────────────────────────────────────────
# After frontiers go quiet and this many seconds pass without Zone B being
# camera-detected, compute Zone B from the SLAM free-space map instead.
ZONE_B_FALLBACK_DELAY = 30.0  # seconds
MIN_ZONE_SEPARATION   = 3.0   # m — zone B must be at least this far from zone A


class MissionState(Enum):
    MAPPING = 'mapping'
    GOTO_A  = 'goto_a'
    SCAN_A  = 'scan_a'
    GOTO_B  = 'goto_b'
    DONE    = 'done'


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        self.start_time         = self.get_clock().now()
        self.map_saved          = False
        self.last_frontier_time = None

        self.map_data   = None
        self.map_width  = 0
        self.map_height = 0
        self.map_res    = 0.02
        self.map_ox     = 0.0
        self.map_oy     = 0.0

        # Mission state machine
        self._state              = MissionState.MAPPING
        self._survey_mode_logged = False

        # Detected zone centres (from zone_detector)
        self._zone_a_center: tuple | None = None   # (cx, cy) in map frame
        self._zone_b_center: tuple | None = None

        # Navigation
        self._navigating          = False
        self._goal_handle         = None
        self._last_status         = None
        self._consecutive_aborts  = 0
        self._abort_backoff_until: int = 0

        # Zone B fallback tracking
        self._frontier_quiet_since_ns: int | None = None

        # Preemption: cancel explore_lite's active goal before first GOTO_A send
        self._preempt_sent = False

        # Zone A scan queue
        self._scan_queue: list = []

        # Detected box clusters (set after SCAN_A, sorted largest-first)
        self._box_clusters: list = []

        if not _SCIPY:
            self.get_logger().warn('scipy not found — cluster detection disabled.')

        cb = ReentrantCallbackGroup()

        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                 callback_group=cb)
        self._cancel_cli = self.create_client(
            CancelGoalSrv,
            '/navigate_to_pose/_action/cancel_goal',
            callback_group=cb)

        if _TF2:
            self._tf_buf      = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        else:
            self._tf_buf = None
            self.get_logger().warn('tf2_ros not available.')

        self._box_pub = self.create_publisher(PoseArray, '/box_poses', 10)

        self.create_subscription(MarkerArray,   '/explore/frontiers',
                                 self._frontier_cb,  10)
        self.create_subscription(OccupancyGrid, '/map',
                                 self._map_cb,        10)
        self.create_subscription(PoseStamped,   '/zone_a_pose',
                                 self._zone_a_cb,     10)
        self.create_subscription(PoseStamped,   '/zone_b_pose',
                                 self._zone_b_cb,     10)

        self.create_timer(CHECK_INTERVAL, self._tick, callback_group=cb)

        self.get_logger().info('Mission manager started — awaiting zone detection.')

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _frontier_cb(self, msg):
        if msg.markers:
            if self.last_frontier_time is None:
                self.get_logger().info('First frontiers detected.')
            self.last_frontier_time = self.get_clock().now()

    def _map_cb(self, msg):
        self.map_data   = msg.data
        self.map_width  = msg.info.width
        self.map_height = msg.info.height
        self.map_res    = msg.info.resolution
        self.map_ox     = msg.info.origin.position.x
        self.map_oy     = msg.info.origin.position.y

    def _zone_a_cb(self, msg: PoseStamped):
        self._zone_a_center = (msg.pose.position.x, msg.pose.position.y)

    def _zone_b_cb(self, msg: PoseStamped):
        self._zone_b_center = (msg.pose.position.x, msg.pose.position.y)

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick(self):
        now     = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        if self._state == MissionState.DONE:
            return

        if elapsed > GLOBAL_TIMEOUT:
            self.get_logger().warn('Global timeout — saving map and stopping.')
            self._save_map()
            self._state = MissionState.DONE
            return

        # ── MAPPING ───────────────────────────────────────────────────────────
        if self._state == MissionState.MAPPING:
            if elapsed < MIN_EXPLORATION_SECONDS:
                return

            frontier_quiet = (
                self.last_frontier_time is not None
                and (now - self.last_frontier_time).nanoseconds / 1e9 >= NO_FRONTIER_SECONDS
            )

            # Track the moment frontiers first go quiet (reset if they come back)
            if frontier_quiet and self._frontier_quiet_since_ns is None:
                self._frontier_quiet_since_ns = now.nanoseconds
            elif not frontier_quiet:
                self._frontier_quiet_since_ns = None

            a_ok = self._zone_a_center is not None
            b_ok = self._zone_b_center is not None

            # SLAM fallback: if Zone B was never camera-detected, estimate its
            # position from the SLAM map after ZONE_B_FALLBACK_DELAY seconds of
            # quiet frontiers.  This handles arenas where the robot never drove
            # close enough to the blue tape for zone_detector to see it.
            if (frontier_quiet and a_ok and not b_ok
                    and self._frontier_quiet_since_ns is not None
                    and (now.nanoseconds - self._frontier_quiet_since_ns) / 1e9
                        >= ZONE_B_FALLBACK_DELAY):
                est = self._estimate_zone_b_from_slam()
                if est is not None:
                    self._zone_b_center = est
                    b_ok = True
                    self.get_logger().warn(
                        f'Zone B not camera-detected after '
                        f'{ZONE_B_FALLBACK_DELAY:.0f} s — '
                        f'using SLAM free-space estimate '
                        f'({est[0]:.1f},{est[1]:.1f}).')

            clusters = self._get_all_clusters()
            n_box = sum(1 for c in clusters if self._in_zone_a(c['centroid'])
                        and c['n_cells'] <= MAX_BOX_CELLS)
            n_obs = len(clusters) - n_box

            quiet_secs = (
                f'{(now.nanoseconds - self._frontier_quiet_since_ns) / 1e9:.0f}s'
                if self._frontier_quiet_since_ns is not None else '—'
            )
            self.get_logger().info(
                f'[MAPPING] frontiers: {"quiet(" + quiet_secs + ")" if frontier_quiet else "active"}  '
                f'Zone A: {"found" if a_ok else "seeking"}  '
                f'Zone B: {"found" if b_ok else "seeking"}  '
                f'SLAM clusters: {n_box} boxes, {n_obs} obstacles')

            if frontier_quiet and a_ok and b_ok:
                if not self._survey_mode_logged:
                    self._survey_mode_logged = True
                    ax, ay = self._zone_a_center
                    bx, by = self._zone_b_center
                    self.get_logger().info(
                        f'Frontiers exhausted. Zone A ({ax:.1f},{ay:.1f}), '
                        f'Zone B ({bx:.1f},{by:.1f}). '
                        f'{n_box} box cluster(s) visible. Entering Zone A.')
                self._state       = MissionState.GOTO_A
                self._last_status = None

            elif frontier_quiet and not (a_ok and b_ok):
                missing = []
                if not a_ok:
                    missing.append('A')
                if not b_ok:
                    if self._frontier_quiet_since_ns is not None:
                        secs_left = max(0, ZONE_B_FALLBACK_DELAY
                                        - (now.nanoseconds - self._frontier_quiet_since_ns) / 1e9)
                        missing.append(f'B (SLAM fallback in {secs_left:.0f}s)')
                    else:
                        missing.append('B')
                self.get_logger().warn(
                    f'Frontiers quiet but Zone {", ".join(missing)} not yet detected.',
                    throttle_duration_sec=5.0)
            return

        # ── All post-MAPPING states involve navigation ─────────────────────────
        if now.nanoseconds < self._abort_backoff_until:
            return
        if self._navigating:
            return

        # ── GOTO_A ────────────────────────────────────────────────────────────
        if self._state == MissionState.GOTO_A:
            if self._last_status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Arrived at Zone A — starting scan sweep.')
                self._scan_queue  = self._make_scan_waypoints(self._zone_a_center)
                self._state       = MissionState.SCAN_A
                self._last_status = None
                # fall through to SCAN_A immediately
            else:
                # Cancel any active nav goal (e.g. explore_lite's last frontier goal)
                # before sending ours, then wait one tick for the cancel to propagate.
                if not self._preempt_sent:
                    if self._cancel_cli.service_is_ready():
                        self._cancel_cli.call_async(CancelGoalSrv.Request())
                        self._preempt_sent = True
                        self.get_logger().info(
                            '[GOTO_A] Cancelling any active nav goal '
                            '(explore_lite holdover) — sending ours next tick.')
                    else:
                        self.get_logger().warn(
                            '[GOTO_A] Cancel service not ready yet — retrying.',
                            throttle_duration_sec=2.0)
                    return
                ax, ay = self._zone_a_center
                bx, by = self._zone_b_center
                entry_yaw = math.atan2(by - ay, bx - ax) + math.pi  # face away from B
                self.get_logger().info(
                    f'[GOTO_A] Navigating to Zone A ({ax:.1f},{ay:.1f}).')
                self._send_goal(ax, ay, entry_yaw)
                return

        # ── SCAN_A ────────────────────────────────────────────────────────────
        if self._state == MissionState.SCAN_A:
            if self._scan_queue:
                x, y, yaw = self._scan_queue.pop(0)
                self.get_logger().info(
                    f'[SCAN_A] ({x:.1f},{y:.1f}) — {len(self._scan_queue)} remaining.')
                self._send_goal(x, y, yaw)
            else:
                self._classify_and_publish()
                self._save_map()
                self.get_logger().info(
                    f'Zone A scan complete — {len(self._box_clusters)} box(es) located. '
                    f'Navigating to Zone B.')
                self._state       = MissionState.GOTO_B
                self._last_status = None
                bx, by = self._zone_b_center
                # Face Zone A when arriving at B so arm is oriented toward pickup side
                ax, ay = self._zone_a_center
                arrive_yaw = math.atan2(ay - by, ax - bx)
                self._send_goal(bx, by, arrive_yaw)
            return

        # ── GOTO_B ────────────────────────────────────────────────────────────
        if self._state == MissionState.GOTO_B:
            if self._last_status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    'Arrived at Zone B — navigation complete. '
                    'Awaiting pick-and-place commands.')
                self._publish_box_poses()
                self._state = MissionState.DONE
            else:
                bx, by = self._zone_b_center
                ax, ay = self._zone_a_center
                arrive_yaw = math.atan2(ay - by, ax - bx)
                self.get_logger().info(f'[GOTO_B] Navigating to ({bx:.1f},{by:.1f}).')
                self._send_goal(bx, by, arrive_yaw)

    # ── Zone geometry helpers ─────────────────────────────────────────────────

    def _in_zone_a(self, centroid):
        """Return True if centroid is within ZONE_HALF of the detected Zone A centre."""
        if self._zone_a_center is None:
            return False
        cx, cy = centroid
        ax, ay = self._zone_a_center
        return abs(cx - ax) <= ZONE_HALF and abs(cy - ay) <= ZONE_HALF

    def _make_scan_waypoints(self, center):
        """
        Four waypoints at SCAN_RADIUS from zone centre in N/S/E/W, each facing
        inward (toward the centre).  If a waypoint falls outside the SLAM free
        region it will be ABORTED by Nav2 and skipped — that is fine; the
        remaining three still give good box coverage.
        """
        cx, cy = center
        r = SCAN_RADIUS
        return [
            (cx + r, cy,  math.pi),          # east  → face west
            (cx,     cy - r,  math.pi / 2),  # south → face north
            (cx - r, cy,  0.0),              # west  → face east
            (cx,     cy + r, -math.pi / 2),  # north → face south
        ]

    # ── SLAM cluster detection ────────────────────────────────────────────────

    def _get_all_clusters(self):
        """
        Detect all SLAM obstacle clusters in the current map.
        Returns list of dicts: {centroid: (cx, cy), n_cells: int}.
        """
        if self.map_data is None or not _SCIPY:
            return []

        data = np.array(self.map_data, dtype=np.int8).reshape(
            self.map_height, self.map_width)

        # Largest connected free region = navigable floor
        free        = (data == 0).astype(np.uint8)
        free_lbl, _ = ndlabel(free)
        sizes       = np.bincount(free_lbl.ravel())
        if len(sizes) <= 1:
            return []
        main_lbl = int(np.argmax(sizes[1:]) + 1)
        if sizes[main_lbl] < ARENA_MIN_FREE_CELLS:
            return []

        main_free = (free_lbl == main_lbl)
        fr, fc    = np.where(main_free)
        rmin, rmax = int(fr.min()), int(fr.max())
        cmin, cmax = int(fc.min()), int(fc.max())

        adj_main = (
            np.roll(main_free,  1, axis=0) | np.roll(main_free, -1, axis=0)
            | np.roll(main_free,  1, axis=1) | np.roll(main_free, -1, axis=1)
        )

        occ      = (data == 100).astype(np.uint8)
        lbl, nlbl = ndlabel(occ)

        raw = []
        for l in range(1, nlbl + 1):
            mask   = (lbl == l)
            n_cells = int(np.sum(mask))
            if n_cells < MIN_POLYGON_CELLS or n_cells > MAX_POLYGON_CELLS:
                continue
            rows, cols = np.where(mask)
            cr = float(rows.mean())
            cc = float(cols.mean())
            if not (rmin <= cr <= rmax and cmin <= cc <= cmax):
                continue
            if not np.any(mask & adj_main):
                continue
            cx = self.map_ox + (cc + 0.5) * self.map_res
            cy = self.map_oy + (cr + 0.5) * self.map_res
            raw.append({'centroid': (cx, cy), 'n_cells': n_cells})

        # Deduplicate — keep largest within merge radius
        raw.sort(key=lambda c: -c['n_cells'])
        accepted: list = []
        deduped:  list = []
        for c in raw:
            cx, cy = c['centroid']
            if any((cx - ax) ** 2 + (cy - ay) ** 2 < _MERGE_DIST_SQ
                   for ax, ay in accepted):
                continue
            accepted.append((cx, cy))
            deduped.append(c)
        return deduped

    def _estimate_zone_b_from_slam(self):
        """
        Estimate Zone B centre from the SLAM map when zone_detector never saw
        the blue tape.

        Strategy: take the 20 % of free SLAM cells that are farthest from the
        known Zone A centre (subject to MIN_ZONE_SEPARATION) and return their
        centroid.  In a two-zone arena Zone B occupies exactly this region.
        Returns (cx, cy) in map frame, or None if the map is too sparse.
        """
        if self.map_data is None or self._zone_a_center is None or not _SCIPY:
            return None

        data = np.array(self.map_data, dtype=np.int8).reshape(
            self.map_height, self.map_width)
        ax, ay = self._zone_a_center

        rows, cols = np.where(data == 0)
        if len(rows) < ARENA_MIN_FREE_CELLS:
            return None

        wx = self.map_ox + (cols + 0.5) * self.map_res
        wy = self.map_oy + (rows + 0.5) * self.map_res

        dist = np.sqrt((wx - ax) ** 2 + (wy - ay) ** 2)

        # Only consider cells genuinely separated from Zone A
        far_mask = dist > MIN_ZONE_SEPARATION
        if np.sum(far_mask) < 100:
            self.get_logger().warn(
                'SLAM fallback: fewer than 100 free cells beyond '
                f'{MIN_ZONE_SEPARATION} m from Zone A — map too sparse.')
            return None

        far_dist = dist[far_mask]
        far_wx   = wx[far_mask]
        far_wy   = wy[far_mask]

        # Centroid of the farthest 20 % → deep inside Zone B territory
        n         = len(far_dist)
        top_start = int(n * 0.80)
        top_idx   = np.argsort(far_dist)[top_start:]
        cx = float(np.mean(far_wx[top_idx]))
        cy = float(np.mean(far_wy[top_idx]))
        return cx, cy

    def _classify_and_publish(self):
        """
        Classify SLAM clusters: those inside Zone A with ≤ MAX_BOX_CELLS are boxes.
        Sort largest-first (largest SLAM footprint = largest physical box = pick first).
        """
        clusters = self._get_all_clusters()
        self._box_clusters = sorted(
            [c for c in clusters
             if self._in_zone_a(c['centroid']) and c['n_cells'] <= MAX_BOX_CELLS],
            key=lambda c: -c['n_cells'],
        )
        obstacle_clusters = [c for c in clusters if c not in self._box_clusters]
        self.get_logger().info(
            f'Classification: {len(self._box_clusters)} box cluster(s) in Zone A, '
            f'{len(obstacle_clusters)} obstacle cluster(s).')
        for i, b in enumerate(self._box_clusters):
            cx, cy = b['centroid']
            self.get_logger().info(
                f'  Box {i + 1}: ({cx:.2f},{cy:.2f})  {b["n_cells"]} cells')
        self._publish_box_poses()

    def _publish_box_poses(self):
        msg                 = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for b in self._box_clusters:
            p = Pose()
            p.position.x, p.position.y = b['centroid']
            p.position.z = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)
        self._box_pub.publish(msg)

    # ── Nav2 action client ────────────────────────────────────────────────────

    def _send_goal(self, x, y, yaw):
        if not self._nav.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('navigate_to_pose server not available.')
            return
        pose                    = PoseStamped()
        pose.header.frame_id    = 'map'
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = pose
        self._navigating = True
        fut = self._nav.send_goal_async(goal_msg)
        fut.add_done_callback(self._on_accepted)

    def _on_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2.')
            self._navigating = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_done)

    def _on_done(self, future):
        status = future.result().status
        label  = {
            GoalStatus.STATUS_SUCCEEDED: 'reached',
            GoalStatus.STATUS_ABORTED:   'aborted',
            GoalStatus.STATUS_CANCELED:  'canceled',
        }.get(status, str(status))
        self.get_logger().info(f'Nav goal {label} (state: {self._state.value}).')

        if status == GoalStatus.STATUS_ABORTED:
            self._consecutive_aborts += 1
            if self._consecutive_aborts >= MAX_CONSEC_ABORTS:
                self._abort_backoff_until = (
                    self.get_clock().now().nanoseconds
                    + int(ABORT_BACKOFF_SEC * 1e9))
                self.get_logger().warn(
                    f'{MAX_CONSEC_ABORTS} consecutive aborts — '
                    f'backing off {ABORT_BACKOFF_SEC:.0f} s.')
        else:
            self._consecutive_aborts = 0

        self._last_status = status
        self._navigating  = False
        self._goal_handle = None

    # ── Map save ──────────────────────────────────────────────────────────────

    def _save_map(self):
        if self.map_saved:
            return
        self.map_saved = True
        self.get_logger().info(f'Saving map → {MAP_SAVE_PATH}')
        result = subprocess.run(
            ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', MAP_SAVE_PATH],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            self.get_logger().info(f'Map saved: {MAP_SAVE_PATH}.pgm')
        else:
            self.get_logger().error(f'Map save failed:\n{result.stderr}')


def main(args=None):
    rclpy.init(args=args)
    node     = ExplorationManager()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
