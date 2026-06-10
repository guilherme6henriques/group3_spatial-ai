#!/usr/bin/env python3
"""
marker_navigator.py  —  ArUco positioning + drive-back for the Mirte Master.

WHAT IT DOES
────────────
1. Loads camera intrinsics from camera_info.yaml (falls back to /camera_info topic).
2. Continuously detects ArUco markers ID 101 and ID 102; prints distance on each hit.
3. Navigates the robot to approach_m in front of the marker midpoint.
4. Publishes /robot_positioned True (latched) → user sends /start_placing manually.
5. When /arm_placed True arrives (box_placer finished lowering arm):
   drives robot BACK using ArUco feedback until seek_dist_m from marker midpoint.
6. Publishes /robot_backed_up True → box_placer opens gripper and returns home.

SIGNAL FLOW
───────────
  marker_navigator ──/robot_positioned──► (user sends /start_placing manually)
  box_placer       ──/arm_placed──────► marker_navigator  (auto, on PLACE_DOWN done)
  marker_navigator ──/robot_backed_up──► box_placer       (auto, on drive-back done)

STATE MACHINE
─────────────
  SEARCHING  → rotate slowly until both markers found
  DRIVE      → P-controller to target XY
  STOP       → settle 1 s
  ROTATE     → pure in-place yaw
  DONE       → /robot_positioned published, waiting for /arm_placed
  DRIVE_BACK → reverse until seek_dist_m from marker midpoint
  BACKED_UP  → /robot_backed_up published

PARAMETERS  (override with --ros-args -p name:=value)
──────────────────────────────────────────────────────
  camera_info_path  str   <script_dir>/camera_info.yaml
  marker_id_left    int   101
  marker_id_right   int   102
  aruco_dict        str   DICT_4X4_250
  marker_size       float 0.08   Physical side length of markers (m)
  marker_z          float 0.05   Known height of marker centre above floor (m)
  approach_m        float 0.40   Stop distance in front of midpoint (m), from base_link.
                                 Camera is ~0.15 m ahead: camera-to-wall ≈ approach_m - 0.15.
  seek_dist_m       float 0.22   Drive-back target distance from midpoint (m), from base_link.
  image_topic       str   /camera/color/image_raw
  info_topic        str   /camera/color/camera_info
  map_frame         str   odom   (no /map frame on real robot — use odom)
  base_frame        str   base_link
  frame_skip        int   5
  scan_vel          float 0.25   Yaw rate while scanning (rad/s)
"""

import math
import os
import time
import yaml
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
import rclpy.time

import tf2_ros

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

try:
    from cv_bridge import CvBridge
    import cv2
    _CV = True
except ImportError:
    _CV = False

# ─────────────────────────────────────────────────────────────────────────────
# Navigation tuning
# ─────────────────────────────────────────────────────────────────────────────
POS_TOL   = 0.030   # m
YAW_TOL   = 0.087   # rad  (~5°) — mecanum drive stalls at ~4°, so 5° clears it
SETTLE_S  = 1.0     # s

KP_LIN = 0.40
KP_ANG = 0.60
MAX_LIN = 0.18   # m/s
MAX_ANG = 0.30   # rad/s

SCAN_TIMEOUT_S = 60.0
EMA_ALPHA      = 0.25
PUBLISH_HZ     = 5.0

# Drive-back P-controller
SEEK_KP        = 0.80
SEEK_MAX_VEL   = 0.08   # m/s
SEEK_MIN_VEL   = 0.03   # m/s  (motor dead-zone minimum)
SEEK_TOL_M     = 0.005  # m    (5 mm tolerance)
SEEK_TIMEOUT_S = 10.0   # s    (fallback if markers lost)


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_rotate(qx, qy, qz, qw, v: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


def _quat_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _matrix_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x = 0.25 / s, (R[2,1]-R[1,2])*s
        y, z = (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x = (R[2,1]-R[1,2])/s, 0.25*s
        y, z = (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s
        y, z = 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s
        y, z = (R[1,2]+R[2,1])/s, 0.25*s
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return x/n, y/n, z/n, w/n


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


# ─────────────────────────────────────────────────────────────────────────────
# State names
# ─────────────────────────────────────────────────────────────────────────────
class S:
    SEARCHING  = 'SEARCHING'
    DRIVE      = 'DRIVE'
    STOP       = 'STOP'
    ROTATE     = 'ROTATE'
    DONE       = 'DONE'
    DRIVE_BACK = 'DRIVE_BACK'
    BACKED_UP  = 'BACKED_UP'


# ─────────────────────────────────────────────────────────────────────────────
class MarkerNavigator(Node):

    def __init__(self):
        super().__init__('marker_navigator')

        if not _CV:
            self.get_logger().fatal(
                'cv_bridge / opencv not available.\n'
                '  Install:  sudo apt install ros-humble-cv-bridge python3-opencv'
            )
            raise RuntimeError('cv_bridge missing')

        self._bridge = CvBridge()

        # ── ROS parameters ────────────────────────────────────────────────────
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self._id_left    = int(self.declare_parameter('marker_id_left',   101).value)
        self._id_right   = int(self.declare_parameter('marker_id_right',  102).value)
        self._msize      = float(self.declare_parameter('marker_size',    0.08).value)
        self._marker_z   = float(self.declare_parameter('marker_z',       0.05).value)
        self._approach_m      = float(self.declare_parameter('approach_m',      0.40).value)
        self._seek_dist       = float(self.declare_parameter('seek_dist_m',     0.22).value)
        self._skip_approach   = bool( self.declare_parameter('skip_approach',   False).value)
        self._fallback_back_m = float(self.declare_parameter('fallback_back_m', 0.25).value)
        self._map_frame  = self.declare_parameter('map_frame',  'odom').value   # real robot has no /map
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._scan_vel   = float(self.declare_parameter('scan_vel',  0.25).value)
        self._frame_skip = int(self.declare_parameter('frame_skip',    5).value)
        dict_name        = self.declare_parameter('aruco_dict',  'DICT_4X4_250').value
        img_topic        = self.declare_parameter('image_topic',
                                                  '/camera/color/image_raw').value
        info_topic       = self.declare_parameter('info_topic',
                                                  '/camera/color/camera_info').value
        calib_path       = self.declare_parameter(
            'camera_info_path',
            os.path.join(_script_dir, 'camera_info.yaml')).value

        # ── Camera calibration — YAML first, topic fallback ───────────────────
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._cam_frame: str = 'camera_optical_frame'
        self._frame_i = 0
        self._load_calibration(calib_path)

        # ── ArUco setup ───────────────────────────────────────────────────────
        a = cv2.aruco
        dict_id = getattr(a, dict_name, None)
        if dict_id is None:
            raise ValueError(f'Unknown aruco_dict: {dict_name}')
        try:
            self._adict = a.getPredefinedDictionary(dict_id)
        except AttributeError:
            self._adict = a.Dictionary_get(dict_id)

        if hasattr(a, 'ArucoDetector'):
            self._detector = a.ArucoDetector(self._adict, a.DetectorParameters())
            self._new_api  = True
        else:
            self._params  = a.DetectorParameters_create()
            self._new_api = False

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Marker state ──────────────────────────────────────────────────────
        self._pos_left:   Optional[Tuple[float, float]] = None
        self._pos_right:  Optional[Tuple[float, float]] = None
        self._quat_left:  Optional[Tuple[float, float, float, float]] = None
        self._quat_right: Optional[Tuple[float, float, float, float]] = None

        # ── Navigation state ──────────────────────────────────────────────────
        self._state            = S.SEARCHING
        self._target_x         = 0.0
        self._target_y         = 0.0
        self._target_yaw       = 0.0
        self._stop_t           = 0.0
        self._search_t         = time.monotonic()
        self._drive_back_start = 0.0
        self._fresh_both       = False   # True only when both markers seen in same frame
        self._fb_start_pos     = None    # odom (x,y) where odom-fallback drive-back began

        # ── Publishers ────────────────────────────────────────────────────────
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_left   = self.create_publisher(PoseStamped, '/aruco_101_pose', 10)
        self._pub_right  = self.create_publisher(PoseStamped, '/aruco_102_pose', 10)
        self._pub_done   = self.create_publisher(Bool, '/robot_positioned', latched)
        self._pub_backed = self.create_publisher(Bool, '/robot_backed_up',  latched)
        self._pub_vel    = self.create_publisher(Twist, '/mirte_base_controller/cmd_vel', 10)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(CameraInfo, info_topic,
                                 self._on_cam_info, qos_profile_sensor_data)
        self.create_subscription(Image, img_topic,
                                 self._on_image, qos_profile_sensor_data)
        self.create_subscription(Bool, '/arm_placed',
                                 self._on_arm_placed, 10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.05,             self._nav_tick)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_poses)

        # ── skip_approach: jump straight to DONE, publish /robot_positioned ──────
        if self._skip_approach:
            self._state = S.DONE
            self._pub_done.publish(Bool(data=True))

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  MarkerNavigator started.\n'
            f'  Markers     : IDs {self._id_left} (left) & {self._id_right} (right)\n'
            f'  Dict        : {dict_name}  size={self._msize} m\n'
            f'  Approach    : {self._approach_m * 100:.0f} cm from midpoint (base_link)\n'
            f'  Drive-back  : {self._seek_dist * 100:.0f} cm from midpoint (base_link)\n'
            f'  Fallback    : {self._fallback_back_m * 100:.0f} cm by odom if markers lost\n'
            f'  Map frame   : {self._map_frame}\n'
            f'  Calibration : {"loaded" if self._K is not None else "waiting for topic"}\n'
            f'\n'
            + (
            f'  *** SKIP-APPROACH MODE — robot_positioned already published ***\n'
            f'  *** Position robot manually, then start box_placer and      ***\n'
            f'  *** send /start_placing when ready.                         ***\n'
            if self._skip_approach else
            f'  Searching for markers...\n'
            ) +
            f'{"="*55}'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Camera calibration
    # ─────────────────────────────────────────────────────────────────────────

    def _load_calibration(self, path: str):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            self._K = np.array(
                data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
            self._D = np.array(
                data['distortion_coefficients']['data'], dtype=np.float64)
            name = data.get('camera_name', 'unknown')
            self.get_logger().info(
                f'Calibration loaded: {path}\n'
                f'  camera={name}  '
                f'fx={self._K[0,0]:.1f}  fy={self._K[1,1]:.1f}  '
                f'cx={self._K[0,2]:.1f}  cy={self._K[1,2]:.1f}'
            )
        except FileNotFoundError:
            self.get_logger().warn(
                f'Calibration file not found: {path}\n'
                f'  Falling back to camera_info topic.'
            )
        except Exception as e:
            self.get_logger().warn(
                f'Failed to load calibration: {e}\n'
                f'  Falling back to camera_info topic.'
            )

    def _on_cam_info(self, msg: CameraInfo):
        # Always update TF frame name — not stored in YAML
        if msg.header.frame_id and msg.header.frame_id != self._cam_frame:
            self.get_logger().info(
                f'Camera TF frame: "{self._cam_frame}" → "{msg.header.frame_id}"')
            self._cam_frame = msg.header.frame_id

        if self._K is not None:
            return   # already loaded from YAML

        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._D = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(
            f'Calibration from topic  frame="{self._cam_frame}"\n'
            f'  fx={self._K[0,0]:.1f}  fy={self._K[1,1]:.1f}  '
            f'cx={self._K[0,2]:.1f}  cy={self._K[1,2]:.1f}'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ArUco detection
    # ─────────────────────────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        self._frame_i += 1
        if self._frame_i % self._frame_skip != 0:
            return
        if self._K is None:
            self.get_logger().warn('No camera calibration yet — waiting.',
                                   throttle_duration_sec=5.0)
            return

        try:
            bgr  = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}',
                                   throttle_duration_sec=5.0)
            return

        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._adict, parameters=self._params)

        if ids is None or len(ids) == 0:
            return

        frame_ids = sorted(int(x) for x in ids.flatten())
        self.get_logger().info(
            f'Frame: detected IDs {frame_ids}',
            throttle_duration_sec=1.0)

        # Track whether BOTH markers appeared in this frame
        frame_has_left  = self._id_left  in frame_ids
        frame_has_right = self._id_right in frame_ids

        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in (self._id_left, self._id_right):
                continue

            try:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], self._msize, self._K, self._D)
            except Exception as e:
                self.get_logger().warn(f'Pose estimate failed for {mid}: {e}')
                continue

            rvec   = rvec[0]
            tvec   = tvec[0]
            dist_m = float(np.linalg.norm(tvec.flatten()))
            side   = 'left ' if mid == self._id_left else 'right'
            self.get_logger().info(
                f'  Marker {mid} ({side}) — dist: {dist_m:.3f} m  '
                f'cam_pos: [{tvec[0][0]:.3f}, {tvec[0][1]:.3f}, {tvec[0][2]:.3f}]'
            )

            pose_map = self._to_map_pose(rvec, tvec, msg.header.stamp)
            if pose_map is None:
                self.get_logger().warn(
                    f'Marker {mid}: TF camera→{self._map_frame} failed '
                    f'(cam="{self._cam_frame}")',
                    throttle_duration_sec=2.0)
                continue

            px = pose_map.pose.position.x
            py = pose_map.pose.position.y
            qt = (pose_map.pose.orientation.x, pose_map.pose.orientation.y,
                  pose_map.pose.orientation.z, pose_map.pose.orientation.w)

            if mid == self._id_left:
                first = self._pos_left is None
                self._pos_left  = self._ema(self._pos_left, px, py)
                self._quat_left = qt
                if first:
                    self.get_logger().info(
                        f'  ✓ Marker {mid} (left) locked  '
                        f'{self._map_frame}: ({px:.3f}, {py:.3f})  dist: {dist_m:.3f} m')
            else:
                first = self._pos_right is None
                self._pos_right  = self._ema(self._pos_right, px, py)
                self._quat_right = qt
                if first:
                    self.get_logger().info(
                        f'  ✓ Marker {mid} (right) locked  '
                        f'{self._map_frame}: ({px:.3f}, {py:.3f})  dist: {dist_m:.3f} m')

        # Signal that this frame had both markers → safe to recompute target
        if frame_has_left and frame_has_right:
            self._fresh_both = True

    # ─────────────────────────────────────────────────────────────────────────
    # Signal from box_placer
    # ─────────────────────────────────────────────────────────────────────────

    def _on_arm_placed(self, msg: Bool):
        if not msg.data:
            return
        if self._state != S.DONE:
            self.get_logger().warn(
                f'/arm_placed in state {self._state} — ignoring.')
            return
        self.get_logger().info(
            f'\n>>> /arm_placed — starting drive-back <<<\n'
            f'    Target: {self._seek_dist * 100:.0f} cm from midpoint\n'
            f'    Fallback: {self._fallback_back_m * 100:.0f} cm by odom if markers lost'
        )
        self._drive_back_start = time.monotonic()
        self._fb_start_pos     = None   # reset odom-fallback origin
        self._state = S.DRIVE_BACK

    # ─────────────────────────────────────────────────────────────────────────
    # TF helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _to_map_pose(self, rvec, tvec, stamp) -> Optional[PoseStamped]:
        if not self._cam_frame:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                self._map_frame, self._cam_frame,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.05))
        except Exception:
            return None

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z
        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w

        mc  = tvec.flatten().astype(np.float64)
        p_w = _quat_rotate(qx, qy, qz, qw, mc) + np.array([tx, ty, tz])

        R_mc, _ = cv2.Rodrigues(rvec.flatten())
        R_mw    = _quat_to_matrix(qx, qy, qz, qw)
        R_res   = R_mw @ R_mc
        ox, oy, oz, ow = _matrix_to_quat(R_res)

        p = PoseStamped()
        p.header.frame_id    = self._map_frame
        p.header.stamp       = stamp
        p.pose.position.x    = float(p_w[0])
        p.pose.position.y    = float(p_w[1])
        p.pose.position.z    = self._marker_z
        p.pose.orientation.x = float(ox)
        p.pose.orientation.y = float(oy)
        p.pose.orientation.z = float(oz)
        p.pose.orientation.w = float(ow)
        return p

    def _get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self._tf_buf.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.05))
        except Exception as e:
            self.get_logger().warn(
                f'Robot TF lookup failed: {e}', throttle_duration_sec=2.0)
            return None
        x   = tf.transform.translation.x
        y   = tf.transform.translation.y
        q   = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return x, y, yaw

    def _marker_midpoint_distance(self) -> Optional[float]:
        """2D distance from base_link to marker midpoint (odom frame)."""
        if self._pos_left is None or self._pos_right is None:
            return None
        mx = (self._pos_left[0]  + self._pos_right[0]) / 2.0
        my = (self._pos_left[1]  + self._pos_right[1]) / 2.0
        pose = self._get_robot_pose()
        if pose is None:
            return None
        rx, ry, _ = pose
        return math.sqrt((rx - mx) ** 2 + (ry - my) ** 2)

    # ─────────────────────────────────────────────────────────────────────────
    # EMA smoothing
    # ─────────────────────────────────────────────────────────────────────────

    def _ema(self, current, nx, ny):
        if current is None:
            return (nx, ny)
        ox, oy = current
        return (ox + EMA_ALPHA * (nx - ox), oy + EMA_ALPHA * (ny - oy))

    # ─────────────────────────────────────────────────────────────────────────
    # Target geometry
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _both_found(self) -> bool:
        return self._pos_left is not None and self._pos_right is not None

    def _compute_target(self):
        lx, ly = self._pos_left
        rx, ry = self._pos_right
        mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0

        dx, dy = rx - lx, ry - ly
        L = math.hypot(dx, dy)
        if L < 1e-6:
            self.get_logger().warn('Markers too close — cannot compute target.')
            return
        dx, dy = dx / L, dy / L

        pa = (-dy,  dx)
        pb = ( dy, -dx)
        pose = self._get_robot_pose()
        if pose is None:
            approach = pa
        else:
            rob_x, rob_y, _ = pose
            dot = pa[0] * (rob_x - mx) + pa[1] * (rob_y - my)
            approach = pa if dot > 0.0 else pb

        self._target_x   = mx + self._approach_m * approach[0]
        self._target_y   = my + self._approach_m * approach[1]
        self._target_yaw = math.atan2(-approach[1], -approach[0])

        self.get_logger().info(
            f'Target: ({self._target_x:.3f}, {self._target_y:.3f})  '
            f'yaw: {math.degrees(self._target_yaw):.1f}°  '
            f'sep: {L:.3f} m',
            throttle_duration_sec=2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # State machine  (20 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _nav_tick(self):
        s = self._state
        if   s == S.SEARCHING:  self._do_searching()
        elif s == S.DRIVE:      self._do_drive()
        elif s == S.STOP:       self._do_stop()
        elif s == S.ROTATE:     self._do_rotate()
        elif s == S.DONE:       self._pub_vel.publish(Twist())
        elif s == S.DRIVE_BACK: self._do_drive_back()
        elif s == S.BACKED_UP:  self._pub_vel.publish(Twist())

    def _do_searching(self):
        elapsed = time.monotonic() - self._search_t

        found = []
        if self._pos_left  is not None: found.append(str(self._id_left))
        if self._pos_right is not None: found.append(str(self._id_right))
        found_str = f'found {found},' if found else 'none found yet,'
        self.get_logger().info(
            f'Scanning...  {found_str}  {elapsed:.0f} s  '
            f'(want {self._id_left} & {self._id_right})',
            throttle_duration_sec=2.0)

        if elapsed > SCAN_TIMEOUT_S:
            self.get_logger().warn('Scan timeout — still searching.',
                                   throttle_duration_sec=10.0)
            self._search_t = time.monotonic()

        if self._both_found:
            self._pub_vel.publish(Twist())
            self._fresh_both = True
            self._compute_target()
            self._fresh_both = False
            self._state = S.DRIVE
            self.get_logger().info(
                f'Both markers found — driving to '
                f'({self._target_x:.3f}, {self._target_y:.3f})')
            return

        t = Twist()
        t.angular.z = self._scan_vel
        self._pub_vel.publish(t)

    def _do_drive(self):
        # Only recompute target when BOTH markers were freshly detected together.
        # If one marker disappears, keep the last good target so the robot
        # keeps heading toward the correct position rather than drifting.
        if self._pos_left is not None and self._pos_right is not None \
                and self._fresh_both:
            self._compute_target()
            self._fresh_both = False

        pose = self._get_robot_pose()
        if pose is None:
            self._pub_vel.publish(Twist())
            return

        rx, ry, ryaw = pose
        dx   = self._target_x - rx
        dy   = self._target_y - ry
        dist = math.hypot(dx, dy)

        self.get_logger().info(
            f'Driving: dist={dist*100:.1f} cm  '
            f'robot=({rx:.3f},{ry:.3f})  target=({self._target_x:.3f},{self._target_y:.3f})',
            throttle_duration_sec=1.0)

        if dist <= POS_TOL:
            self._pub_vel.publish(Twist())
            self._stop_t = time.monotonic()
            self._state  = S.STOP
            self.get_logger().info(f'Position reached (err={dist*100:.1f} cm). Settling...')
            return

        c, s = math.cos(ryaw), math.sin(ryaw)
        t = Twist()
        t.linear.x  = _clamp(KP_LIN * ( c*dx + s*dy), MAX_LIN)
        t.linear.y  = _clamp(KP_LIN * (-s*dx + c*dy), MAX_LIN)
        t.angular.z = _clamp(KP_ANG * _wrap(math.atan2(dy, dx) - ryaw), MAX_ANG)
        self._pub_vel.publish(t)

    def _do_stop(self):
        self._pub_vel.publish(Twist())
        if time.monotonic() - self._stop_t >= SETTLE_S:
            self.get_logger().info('Settled. Correcting yaw...')
            self._state = S.ROTATE

    def _do_rotate(self):
        # Do NOT re-compute target here — if one marker is lost the midpoint
        # drifts and causes the robot to over-rotate and end up too far away.
        # The target_yaw was locked correctly when we left _do_drive().
        pose = self._get_robot_pose()
        if pose is None:
            return
        _, _, ryaw = pose
        err = _wrap(self._target_yaw - ryaw)

        self.get_logger().info(
            f'Rotating: err={math.degrees(err):.1f}°  '
            f'target={math.degrees(self._target_yaw):.1f}°',
            throttle_duration_sec=1.0)

        if abs(err) <= YAW_TOL:
            self._pub_vel.publish(Twist())
            self._state = S.DONE
            self._pub_done.publish(Bool(data=True))
            dist = self._marker_midpoint_distance()
            bl   = f'{dist*100:.1f} cm'       if dist is not None else 'unknown'
            cam  = f'{(dist-0.15)*100:.1f} cm' if dist is not None else 'unknown'
            self.get_logger().info(
                f'\n{"="*55}\n'
                f'  Robot positioned!\n'
                f'  base_link → midpoint : {bl}\n'
                f'  camera    → midpoint : {cam}  (camera ~15 cm ahead)\n'
                f'  Yaw error : {math.degrees(err):.1f}°\n'
                f'\n'
                f'  ► Trigger box_placer:\n'
                f"    ros2 topic pub --once /start_placing "
                f"std_msgs/msg/Bool '{{data: true}}'\n"
                f'{"="*55}'
            )
            return

        t = Twist()
        t.angular.z = _clamp(KP_ANG * err, MAX_ANG)
        self._pub_vel.publish(t)

    def _do_drive_back(self):
        dist    = self._marker_midpoint_distance()
        elapsed = time.monotonic() - self._drive_back_start

        if dist is None:
            # ── Odom-fallback: drive back a hardcoded distance ────────────────
            pose = self._get_robot_pose()
            if pose is None:
                # No odom either — last resort: creep until timeout
                if elapsed > SEEK_TIMEOUT_S:
                    self.get_logger().warn('Drive-back: no markers, no odom, timeout — finishing.')
                    self._pub_vel.publish(Twist())
                    self._finish_drive_back()
                else:
                    t = Twist()
                    t.linear.x = -SEEK_MIN_VEL
                    self._pub_vel.publish(t)
                return

            rx, ry, _ = pose
            if self._fb_start_pos is None:
                self._fb_start_pos = (rx, ry)
                self.get_logger().warn(
                    f'Drive-back: markers lost — odom fallback '
                    f'({self._fallback_back_m * 100:.0f} cm)')

            driven = math.hypot(rx - self._fb_start_pos[0],
                                ry - self._fb_start_pos[1])
            self.get_logger().info(
                f'Drive-back fallback: {driven*100:.1f} / '
                f'{self._fallback_back_m*100:.0f} cm  (odom)',
                throttle_duration_sec=0.5)

            if driven >= self._fallback_back_m:
                self.get_logger().warn(
                    f'Drive-back fallback done: moved {driven*100:.1f} cm.')
                self._pub_vel.publish(Twist())
                self._finish_drive_back()
                return

            t = Twist()
            t.linear.x = -SEEK_MIN_VEL
            self._pub_vel.publish(t)
            return

        error = dist - self._seek_dist

        self.get_logger().info(
            f'Drive-back: dist={dist*100:.1f} cm  '
            f'target={self._seek_dist*100:.0f} cm  '
            f'err={error*100:+.1f} cm',
            throttle_duration_sec=0.25)

        if abs(error) <= SEEK_TOL_M:
            self._pub_vel.publish(Twist())
            self.get_logger().info(
                f'Drive-back done: {dist*100:.1f} cm from markers.')
            self._finish_drive_back()
            return

        vel = SEEK_KP * error
        vel = max(-SEEK_MAX_VEL, min(SEEK_MAX_VEL, vel))
        if 0.0 < abs(vel) < SEEK_MIN_VEL:
            vel = math.copysign(SEEK_MIN_VEL, vel)

        t = Twist()
        t.linear.x = vel
        self._pub_vel.publish(t)

    def _finish_drive_back(self):
        self._pub_backed.publish(Bool(data=True))
        self._state = S.BACKED_UP
        self.get_logger().info(
            '\n>>> /robot_backed_up published <<<\n'
            '    box_placer will open gripper and return home.'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Continuous pose publisher  (5 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_poses(self):
        now = self.get_clock().now().to_msg()
        if self._pos_left is not None and self._quat_left is not None:
            self._pub_left.publish(
                self._make_pose(self._pos_left, self._quat_left, now))
        if self._pos_right is not None and self._quat_right is not None:
            self._pub_right.publish(
                self._make_pose(self._pos_right, self._quat_right, now))

    def _make_pose(self, xy, quat, stamp) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id    = self._map_frame
        p.header.stamp       = stamp
        p.pose.position.x    = float(xy[0])
        p.pose.position.y    = float(xy[1])
        p.pose.position.z    = self._marker_z   # known physical height
        p.pose.orientation.x = float(quat[0])
        p.pose.orientation.y = float(quat[1])
        p.pose.orientation.z = float(quat[2])
        p.pose.orientation.w = float(quat[3])
        return p


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MarkerNavigator()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
