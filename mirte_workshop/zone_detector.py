#!/usr/bin/env python3
"""
Zone detector — locates Zone A (one marker) and Zone B (the midpoint of TWO
markers) using ArUco, for the A↔B shuttle.

  Zone A  : one marker (zone_a_id)                  →  /zone_a_pose
  Zone B  : midpoint of two markers (zone_b_left_id, →  /zone_b_pose
            zone_b_right_id) — the precision team's
            placement stand
  Any other ID                                       →  ignored

Why B is two markers: Zone B is the precision-team's placement stand, which
carries a marker pair (e.g. 101 left / 102 right).  We publish their MIDPOINT as
/zone_b_pose so the shuttle navigates straight to the stand and stops there; the
precision dock (marker_navigator) then does the cm-level alignment between the
two markers.  (The old single Zone-B marker is gone — erase it from the arena.)

Each detection:
  1. estimatePoseSingleMarkers gives tvec/rvec in camera frame using the
     physical marker size.
  2. TF transforms the marker position into the map frame.
  3. Each marker's map position is EMA-smoothed with outlier rejection.
  4. Zone A pose = its marker; Zone B pose = midpoint of the two B markers
     (orientation carried from one of them — the shuttle computes its own
     facing yaw, so B's orientation is not critical).
  5. Poses are re-published at PUBLISH_RATE_HZ even between detections.

Requires:
  cv2.aruco (OpenCV 4.5.4 old API — Dictionary_get / DetectorParameters_create)
  /camera/camera_info  →  fills camera_matrix and dist_coeffs
  /camera/image_raw    →  BGR8 image for detection
  TF: map ← ... ← camera_optical_frame
"""

import math
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time

import tf2_ros
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped

try:
    from cv_bridge import CvBridge
    import cv2
    _CV = True
except ImportError:
    _CV = False

# ── ArUco config ──────────────────────────────────────────────────────────────
MARKER_DICT      = cv2.aruco.DICT_4X4_50 if _CV else None
ZONE_MARKER_SIZE = 0.08   # physical side length in metres

# EMA smoothing factor for position (lower = smoother, slower to converge)
EMA_ALPHA = 0.20

# Outlier rejection: a single detection that lands more than JUMP_THRESH_M from
# the running estimate is ignored (small marker seen far away / while spinning
# gives noisy poses, and SLAM drift bounces the map position).  Only after
# JUMP_PERSIST consecutive far readings do we accept it (re-seed).
JUMP_THRESH_M = 0.6
JUMP_PERSIST  = 5

# Re-publish rate even when no new detection arrives
PUBLISH_RATE_HZ = 5.0


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_rotate(qx, qy, qz, qw, v):
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


def _quat_to_matrix(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return x/n, y/n, z/n, w/n


class ZoneDetector(Node):

    def __init__(self):
        super().__init__('zone_detector')

        if not _CV:
            self.get_logger().error(
                'cv_bridge / opencv not found — zone detection disabled.')
            return

        self._bridge = CvBridge()

        # Marker config as ROS params so the SAME node works in sim and on the
        # real robot.  Zone A is one marker; Zone B is a pair (left/right) whose
        # midpoint is the stand centre.
        dict_name = self.declare_parameter('aruco_dict', 'DICT_4X4_50').value
        self._a_id  = int(self.declare_parameter('zone_a_id',        0).value)
        self._bl_id = int(self.declare_parameter('zone_b_left_id',   1).value)
        self._br_id = int(self.declare_parameter('zone_b_right_id',  2).value)
        self._marker_size = float(
            self.declare_parameter('zone_marker_size', ZONE_MARKER_SIZE).value)

        a = cv2.aruco
        dict_id = getattr(a, dict_name)
        try:
            self._aruco_dict = a.getPredefinedDictionary(dict_id)   # 4.5 & 4.7+
        except AttributeError:
            self._aruco_dict = a.Dictionary_get(dict_id)
        if hasattr(a, 'ArucoDetector'):            # OpenCV >= 4.7 (new API)
            self._detector = a.ArucoDetector(self._aruco_dict, a.DetectorParameters())
            self._new_aruco_api = True
        else:                                      # OpenCV 4.5 / 4.6 (old API)
            self._aruco_params = a.DetectorParameters_create()
            self._new_aruco_api = False
        self.get_logger().info(
            f'ArUco dict={dict_name}, Zone A=id{self._a_id}, '
            f'Zone B=midpoint(id{self._bl_id}, id{self._br_id}), '
            f'size={self._marker_size} m, api={"new" if self._new_aruco_api else "old"}')

        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs:   np.ndarray | None = None
        # Camera optical frame for the marker→map transform.  Taken from a param so
        # the detector still works when the camera_info TOPIC isn't reaching us
        # (it's what normally carries the frame); the topic updates it if it comes.
        self._cam_frame: str = str(self.declare_parameter(
            'camera_frame', 'camera_color_optical_frame').value)
        # Optional: load intrinsics from a camera_info.yaml so we DON'T have to
        # wait for the /camera_info topic (which sometimes doesn't cross wifi).
        # If set + loadable, the topic becomes a non-blocking fallback.
        _cal_path = str(self.declare_parameter('camera_info_path', '').value)
        if _cal_path:
            self._load_calibration(_cal_path)

        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # Per-marker tracks: id -> {'xy': (x,y)|None, 'quat': (..)|None, 'far': int}
        self._track = {mid: {'xy': None, 'quat': None, 'far': 0}
                       for mid in (self._a_id, self._bl_id, self._br_id)}

        # Publishers
        self._pub_a = self.create_publisher(PoseStamped, '/zone_a_pose', 10)
        self._pub_b = self.create_publisher(PoseStamped, '/zone_b_pose', 10)

        # Cameras publish with SensorData QoS (BEST_EFFORT); subscribe with sensor
        # QoS or a default RELIABLE subscriber silently gets nothing.
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self._camera_info_cb, qos_profile_sensor_data)
        self._frame_skip = int(self.declare_parameter('frame_skip', 5).value)
        self._frame_i = 0
        self._use_compressed = bool(self.declare_parameter('use_compressed', False).value)
        if self._use_compressed:
            self.create_subscription(CompressedImage, '/camera/image_raw/compressed',
                                     self._image_cb_compressed, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, '/camera/image_raw',
                                     self._image_cb, qos_profile_sensor_data)

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_zones)

        self.get_logger().info(
            f'Zone detector started — Zone A=id{self._a_id}, '
            f'Zone B=midpoint(id{self._bl_id},id{self._br_id}); other IDs ignored.')

    # ── Camera intrinsics ─────────────────────────────────────────────────────

    def _load_calibration(self, path: str) -> bool:
        """Load camera_matrix + distortion from a ROS camera_info YAML, so the
        detector doesn't depend on the /camera_info topic crossing the network."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            self._camera_matrix = np.array(
                data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
            self._dist_coeffs = np.array(
                data['distortion_coefficients']['data'], dtype=np.float64)
            self.get_logger().info(
                f'Camera intrinsics loaded from FILE {path}: '
                f'fx={self._camera_matrix[0,0]:.1f} fy={self._camera_matrix[1,1]:.1f} '
                f'frame="{self._cam_frame}"')
            return True
        except FileNotFoundError:
            self.get_logger().warn(
                f'camera_info_path not found: {path} — waiting for /camera_info topic.')
        except Exception as e:
            self.get_logger().warn(
                f'Failed to load {path}: {e} — waiting for /camera_info topic.')
        return False

    def _camera_info_cb(self, msg: CameraInfo):
        if self._camera_matrix is not None:
            return
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64)
        self._camera_matrix = k
        self._dist_coeffs   = d
        self._cam_frame     = msg.header.frame_id
        self.get_logger().info(
            f'Camera intrinsics loaded: fx={k[0,0]:.1f} fy={k[1,1]:.1f} '
            f'frame="{self._cam_frame}"')

    # ── Image processing ──────────────────────────────────────────────────────

    def _should_process(self) -> bool:
        if self._camera_matrix is None:
            return False
        self._frame_i += 1
        return self._frame_i % self._frame_skip == 0   # process every Nth frame

    def _image_cb(self, msg: Image):
        if not self._should_process():
            return
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge: {exc}', throttle_duration_sec=5.0)
            return
        self._process(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), msg.header.stamp)

    def _image_cb_compressed(self, msg: CompressedImage):
        if not self._should_process():
            return
        try:
            bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            self.get_logger().warn(f'imdecode: {exc}', throttle_duration_sec=5.0)
            return
        if bgr is None:
            return
        self._process(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), msg.header.stamp)

    def _label(self, mid) -> str:
        if mid == self._a_id:
            return 'A'
        if mid == self._bl_id:
            return 'B-left'
        return 'B-right'

    def _process(self, gray, stamp):
        if self._new_aruco_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._aruco_params)

        if ids is None or len(ids) == 0:
            return

        self.get_logger().info(
            f'ArUco detected IDs={sorted(int(x) for x in ids.flatten())} '
            f'(want A={self._a_id}, B={self._bl_id}/{self._br_id})',
            throttle_duration_sec=1.0)

        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self._track:
                continue
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], self._marker_size, self._camera_matrix, self._dist_coeffs)
            rvec, tvec = rvec[0], tvec[0]

            pose_map = self._to_map_pose(rvec, tvec, stamp)
            if pose_map is None:
                self.get_logger().warn(
                    f'Marker {marker_id} seen but map transform failed '
                    f'(cam frame "{self._cam_frame}").', throttle_duration_sec=2.0)
                continue

            px = pose_map.pose.position.x
            py = pose_map.pose.position.y
            quat = (pose_map.pose.orientation.x, pose_map.pose.orientation.y,
                    pose_map.pose.orientation.z, pose_map.pose.orientation.w)

            if self._accept_xy(marker_id, px, py):     # reject outlier jumps
                self._track[marker_id]['quat'] = quat
                xy = self._track[marker_id]['xy']
                self.get_logger().info(
                    f'{self._label(marker_id)} (id{marker_id}) → map '
                    f'({xy[0]:.2f}, {xy[1]:.2f})', throttle_duration_sec=2.0)

    # ── Coordinate transform ──────────────────────────────────────────────────

    def _to_map_pose(self, rvec, tvec, stamp) -> PoseStamped | None:
        if not self._cam_frame:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                'map', self._cam_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except Exception:
            return None

        ctx = tf.transform.translation.x
        cty = tf.transform.translation.y
        ctz = tf.transform.translation.z
        cqx = tf.transform.rotation.x
        cqy = tf.transform.rotation.y
        cqz = tf.transform.rotation.z
        cqw = tf.transform.rotation.w

        mc  = tvec.flatten().astype(np.float64)
        p_w = _quat_rotate(cqx, cqy, cqz, cqw, mc) + np.array([ctx, cty, ctz])

        R_marker_cam, _ = cv2.Rodrigues(rvec.flatten())
        R_cam_map       = _quat_to_matrix(cqx, cqy, cqz, cqw)
        R_marker_map    = R_cam_map @ R_marker_cam
        qx, qy, qz, qw  = _matrix_to_quat(R_marker_map)

        p = PoseStamped()
        p.header.frame_id    = 'map'
        p.header.stamp       = stamp
        p.pose.position.x    = float(p_w[0])
        p.pose.position.y    = float(p_w[1])
        p.pose.position.z    = float(p_w[2])
        p.pose.orientation.x = float(qx)
        p.pose.orientation.y = float(qy)
        p.pose.orientation.z = float(qz)
        p.pose.orientation.w = float(qw)
        return p

    # ── EMA smoothing + outlier rejection (per marker id) ───────────────────────

    def _accept_xy(self, mid, px, py) -> bool:
        """Update marker `mid`'s smoothed (x,y) with outlier rejection.  Returns
        True if the reading was accepted.  A reading >JUMP_THRESH_M from the
        running estimate is rejected as noise unless JUMP_PERSIST consecutive far
        readings arrive (then re-seed)."""
        tr  = self._track[mid]
        cur = tr['xy']
        if cur is None:
            tr['xy'], tr['far'] = (px, py), 0          # first lock
            return True
        if math.hypot(px - cur[0], py - cur[1]) > JUMP_THRESH_M:
            tr['far'] += 1
            if tr['far'] >= JUMP_PERSIST:
                tr['xy'], tr['far'] = (px, py), 0      # persistent → re-seed
                return True
            return False                                # transient → reject
        tr['xy'] = (cur[0] + EMA_ALPHA * (px - cur[0]),
                    cur[1] + EMA_ALPHA * (py - cur[1]))
        tr['far'] = 0
        return True

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_zones(self):
        now = self.get_clock().now().to_msg()

        ta = self._track[self._a_id]
        if ta['xy'] is not None and ta['quat'] is not None:
            self._pub_a.publish(self._make_pose(ta['xy'], ta['quat'], now))

        # Zone B = midpoint of the two B markers (only once BOTH are located).
        tl = self._track[self._bl_id]
        tr = self._track[self._br_id]
        if tl['xy'] is not None and tr['xy'] is not None:
            mid_xy = ((tl['xy'][0] + tr['xy'][0]) / 2.0,
                      (tl['xy'][1] + tr['xy'][1]) / 2.0)
            quat = tl['quat'] or tr['quat'] or (0.0, 0.0, 0.0, 1.0)
            self._pub_b.publish(self._make_pose(mid_xy, quat, now))

    def _make_pose(self, xy, quat, stamp) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id    = 'map'
        p.header.stamp       = stamp
        p.pose.position.x    = float(xy[0])
        p.pose.position.y    = float(xy[1])
        p.pose.position.z    = 0.0
        p.pose.orientation.x = float(quat[0])
        p.pose.orientation.y = float(quat[1])
        p.pose.orientation.z = float(quat[2])
        p.pose.orientation.w = float(quat[3])
        return p


def main(args=None):
    rclpy.init(args=args)
    node = ZoneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
