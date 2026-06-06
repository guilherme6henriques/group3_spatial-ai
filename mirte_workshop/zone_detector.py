#!/usr/bin/env python3
"""
Zone detector — locates Zone A and Zone B using ArUco markers.

  Marker ID=0  DICT_4X4_50  →  Zone A pole entrance  →  /zone_a_pose
  Marker ID=1  DICT_4X4_50  →  Zone B stand centre   →  /zone_b_pose
  Any other ID              →  ignored

This node detects ONLY the two zone markers (A and B).  The pickup boxes are
unmarked — they are not this node's concern.

Each detection:
  1. estimatePoseSingleMarkers gives tvec/rvec in camera frame using the
     physical zone-marker size.
  2. TF transforms the marker position and orientation to map frame.
  3. Position is EMA-smoothed; orientation is taken from the most recent frame.
  4. The PoseStamped orientation encodes the marker's facing direction (+Z in
     its own frame) so navigation can compute approach waypoints.
  5. All poses are re-published at PUBLISH_RATE_HZ even between detections.

Requires:
  cv2.aruco (OpenCV 4.5.4 old API — Dictionary_get / DetectorParameters_create)
  /camera/camera_info  →  fills camera_matrix and dist_coeffs
  /camera/image_raw    →  BGR8 image for detection
  TF: map ← ... ← camera_optical_frame
"""

import math
import numpy as np

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
ZONE_MARKER_IDS  = frozenset({0, 1})
ZONE_MARKER_SIZE = 0.20   # physical side length in metres — ID 0, 1

# EMA smoothing factor for position (lower = smoother, slower to converge)
EMA_ALPHA = 0.20

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

        # Marker config as ROS params so the SAME node works in sim (ids 0/1,
        # DICT_4X4_50) and on the real robot (e.g. ids 104/100, DICT_4X4_250).
        dict_name = self.declare_parameter('aruco_dict', 'DICT_4X4_50').value
        self._a_id = int(self.declare_parameter('zone_a_id', 0).value)
        self._b_id = int(self.declare_parameter('zone_b_id', 1).value)
        self._marker_size = float(
            self.declare_parameter('zone_marker_size', ZONE_MARKER_SIZE).value)

        a = cv2.aruco
        dict_id = getattr(a, dict_name)
        try:
            self._aruco_dict = a.getPredefinedDictionary(dict_id)   # works on 4.5 & 4.7+
        except AttributeError:
            self._aruco_dict = a.Dictionary_get(dict_id)
        if hasattr(a, 'ArucoDetector'):            # OpenCV >= 4.7 (new API)
            self._detector = a.ArucoDetector(self._aruco_dict, a.DetectorParameters())
            self._new_aruco_api = True
        else:                                      # OpenCV 4.5 / 4.6 (old API)
            self._aruco_params = a.DetectorParameters_create()
            self._new_aruco_api = False
        self.get_logger().info(
            f'ArUco dict={dict_name}, Zone A=id{self._a_id}, Zone B=id{self._b_id}, '
            f'size={self._marker_size} m, api={"new" if self._new_aruco_api else "old"}')

        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs:   np.ndarray | None = None
        self._cam_frame: str = ''

        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # Zone poses
        self._zone_a_xy:   tuple | None = None
        self._zone_b_xy:   tuple | None = None
        self._zone_a_quat: tuple | None = None
        self._zone_b_quat: tuple | None = None

        # Publishers
        self._pub_a = self.create_publisher(PoseStamped, '/zone_a_pose', 10)
        self._pub_b = self.create_publisher(PoseStamped, '/zone_b_pose', 10)

        # Cameras publish with SensorData QoS (BEST_EFFORT); a default RELIABLE
        # subscriber silently gets nothing from them (esp. on the real robot),
        # so subscribe with sensor QoS to match any camera.
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self._camera_info_cb, qos_profile_sensor_data)
        # Decode/detect on only every Nth frame.  The orbbec streams ~30 Hz, but
        # ArUco decode of a 640x480 frame is costly and the SBC also runs SLAM +
        # Nav2; processing every frame starves Nav2 (its lifecycle activation
        # times out).  ~6 Hz is ample to catch a marker during a slow spin.
        self._frame_skip = int(self.declare_parameter('frame_skip', 5).value)
        self._frame_i = 0
        # Subscribing to the raw 30 Hz image and deserializing every frame costs
        # ~40% of a CPU core even when we only decode every Nth.  The compressed
        # (JPEG) stream is ~20x smaller, so receiving it is cheap and we only
        # cv2.imdecode the frames we actually process.  use_compressed:=true on
        # the real robot; sim publishes raw, so it defaults false.
        self._use_compressed = bool(self.declare_parameter('use_compressed', False).value)
        if self._use_compressed:
            self.create_subscription(CompressedImage, '/camera/image_raw/compressed',
                                     self._image_cb_compressed, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, '/camera/image_raw',
                                     self._image_cb, qos_profile_sensor_data)

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_zones)

        self.get_logger().info(
            f'Zone detector started — Zone A=id{self._a_id}, Zone B=id{self._b_id}; '
            f'any other marker ID is ignored (boxes are unmarked).')

    # ── Camera intrinsics ─────────────────────────────────────────────────────

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

    def _process(self, gray, stamp):
        if self._new_aruco_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._aruco_params)

        if ids is None or len(ids) == 0:
            return

        # DEBUG: what IDs is the camera actually seeing?  Tells us at a glance
        # whether a "looked at but not found" marker is simply a different ID
        # than zone_a_id/zone_b_id (or is being misread by the wrong dictionary).
        self.get_logger().info(
            f'ArUco detected IDs={sorted(int(x) for x in ids.flatten())} '
            f'(want A={self._a_id}, B={self._b_id})',
            throttle_duration_sec=1.0)

        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            # Only the two ZONE markers are handled here; ignore any other ID.
            if marker_id not in (self._a_id, self._b_id):
                continue
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], self._marker_size, self._camera_matrix, self._dist_coeffs)
            rvec, tvec = rvec[0], tvec[0]   # unwrap batch dimension

            pose_map = self._to_map_pose(rvec, tvec, stamp)
            if pose_map is None:
                # Detected a zone marker but couldn't place it in map — TF gap.
                self.get_logger().warn(
                    f'Marker {marker_id} seen but map transform failed '
                    f'(cam frame "{self._cam_frame}").', throttle_duration_sec=2.0)
                continue

            px = pose_map.pose.position.x
            py = pose_map.pose.position.y
            qx = pose_map.pose.orientation.x
            qy = pose_map.pose.orientation.y
            qz = pose_map.pose.orientation.z
            qw = pose_map.pose.orientation.w

            if marker_id == self._a_id:
                self._zone_a_xy   = self._ema(self._zone_a_xy, px, py)
                self._zone_a_quat = (qx, qy, qz, qw)
                self.get_logger().info(
                    f'Zone A marker detected → map ({px:.2f}, {py:.2f})',
                    throttle_duration_sec=2.0)
            elif marker_id == self._b_id:
                self._zone_b_xy   = self._ema(self._zone_b_xy, px, py)
                self._zone_b_quat = (qx, qy, qz, qw)
                self.get_logger().info(
                    f'Zone B marker detected → map ({px:.2f}, {py:.2f})',
                    throttle_duration_sec=2.0)

    # ── Coordinate transform ──────────────────────────────────────────────────

    def _to_map_pose(self, rvec, tvec, stamp) -> PoseStamped | None:
        if not self._cam_frame:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                'map', self._cam_frame,
                rclpy.time.Time(),
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

    # ── EMA smoothing ─────────────────────────────────────────────────────────

    def _ema(self, current, cx, cy):
        if current is None:
            return (cx, cy)
        ox, oy = current
        return (ox + EMA_ALPHA * (cx - ox), oy + EMA_ALPHA * (cy - oy))

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_zones(self):
        now = self.get_clock().now().to_msg()
        if self._zone_a_xy is not None and self._zone_a_quat is not None:
            self._pub_a.publish(
                self._make_pose(self._zone_a_xy, self._zone_a_quat, now))
        if self._zone_b_xy is not None and self._zone_b_quat is not None:
            self._pub_b.publish(
                self._make_pose(self._zone_b_xy, self._zone_b_quat, now))

    def _make_pose(self, xy, quat, stamp) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id    = 'map'
        p.header.stamp       = stamp
        p.pose.position.x    = xy[0]
        p.pose.position.y    = xy[1]
        p.pose.position.z    = 0.0
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        return p


def main(args=None):
    rclpy.init(args=args)
    node = ZoneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
