#!/usr/bin/env python3
"""
Zone detector — identifies Zone A (red tape) and Zone B (blue tape) from the
Orbbec RGB camera during explore_lite exploration.

Each detected tape pixel's centroid is projected onto the ground plane (z=0 in map
frame) using the full TF chain and the pinhole camera model.  Zone centres are smoothed
with an exponential moving average so a handful of noisy frames don't distort the
estimate.

Publishes:
  /zone_a_pose  (geometry_msgs/PoseStamped, frame: map) — red-tape zone centre
  /zone_b_pose  (geometry_msgs/PoseStamped, frame: map) — blue-tape zone centre
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time

import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

try:
    from cv_bridge import CvBridge
    import cv2
    _CV = True
except ImportError:
    _CV = False


# ── HSV colour ranges (OpenCV convention: H 0-179, S/V 0-255) ────────────────
# Red tape: ambient (0.85, 0.10, 0.10) → BGR (26, 26, 217) → H≈0, S≈224, V≈217
RED_LOWER1 = np.array([  0, 100,  80], dtype=np.uint8)
RED_UPPER1 = np.array([ 15, 255, 255], dtype=np.uint8)
RED_LOWER2 = np.array([160, 100,  80], dtype=np.uint8)
RED_UPPER2 = np.array([179, 255, 255], dtype=np.uint8)

# Blue tape: ambient (0.10, 0.20, 0.85) → BGR (217, 51, 26) → H≈116, S≈224, V≈217
BLUE_LOWER = np.array([ 95, 100,  80], dtype=np.uint8)
BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# Minimum blob pixel area to be considered a real tape detection
MIN_BLOB_PIXELS = 50

# EMA smoothing factor — lower = more smoothing, slower convergence
EMA_ALPHA = 0.15

# How far above floor the tape surface is (z = 0 = floor, tape is 3 mm thick)
TAPE_Z = 0.003

# Re-publish detected zones at this rate even when no new detection arrives
PUBLISH_RATE_HZ = 2.0


def _quat_rotate(qx, qy, qz, qw, v):
    """Rotate 3-vector v by quaternion (qx, qy, qz, qw)."""
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


class ZoneDetector(Node):

    def __init__(self):
        super().__init__('zone_detector')

        if not _CV:
            self.get_logger().error(
                'cv_bridge / opencv-python not found — zone detection disabled.')

        self._bridge  = CvBridge() if _CV else None

        # Camera intrinsics (filled from camera_info)
        self._fx = self._fy = self._ppx = self._ppy = None
        self._cam_frame: str = ''

        # TF
        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # Smoothed zone centres (None until first detection)
        self._zone_a: tuple | None = None   # (cx, cy) in map frame
        self._zone_b: tuple | None = None

        # Publishers
        self._pub_a = self.create_publisher(PoseStamped, '/zone_a_pose', 10)
        self._pub_b = self.create_publisher(PoseStamped, '/zone_b_pose', 10)

        # Subscribers
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self._camera_info_cb, 10)
        self.create_subscription(Image, '/camera/image_raw',
                                 self._image_cb, 10)

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_zones)

        self.get_logger().info('Zone detector started — watching for red and blue tape.')

    # ── Camera intrinsics ─────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self._fx is not None:
            return  # only need to read once
        k = msg.k          # row-major 3×3 intrinsic matrix
        self._fx  = k[0]
        self._fy  = k[4]
        self._ppx = k[2]
        self._ppy = k[5]
        self._cam_frame = msg.header.frame_id
        self.get_logger().info(
            f'Camera intrinsics: fx={self._fx:.1f} fy={self._fy:.1f} '
            f'pp=({self._ppx:.1f},{self._ppy:.1f}) frame="{self._cam_frame}"')

    # ── Image processing ──────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        if not _CV or self._bridge is None:
            return
        if self._fx is None:
            return  # no intrinsics yet

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge error: {exc}', throttle_duration_sec=5.0)
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Red mask (wraps around H=0)
        red_mask = (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
                    | cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))

        # Blue mask
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        for mask, color in ((red_mask, 'a'), (blue_mask, 'b')):
            pt = self._blob_centroid(mask)
            if pt is None:
                continue
            u, v = pt
            world_xy = self._project_to_ground(u, v, msg.header.stamp)
            if world_xy is None:
                continue
            cx, cy = world_xy
            if color == 'a':
                self._zone_a = self._ema(self._zone_a, cx, cy)
            else:
                self._zone_b = self._ema(self._zone_b, cx, cy)

    def _blob_centroid(self, mask):
        """Return (u, v) centroid of the largest blob in mask, or None."""
        nnz = int(np.count_nonzero(mask))
        if nnz < MIN_BLOB_PIXELS:
            return None
        # Find largest connected component
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return None
        # Skip label 0 (background)
        best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        if stats[best, cv2.CC_STAT_AREA] < MIN_BLOB_PIXELS:
            return None
        return float(centroids[best][0]), float(centroids[best][1])

    def _project_to_ground(self, u, v, stamp):
        """
        Project image pixel (u, v) to the ground plane (z = TAPE_Z in map frame).
        Returns (cx, cy) in map frame, or None if ray doesn't reach the ground.
        """
        if not self._cam_frame:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                'map', self._cam_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except Exception:
            return None

        # Camera position in map frame
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z

        # Ray direction in camera optical frame (z forward, x right, y down)
        ray_cam = np.array([
            (u - self._ppx) / self._fx,
            (v - self._ppy) / self._fy,
            1.0,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate ray into map frame
        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        ray_world = _quat_rotate(qx, qy, qz, qw, ray_cam)

        # Intersect ray with z = TAPE_Z plane
        # P = (tx, ty, tz) + t * ray_world,  P.z = TAPE_Z
        # t = (TAPE_Z - tz) / ray_world[2]
        if abs(ray_world[2]) < 1e-6:
            return None
        t = (TAPE_Z - tz) / ray_world[2]
        if t <= 0:
            return None  # intersection is behind the camera

        gx = tx + t * ray_world[0]
        gy = ty + t * ray_world[1]

        # Sanity: projected point must be at a plausible distance (<10 m)
        dist = math.hypot(gx - tx, gy - ty)
        if dist > 10.0:
            return None

        return gx, gy

    def _ema(self, current, cx, cy):
        if current is None:
            return (cx, cy)
        ox, oy = current
        return (ox + EMA_ALPHA * (cx - ox), oy + EMA_ALPHA * (cy - oy))

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_zones(self):
        now = self.get_clock().now().to_msg()
        if self._zone_a is not None:
            self._pub_a.publish(self._make_pose(self._zone_a, now))
        if self._zone_b is not None:
            self._pub_b.publish(self._make_pose(self._zone_b, now))

    def _make_pose(self, xy, stamp):
        p = PoseStamped()
        p.header.frame_id    = 'map'
        p.header.stamp       = stamp
        p.pose.position.x    = xy[0]
        p.pose.position.y    = xy[1]
        p.pose.position.z    = 0.0
        p.pose.orientation.w = 1.0
        return p


def main(args=None):
    rclpy.init(args=args)
    node = ZoneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
