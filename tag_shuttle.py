#!/usr/bin/env python3
"""
tag_shuttle.py — minimal REAL-MIRTE test: shuttle between two ArUco tags.

Finds tag A (id 104) and tag B (id 100), drives to A, then to B, repeating
N_ROUND_TRIPS times.  Purely REACTIVE: spin in place to find the target tag,
steer to centre it in the image, drive forward until it looks close, stop,
then go for the next tag.

  NO map, NO Nav2, NO SLAM, NO obstacle avoidance.
  → Run it in an OPEN space and keep a hand on Ctrl-C.

WHERE TO RUN: on the ROBOT itself (its camera + motors are there), from the
MIRTE web VSCode terminal:

    source /opt/ros/humble/setup.bash        # + your robot workspace if any
    ros2 topic list                          # confirm the topic names below
    python3 tag_shuttle.py

The two settings that most commonly need changing for your robot are
IMAGE_TOPIC, CMDVEL_TOPIC and ARUCO_DICT — see CONFIG.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np
import cv2
from cv_bridge import CvBridge

# ════════════════ CONFIG — EDIT FOR YOUR ROBOT ════════════════
# Find these with `ros2 topic list` on the robot.
IMAGE_TOPIC  = '/camera/image_raw'                       # RGB image (maybe /camera/color/image_raw)
CMDVEL_TOPIC = '/mirte_base_controller/cmd_vel_unstamped'  # maybe just '/cmd_vel'

# Which printed marker is which zone.
TAG_A_ID = 104
TAG_B_ID = 100

# IMPORTANT: this MUST match the dictionary you generated on the website.
# IDs 100 and 104 only exist in a 250- or 1000-marker dictionary — they are
# NOT in any *_50 or *_100 dict.  Common matches:
#   DICT_4X4_250, DICT_5X5_250, DICT_6X6_250, DICT_4X4_1000, ...
ARUCO_DICT = cv2.aruco.DICT_4X4_250

N_ROUND_TRIPS = 3        # go (A then B) this many times

# Motion (start gentle on a real robot)
SEARCH_ANGULAR = 0.40    # rad/s, spin speed while searching for a tag
DRIVE_LINEAR   = 0.12    # m/s, forward speed toward a tag
STEER_GAIN     = 1.2     # turn effort to centre the tag (×normalised error)
MAX_ANGULAR    = 0.6     # rad/s clamp
CLOSE_FRAC     = 0.35    # "arrived" when marker width > this fraction of image width
LOST_TIMEOUT   = 1.0     # s without a fresh detection → go back to searching
ARRIVE_PAUSE   = 1.0     # s to sit still after reaching a tag
CONTROL_HZ     = 10.0
# ═══════════════════════════════════════════════════════════════


class TagShuttle(Node):
    def __init__(self):
        super().__init__('tag_shuttle')
        self._bridge = CvBridge()
        self._setup_aruco()

        # latest detection per id: id -> (centre_x_px, width_px, stamp_s)
        self._seen: dict[int, tuple] = {}
        self._img_w = None

        # visit sequence: A, B, A, B, ...
        self._targets = [TAG_A_ID, TAG_B_ID] * N_ROUND_TRIPS
        self._idx = 0
        self._pause_until = 0.0
        self._done = False

        self._cmd = self.create_publisher(Twist, CMDVEL_TOPIC, 10)
        self.create_subscription(Image, IMAGE_TOPIC, self._image_cb, 5)
        self.create_timer(1.0 / CONTROL_HZ, self._control)

        self.get_logger().info(
            f'tag_shuttle up. A=id{TAG_A_ID}, B=id{TAG_B_ID}, '
            f'plan={self._targets}. image="{IMAGE_TOPIC}" cmd="{CMDVEL_TOPIC}".')

    # ── ArUco (works on old OpenCV 4.5 and new 4.7+ APIs) ──────────────────
    def _setup_aruco(self):
        a = cv2.aruco
        try:
            self._dict = a.getPredefinedDictionary(ARUCO_DICT)
        except AttributeError:
            self._dict = a.Dictionary_get(ARUCO_DICT)
        if hasattr(a, 'ArucoDetector'):                 # OpenCV >= 4.7
            self._detector = a.ArucoDetector(self._dict, a.DetectorParameters())
            self._new_api = True
        else:                                           # OpenCV 4.5/4.6
            self._params = a.DetectorParameters_create()
            self._new_api = False

    def _detect(self, gray):
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dict, parameters=self._params)
        return corners, ids

    def _image_cb(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge: {exc}', throttle_duration_sec=5.0)
            return
        self._img_w = img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        if ids is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(-1, 2)
            cx = float(pts[:, 0].mean())
            width = float(pts[:, 0].max() - pts[:, 0].min())
            self._seen[int(i)] = (cx, width, now)

    # ── Reactive control loop ──────────────────────────────────────────────
    def _control(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._done:
            return
        if self._idx >= len(self._targets):
            self._stop()
            self.get_logger().info('Plan complete — all tags visited.')
            self._done = True
            return
        if now < self._pause_until:          # sit still briefly after arriving
            self._stop()
            return
        if self._img_w is None:
            return                            # no image yet

        tid = self._targets[self._idx]
        det = self._seen.get(tid)
        fresh = det is not None and (now - det[2]) < LOST_TIMEOUT

        tw = Twist()
        if not fresh:
            # Can't see the target → spin in place to look for it.
            tw.angular.z = SEARCH_ANGULAR
            self.get_logger().info(f'Searching for tag {tid}…',
                                   throttle_duration_sec=2.0)
        else:
            cx, width, _ = det
            if width >= CLOSE_FRAC * self._img_w:
                # Close enough → arrived.
                self._stop()
                self.get_logger().info(
                    f'Reached tag {tid}  ({self._idx + 1}/{len(self._targets)}).')
                self._idx += 1
                self._pause_until = now + ARRIVE_PAUSE
                return
            # Steer to centre the tag, drive forward.
            err = (cx - self._img_w / 2.0) / (self._img_w / 2.0)   # -1..+1
            tw.linear.x = DRIVE_LINEAR
            tw.angular.z = float(np.clip(-STEER_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR))

        self._cmd.publish(tw)

    def _stop(self):
        self._cmd.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = TagShuttle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()                          # leave the robot stopped
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
