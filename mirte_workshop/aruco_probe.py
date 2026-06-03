#!/usr/bin/env python3
"""
aruco_probe.py — standalone ArUco detection sanity check.

No SLAM, no Nav2, no TF, no spinning.  Just: subscribe to the camera image and,
for EVERY common ArUco dictionary, report which marker IDs are visible.  Park
the robot facing the marker and run this to answer "is the camera even seeing
it, and in which dictionary / ID?".

Run on the robot (ROS env sourced), no build needed:

    python3 ~/mirte_ws/src/mirte_workshop/mirte_workshop/aruco_probe.py
    python3 .../aruco_probe.py /camera/color/image_raw      # custom topic

It prints, ~twice a second:
  * a heartbeat with image size if NO markers are found (camera is alive but
    nothing detected) — points to blur, distance, lighting, or wrong marker;
  * for each dictionary that matches, the IDs found, e.g.
        DICT_4X4_250 -> [104]
    which tells you exactly what to pass as aruco_dict / zone_a_id.
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Dictionaries worth checking — covers the usual printed sets.
DICT_NAMES = [
    'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
    'DICT_5X5_50', 'DICT_5X5_250',
    'DICT_6X6_50', 'DICT_6X6_250',
    'DICT_7X7_250',
    'DICT_ARUCO_ORIGINAL',
]


class ArucoProbe(Node):
    def __init__(self, topic):
        super().__init__('aruco_probe')
        self._bridge = CvBridge()
        self._new_api = hasattr(cv2.aruco, 'ArucoDetector')

        # Build a detector (new API) or (dict, params) pair (old API) per name.
        self._dicts = {}
        for name in DICT_NAMES:
            const = getattr(cv2.aruco, name, None)
            if const is None:
                continue
            if self._new_api:
                d = cv2.aruco.getPredefinedDictionary(const)
                self._dicts[name] = cv2.aruco.ArucoDetector(
                    d, cv2.aruco.DetectorParameters())
            else:
                self._dicts[name] = (
                    cv2.aruco.Dictionary_get(const),
                    cv2.aruco.DetectorParameters_create())

        self._frames = 0
        self.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)
        self.get_logger().info(
            f'aruco_probe listening on "{topic}" '
            f'({"new" if self._new_api else "old"} OpenCV API, '
            f'{len(self._dicts)} dictionaries). Show it a marker…')
        self.create_timer(0.5, self._report)
        self._last = {}      # name -> list of ids (latest frame)
        self._size = None

    def _detect(self, gray, entry):
        if self._new_api:
            corners, ids, _ = entry.detectMarkers(gray)
        else:
            d, params = entry
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
        if ids is None:
            return []
        return sorted(int(x) for x in ids.flatten())

    def _cb(self, msg):
        self._frames += 1
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge: {exc}', throttle_duration_sec=3.0)
            return
        self._size = (msg.width, msg.height, msg.encoding)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._last = {name: self._detect(gray, e) for name, e in self._dicts.items()}

    def _report(self):
        if self._frames == 0:
            self.get_logger().warn(
                'No images received yet — wrong topic, or camera not publishing?')
            return
        hits = {n: ids for n, ids in self._last.items() if ids}
        if not hits:
            self.get_logger().info(
                f'frames={self._frames} size={self._size} — NO markers in any '
                f'dictionary (try closer / steadier / better lit).')
            return
        for name, ids in hits.items():
            self.get_logger().info(f'  {name} -> {ids}')


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else '/camera/color/image_raw'
    rclpy.init()
    node = ArucoProbe(topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
