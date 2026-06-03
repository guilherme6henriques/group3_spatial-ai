#!/usr/bin/env python3
"""
Box perception — finds the floor boxes with the DEPTH camera (no risers, no
ArUco on boxes) and publishes their pose + relative size.

Why this exists
───────────────
The pickup boxes are short (~3 cm) objects sitting on the floor.  A 2D LIDAR
beam rides at ~0.13 m, so it never sees them — which is exactly what makes the
discriminator reliable:

      SHORT thing on the floor  =  a box
      TALL thing (wall, pillar) =  not a box

We don't need to *recognise* "box-ness" or read a marker; we exploit that
height signature.  The RGB-D camera produces a point cloud; we:

  1. transform it to the map frame,
  2. drop the floor and the ceiling,
  3. cluster what's left,
  4. KEEP only clusters that are LOW (max height < BOX_MAX_Z) — that throws out
     walls/pillars, whose clusters reach far higher,
  5. track each surviving cluster across frames (boxes are static and ≥1.3 m
     apart, so nearest-neighbour association is unambiguous) and accumulate its
     footprint extent over multiple viewpoints,
  6. publish, per box:
        /box_pose/id<k>   geometry_msgs/PoseStamped   (map frame, yaw = footprint axis)
        /box_size/id<k>   std_msgs/Float32            (footprint extent, metres)
     plus /box_markers (visualization_msgs/MarkerArray) for RViz.

IDs are assigned 2,3,4,… in order of first detection, matching the IDs the
mission manager already subscribes to.  Navigation around the boxes is handled
separately by Nav2's obstacle layer (it consumes the same depth cloud, height
gated) — this node is purely perception + ranking.

Requires: sensor_msgs_py (PointCloud2 parsing), scipy.ndimage (clustering), TF
map ← camera_depth_optical_frame.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time

import tf2_ros
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, ColorRGBA, Header
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _PC2 = True
except ImportError:
    _PC2 = False

try:
    from scipy import ndimage
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _quat_to_matrix(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


# Ground-truth box poses + footprint side (cm) for the exploration_arena sim.
# DIAGNOSTIC ONLY: if non-empty, each consolidated box is compared to the
# nearest truth in the log so we can see detection accuracy.  Set [] for real
# deployments.
_TRUTH_BOXES = [(-5.0, 0.9, 10), (-3.0, 0.9, 12), (-4.0, 2.1, 14),
                (-5.0, 3.2, 16), (-3.0, 3.2, 18)]


class BoxPerception(Node):

    def __init__(self):
        super().__init__('box_perception')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('cloud_topic', '/camera/points')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('first_box_id', 2)      # match mission manager IDs
        self.declare_parameter('floor_z',      0.008)  # keep points THIS far above est. ground
        self.declare_parameter('ceiling_z',    1.20)   # drop points above (noise)
        self.declare_parameter('box_max_z',    0.10)   # cluster taller than this = NOT a box
        self.declare_parameter('grid_res',     0.025)  # clustering raster (m/cell)
        self.declare_parameter('min_cell_pts', 2)      # points to call a cell occupied
        self.declare_parameter('min_points',   18)     # min points for a box cluster
        self.declare_parameter('min_size',     0.04)   # reject specks (m)
        self.declare_parameter('max_size',     0.40)   # reject walls/large (m)
        # Reject FAR points: floor leaks above floor_z grow with range (slight
        # camera pitch), and small boxes are only reliable up close anyway.
        self.declare_parameter('max_range',    2.5)    # m, in camera frame
        # A real box has a vertical face → some z-spread.  A leaked floor patch
        # is flat.  Require this minimum vertical extent to count as a box.
        self.declare_parameter('min_box_height', 0.010)  # m
        # Association radius: a single-view footprint centroid sits on the box's
        # near FACE, so it shifts by up to ~half the box as the robot circles —
        # this must be wide enough to merge those views into one track, but well
        # under the ~1.3 m spacing between real boxes.
        self.declare_parameter('assoc_radius', 0.55)   # track association (m)
        # A box must be seen this many times before it's published — filters
        # one-/two-frame false positives (e.g. a pillar base glimpsed at range).
        self.declare_parameter('min_hits',    4)
        # Consolidation: per-frame detections of a box are noisy and drift on its
        # near face, so we DON'T track incrementally (that smeared the size to
        # ~60 cm).  Instead we collect raw detections and, each cycle, cluster
        # them: detections within merge_radius form one box; its pose is the
        # MEDIAN position and its size the MEDIAN per-frame extent (both robust).
        self.declare_parameter('merge_radius',    0.65)  # m, group detections of one box
                                                         # (face-bias scatters them ~0.5 m; boxes are ≥1.5 m apart)
        self.declare_parameter('min_obs',         5)     # detections to CONFIRM a box (publish + rank)
        self.declare_parameter('obstacle_min_obs', 2)    # detections to mark it as a costmap obstacle
        self.declare_parameter('max_obs',         2000)  # cap stored detections
        # Zone-A gate: accept boxes only within this distance of the pole AND on
        # the into-the-zone side of it.  Covers the box field (~3.4 m) without
        # admitting the pillars/stand around/behind the pole.
        self.declare_parameter('zone_radius', 4.0)
        # A box must be at least this far INTO the zone from the pole — rejects
        # the pole itself and near-pole artifacts (nearest real box is ~1.1 m in).
        self.declare_parameter('zone_min_forward', 0.5)
        self.declare_parameter('process_period', 0.2)  # min seconds between clouds
        self.declare_parameter('publish_rate',  5.0)   # Hz, re-publish tracks
        self.declare_parameter('max_points',  30000)   # stride cap (CPU budget)
        self.declare_parameter('obstacle_z',  0.12)    # height to mark boxes in costmap

        gp = self.get_parameter
        self._cloud_topic  = gp('cloud_topic').value
        self._target_frame = gp('target_frame').value
        self._first_id     = int(gp('first_box_id').value)
        self._floor_z      = float(gp('floor_z').value)
        self._ceiling_z    = float(gp('ceiling_z').value)
        self._box_max_z    = float(gp('box_max_z').value)
        self._grid_res     = float(gp('grid_res').value)
        self._min_cell_pts = int(gp('min_cell_pts').value)
        self._min_points   = int(gp('min_points').value)
        self._min_size     = float(gp('min_size').value)
        self._max_size     = float(gp('max_size').value)
        self._assoc_radius = float(gp('assoc_radius').value)
        self._min_hits     = int(gp('min_hits').value)
        self._zone_radius  = float(gp('zone_radius').value)
        self._zone_min_fwd = float(gp('zone_min_forward').value)
        self._merge_radius = float(gp('merge_radius').value)
        self._min_obs      = int(gp('min_obs').value)
        self._obstacle_min_obs = int(gp('obstacle_min_obs').value)
        self._max_obs      = int(gp('max_obs').value)
        self._proc_period  = float(gp('process_period').value)
        self._max_points   = int(gp('max_points').value)
        self._obstacle_z   = float(gp('obstacle_z').value)
        self._max_range    = float(gp('max_range').value)
        self._min_box_h    = float(gp('min_box_height').value)

        if not _PC2:
            self.get_logger().error('sensor_msgs_py not found — box perception disabled.')
            return
        if not _SCIPY:
            self.get_logger().error('scipy not found — box perception disabled.')
            return

        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        self._obs: list = []          # raw accepted detections: [x, y, extent]
        self._published_ids: set = set()
        self._last_proc_ns = 0

        # Per-ID publishers, created on first detection
        self._pose_pubs: dict[int, object] = {}
        self._size_pubs: dict[int, object] = {}
        self._marker_pub = self.create_publisher(MarkerArray, '/box_markers', 10)
        # Clean box-only obstacle cloud for the Nav2 costmap (real boxes only —
        # never the floor — so the planner routes around them safely).
        self._obstacle_pub = self.create_publisher(PointCloud2, '/box_obstacles', 5)

        # Zone-A gate: only accept boxes inside the Zone A region (the box
        # field), once the pole is known.  Boxes lie "into the zone" from the
        # pole; pillars/stand/walls do not — so we gate on direction + distance.
        self._zone_a = None
        self.create_subscription(PoseStamped, '/zone_a_pose', self._zone_a_cb, 10)

        self.create_subscription(PointCloud2, self._cloud_topic, self._cloud_cb, 5)
        self.create_timer(1.0 / float(gp('publish_rate').value), self._publish_tracks)

        self.get_logger().info(
            f'Box perception started — cloud="{self._cloud_topic}", '
            f'frame="{self._target_frame}", keep clusters with max-z < '
            f'{self._box_max_z:.2f} m as boxes.')

    # ── Cloud processing ────────────────────────────────────────────────────

    def _cloud_cb(self, msg: PointCloud2):
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_proc_ns) / 1e9 < self._proc_period:
            return
        self._last_proc_ns = now_ns

        # Read xyz as an (N,3) array.
        arr = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'),
                                    skip_nans=True)
        if arr is None or arr.shape[0] == 0:
            return
        pts_cam = np.asarray(arr, dtype=np.float64).reshape(-1, 3)

        # Depth no-returns come through as inf (skip_nans doesn't catch those);
        # drop every non-finite point or the transform produces NaNs.
        pts_cam = pts_cam[np.isfinite(pts_cam).all(axis=1)]
        if pts_cam.shape[0] == 0:
            return

        # Reject far points (camera-frame range): far floor leaks above floor_z
        # due to slight pitch, and small boxes aren't reliable past a couple m.
        rng2 = np.einsum('ij,ij->i', pts_cam, pts_cam)
        pts_cam = pts_cam[rng2 < self._max_range * self._max_range]
        if pts_cam.shape[0] == 0:
            return

        pts_map = self._transform_to_target(pts_cam, msg.header.frame_id)
        if pts_map is None:
            return

        # Keep everything above the floor and below the ceiling.  NOTE: we do
        # NOT stride here — a 3 cm box is only a few hundred points and striding
        # the whole cloud drops it below min_points (that's why only 1 box was
        # ever seen).  _cluster keeps the low/box points dense and strides only
        # the tall wall/pillar points (used solely for rejection).
        z = pts_map[:, 2]
        # Threshold height RELATIVE to the detected ground, not an absolute z.
        # The diff-drive odom is planar (base_link at z=0) while the robot is
        # physically ~2 cm up, so absolute map-z is offset by ~that much — enough
        # to push 3 cm box tops below a fixed floor threshold and lose every box.
        # Estimating the floor from the cloud (a low percentile) is immune to it.
        floor = float(np.percentile(z, 5))
        rel = z - floor
        keep = (rel > self._floor_z) & (rel < self._ceiling_z)
        pts = pts_map[keep].copy()
        if pts.shape[0]:
            pts[:, 2] = rel[keep]          # hand _cluster floor-relative z

        # DIAGNOSTIC funnel — shows where boxes are lost (remove once tuned).
        self.get_logger().info(
            f'cloud: finite={pts_cam.shape[0]} floor={floor:.3f} '
            f'fullZ[{z.min():.3f},{z.max():.3f}] band={pts.shape[0]}',
            throttle_duration_sec=2.0)

        if pts.shape[0] < self._min_points:
            return

        detections = self._cluster(pts, now_ns)
        if detections:
            self._add_observations(detections)

    def _transform_to_target(self, pts_cam, src_frame):
        if not src_frame:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                self._target_frame, src_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            self.get_logger().warn(
                f'TF {self._target_frame}<-{src_frame} unavailable.',
                throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = _quat_to_matrix(q.x, q.y, q.z, q.w)
        return pts_cam @ R.T + np.array([t.x, t.y, t.z])

    def _cluster(self, pts, now_ns):
        """Cluster the LOW (box-height) points and reject any cluster sitting
        under a TALL structure (wall/pillar base).  Returns a list of
        (cx, cy, pts2d) box detections in the target frame, where pts2d are the
        cluster's 2D points (the tracker accumulates them into a footprint area).

        The low points (boxes) are kept at full density; only the tall points,
        used purely to build a rejection mask, are strided for CPU."""
        z = pts[:, 2]
        low  = pts[z < self._box_max_z]            # boxes + low wall/pillar slices
        tall = pts[z >= self._box_max_z][:, :2]    # walls/pillars → rejection mask
        if low.shape[0] < self._min_points:
            return []

        res = self._grid_res
        ox, oy = pts[:, 0].min(), pts[:, 1].min()
        ncol = int((pts[:, 0].max() - ox) / res) + 1
        nrow = int((pts[:, 1].max() - oy) / res) + 1
        if ncol < 1 or nrow < 1:
            return []

        lc = ((low[:, 0] - ox) / res).astype(np.int32)
        lr = ((low[:, 1] - oy) / res).astype(np.int32)
        counts = np.zeros((nrow, ncol), dtype=np.int32)
        np.add.at(counts, (lr, lc), 1)
        occupied = counts >= self._min_cell_pts

        # Cells that have a TALL return → a wall/pillar stands there.  Dilate so
        # the base footprint of a tall structure is masked out generously.
        tallgrid = np.zeros((nrow, ncol), dtype=bool)
        if tall.shape[0]:
            if tall.shape[0] > self._max_points:
                tall = tall[::int(tall.shape[0] // self._max_points) + 1]
            tc = ((tall[:, 0] - ox) / res).astype(np.int32)
            tr = ((tall[:, 1] - oy) / res).astype(np.int32)
            v = (tc >= 0) & (tc < ncol) & (tr >= 0) & (tr < nrow)
            tallgrid[tr[v], tc[v]] = True
            tallgrid = ndimage.binary_dilation(tallgrid, iterations=2)

        labels, nlab = ndimage.label(occupied, structure=np.ones((3, 3)))
        if nlab == 0:
            return []
        point_labels = labels[lr, lc]

        detections = []
        for lab in range(1, nlab + 1):
            sel = point_labels == lab
            if int(sel.sum()) < self._min_points:
                continue
            # Reject if the cluster overlaps a tall column (wall/pillar base).
            if tallgrid[lr[sel], lc[sel]].any():
                continue
            cl = low[sel]
            # A real box has a vertical face → z-spread; a leaked floor patch is
            # flat.  Reject clusters with too little vertical extent.
            if (cl[:, 2].max() - cl[:, 2].min()) < self._min_box_h:
                continue
            cx = float(cl[:, 0].mean())
            cy = float(cl[:, 1].mean())
            extent, minor, yaw = self._footprint(cl[:, :2])
            if extent < self._min_size or extent > self._max_size:
                continue
            detections.append((cx, cy, cl[:, :2]))
        return detections

    def _footprint(self, xy):
        """PCA of a cluster's 2D points → (major span, minor span, yaw) of the
        footprint, orientation-invariant.  Quantisation-corrected by one cell."""
        mean = xy.mean(axis=0)
        d = xy - mean
        cov = np.cov(d.T) if d.shape[0] > 2 else np.eye(2) * 1e-6
        evals, evecs = np.linalg.eigh(cov)
        proj = d @ evecs
        spans = proj.max(axis=0) - proj.min(axis=0) + self._grid_res
        major_i = int(np.argmax(spans))
        major = float(spans[major_i])
        minor = float(spans[1 - major_i])
        axis = evecs[:, major_i]
        yaw = math.atan2(axis[1], axis[0])
        return major, minor, yaw

    # ── Tracking ──────────────────────────────────────────────────────────────

    def _zone_a_cb(self, msg: PoseStamped):
        self._zone_a = msg

    def _in_zone_a(self, cx, cy) -> bool:
        """True if (cx, cy) is inside the Zone A box field: within zone_radius of
        the pole AND on the into-the-zone side of it.  Until the pole is known,
        nothing qualifies (so no boxes are accepted before the robot reaches A)."""
        za = self._zone_a
        if za is None:
            return False
        px, py = za.pose.position.x, za.pose.position.y
        dx, dy = cx - px, cy - py
        if math.hypot(dx, dy) > self._zone_radius:
            return False
        # Into-zone direction = opposite the marker's facing (+Z → R[:,2]).
        q = za.pose.orientation
        R = _quat_to_matrix(q.x, q.y, q.z, q.w)
        into = math.atan2(-R[1, 2], -R[0, 2])
        proj = dx * math.cos(into) + dy * math.sin(into)
        return proj > self._zone_min_fwd

    def _add_observations(self, detections):
        """Store each in-zone per-frame detection as (x, y, extent).  No
        incremental tracking — consolidation happens at publish time."""
        for (cx, cy, pts2d) in detections:
            if not self._in_zone_a(cx, cy):       # Zone-A gate
                continue
            extent, _minor, _yaw = self._footprint(np.asarray(pts2d))
            self._obs.append([cx, cy, extent])
        if len(self._obs) > self._max_obs:        # keep the most recent
            self._obs = self._obs[-self._max_obs:]

    def _consolidate(self):
        """Cluster stored detections (within merge_radius) into boxes.  Each
        box: MEDIAN position + MEDIAN per-frame extent + observation count.
        Robust to the per-frame face-bias/jitter that fragmented the old tracker."""
        if not self._obs:
            return []
        obs = np.asarray(self._obs, dtype=np.float64)
        centers, members = [], []
        r = self._merge_radius
        for i in range(obs.shape[0]):
            x, y = obs[i, 0], obs[i, 1]
            best, bd = -1, r
            for gi, c in enumerate(centers):
                d = math.hypot(c[0] - x, c[1] - y)
                if d < bd:
                    best, bd = gi, d
            if best < 0:
                centers.append([x, y]); members.append([i])
            else:
                members[best].append(i)
                idx = members[best]
                centers[best] = [obs[idx, 0].mean(), obs[idx, 1].mean()]
        groups = []
        for idx in members:
            pts = obs[idx]
            groups.append({'x': float(np.median(pts[:, 0])),
                           'y': float(np.median(pts[:, 1])),
                           'size': float(np.median(pts[:, 2])),
                           'n': len(idx)})
        return groups

    # ── Publishing ──────────────────────────────────────────────────────────

    def _publish_tracks(self):
        now = self.get_clock().now().to_msg()
        groups = self._consolidate()

        # CONFIRMED boxes (enough observations), sorted by position so IDs are
        # stable across cycles; IDs start at first_box_id to match the manager.
        confirmed = sorted((g for g in groups if g['n'] >= self._min_obs),
                           key=lambda g: (round(g['x'], 1), round(g['y'], 1)))
        markers = MarkerArray()
        for i, g in enumerate(confirmed):
            bid = self._first_id + i
            if bid not in self._pose_pubs:
                self._pose_pubs[bid] = self.create_publisher(
                    PoseStamped, f'/box_pose/id{bid}', 10)
                self._size_pubs[bid] = self.create_publisher(
                    Float32, f'/box_size/id{bid}', 10)

            ps = PoseStamped()
            ps.header.frame_id = self._target_frame
            ps.header.stamp    = now
            ps.pose.position.x = g['x']
            ps.pose.position.y = g['y']
            ps.pose.position.z = 0.015
            ps.pose.orientation.w = 1.0
            self._pose_pubs[bid].publish(ps)
            self._size_pubs[bid].publish(Float32(data=g['size']))
            markers.markers.append(self._make_marker(bid, g, now))

        if markers.markers:
            self._marker_pub.publish(markers)

        # Obstacle cloud — mark boxes for the costmap as soon as they have a few
        # observations (lower bar than publishing) so the robot stops stepping
        # on them sooner.  Only real in-zone boxes are ever included.
        obstacle_pts = []
        for g in groups:
            if g['n'] >= self._obstacle_min_obs:
                obstacle_pts.extend(self._obstacle_points(g))
        header = Header(stamp=now, frame_id=self._target_frame)
        self._obstacle_pub.publish(pc2.create_cloud_xyz32(header, obstacle_pts))

        self._report_truth(confirmed)

    def _report_truth(self, confirmed):
        """DIAGNOSTIC: compare each confirmed box to the nearest ground-truth box."""
        if not _TRUTH_BOXES or not confirmed:
            return
        lines = []
        for i, g in enumerate(confirmed):
            tx, ty, tcm = min(_TRUTH_BOXES,
                              key=lambda t: math.hypot(t[0] - g['x'], t[1] - g['y']))
            err = math.hypot(tx - g['x'], ty - g['y'])
            lines.append(f'id{self._first_id + i}: det({g["x"]:+.2f},{g["y"]:+.2f}) '
                         f'{g["size"]*100:.0f}cm | true({tx:+.1f},{ty:+.1f}) {tcm}cm | '
                         f'err={err*100:.0f}cm')
        self.get_logger().info('DET vs TRUTH:\n  ' + '\n  '.join(lines),
                               throttle_duration_sec=5.0)

    def _obstacle_points(self, g):
        """A footprint-sized patch of points for one box so the costmap marks it
        (inflation then turns it into a keep-out)."""
        h = max(g['size'], 0.08) / 2.0
        z = self._obstacle_z
        return [(g['x'] + dx, g['y'] + dy, z)
                for dx in (-h, 0.0, h) for dy in (-h, 0.0, h)]

    def _make_marker(self, bid, g, stamp):
        m = Marker()
        m.header.frame_id = self._target_frame
        m.header.stamp    = stamp
        m.ns   = 'boxes'
        m.id   = bid
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = g['x']
        m.pose.position.y = g['y']
        m.pose.position.z = 0.015
        m.pose.orientation.w = 1.0
        m.scale.x = max(g['size'], 0.02)
        m.scale.y = max(g['size'], 0.02)
        m.scale.z = 0.03
        m.color = ColorRGBA(r=0.9, g=0.6, b=0.1, a=0.8)
        return m


def main(args=None):
    rclpy.init(args=args)
    node = BoxPerception()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
