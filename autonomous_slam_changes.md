# Autonomous SLAM — Changes on Top of Fork

All changes were made on branch `ros2_humble` on top of the upstream `mirte_workshop` fork.

---

## New Files Added

| File | Purpose |
|---|---|
| `mirte_workshop/exploration_manager.py` | Polygon-aware exploration manager (described in detail below) |
| `mirte_workshop/scan_filter.py` | Filters LIDAR self-returns from Mirte's chassis |
| `mirte_workshop/odom_to_tf.py` | Publishes odom→base_link TF from odometry topic |
| `mirte_workshop/arm_joint_controller.py` | Arm joint position controller |

---

## Modified Files

### `setup.py`
Added entry points so the new Python nodes are discoverable by ROS 2:

```python
'exploration_manager.py = mirte_workshop.exploration_manager:main',
'scan_filter.py         = mirte_workshop.scan_filter:main',
'odom_to_tf.py          = mirte_workshop.odom_to_tf:main',
```

### `package.xml`
Added runtime dependencies: `slam_toolbox`, `nav2_*`, `explore_lite`, `topic_tools`.

---

## New Package: `mirte_navigation`

A separate package (`/home/gui/spatial-ai/ws/src/mirte_navigation/`) was created to hold all navigation configuration. Key files:

### `launch/autonomous_exploration.launch.py`

Launches the full autonomous SLAM pipeline in a single command:

```
ros2 launch mirte_navigation autonomous_exploration.launch.py
```

Start order (with timers):
1. **t=0 s** — `scan_filter` (filters LIDAR self-returns)
2. **t=0 s** — `slam_toolbox sync_slam_toolbox_node` (SLAM, provides `/map` + map→odom TF)
3. **t=0 s** — Two `static_transform_publisher` nodes:
   - `base_link → base_footprint` (SLAM toolbox default base frame)
   - `base_link → base_frame`
4. **t=0 s** — `topic_tools relay` (`/mirte_base_controller/odom → /odom`)
5. **t=10 s** — Nav2 stack (planner, controller, behavior server, BT navigator, lifecycle manager)
6. **t=20 s** — `explore_lite` (frontier-based autonomous exploration)
7. **t=20 s** — `exploration_manager.py` (polygon tracking and survey mode)

### `params/slam_params.yaml`
SLAM Toolbox configuration: `use_sim_time: true`, scan topic `/scan_filtered`.

### `params/explore_params.yaml`
explore_lite configuration: `use_sim_time: true`, matched to Nav2.

### `params/exploration_nav2_params.yaml`

Full Nav2 parameter file for the exploration scenario. Key differences from the default:
- No `amcl` or `map_server` (SLAM toolbox handles both)
- **Planner**: `SmacPlanner2D` with `allow_unknown: true` (navigates into unmapped space)
- **Controller**: `RegulatedPurePursuitController`, `desired_linear_vel: 0.2 m/s`
- **Costmaps**: subscribe to `/scan_filtered` (not `/scan`), asymmetric Mirte footprint `[[0.15,0.19],[0.15,-0.14],[-0.15,-0.14],[-0.15,0.19]]`, `inflation_radius: 0.30 m`
- **`behavior_server`**: Spin, BackUp, Wait recovery behaviors with `enable_stamped_cmd_vel: false`
- **`bt_navigator`**: references `nav2_minimal_tree.xml`

### `trees/nav2_minimal_tree.xml`

Behaviour Tree with recovery actions:
- `ClearEntireCostmap` (local + global)
- `Spin` (1.57 rad)
- `BackUp` (0.15 m at 0.05 m/s)
- `Wait` (5 s)

This was added (vs default which had only costmap clearing) to let the robot escape tight corners.

### `maps/`
Directory where the final map is saved (`default.pgm` + `default.yaml`).

---

## `mirte_workshop/scan_filter.py`

Subscribes to `/scan` and republishes on `/scan_filtered` with all ranges below 0.12 m set to `inf`.

**Why**: Gazebo's simulated LIDAR has `range_min=0.0` and sees Mirte's own chassis. These near-zero returns mark the robot's own occupancy grid cell as `LETHAL`, blocking the planner from the robot's position. Setting them to `inf` makes them invisible to the costmap.

---

## `mirte_workshop/exploration_manager.py`

The main new piece of software. Runs as a ROS 2 node alongside SLAM + Nav2 + explore_lite and implements:

### Architecture

```
/map  ──────────────────────► _map_cb (stores OccupancyGrid)
/explore/frontiers ─────────► _frontier_cb (records last frontier time)
(2 s timer) ────────────────► _tick()
                                  │
                                  ├─ _analyse_polygons()  ◄─── key function
                                  │     - finds obstacle clusters in SLAM map
                                  │     - updates persistent polygon registry
                                  │     - classifies: detected / complete / incomplete
                                  │
                                  ├─ stuck detection (per polygon)
                                  │
                                  ├─ stopping check (all complete + frontiers quiet)
                                  │
                                  └─ navigation decision
                                        - incomplete polygon detected → survey mode
                                        - no incomplete → let explore_lite drive
                                              │
                                              ▼
                                        _pick_viewpoint()
                                              │
                                              ▼
                                        _send_goal() → NavigateToPose action
```

### Polygon Detection (`_analyse_polygons()`)

1. **Arena detection**: Finds the largest connected free-space region in the SLAM map (the navigable floor). Only obstacle clusters whose centroid is inside this region count — eliminates arena walls and exterior SLAM noise.

2. **Cluster labelling**: `scipy.ndimage.label` on occupied cells. Filters:
   - `MIN_POLYGON_CELLS = 20` (≥20 cells at 0.05 m/px ≈ 5 cm × 5 cm minimum)
   - `MAX_POLYGON_CELLS = 600`
   - Centroid inside arena bounding box
   - At least one cell adjacent to the navigable floor

3. **Face analysis (N/S/E/W)**: For each cluster, the 4 bounding-box face slices are checked:
   - **accessible**: face cells have a non-occupied neighbour (robot can reach that side)
   - **seen**: face cells have a free-space neighbour (robot has been on that side)
   - Inaccessible faces (flush against a wall) auto-count as seen

4. **Deduplication**: Clusters sorted by size descending; any cluster whose centroid is within 1.2 m of an already-accepted centroid is discarded. This collapses SLAM fragments of the same physical object.

5. **Persistent registry (`_poly_db`)**: Each surviving cluster is matched to the nearest existing registry entry (within 1.2 m). If matched, the entry is updated:
   - Centroid: exponential moving average (α = 0.15)
   - `n_faces_seen`: max(old, current) — monotonically increasing, never forgets
   - `face_seen_flags[4]`: boolean OR of all ticks — once a face is seen it stays seen
   - Per-tick data (mask, undet, bbox) refreshed from current SLAM scan
   
   If no match: new entry created with a stable integer ID.

6. **Detection threshold**: A polygon enters the count only once `n_faces_seen ≥ MIN_FACES_FOR_DETECTION = 2`. This satisfies the requirement that **two distinct connected edges must be observed before a polygon is registered**.

7. **Completion**: A polygon is complete when `n_faces_seen ≥ faces_required` where `faces_required = min(3, n_faces_accessible)`. For a box with 3 accessible sides (1 against a wall), all 3 must be seen.

### Why n_total was Previously Fluctuating

The old design recomputed `n_total` from scratch each tick. As SLAM updated, clusters appeared and disappeared, making the count jump (6→8→7→6 in the logs). The same polygon also got re-logged when its centroid drifted enough to change its 0.5 m grid key.

The registry fixes both: `n_total` = `len(registry entries with n_faces_seen ≥ 2)`, which only grows. Logging uses the stable registry ID.

### Survey Mode and Priority

**Old behaviour**: Survey only started after `explore_lite` reported no frontiers (10 s of silence).

**New behaviour**: As soon as a polygon has been seen from ≥2 faces (registered and incomplete), the manager sends survey goals to cover the remaining faces — even during active frontier exploration. If no registered incomplete polygon exists, `explore_lite` drives uninterrupted.

### Viewpoint Generation (`_pick_viewpoint()`)

For each incomplete polygon, candidate viewpoints are generated at distances `[1.1, 1.6]` m from each face that still needs coverage:

- **Cardinal viewpoints** (N/S/E/W): perpendicular to each face
- **Diagonal corner viewpoints** (NE/NW/SE/SW): at 45° between adjacent faces — added so the robot can see into the gap between a box and the arena wall where cardinal viewpoints can't reach

A face "needs coverage" if it is not yet `seen` (in the registry's `face_seen_flags`) OR still has cells adjacent to unknown space (`undet`).

Viewpoints are filtered by:
- `_is_free()`: cell not occupied, within arena bounding box, no occupied cell within 0.60 m
- `_is_clear_of_polygon()`: at least 0.50 m from the target obstacle's cells
- `_attempted` blacklist: not tried in the last 60 s (20 s on planner abort)

### Stuck Detection

Per polygon, the stuck counter advances only when the robot is committed to that specific polygon (not all incomplete polygons simultaneously). After `MAX_STUCK_TICKS = 20` consecutive ticks (40 s) without faces improving, the polygon is skipped.

### Stopping Condition

```
frontier_quiet AND incomplete list is empty
```

"Frontier quiet" = no frontier message received for 10 s.  
"Incomplete empty" = all registered polygons are complete or skipped.

On success: `ros2 run nav2_map_server map_saver_cli` saves the map and `eog` opens it.

### Global Timeout

After `GLOBAL_TIMEOUT = 480 s` (8 min) the map is saved regardless of polygon state.

---

## Key Constants (tunable)

| Constant | Value | Meaning |
|---|---|---|
| `MIN_POLYGON_CELLS` | 20 | Minimum cluster size to consider as obstacle |
| `MIN_FACES_FOR_DETECTION` | 2 | Faces seen before polygon enters the count |
| `MIN_FACES_REQUIRED` | 3 | Faces needed for completion |
| `VIEWPOINT_DISTS` | [1.1, 1.6] m | Standoff distances for survey viewpoints |
| `VIEWPOINT_CLEARANCE` | 0.60 m | Safety margin from walls at viewpoint |
| `POLYGON_CLEARANCE` | 0.50 m | Min distance from target obstacle cells |
| `_MERGE_DIST_SQ` | 1.2² m² | Proximity threshold for cluster deduplication |
| `NO_FRONTIER_SECONDS` | 10 s | Silence needed before declaring exploration done |
| `VIEWPOINT_RETRY_SECONDS` | 60 s | Blacklist duration for a visited viewpoint |
| `MAX_STUCK_TICKS` | 20 | Ticks without progress before skipping polygon |
| `GLOBAL_TIMEOUT` | 480 s | Hard timeout before forced map save |
| `CHECK_INTERVAL` | 2 s | How often the manager tick runs |
