# Autonomous Pick-and-Place Mission — Deep Dive Reference

> **Audience**: Students who built this system and need to explain, defend, and extend it.  
> **What this covers**: Every file in the workspace, ROS2 concepts from first principles, every code change made, and a clear navigation / mapping division for the oral exam.

---

## Table of Contents

1. [ROS2 Primer — Key Concepts](#1-ros2-primer--key-concepts)
2. [Workspace Layout — The Big Picture](#2-workspace-layout--the-big-picture)
3. [Package-by-Package File Reference](#3-package-by-package-file-reference)
   - 3.1 [mirte-ros-packages — the robot itself](#31-mirte-ros-packages--the-robot-itself)
   - 3.2 [mirte-gazebo — the simulation world](#32-mirte-gazebo--the-simulation-world)
   - 3.3 [mirte_navigation — your navigation package](#33-mirte_navigation--your-navigation-package)
   - 3.4 [mirte_workshop — your mission logic package](#34-mirte_workshop--your-mission-logic-package)
   - 3.5 [Supporting packages](#35-supporting-packages)
4. [System Architecture — How Everything Connects](#4-system-architecture--how-everything-connects)
5. [The Full Data Flow](#5-the-full-data-flow)
6. [All Code Changes Made](#6-all-code-changes-made)
7. [Navigation vs Mapping — Exam Division](#7-navigation-vs-mapping--exam-division)
8. [Putting it on GitHub](#8-putting-it-on-github)

---

## 1. ROS2 Primer — Key Concepts

Before explaining what every file does, you need the vocabulary. Here are the concepts you will be asked about.

### 1.1 The ROS2 Communication Model

ROS2 is a **middleware** — it lets multiple programs (nodes) talk to each other over a standard API without caring whether they run on the same machine, a different machine, or even a different OS.

```
┌──────────────────────────────────────────────────────────────────┐
│                         ROS2 Network                             │
│                                                                  │
│  slam_toolbox ──/map──► global_costmap                          │
│                │                                                 │
│                └──/tf──► controller_server                       │
│                          zone_detector                           │
│                          exploration_manager                     │
│                                                                  │
│  scan_filter ──/scan_filtered──► slam_toolbox                   │
│                                  local_costmap                   │
│                                  global_costmap                  │
│                                                                  │
│  camera ──/camera/image_raw──► zone_detector                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Nodes

A **node** is a single running process. Each node has one job.

| Node | Package | Job |
|---|---|---|
| `slam_toolbox` | slam_toolbox (external) | Build the occupancy map AND localise the robot |
| `planner_server` | nav2_planner | Compute a collision-free global path |
| `controller_server` | nav2_controller | Follow a path with real-time obstacle checks |
| `bt_navigator` | nav2_bt_navigator | Orchestrate plan+follow+recovery via a Behavior Tree |
| `scan_filter` | mirte_workshop | Filter chassis self-returns from LIDAR |
| `zone_detector` | mirte_workshop | Detect ArUco markers, publish zone/box poses |
| `exploration_manager` | mirte_workshop | Run the full mission state machine |

### 1.3 Topics

A **topic** is a named message stream. Any node can publish; any node can subscribe. Neither knows about the other.

```
Publisher                  Topic                  Subscriber(s)
─────────────────────────────────────────────────────────────────
LIDAR Gazebo plugin   →   /scan              →   scan_filter
scan_filter           →   /scan_filtered     →   slam_toolbox
                                                 local_costmap
                                                 global_costmap
slam_toolbox          →   /map               →   global_costmap (static layer)
                                                 rviz
diff_drive plugin     →   /odom              →   relay → /odom (for Nav2)
zone_detector         →   /zone_a_pose       →   exploration_manager
zone_detector         →   /zone_b_pose       →   exploration_manager
zone_detector         →   /box_pose/id2..6   →   exploration_manager
exploration_manager   →   /mirte_base_controller/cmd_vel_unstamped  →  diff drive (spin)
```

### 1.4 TF2 — The Coordinate Frame Tree

TF is how ROS2 tracks where everything is in 3D space. Every sensor, wheel, and frame is defined relative to a parent frame. The full chain for this robot is:

```
map
 └── odom                   (published by slam_toolbox: map→odom)
      └── base_link         (published by diff_drive plugin: odom→base_link)
           ├── base_footprint   (static: identity)
           ├── base_frame       (static: identity)
           ├── frame_link       (from robot_state_publisher + URDF)
           │    ├── lidar_base  → lidar_link   (LIDAR sensor frame)
           │    ├── front_left_wheel
           │    ├── front_right_wheel
           │    ├── camera_link → camera_depth_optical_frame  (camera)
           │    └── ... (arm links, gripper, etc.)
```

**Why this matters**: Nav2 needs `map → base_link` to know where the robot is on the map. SLAM needs `odom → base_link` to compute odometry. The zone_detector needs `map ← ... ← camera_depth_optical_frame` to convert a detected marker from camera coordinates to map coordinates.

If any link in this chain is missing, nothing works. The two most critical:
- `odom → base_link` — comes from Gazebo's diff-drive plugin (only exists when Gazebo is running)
- `map → odom` — comes from slam_toolbox (only exists after SLAM has processed at least one scan)

### 1.5 Actions

An **action** is like a service but designed for long-running tasks. It has:
- A **goal** (what you want)
- **Feedback** (progress updates while running)
- A **result** (success/failure when done)
- **Cancellation** (you can abort it)

The key action in this project: `navigate_to_pose` (provided by bt_navigator, used by exploration_manager).

```python
# Sending a nav goal (simplified):
goal = NavigateToPose.Goal()
goal.pose.pose.position.x = 2.0
goal.pose.pose.position.y = 1.5
future = self._nav.send_goal_async(goal)
future.add_done_callback(self._goal_accepted_cb)
```

### 1.6 Services

A **service** is a one-shot synchronous call. Request → Response.

Used in this project for:
- Clearing costmaps: `local_costmap/clear_entirely_local_costmap`
- (Previously) changing robot footprint: `local_costmap/local_costmap/set_parameters`

### 1.7 Parameters

ROS2 nodes declare typed parameters. You pass them via YAML files in launch files.

```yaml
# In exploration_nav2_params.yaml:
controller_server:
  ros__parameters:
    use_sim_time: true          # ← parameter name: use_sim_time, value: true
    desired_linear_vel: 0.35    # ← max forward speed in m/s
```

`use_sim_time: true` is the single most important parameter — it tells every node to use the Gazebo simulated clock (`/clock` topic) instead of wall-clock time. Without it, timestamps don't match and TF lookups fail.

### 1.8 The Costmap

The costmap is a 2D grid overlaid on the world. Each cell has a cost from 0 (free) to 254 (lethal obstacle). The **inflation layer** expands obstacles:

```
Raw SLAM map:          After inflation (radius = 0.30 m):
                       
  ░░░░░░               ░░░░░░
  ░▓▓▓░░               ░▓▓▓░░    ← lethal (254)
  ░▓▓▓░░         →     ░░░░░░
  ░░░░░░               ░░░░░░    ← high cost (e.g. 100)
                       ░░░░░░
                       ░░░░░░    ← low cost (~10)
                                  ← free beyond inflation radius
```

**Why inflation_radius = 0.30 m matters**: The NavFn global planner avoids cells with cost above a threshold. With 0.30 m inflation the planner routes paths at least 0.30 m from obstacle edges — which is enough clearance for the robot footprint (0.25 m half-width) plus margin. Reducing it to 0.20 m caused NavFn to route paths through the INSCRIBED zone (cost ≥ 253), which immediately aborts the RPP local controller.

### 1.9 SLAM vs Localisation

- **SLAM** (Simultaneous Localisation And Mapping): builds the map AND figures out where the robot is at the same time. Used here — `slam_toolbox` in `mapping` mode.
- **Localisation only** (AMCL): uses a pre-built map, estimates where the robot is within it. Not used in the autonomous mission.

### 1.10 The Behavior Tree (BT)

The BT navigator uses an XML tree to describe what to do when navigation fails. Nodes are:

| BT Node type | Behaviour |
|---|---|
| `Sequence` | Run children in order; stop at first failure |
| `Fallback` | Try children in order; stop at first success |
| `RecoveryNode` | Run main child; on failure run recovery child up to N times |
| `ComputePathToPose` | Call the global planner |
| `FollowPath` | Call the local controller |
| `ClearEntireCostmap` | Service call to clear a costmap |
| `BackUp` | Drive backward a fixed distance |
| `Wait` | Pause for N seconds |

---

## 2. Workspace Layout — The Big Picture

```
/home/gui/spatial-ai/ws/         ← colcon workspace root
├── src/                         ← all source packages live here
│   ├── mirte-ros-packages/      ← robot hardware & description (DO NOT MODIFY)
│   ├── mirte-gazebo/            ← Gazebo simulation world (YOU MODIFIED)
│   ├── mirte_navigation/        ← YOUR navigation package
│   ├── mirte_workshop/          ← YOUR mission logic package
│   ├── m-explore-ros2/          ← frontier exploration library (dependency, not used)
│   ├── gazebo_grasp_fix/        ← gripper physics plugin (dependency)
│   └── mirte_location_markers/  ← location service helper (not used)
├── build/                       ← colcon build output (auto-generated, gitignore)
├── install/                     ← colcon install output (auto-generated, gitignore)
└── log/                         ← colcon build logs (auto-generated, gitignore)
```

Run `colcon build --symlink-install` from `ws/` to compile everything. `--symlink-install` means Python files are symlinked from `src/` into `install/` so edits take effect without rebuilding.

---

## 3. Package-by-Package File Reference

### 3.1 `mirte-ros-packages/` — the robot itself

This is the upstream MIRTE robot software. You do not modify it. Understanding it helps you debug.

#### `mirte_description/mirte_master_description/`

The robot's physical description. This is what tells every tool (Gazebo, RViz, Nav2) how the robot is built.

| File | Purpose |
|---|---|
| `urdf/mirte_master.xacro` | Master file — `<xacro:include>` all sub-files to build the complete robot |
| `urdf/lidar.xacro` | Defines `lidar_base` link + `lidar_joint` + the Gazebo ray sensor plugin. The plugin publishes `/scan` at 10 Hz, 360°, range 0.03–12 m |
| `urdf/mirte_master_base.gazebo.xacro` | The **differential drive plugin** — converts `/cmd_vel` commands to wheel velocities; publishes `/mirte_base_controller/odom` topic AND the `odom → base_link` TF |
| `urdf/orbbec_astra_plus_pro.xacro` | Depth camera definition — publishes `/camera/image_raw` and `/camera/camera_info` |
| `urdf/arm.xacro` | 4-DOF robot arm definition |
| `urdf/wheel.xacro` | Wheel geometry |
| `meshes/*.STL` | 3D mesh files for every visible robot part. Referenced in the URDF via `package://mirte_master_description/meshes/lidar.STL` |

**Key URDF concepts**:
- **Link**: a rigid body (e.g. `base_link`, `lidar_link`)
- **Joint**: connects two links with a transform + optional axis (e.g. `lidar_joint` is `fixed`, wheels are `continuous`)
- **Gazebo plugin**: a `<gazebo>` block inside the URDF that Gazebo reads to add sensors/actuators

#### `mirte_bringup/`

Launch files for the **physical robot** (not simulation):
- `minimal_master.launch.py` — starts telemetrix (hardware comms), state publishers, base control
- `ros2_control.launch.py` — starts ros2_control infrastructure for the physical robot

#### `mirte_control/mirte_master_base_control/`

The **ros2_control** hardware interface for the base. In Gazebo this is replaced by the diff_drive Gazebo plugin; on the real robot this interfaces with telemetrix.

#### `mirte_msgs/`

Custom message and service types. Examples:
- `msg/Encoder.msg` — encoder tick counts
- `srv/SetMotorSpeed.srv` — set left/right motor speeds
- `srv/GetRange.srv` — read ultrasonic distance sensor

---

### 3.2 `mirte-gazebo/` — the simulation world

#### `worlds/exploration_arena.world`

The complete Gazebo SDF world. Every object in the simulation is defined here.

**Structure of the world file**:
```xml
<world name="default">
  <!-- Physics settings -->
  <!-- Lighting (sun) -->
  <!-- Ground plane -->
  <!-- Arena walls (outer boundary) -->
  <!-- Corridor pillars (7 × 0.30×0.30×0.80 m) -->
  <!-- Zone A pole + ArUco marker ID=0 -->
  <!-- Zone B stand + ArUco marker ID=1 -->
  <!-- 5 pickup boxes on risers -->
  <!-- 5 ArUco box markers (ID=2–6) on riser east faces -->
</world>
```

**The 7 corridor pillars** — these are 0.30 × 0.30 × 0.80 m red square columns arranged in three rows:

```
Y=3.2:   (-0.8, 3.2)    (0.8, 3.2)
Y=2.0:  (-1.0, 2.0)    (1.0, 2.0)     ← Zone A (-1.8, 2.0)  Zone B (3.0, 2.0)
Y=1.0: (-1.5, 1.0)  (0.0, 0.8)  (1.5, 1.0)

          Zone A side (boxes)   ←corridor→   Zone B side
          X: -6.7 to -1.8          -1.8 to 3.0          3.0 to 4.7
```

The corridor between Zone A and Zone B at y=2.0 passes between pillars at (-1.0, 2.0) and (1.0, 2.0). Gap = 2.0 m − 2 × 0.15 m (half-size) = 1.70 m clear, minus 2 × 0.30 m inflation = **1.10 m navigable corridor**.

**The 5 pickup boxes**: boxes 80–160 mm footprint (STL geometry, 30 mm tall), placed on 110 mm risers. LIDAR at 129 mm height passes over the riser top, only scanning the box walls.

**The risers** (after changes): all identical 180 × 220 × 110 mm grey blocks. Below the LIDAR plane so SLAM only sees box walls (not riser walls). ArUco markers glued to the riser east face.

**The ArUco markers** (after changes): 12 cm panels at z = 0.070 m (centre of riser east face). Pose published by zone_detector tells exploration_manager where each box/zone is.

| ID | Meaning | Location |
|---|---|---|
| 0 | Zone A pole (entrance marker) | (-1.8, 2.0) |
| 1 | Zone B stand (delivery marker) | (3.0, 2.0) |
| 2 | Box on riser_1 | Riser at (-4.960, 0.920) |
| 3 | Box on riser_2 | Riser at (-2.950, 0.930) |
| 4 | Box on riser_3 | Riser at (-3.940, 2.140) |
| 5 | Box on riser_4 | Riser at (-4.930, 3.250) |
| 6 | Box on riser_5 | Riser at (-2.920, 3.260) |

#### `models/aruco_box_2/` through `aruco_box_6/`

Each ArUco box marker is a Gazebo model:
```
aruco_box_2/
├── model.config          ← model metadata (name, version, description)
├── model.sdf             ← geometry: 0.001×0.12×0.12 m thin panel, visual only
└── materials/
    ├── scripts/
    │   └── aruco_box_2.material   ← links the texture to this material name
    └── textures/
        └── aruco_box_2.png        ← the actual ArUco ID=2 pattern as a PNG
```

The panel is 1 mm thick in X (facing east). The PNG texture is the ArUco marker image that the camera and OpenCV detect.

#### `models/arena_obstacles/meshes/`

STL files for the box shapes:
- `box_80.stl` — 80 × 120 × 30 mm box
- `box_100.stl` — 100 × 140 × 30 mm box
- `box_120.stl` — 120 × 160 × 30 mm box
- `box_140.stl` — 140 × 180 × 30 mm box
- `box_160.stl` — 160 × 200 × 30 mm box

These are only 30 mm tall. The riser lifts them to 110–140 mm height where the LIDAR can see them.

#### `launch/gazebo_exploration_arena.launch.py`

**Your file.** Does the following:

1. Runs `_random_spawn()` — rejection sampling to find a valid spawn position:
   - Arena bounds: X ∈ [−6.7, 4.7], Y ∈ [−1.2, 4.7]
   - Clearance: 1.00 m from every known obstacle centre (pillars, boxes, ArUco stands)
   - Up to 2000 attempts; fallback to (0.0, 1.75) if all fail

2. Starts Gazebo with `exploration_arena.world`

3. Spawns the robot URDF into Gazebo via `spawn_mirte_master.launch.xml`:
   - This also starts `robot_state_publisher` which publishes the full TF chain from the URDF

4. Starts the ros2_control spawners:
   - `joint_state_broadcaster` — publishes joint states
   - `mirte_base_controller` — diff drive controller (publishes `/mirte_base_controller/cmd_vel_unstamped` subscriber + odom TF)
   - `mirte_master_arm_controller` / `mirte_master_gripper_controller`

**Why spawn clearance = 1.00 m matters**: The corridor pillars are 0.30 × 0.30 m. The INSCRIBED robot zone (minimum costmap clearance) extends `pillar_half + robot_inscribed = 0.15 + 0.15 = 0.30 m` from pillar centre. NavFn routes paths at least inflation_radius (0.30 m) from obstacle edges. So for the initial path from spawn to Zone A to never graze the INSCRIBED zone, the robot must start ≥ `pillar_half + inflation + some margin ≈ 0.15 + 0.30 + 0.55 = 1.00 m` from the pillar centre.

---

### 3.3 `mirte_navigation/` — your navigation package

This package configures and launches the navigation stack. It contains **no executable logic** — just configuration files and launch structure.

#### `launch/autonomous_exploration.launch.py`

The main launch file for the autonomous mission. Starts everything needed for navigation in a staggered sequence:

```
t = 0 s   scan_filter        (needs: /scan from Gazebo)
t = 0 s   zone_detector      (needs: /camera/image_raw, TF)
t = 0 s   static_transform_publisher × 2  (base_link→base_footprint, base_link→base_frame)
t = 0 s   topic relay        (/mirte_base_controller/odom → /odom)
t = 5 s   slam_toolbox       (needs: /scan_filtered; waits for Gazebo to stabilise)
t = 20 s  Nav2 stack         (needs: map→odom TF from SLAM; 15 s for SLAM to build first map)
t = 30 s  exploration_manager (needs: Nav2 to be fully activated; ~5–10 s after Nav2 starts)
```

**Why the topic relay?** Nav2 expects `/odom`. The MIRTE diff_drive controller publishes to `/mirte_base_controller/odom`. The relay bridges them.

**Why static TFs for base_footprint and base_frame?** SLAM is configured with `base_frame: base_footprint`. Nav2 is configured with `robot_base_frame: base_link`. Both frames are needed. Since neither moves relative to base_link, static identity transforms work fine.

#### `params/slam_params.yaml`

Configures slam_toolbox in online synchronous mapping mode.

```yaml
slam_toolbox:
  ros__parameters:
    use_sim_time: true
    mode: "mapping"           # builds map from scratch (vs "localization" which uses a pre-built map)
    resolution: 0.02          # map cell size in metres. 2 cm = high detail but large memory
    map_update_interval: 1.0  # seconds between full map republishes to /map
    use_scan_matching: true   # align scans against existing map for better localisation
    map_topic: /map
    scan_topic: /scan_filtered  # ← uses the filtered scan (no chassis returns)
    base_frame: base_footprint  # robot body frame for SLAM's odometry integration
    odom_frame: odom
    map_frame: map
    transform_publish_period: 0.02  # 50 Hz map→odom TF (smooth navigation needs fast TF)
    transform_timeout: 0.5
    minimum_travel_distance: 0.05   # robot must move 5 cm before adding a new keyframe
    minimum_travel_heading: 0.1     # or rotate 0.1 rad (~6°)
    throttle_scans: 1               # process every scan (set to 2 to halve CPU usage)
```

**What is a keyframe?** SLAM doesn't store every scan. It stores "keyframes" — scans taken when the robot has moved enough. `minimum_travel_distance: 0.05` means a new keyframe is added every 5 cm of movement. Too small = too many keyframes = slow. Too large = gaps in the map.

**Why `resolution: 0.02` and not `0.05`?** The boxes are 80–160 mm. At 0.05 m/cell, the smallest box (80 mm) is only 1–2 cells wide — barely detectable. At 0.02 m/cell it's 4 cells wide — enough to count and compare. However, the **costmap** uses 0.05 m/cell (defined in nav2 params) — SLAM and costmap resolutions are independent.

#### `params/exploration_nav2_params.yaml`

The most important configuration file for navigation. Every section controls a different Nav2 component.

**`planner_server` section — the global planner:**
```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5        # goal tolerance in metres
      use_astar: false       # false = Dijkstra (guaranteed shortest); true = A* (faster)
      allow_unknown: true    # plan through unknown (grey) cells
```

NavFn/Dijkstra explores the costmap from the start, wavefront-style, until it reaches the goal. It finds the globally optimal path. `allow_unknown: true` lets the robot navigate into unmapped areas (necessary during SLAM when the map is incomplete).

**`controller_server` section — the local controller:**
```yaml
controller_server:
  ros__parameters:
    controller_frequency: 10.0   # Hz — how often the controller re-computes cmd_vel
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      use_collision_detection: true
      max_allowed_time_to_collision_up_to_carrot: 0.5   # s × speed = collision lookahead
      desired_linear_vel: 0.35   # m/s forward speed
      lookahead_dist: 0.6        # m — distance ahead on path to target ("carrot")
      use_velocity_scaled_lookahead_dist: true  # faster = look further ahead
      max_lookahead_dist: 1.0    # cap on lookahead
      min_lookahead_dist: 0.3    # minimum lookahead
      use_rotate_to_heading: false  # ← KEY: do NOT snap-rotate before moving
      regulated_linear_scaling_min_radius: 0.3  # slow down on sharp curves
      regulated_linear_scaling_min_speed: 0.15  # minimum speed when slowing
```

**How RPP (Regulated Pure Pursuit) works:**
1. Find the "carrot point" — the point on the global path `lookahead_dist` ahead
2. Compute the arc radius needed to reach the carrot from the robot's current pose
3. Convert arc radius to `(linear_vel, angular_vel)` command
4. Optionally check for collision along the path up to `max_allowed_time_to_collision × speed`

**Why `use_rotate_to_heading: false`?** With `true`, RPP first rotates in place to face the path direction before moving. During this in-place rotation, the robot's footprint sweeps through a circular area. If any pillar or box is within ~0.45 m during this sweep, the costmap shows INSCRIBED cost and RPP aborts with "collision ahead" before the robot moves a millimetre. Setting `false` makes the robot arc toward the goal instead — no in-place rotation, no swept collision.

**`local_costmap` section:**
```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      rolling_window: true    # 3×3 m window centred on robot (moves with it)
      width: 3                # metres
      height: 3
      resolution: 0.05        # 5 cm cells
      footprint: "[[0.20, 0.25], [0.20, -0.25], [-0.15, -0.25], [-0.15, 0.25]]"
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        scan:
          topic: /scan_filtered
          raytrace_min_range: 0.25   # don't clear obstacles within 25 cm of robot
          obstacle_min_range: 0.25   # don't mark obstacles within 25 cm of robot
      inflation_layer:
        inflation_radius: 0.30       # expand obstacles by 30 cm
        cost_scaling_factor: 3.5     # how fast cost drops off with distance
```

**Why `raytrace_min_range: 0.25`?** The scan_filter removes LIDAR readings below 0.25 m (chassis self-returns). If the costmap tried to raytrace (clear free space) all the way to the robot, it would incorrectly clear cells that should remain occupied. This threshold matches the filter.

**The robot footprint** `[[0.20, 0.25], [0.20, -0.25], [-0.15, -0.25], [-0.15, 0.25]]` is a rectangle:
- Front: +0.20 m from centre
- Back: −0.15 m from centre
- Left/right: ±0.25 m from centre

The **inscribed radius** (minimum distance from centre to any edge) = min(0.20, 0.15, 0.25) = **0.15 m**. Cells within 0.15 m of an obstacle edge get LETHAL cost (254). Cells 0.15–0.45 m get decreasing cost.

**`global_costmap` section:** Similar to local but uses a static layer (the SLAM map) plus the obstacle layer from real-time LIDAR. The global costmap is large (100 × 100 m in this config) and doesn't roll.

#### `trees/nav2_minimal_tree.xml`

The Behavior Tree that Nav2's bt_navigator executes for each navigation goal.

```xml
<RecoveryNode number_of_retries="6" name="NavigateRecovery">
  <!-- PRIMARY SEQUENCE -->
  <PipelineSequence name="NavigateWithReplanning">
    <RateController hz="1.0">
      <!-- Replan global path at 1 Hz -->
      <RecoveryNode number_of_retries="1" name="ComputePathToPose">
        <ComputePathToPose .../>          <!-- ask planner_server for a path -->
        <ClearEntireCostmap .../>          <!-- on failure: clear global costmap, retry once -->
      </RecoveryNode>
    </RateController>
    <RecoveryNode number_of_retries="1" name="FollowPath">
      <FollowPath .../>                    <!-- ask controller_server to follow the path -->
      <ClearEntireCostmap .../>            <!-- on failure: clear local costmap, retry once -->
    </RecoveryNode>
  </PipelineSequence>
  <!-- RECOVERY SEQUENCE (runs when primary fails) -->
  <ReactiveFallback name="RecoveryFallback">
    <GoalUpdated/>                         <!-- exit recovery if goal changed -->
    <RoundRobin name="RecoveryActions">
      <!-- Cycle through these on each retry: -->
      <Sequence>
        <ClearEntireCostmap .../>          <!-- clear both costmaps -->
        <ClearEntireCostmap .../>
      </Sequence>
      <BackUp backup_dist="0.15" backup_speed="0.05"/>   <!-- back up 15 cm -->
      <Wait wait_duration="1.0"/>          <!-- wait 1 second -->
    </RoundRobin>
  </ReactiveFallback>
</RecoveryNode>
```

**What was removed**: `<Spin spin_dist="1.57"/>` — a 90° in-place spin recovery. In a cluttered arena, spinning while stuck just re-enters the same collision state. With `use_rotate_to_heading: false`, the collision detection is far less aggressive, so spin recovery is no longer needed.

#### `maps/*.pgm + *.yaml`

Saved SLAM maps. The `.pgm` is a greyscale image where:
- White (255) = free space
- Black (0) = occupied (wall/obstacle)
- Grey (205) = unknown

The `.yaml` references the `.pgm` and provides metadata:
```yaml
image: default.pgm
resolution: 0.05    # metres per pixel
origin: [-50.0, -50.0, 0.0]   # real-world coordinates of the image's bottom-left corner
negate: 0
occupied_thresh: 0.65   # pixels darker than this threshold → occupied
free_thresh: 0.196      # pixels lighter than this threshold → free
```

---

### 3.4 `mirte_workshop/` — your mission logic package

This package contains the actual application intelligence — the nodes that make the robot DO the mission.

#### `mirte_workshop/scan_filter.py`

**A ROS2 node** (53 lines). Subscribes to `/scan` (`sensor_msgs/LaserScan`), filters all readings below `MIN_RANGE = 0.25 m` by setting them to `inf`, republishes on `/scan_filtered`.

**Why is this necessary?** The MIRTE robot's LIDAR is mounted on the robot chassis. When the LIDAR scans in a full circle, it sees parts of the robot's own body (wheels, frame, arm) at very short ranges (0–0.20 m). If these readings reach SLAM or the costmap, the robot would think there are obstacles all around it and never move. Setting them to `inf` tells SLAM and the costmap "no obstacle here" for those angles.

```python
# Core logic (simplified):
for i, r in enumerate(msg.ranges):
    if r < MIN_RANGE:
        msg.ranges[i] = float('inf')   # treat as free space
self._pub.publish(msg)
```

#### `mirte_workshop/zone_detector.py`

**A ROS2 node** (315 lines). The perception system for this project.

**What it does step by step:**

1. **Camera intrinsics** — subscribes to `/camera/camera_info` once, extracts the camera matrix K and distortion coefficients d. These are needed for ArUco pose estimation.

2. **Image callback** — subscribes to `/camera/image_raw`. On each frame:
   - Convert ROS image message to OpenCV BGR array via cv_bridge
   - Convert to greyscale
   - Run `cv2.aruco.detectMarkers()` using DICT_4X4_50
   - For each detected marker ID, call `estimatePoseSingleMarkers()` with the correct physical size:
     - IDs 0, 1 (zones): `ZONE_MARKER_SIZE = 0.20 m`
     - IDs 2–6 (boxes): `BOX_MARKER_SIZE = 0.12 m`
   - Call `_to_map_pose()` to transform from camera frame to map frame

3. **TF transform** (`_to_map_pose`):
   - Look up `map ← camera_depth_optical_frame` from the TF tree
   - Apply the camera-to-map rotation to the ArUco position vector
   - Apply the camera-to-map rotation to the ArUco orientation matrix
   - Return a `PoseStamped` in the `map` frame

4. **EMA smoothing** — position is smoothed with Exponential Moving Average (`alpha = 0.20`): `new = old + 0.20 × (measurement − old)`. This reduces noise from single-frame ArUco detection errors. Orientation is taken from the most recent frame (not EMA'd, as quaternion averaging is complex).

5. **Publishing** — a 5 Hz timer publishes all known poses even when no new detection arrives. This ensures subscribers always have fresh data.

**Topic map:**
```
/camera/image_raw    → [ArUco detection] → /zone_a_pose  (ID=0)
/camera/camera_info  → [intrinsics]      → /zone_b_pose  (ID=1)
/tf                  → [frame transform] → /box_pose/id2 (ID=2)
                                         → /box_pose/id3 (ID=3)
                                         → /box_pose/id4 (ID=4)
                                         → /box_pose/id5 (ID=5)
                                         → /box_pose/id6 (ID=6)
```

**Why `/box_pose/id2` not `/box_pose/2`?** ROS2 topic names are validated. Each `/`-separated token must start with a letter or underscore, not a digit. `id2` is valid; `2` is not.

#### `mirte_workshop/exploration_manager.py`

**A ROS2 node** (~1050 lines). The brain of the autonomous mission. A state machine that drives through 6 states.

**State machine overview:**

```
┌──────────────┐
│  INIT_SPIN   │ Robot spins in place at 0.4 rad/s (~15.7 s/revolution)
│   Phase 1    │ Watching for Zone A ArUco (ID=0) via /zone_a_pose
└──────┬───────┘
       │ Zone A detected → stop spin
       ▼
┌──────────────┐
│  GOTO_POLE   │ Navigate to 1.0 m before Zone A on the straight robot→A line
│   Phase 2    │ Uses _direct_approach() — no orientation inference
└──────┬───────┘
       │ Arrived
       ▼
┌──────────────┐
│   SURVEY     │ Compute scan grid from Zone A orientation
│   Phase 3    │ Navigate + 360° spin at each position
└──────┬───────┘ Detect box markers (IDs 2–6)
       │ All 5 found OR stale count exhausted
       ▼
┌──────────────┐
│  GOTO_BOX    │ Navigate to 1.0 m before current box (largest→smallest)
│   Phase 4    │ Uses _find_approach_with_los() — marker orientation matters here
└──────┬───────┘ (approach from the east, facing the ArUco)
       │ Arrived
       ▼
┌──────────────┐  Phase 0: Navigate to Zone A relay (direct approach)
│   GOTO_B     │  Phase 1: Navigate east to Zone B (direct approach)
│   Phase 5    │
└──────┬───────┘
       │ Arrived at Zone B
       ├─── More boxes? → GOTO_BOX (next box, idx++)
       ▼
┌──────────────┐
│     DONE     │ Save SLAM map as .pgm + .yaml, open in image viewer
│   Phase 6    │
└──────────────┘
```

**Key methods explained:**

`_robot_xy()`:
```python
def _robot_xy(self):
    # Looks up the robot's current position in map frame via TF2
    # Returns (x, y) or None if TF unavailable
    tf = self._tf_buf.lookup_transform('map', 'base_link', ...)
    return (tf.transform.translation.x, tf.transform.translation.y)
```

`_direct_approach(target_pose, dist=1.0)`:
```python
# Gets robot current position
# Draws a line: robot → target
# Places approach point 'dist' metres before the target on that line
# Checks that the approach point has 0.55 m map clearance
# Falls back to shorter distances if needed
# Returns (x, y, yaw) where yaw faces the target
```

Why this replaces `_find_approach_with_los` for zones: `_find_approach_with_los` used the ArUco marker's detected `+Z` axis (outward direction) to decide which side of the marker to approach from. Early EMA readings produce noisy quaternions. A noisy quaternion can point the `+Z` axis in the wrong direction — causing the approach to be placed on the wrong side of the marker. `_direct_approach` doesn't care about the quaternion at all; it just goes toward the target from wherever the robot currently is.

`_plan_survey()`:
```python
# Uses Zone A pole's detected orientation to find "into zone" direction
# Lays a grid: 3 depths (0.8, 1.8, 2.8 m) × lateral offsets (0, ±1, ±2, ±3, ±4 m)
# Filters positions without 0.45 m map clearance
# Returns list of (x, y, yaw) scan positions, yaw facing back toward the pole
```

`_rank_boxes()`:
```python
# For each detected box marker:
#   1. Get marker position (x, y) from _box_poses
#   2. Get marker orientation → compute facing direction (+Z axis)
#   3. Offset 0.08 m inward (into the box) to get counting centre
#   4. Count SLAM occupied cells within 0.30 m radius (circular window)
# Sort by count descending → largest box first
```

`_save_map()`:
```python
# Spawns a background thread (so node doesn't block)
# Runs: ros2 run nav2_map_server map_saver_cli -f ~/mirte_maps/default
# Waits up to 30 s for it to complete
# On success: opens the .pgm file with eog / xdg-open / feh / display
```

**Constants table (defined at module level):**

| Constant | Value | Meaning |
|---|---|---|
| `SLAM_WAIT_TIMEOUT` | 30.0 s | Max wait for map→odom TF before aborting |
| `GLOBAL_TIMEOUT` | 900.0 s | Total mission timeout (15 min) |
| `TICK_HZ` | 2.0 Hz | State machine evaluation rate |
| `SPIN_RATE_RAD_S` | 0.4 rad/s | Angular velocity during init spin |
| `SPIN_DURATION` | 15.7 s | Time for one full revolution at 0.4 rad/s |
| `MAX_SEARCH_CYCLES` | 8 | Max spin cycles before giving up on Zone A |
| `APPROACH_DIST_MIN` | 1.00 m | How far before a target to stop |
| `APPROACH_DIST_MAX` | 3.00 m | Max distance searched for clear approach |
| `MAX_CONSEC_ABORTS` | 3 | Nav failures before skipping |
| `GOAL_TIMEOUT` | 90.0 s | Max time for one nav goal |
| `SURVEY_DEPTHS` | [0.8, 1.8, 2.8] m | Depths into zone to place scan positions |
| `SURVEY_LAT_STEP` | 1.0 m | Lateral spacing between scan columns |
| `SURVEY_MAX_STALE` | 2 | Consecutive no-new-marker spins before ending survey |
| `SLAM_COUNT_RADIUS` | 0.30 m | Radius for box size counting |

---

### 3.5 Supporting Packages

#### `m-explore-ros2/` — frontier exploration

`explore_lite` is a frontier-based autonomous exploration algorithm. A **frontier** is a boundary between mapped free space and unknown space. `explore_lite` finds frontiers and sends the robot toward them to fill in the map.

It's installed as a dependency but **not used** in the autonomous mission (the SURVEY state handles exploration manually). The `params/explore_params.yaml` file in mirte_navigation is a leftover config.

#### `gazebo_grasp_fix/`

A Gazebo plugin that makes grasping simulation work. Real Gazebo physics doesn't handle gripper contact well (objects slide through). This plugin adds an artificial attachment joint when the gripper closes around an object. Not used in the pick-and-place simulation (the mission doesn't physically pick up boxes — it's conceptual).

#### `mirte_location_markers/`

Provides ROS2 services to store named poses (`StorePose`) and navigate to them (`MoveTo`). Pre-existing workshop utility, not used in the autonomous mission.

---

## 4. System Architecture — How Everything Connects

```
                        ┌─────────────────────────────────┐
                        │           GAZEBO                │
                        │  diff_drive  camera  lidar      │
                        └──┬──────────┬──────────┬────────┘
                           │/odom     │/camera   │/scan
                           ▼          ▼          ▼
                    relay→/odom  zone_detector  scan_filter→/scan_filtered
                           │          │                │
                           │      /zone_a_pose      slam_toolbox
                           │      /zone_b_pose         │/map    │/tf(map→odom)
                           │      /box_pose/idN         │         │
                           │          │         global_costmap   │
                           ▼          ▼          │               ▼
                     bt_navigator ←── exploration_manager   Nav2 TF chain
                           │          │(navigate_to_pose actions)
                    controller_server ← planner_server
                           │                │
                    local_costmap    global_costmap
                           │                │
                    cmd_vel→diff_drive (Gazebo moves the robot)
```

---

## 5. The Full Data Flow

**From boot to Zone A detection:**

1. Gazebo starts → diff_drive plugin publishes `/mirte_base_controller/odom` + `odom→base_link` TF at 50 Hz
2. relay node bridges `/mirte_base_controller/odom` → `/odom`
3. robot_state_publisher reads the URDF → publishes all fixed joint TFs (base_link→lidar_link etc.)
4. scan_filter starts → passes through `/scan_filtered`
5. slam_toolbox starts → receives `/scan_filtered`, integrates odometry from `base_footprint` TF, builds occupancy map, publishes `/map` + `map→odom` TF
6. Nav2 lifecycle activates → costmaps load the `/map` static layer, RPP controller activates
7. exploration_manager starts → waits for `map→odom` TF
8. Manager starts INIT_SPIN → publishes angular velocity to cmd_vel at 10 Hz
9. Robot rotates → camera sees Zone A ArUco ID=0
10. zone_detector: `estimatePoseSingleMarkers` → tvec/rvec in camera frame → `lookup_transform(map ← camera_depth_optical_frame)` → publish `/zone_a_pose`
11. Manager's `_zone_a_cb` fires → stops spin → transitions to GOTO_POLE

**From GOTO_POLE to SURVEY:**

12. `_direct_approach(zone_a_pose)` → get robot TF position → compute point 1.0 m before Zone A on the straight line → check map clearance
13. `_send_nav_goal(ax, ay, ayaw)` → sends `NavigateToPose` action to bt_navigator
14. bt_navigator calls `ComputePathToPose` → NavFn searches the global costmap via Dijkstra → returns global path
15. bt_navigator calls `FollowPath` → RPP computes cmd_vel each control cycle:
    - Find carrot point 0.6 m ahead on path
    - Compute arc to reach carrot
    - Check local costmap 0.175 m ahead for collision
    - Publish `cmd_vel`
16. Robot moves → SLAM updates map → global path replanned at 1 Hz
17. Robot reaches approach → `_goal_done_cb` → SUCCESS → `_on_goal_succeeded` → transition to SURVEY

---

## 6. All Code Changes Made

### 6.1 `mirte-gazebo/worlds/exploration_arena.world`

| Change | Before | After | Reason |
|---|---|---|---|
| Risers | 5 different sizes matching their boxes | All identical 180×220×110 mm | Uniform platform; LIDAR still only sees box walls (riser below LIDAR plane) |
| ArUco z-height | 0.125 m (middle of box) | 0.070 m (middle of riser face) | Marker now on riser, not box |
| ArUco x-position | Box east face | Riser east face (+0.090 m from riser centre) | Shifted outward by riser half-width |

### 6.2 `mirte-gazebo/launch/gazebo_exploration_arena.launch.py`

| Change | Before | After | Reason |
|---|---|---|---|
| `_CLEARANCE` | 0.55 m | 1.00 m | Prevent spawn inside pillar's inflation zone, which caused NavFn to route through INSCRIBED zone |

### 6.3 `mirte_navigation/params/exploration_nav2_params.yaml`

| Parameter | Before | After | Reason |
|---|---|---|---|
| `use_rotate_to_heading` | `true` | `false` | In-place rotation sweeps footprint through nearby obstacles → immediate collision abort |
| `max_allowed_time_to_collision_up_to_carrot` | 1.0 | 0.5 | Reduces lookahead from 0.35 m to 0.175 m; avoids false positives from NavFn paths that marginally enter high-cost zones |
| `inflation_radius` | 0.30 | **reverted to 0.30** | Reducing to 0.20 made NavFn route paths into the INSCRIBED zone (cost ≥ 253), making RPP abort more frequently |

### 6.4 `mirte_navigation/trees/nav2_minimal_tree.xml`

| Change | Before | After | Reason |
|---|---|---|---|
| Spin recovery | `<Spin spin_dist="1.57"/>` present | Removed | Spinning in cluttered arena re-enters same collision state; worse than just waiting |
| Wait duration | 5.0 s | 1.0 s | Shorter recovery cycle |
| Outer retries | 3 | 6 | More attempts before giving up |

### 6.5 `mirte_navigation/launch/autonomous_exploration.launch.py`

| Change | Before | After | Reason |
|---|---|---|---|
| Nav2 start delay | 15 s | 20 s | Give Gazebo's diff_drive plugin time to publish `odom→base_link` TF |
| Manager start delay | 20 s | 30 s | Give Nav2 lifecycle activation time (~5–10 s after Nav2 starts) |

### 6.6 `mirte_workshop/mirte_workshop/zone_detector.py`

| Change | Before | After | Reason |
|---|---|---|---|
| Box publisher topic | `/box_pose/2` | `/box_pose/id2` | ROS2 forbids topic tokens starting with digits |
| Per-detection log level | `INFO` (throttled 2 s) | `DEBUG` | Reduces log spam; first detection stays at INFO |

### 6.7 `mirte_workshop/mirte_workshop/exploration_manager.py`

**Additions:**

| Addition | Purpose |
|---|---|
| `import threading` | For non-blocking map save |
| `self._nav_backoff_until_ns` | 3 s backoff after rejected goals (prevents spam when Nav2 not ready) |
| `self._expected_box_ids` | Frozenset of expected IDs for early survey exit |
| `self._goto_b_phase` | Phase tracker for two-step GOTO_B |
| `_robot_xy()` | TF lookup for robot's current map position |
| `_direct_approach()` | Straight-line approach that ignores ArUco orientation |
| Phase banners | `▶▶ PHASE N: description ◀◀` in bold yellow |

**Removals:**

| Removed | Was doing | Why removed |
|---|---|---|
| `_FP_EMPTY`, `_FP_CARRYING` constants | Defined empty/carrying footprint strings | Carrying footprint made navigation harder (wider footprint enters inflation zones) |
| `_fp_clients` | Service clients for footprint switching | No longer needed |
| `_set_footprint()` | Called costmap set_parameters service | No longer needed |
| `rcl_interfaces` imports | Needed for footprint switching | No longer needed |
| `_approach_a`, `_approach_b` instance variables | Pre-computed approach waypoints | Approach now computed fresh in each tick using _direct_approach |
| `_find_approach_with_los` for zones | Orientation-based approach | Replaced by _direct_approach (more reliable) |

**Modified behaviours:**

| Method | Change | Effect |
|---|---|---|
| `_zone_a_cb` / `_zone_b_cb` | Removed approach computation on detection | Approach computed fresh in tick with current robot position |
| `_tick_goto_pole` | Replaced with `_direct_approach` | Robot goes directly toward Zone A, no orientation dependency |
| `_tick_survey` | Added early-exit when all expected IDs found | Survey ends immediately when all 5 box markers are detected, no waiting for stale counter |
| `_tick_goto_b` | Two-phase: Zone A relay then Zone B | Uses known A→B east corridor instead of cross-arena pathfinding |
| `_on_goal_succeeded` / `_on_nav_failed` | Handle GOTO_B two-phase logic | Phase 0 failure falls through to phase 1 (direct Zone B); each phase has independent abort counter |
| `_count_occupied_near` | Added circular check `dr² + dc² > r²` | Count window is now a circle not a square (consistent with `_has_clearance`) |
| `_rank_boxes` | Offset count centre 0.08 m inward | Counts box footprint cells, not cells in front of the marker face |
| `_save_map` | Threaded synchronous save + open viewer | Node stays responsive; map displays automatically on completion |
| `APPROACH_DIST_MIN` | 0.70 → 1.00 m | Wider clearance from Zone B stand's inflation zone |
| `_find_approach_with_los` clearance | 0.45 → 0.55 m | Accounts for inflation (0.20) + footprint half-width (0.25) + margin (0.10) |

---

## 7. Navigation vs Mapping — Exam Division

### 7.1 What "Mapping" means here

Mapping is the process of **representing the unknown environment** in a form the robot can reason about. It includes:
- Building the occupancy grid (SLAM)
- Filtering sensor data to prevent bad map entries
- Detecting and localising objects of interest (ArUco markers)
- Computing relative box sizes from the map
- Designing the survey strategy to find all objects

### 7.2 What "Navigation" means here

Navigation is **moving the robot safely through the known/partially-known environment** toward a goal. It includes:
- Computing globally optimal paths
- Following paths with real-time obstacle avoidance
- Designing recovery behaviours when navigation fails
- Computing safe approach waypoints
- Designing the high-level state machine flow

---

### 7.3 Your Code (Navigation)

**Files you own and explain:**

**`params/exploration_nav2_params.yaml`** — You tuned every parameter here. Key talking points:
- Why NavFn over A*? Dijkstra finds the globally optimal path; A* is faster but the arena is small enough that NavFn's 1 Hz replanning is sufficient.
- Why RPP over DWB? RPP is simpler and predictable for this environment; DWB's trajectory sampling is better for highly dynamic environments.
- Why `use_rotate_to_heading: false`? The arena has pillars close to valid approach paths. In-place rotation sweeps the footprint through nearby obstacles → immediate abort. Arcing avoids this entirely.
- Why `inflation_radius: 0.30`? Calibrated against the pillar geometry. At 0.20 m, NavFn routes paths into the INSCRIBED zone. At 0.40 m, some corridors become too narrow to navigate.
- What does `max_allowed_time_to_collision_up_to_carrot: 0.5` do? At 0.35 m/s, RPP only checks 0.175 m ahead for collision. This prevents false positives from paths that marginally enter high-cost (non-lethal) zones.

**`trees/nav2_minimal_tree.xml`** — Explain every node type and why you designed the recovery this way:
- Why no Spin? Spinning in a corridor just knocks into the other wall.
- Why BackUp before Wait? BackUp moves the robot away from what it collided with; then Wait lets dynamic obstacles clear.
- Why 6 outer retries? Three backup-wait cycles before giving up, matching the 3-strike abort logic in exploration_manager.

**`exploration_manager.py` — navigation parts:**
- The entire state machine structure (which states exist, how transitions work)
- `_direct_approach()` — why you replaced orientation-based approach with position-based approach (noisy quaternion → wrong direction)
- `_tick_goto_b()` two-phase relay — why Zone A as relay is clever: it guarantees the robot is in the correct position for the well-understood A→B east corridor. Without relay, the robot tries random cross-arena paths that may go through unseen corridors.
- The goal backoff mechanism — why needed (Nav2 rejects goals while still activating; without backoff → 2 Hz goal spam)
- The `_nav_backoff_until_ns` pattern — explain the async nature of ROS2 callbacks and why you can't just use a blocking sleep

**`launch/autonomous_exploration.launch.py`** — Explain the stagger:
- Gazebo Gazebo Gazebo starts first → diff_drive publishes odom+TF
- scan_filter and zone_detector at t=0 (need camera/scan immediately)
- SLAM at t=5 s (small buffer for Gazebo to stabilise)
- Nav2 at t=20 s (needs SLAM TF; 15 s buffer for SLAM to publish first map→odom)
- Manager at t=30 s (needs Nav2 to be activated; ~5–10 s for lifecycle activation)

---

### 7.4 Your Teammate's Code (Mapping)

**Files they own and explain:**

**`params/slam_params.yaml`** — Every parameter has a purpose:
- `resolution: 0.02` — why 2 cm for this task: the smallest box is 80 mm wide = 4 cells at 2 cm/cell. At 5 cm/cell it's only 1–2 cells. Insufficient for size ranking.
- `minimum_travel_distance: 0.05` — keyframe spacing. Too small = too many keyframes = high CPU. Too large = gaps. 5 cm is a good balance for 0.35 m/s robot speed.
- `transform_publish_period: 0.02` — 50 Hz map→odom TF. Nav2's controller runs at 10 Hz. It needs TF at least that fast. 50 Hz provides smooth interpolation.
- `base_frame: base_footprint` — SLAM uses base_footprint (not base_link) because it's the projection onto the ground plane. Odometry should be 2D-consistent.

**`zone_detector.py`** — The full ArUco pipeline:
- Why DICT_4X4_50? 50-marker dictionary, 4×4 bit pattern. Simple enough for fast detection, enough markers for this application.
- What is `estimatePoseSingleMarkers`? Takes the 4 corner pixel positions of a detected marker, the known physical size, and the camera matrix. Returns a rotation vector (rvec) and translation vector (tvec) — the marker's 6DOF pose in camera frame.
- Why EMA smoothing? ArUco pose estimation has ~1–5 cm noise at 1 m distance. EMA with α=0.20 averages ~5 frames, reducing noise by ~√5 ≈ 2.2×.
- The quaternion helpers — explain `_quat_rotate`, `_matrix_to_quat`. Why not use scipy? The node is meant to have minimal dependencies; these are small enough to implement inline.

**`scan_filter.py`** — Simple but important:
- Without it, the robot sees itself as an obstacle everywhere it goes.
- The 0.25 m threshold matches `raytrace_min_range: 0.25` in the costmap config.
- This is a direct relationship: the filter threshold and the costmap min range must agree.

**`exploration_manager.py` — mapping parts:**

`_plan_survey()`:
```python
# The survey grid is derived entirely from Zone A's detected orientation:
# 1. Get Zone A's +Z axis in world frame (marker facing direction)
# 2. "Into zone" direction = opposite of facing (+180°)
# 3. "Lateral" direction = perpendicular to into-zone
# 4. Grid: depth ∈ {0.8, 1.8, 2.8} m × lateral ∈ {0, ±1, ±2, ±3, ±4} m
# 5. Filter: only positions with 0.45 m SLAM clearance
# 6. Deduplicate: positions closer than 1.2 m to existing kept position are dropped
```

Why this design: The robot doesn't know the zone size in advance. Three depth levels cover zones of varying extent without hardcoding dimensions.

`_rank_boxes()`:
```python
# The counting centre is offset 0.08 m inward from the marker face because:
# - ArUco is on the RISER east face (not box centre)
# - Riser half-depth = 0.090 m
# - 0.08 m offset ≈ riser centre ≈ box centre
# - Counting at the box centre captures the full footprint, not empty air
```

Why circular window in `_count_occupied_near`: a square window counts corner cells that are actually farther away than the radius. Circles give a consistent, distance-based count. At radius = 0.30 m, `π × (0.30/0.05)² ≈ 113 cells` in the circle vs 169 cells in the square.

**`exploration_arena.world` — physical setup:**
- Why riser height = 110 mm? The LIDAR is at 129 mm. Boxes are 30 mm tall. Box bottom = 110 mm, box top = 140 mm. LIDAR at 129 mm = 19 mm above box bottom = scanning the lower 1/3 of the box wall. Any height below 129 mm - 30 mm = 99 mm would work; 110 mm is chosen to give a 19 mm margin above the riser top.
- Why all risers the same size (after changes)? Simplifies world construction; LIDAR still only sees box walls because riser is below the LIDAR plane regardless of riser footprint size.
- Why ArUco on riser not box? The ArUco gives the box's location. The LIDAR gives the box's size (via SLAM footprint counting). Separating the two means size estimation uses the actual box geometry, not the marker position.

---

### 7.5 Shared Components

Both parties contributed to `exploration_manager.py`. For the exam, know which parts you each own:

| Component | Owner |
|---|---|
| State machine skeleton (states, transitions, timers) | Navigation |
| `_tick_goto_pole`, `_tick_goto_box`, `_tick_goto_b` | Navigation |
| `_direct_approach`, `_find_approach_with_los` | Navigation |
| `_send_nav_goal`, goal callbacks, backoff | Navigation |
| `_plan_survey`, `_tick_survey` | Mapping |
| `_rank_boxes`, `_count_occupied_near` | Mapping |
| `_save_map` | Mapping (map output) |
| `_slam_ready`, `_has_clearance`, `_has_los` | Shared (boundary of nav + mapping) |

---

## 8. Putting it on GitHub

### 8.1 What to include in your repo

You own two packages. Everything else is an external dependency.

```bash
mkdir ~/autonomous_mission
cd ~/autonomous_mission
git init

# Copy your packages
cp -r /home/gui/spatial-ai/ws/src/mirte_navigation .
cp -r /home/gui/spatial-ai/ws/src/mirte_workshop .

# Remove build artefacts from mirte_workshop
rm -rf mirte_workshop/build mirte_workshop/install mirte_workshop/log

# Remove editor/tool files (optional — they're not part of the package)
rm -f mirte_workshop/.claude/settings.local.json

# Add gitignore
cat > .gitignore << 'EOF'
build/
install/
log/
*.pyc
__pycache__/
*.egg-info/
.vscode/browse.vc.db*
EOF

git add .
git commit -m "Initial: autonomous pick-and-place navigation + mission stack"
```

### 8.2 The world changes (mirte-gazebo)

Your changes to `mirte-gazebo` are in an external repository. Two options:

**Option A (fork):** Fork `mirte-gazebo` on GitHub and push your changes there. Reference the fork URL in your README.

**Option B (copy into your repo):** Copy just the two changed files:
```bash
mkdir -p overrides/mirte-gazebo/worlds
mkdir -p overrides/mirte-gazebo/launch
cp /home/gui/spatial-ai/ws/src/mirte-gazebo/worlds/exploration_arena.world overrides/mirte-gazebo/worlds/
cp /home/gui/spatial-ai/ws/src/mirte-gazebo/launch/gazebo_exploration_arena.launch.py overrides/mirte-gazebo/launch/
```
Add a README explaining these override the corresponding files in the mirte-gazebo dependency.

### 8.3 Workspace setup README

Add a `README.md` at the repo root explaining how to set up the full workspace:

```markdown
## Dependencies (clone alongside this repo in a colcon workspace)

- mirte-ros-packages: [URL]
- mirte-gazebo:       [URL] (or use the override files in overrides/)
- slam_toolbox:       sudo apt install ros-humble-slam-toolbox
- nav2:               sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
- gazebo_grasp_fix:   [URL]

## Running

Terminal 1 (simulation):
  ros2 launch mirte_gazebo gazebo_exploration_arena.launch.py

Terminal 2 (navigation + mission, wait for Gazebo to fully load first):
  ros2 launch mirte_navigation autonomous_exploration.launch.py
```

### 8.4 Recommended repo structure

```
autonomous_mission/
├── mirte_navigation/
│   ├── package.xml
│   ├── setup.py
│   ├── launch/
│   │   └── autonomous_exploration.launch.py     ← yours
│   ├── params/
│   │   ├── slam_params.yaml                     ← yours
│   │   └── exploration_nav2_params.yaml         ← yours
│   ├── trees/
│   │   └── nav2_minimal_tree.xml                ← yours
│   └── maps/                                    ← saved maps
│
├── mirte_workshop/
│   ├── package.xml
│   ├── setup.py
│   └── mirte_workshop/
│       ├── exploration_manager.py               ← yours (main)
│       ├── zone_detector.py                     ← yours
│       └── scan_filter.py                       ← yours
│
├── overrides/
│   └── mirte-gazebo/
│       ├── worlds/exploration_arena.world       ← your world
│       └── launch/gazebo_exploration_arena.launch.py  ← your spawn
│
├── PROJECT_DEEP_DIVE.md                         ← this file
└── README.md                                    ← setup instructions
```

---

*Document generated from session history and code analysis. Last updated: 2026-05-29.*
