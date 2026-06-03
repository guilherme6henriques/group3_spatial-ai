# Session Recap — Navigation, SLAM & Arm Control

## What was accomplished

### 1. Fixed `arm_task_server.py`

**Problem:** The original skeleton had two bugs:
- Used `Float64MultiArray` published to `/mirte_master_gripper_controller/commands` — but the actual gripper controller is a `GripperActionController`, which only accepts `control_msgs/action/GripperCommand` actions, not raw topic messages.
- Used `rclpy.spin_until_future_complete()` inside a service callback that was already being spun by `rclpy.spin()` — this is a deadlock: the single thread is stuck waiting for a future that can only complete once the callback returns.

**Fix applied:**
- Replaced the `Float64MultiArray` publisher with an `ActionClient` for `GripperCommand` (matching the pattern in `gripper_server.py`).
- Replaced `rclpy.spin_until_future_complete()` with `send_goal_async()` + `time.sleep(2)` — fire-and-forget, consistent with how the arm trajectory publisher works.
- Changed `main()` to use `MultiThreadedExecutor` + `ReentrantCallbackGroup` so that the gripper action callbacks can be processed by background threads while the service handler is sleeping.

**File:** `mirte_workshop/arm_task_server.py`

**To run:**
```bash
# Terminal 1: Gazebo already running
# Terminal 2:
ros2 run mirte_workshop arm_task_server.py
# Terminal 3:
ros2 service call /deliver_package_1 std_srvs/srv/Trigger
```

**Architecture note:** `arm_task_server.py` is a *server* — it does nothing until a service is called. Two terminals are needed: one keeps the server alive and listening, the other sends the trigger. This is unlike `ros2 topic pub` which directly commands the hardware.

---

### 2. Fixed `minimal_nav2_params.yaml` for Gazebo

**File:** `mirte_navigation/params/minimal_nav2_params.yaml`

**Problems found and fixed:**

| Issue | Old value | New value |
|---|---|---|
| `use_sim_time` (all nav2 nodes) | `false` | `true` — required for Gazebo simulated clock |
| Map server path | `/home/mirte/mirte_ws/...` | `/home/gui/spatial-ai/ws/...` — hardcoded to wrong machine |
| BT navigator XML path | `/home/mirte/mirte_ws/...` | `/home/gui/spatial-ai/ws/...` |
| `min_vel_x` | `0.3` | `0.0` — 0.3 m/s minimum prevented the robot from slowing down near goals |
| `min_speed_xy` | `0.3` | `0.0` — same reason |
| `desired_linear_vel` (RPP) | `0.3` | `0.2` — more conservative for Mirte Master in small spaces |
| `lookahead_dist` (RPP) | `0.35` | `0.5` — smoother path following, less oscillation |

> **Important for real robot:** When switching from Gazebo to the real robot, `use_sim_time` must be set back to `false` and the paths must match the robot's filesystem (usually `/home/mirte/mirte_ws/...`).

---

### 3. Fixed `slam_params.yaml` for Gazebo

**File:** `mirte_navigation/params/slam_params.yaml`

Changed `use_sim_time: false` → `use_sim_time: true`. Same reason as above — without this, SLAM ignores the simulated clock and timestamps desync from the lidar data.

> **For real robot:** revert to `false`.

---

### 4. Nav2 algorithm selection — rationale

**Global planner candidates:**

| Planner | Plugin string | Tradeoff |
|---|---|---|
| NavfnPlanner | `nav2_navfn_planner/NavfnPlanner` | Dijkstra/A*, fast, reliable in open environments |
| SmacPlanner2D | `nav2_smac_planner/SmacPlanner2D` | A* with path smoothing, better curve quality |
| SmacPlannerHybrid | `nav2_smac_planner/SmacPlannerHybrid` | Nonholonomic-aware SE2 planning, overkill for diff-drive |

**Local controller candidates:**

| Controller | Plugin string | Tradeoff |
|---|---|---|
| RegulatedPurePursuitController (RPP) | `nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController` | Smooth, easy to tune, designed for diff-drive |
| DWB | `dwb_core::DWBLocalPlanner` | Robust default, more parameters to tune |
| MPPI | `nav2_mppi_controller::MPPIController` | High quality trajectories, computationally expensive |

**Chosen combination: NavfnPlanner + RPP**

Mirte Master is a small differential-drive robot operating in a structured, low-clutter indoor environment (the Gazebo empty world / workshop floor). NavfnPlanner is sufficient — the environment has no narrow corridors requiring SE2-aware planning. RPP is explicitly designed for this robot class, produces smooth velocity profiles, and has far fewer parameters to tune than DWB. MPPI is too computationally heavy for the onboard hardware.

---

### 5. SLAM mapping in Gazebo

**Packages used:** `slam_toolbox` (synchronous mode)

**Steps performed:**
1. Launched Gazebo with Mirte Master (`gazebo_mirte_master_empty.launch.xml`)
2. Launched SLAM (`minimal_slam_launch.py`) — starts `sync_slam_toolbox_node`, static TF publishers, and an odom relay
3. Drove the robot around using `teleop_twist_keyboard` remapped to `/mirte_base_controller/cmd_vel_unstamped`
4. Saved the map with `map_saver_cli`

**Map saved to:**
```
mirte_navigation/maps/default.pgm   (occupancy grid image)
mirte_navigation/maps/default.yaml  (metadata: resolution, origin, thresholds)
```

**Map quality tips:**
- White = free space, Black = walls, Grey = unknown
- Drive slowly — fast turns introduce scan-matching errors and doubled walls
- Re-save and check in an image viewer (`eog default.pgm`) while still mapping
- If walls are doubled or distorted, restart SLAM and remap

**RViz note:** The map display (`/map` topic) failed to render due to a GLSL shader bug on this machine's GPU driver (`active samplers with a different type refer to the same texture image unit`). SLAM itself worked correctly — the bug is display-only. Workaround: `LIBGL_ALWAYS_SOFTWARE=1 rviz2`.

---

## Pending tasks

- **Station A / Station B:** Drive to each location with nav stack running, then:
  ```bash
  ros2 run mirte_location_markers pose_manager.py
  ros2 service call /store_current_pose mirte_location_markers/srv/StorePose "{label: 'station_a'}"
  # drive to B
  ros2 service call /store_current_pose mirte_location_markers/srv/StorePose "{label: 'station_b'}"
  ```
  Locations are stored in `mirte_location_markers/locations/stored_poses.yaml` and can be re-recorded any time.

- **Nav2 testing:** Launch nav stack and test a 2D Nav Goal in RViz (after setting a 2D Pose Estimate first).

---

## Running on the real robot

### Architecture overview

The real Mirte Master runs Ubuntu + ROS2 Humble on an onboard Raspberry Pi. Your laptop connects to it over WiFi. ROS2 uses DDS (Data Distribution Service) for communication — topics, services, and actions are automatically shared across the network as long as:
1. Both machines are on the **same WiFi network**
2. Both have the same `ROS_DOMAIN_ID` (default is 0 — usually fine)

### What runs where

| Component | Where it runs | Why |
|---|---|---|
| Hardware drivers (`mirte_bringup`) | **Robot** | Direct access to motors, lidar, servos |
| SLAM / Nav2 stack | **Robot** (recommended) or laptop | Lower latency for sensor data; laptop works but adds network delay |
| RViz | **Laptop** | GUI requires a display; subscribes to topics over WiFi |
| `teleop_twist_keyboard` | **Laptop** | Publishes cmd_vel over WiFi to the robot |
| `arm_task_server.py` | **Robot** or laptop | Either works; robot is cleaner |

### Key differences from simulation

| Parameter | Gazebo (simulation) | Real robot |
|---|---|---|
| `use_sim_time` | `true` | `false` |
| Map/XML paths | `/home/gui/spatial-ai/ws/...` | `/home/mirte/mirte_ws/...` |
| Motor drivers | Gazebo plugins | `mirte_bringup` + `telemetrix` |
| Lidar | Simulated `/scan` | Real RPLidar on `/scan` |
| Odometry | Gazebo physics | Wheel encoders via `mirte_base_control` |

### Workflow on the real robot

```bash
# On the ROBOT (via SSH):
ros2 launch mirte_bringup minimal_master.launch.py   # starts hardware

# SLAM (on robot):
ros2 launch mirte_navigation minimal_slam_launch.py

# On the LAPTOP (to visualise):
rviz2   # add Map (/map) and LaserScan (/scan) displays

# Drive (on laptop):
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r /cmd_vel:=/mirte_base_controller/cmd_vel_unstamped

# Save map (on robot):
ros2 run nav2_map_server map_saver_cli \
  -f /home/mirte/mirte_ws/src/mirte_navigation/maps/default
```

Remember to set `use_sim_time: false` in both `slam_params.yaml` and `minimal_nav2_params.yaml` before running on the real robot, and restore the `/home/mirte/mirte_ws/` paths.

---

## What to commit to git

### `mirte_workshop` (your fork: `guilherme6henriques/group3_spatial-ai`)
These are your team's work — commit and push:

| File | What changed |
|---|---|
| `mirte_workshop/arm_task_server.py` | Full rewrite — gripper action client, MultiThreadedExecutor, deadlock fix |
| `mirte_workshop/gripper_server.py` | Minor fixes |
| `setup.py` | Entry point additions |

```bash
cd /home/gui/spatial-ai/ws/src/mirte_workshop
git add mirte_workshop/arm_task_server.py mirte_workshop/gripper_server.py setup.py
git commit -m "Fix arm_task_server: gripper action client, MultiThreadedExecutor, velocity tuning"
git push
```

### `mirte_navigation` (upstream: `MartijnWisse/mirte_navigation`)
This is the professor's repo — you do **not** have push access. Options:

1. **Keep changes local only** (simplest — fine for the workshop)
2. **Fork `mirte_navigation`** on GitHub, push there, and update your `.repos` or `ros2_ws` file to point to your fork
3. **Ask the professor** if he wants a PR with the `use_sim_time` and path fixes (they are valid improvements)

The map files (`default.pgm`, `default.yaml`) and param changes are the main things worth preserving. At minimum, copy the map files somewhere safe.
