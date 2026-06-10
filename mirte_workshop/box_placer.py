#!/usr/bin/env python3
"""
box_placer.py  —  Mirte Master box stacking (no IK, hardcoded joints).

TUNING CONSTANTS  (top of file)
────────────────────────────────
  POSE_CARRY_FALLBACK  — arm joints while carrying a box
  POSE_PLACE_FALLBACK  — arm joints at fully lowered (place) position
  WRIST_CARRY          — wrist angle at top of motion  (-0.3 rad)
  WRIST_PLACE          — wrist angle at bottom of motion (-0.7 rad)
  GRIPPER_OPEN         — gripper open position  (-0.3 rad)
  GRIPPER_CLOSED       — gripper closed position (0.35 rad)
  GRIP_EFFORT          — gripper motor effort (increase if it won't close)

MOVEMENT SEQUENCE  (per box)
──────────────────────────────
  1. CLOSE_GRIP    — gripper closes
  2. PLACE_DOWN    — arm lowers; wrist sweeps WRIST_CARRY → WRIST_PLACE
                     → publishes /arm_placed True
  3. WAIT_FOR_BACK — waits for /robot_backed_up from marker_navigator
  4. OPEN_GRIP     — gripper opens, box released
  5. SETTLING      — brief pause
  6. RETURN_HOME   — arm raises; wrist sweeps WRIST_PLACE → WRIST_CARRY
  7. DONE          — box count saved

SIGNAL FLOW
───────────
  box_placer       ──/arm_placed──────► marker_navigator
  marker_navigator ──/robot_backed_up──► box_placer
  (you)            ──/start_placing───► box_placer

HOW TO USE
──────────
  Step 1 — move arm to carry position:
      ros2 action send_goal /mirte_master_arm_controller/follow_joint_trajectory \
        control_msgs/action/FollowJointTrajectory \
        "{trajectory: {joint_names: [shoulder_pan_joint, shoulder_lift_joint, \
elbow_joint, wrist_joint], points: [{positions: [0.0, -0.4329, -0.8916, -0.3], \
time_from_start: {sec: 3, nanosec: 0}}]}}"

  Step 2 — open gripper:
      ros2 action send_goal /mirte_master_gripper_controller/gripper_cmd \
        control_msgs/action/GripperCommand \
        "{command: {position: -0.3, max_effort: 10.0}}"

  Step 3 — place box in gripper

  Step 4 — trigger:
      ros2 topic pub --once /start_placing std_msgs/msg/Bool '{data: true}'

  Repeat steps 2-4 for each box.
  Reset stack count:  rm ~/.mirte_stack_state.json
"""

import json
import os
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─────────────────────────────────────────────────────────────────────────────
# Persistent state
# ─────────────────────────────────────────────────────────────────────────────
STATE_FILE = os.path.expanduser('~/.mirte_stack_state.json')


def _load_state() -> Tuple[int, float]:
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        return int(d.get('box_count', 0)), float(d.get('stack_z_offset', 0.0))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0, 0.0


def _save_state(box_count: int, stack_z_offset: float):
    with open(STATE_FILE, 'w') as f:
        json.dump({'box_count': box_count,
                   'stack_z_offset': stack_z_offset}, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Arm joints
# ─────────────────────────────────────────────────────────────────────────────
ARM_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_joint',
]

# ─────────────────────────────────────────────────────────────────────────────
# Arm poses  — tune these to match your robot
# ─────────────────────────────────────────────────────────────────────────────
# Wrist sweeps from WRIST_CARRY → WRIST_PLACE as the arm lowers,
# and back from WRIST_PLACE → WRIST_CARRY as the arm rises.
WRIST_CARRY = -0.3   # rad — wrist at carry height
WRIST_PLACE = -0.7   # rad — wrist at place height

# Carry = arm raised, holding box forward.  Wrist already embedded.
POSE_CARRY = [0.0, -0.4329, -0.8916, WRIST_CARRY]

# Place = arm lowered to drop zone.  Tune shoulder_lift/elbow as needed.
POSE_PLACE = [0.0, -0.9500, -0.8916, WRIST_PLACE]

# ─────────────────────────────────────────────────────────────────────────────
# Gripper
# ─────────────────────────────────────────────────────────────────────────────
GRIPPER_OPEN   = -0.3    # rad
GRIPPER_CLOSED =  0.35   # rad  — increase if grip is too weak
GRIPPER_JOINT  = 'gripper_joint'
GRIP_EFFORT    =  10.0   # N    — increase if gripper won't close
GRIP_DURATION  =   3.5   # s    — time to wait after sending gripper goal

# ─────────────────────────────────────────────────────────────────────────────
# Motion timing  (seconds)
# ─────────────────────────────────────────────────────────────────────────────
T_WRIST_SET  =  0.6   # s — snap wrist to carry angle before descending
T_PLACE_DOWN =  3.5   # s — arm lowers (full descent duration)
T_RETURN     =  3.5   # s — arm raises back to carry height
T_SETTLE     =  1.5   # s — pause after gripper opens

# Height added per placed box (for future multi-layer stacking)
BOX_HEIGHT_STEP = 0.030   # m


# ─────────────────────────────────────────────────────────────────────────────
# State names
# ─────────────────────────────────────────────────────────────────────────────
class S:
    IDLE          = 'IDLE'
    CLOSE_GRIP    = 'CLOSE_GRIP'
    PLACE_DOWN    = 'PLACE_DOWN'
    WAIT_FOR_BACK = 'WAIT_FOR_BACK'
    OPEN_GRIP     = 'OPEN_GRIP'
    SETTLING      = 'SETTLING'
    RETURN_HOME   = 'RETURN_HOME'
    DONE          = 'DONE'


# ─────────────────────────────────────────────────────────────────────────────
class BoxPlacer(Node):

    def __init__(self):
        super().__init__('box_placer')

        # ── Persistent state ──────────────────────────────────────────────────
        self._box_count, self._stack_z_offset = _load_state()

        # ── Action clients ────────────────────────────────────────────────────
        self._arm = ActionClient(
            self, FollowJointTrajectory,
            '/mirte_master_arm_controller/follow_joint_trajectory',
        )
        self._grip_client = ActionClient(
            self, GripperCommand,
            '/mirte_master_gripper_controller/gripper_cmd',
        )

        # ── Publishers ────────────────────────────────────────────────────────
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._arm_placed_pub = self.create_publisher(Bool,   '/arm_placed',  latched)
        self._placed_pub     = self.create_publisher(String, '/box_placed',  10)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Bool,       '/start_placing',    self._on_start,           10)
        self.create_subscription(JointState, '/joint_states',     self._on_joint_states,    10)
        self.create_subscription(Bool,       '/robot_backed_up',  self._on_robot_backed_up, 10)

        # ── Internal state ────────────────────────────────────────────────────
        self._state          = S.IDLE
        self._arm_busy       = False
        self._grip_busy      = False
        self._grip_finish_ns = 0
        self._grip_finish_cb = None
        self._wait_until_ns  = 0
        self._joint_pos: dict = {}

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(0.05, self._tick)   # 20 Hz

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  BoxPlacer ready — box #{self._box_count + 1}\n'
            f'  Carry joints : {[f"{v:.3f}" for v in POSE_CARRY]}\n'
            f'  Place joints : {[f"{v:.3f}" for v in POSE_PLACE]}\n'
            f'  Wrist sweep  : {WRIST_CARRY} → {WRIST_PLACE} rad\n'
            f'\n'
            f'  Move arm to carry, open gripper, place box, then:\n'
            f"  ros2 topic pub --once /start_placing "
            f"std_msgs/msg/Bool '{{data: true}}'\n"
            f'{"="*55}'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Subscription callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_start(self, msg: Bool):
        if msg.data and self._state == S.IDLE:
            self.get_logger().info(
                f'>>> /start_placing — box #{self._box_count + 1} <<<'
            )
            self._set(S.CLOSE_GRIP)

    def _on_joint_states(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._joint_pos[name] = msg.position[i]

    def _on_robot_backed_up(self, msg: Bool):
        if not msg.data:
            return
        if self._state != S.WAIT_FOR_BACK:
            self.get_logger().warn(
                f'/robot_backed_up in state {self._state} — ignoring.')
            return
        self.get_logger().info('>>> /robot_backed_up — opening gripper <<<')
        self._set(S.OPEN_GRIP)

    # ─────────────────────────────────────────────────────────────────────────
    # State machine  (20 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self):
        # Gripper timer — wait GRIP_DURATION seconds then fire callback
        if self._grip_busy:
            if self.get_clock().now().nanoseconds >= self._grip_finish_ns:
                self._grip_busy      = False
                cb                   = self._grip_finish_cb
                self._grip_finish_cb = None
                if cb:
                    cb()
            return

        s = self._state

        if s in (S.IDLE, S.DONE):
            return

        elif s == S.CLOSE_GRIP:
            self.get_logger().info('[1/6] Closing gripper...')
            self._grip_move(GRIPPER_CLOSED, done=lambda: self._set(S.PLACE_DOWN))

        elif s == S.PLACE_DOWN:
            if not self._arm_busy:
                self.get_logger().info(
                    f'[2/6] Lowering arm — wrist {WRIST_CARRY} → {WRIST_PLACE} rad'
                )
                # Two-point trajectory:
                #   t=T_WRIST_SET   : carry joints with wrist=-0.3 (set start angle)
                #   t=T_PLACE_DOWN  : place joints with wrist=-0.7 (lower + sweep)
                self._arm_go_multi([
                    (POSE_CARRY, T_WRIST_SET),
                    (POSE_PLACE, T_PLACE_DOWN),
                ], done=self._on_arm_placed_done)

        elif s == S.WAIT_FOR_BACK:
            self.get_logger().info(
                'Waiting for /robot_backed_up from marker_navigator...',
                throttle_duration_sec=5.0)

        elif s == S.OPEN_GRIP:
            self.get_logger().info('[4/6] Opening gripper — releasing box...')
            self._grip_move(GRIPPER_OPEN, done=self._begin_settling)
            self._state = S.SETTLING

        elif s == S.SETTLING:
            if self.get_clock().now().nanoseconds >= self._wait_until_ns:
                self._set(S.RETURN_HOME)

        elif s == S.RETURN_HOME:
            if not self._arm_busy:
                self.get_logger().info(
                    f'[5/6] Raising arm — wrist {WRIST_PLACE} → {WRIST_CARRY} rad'
                )
                # Single point — JTC interpolates wrist from -0.7 back to -0.3
                self._arm_go(POSE_CARRY, T_RETURN, done=self._on_box_complete)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_arm_placed_done(self):
        self.get_logger().info(
            '[3/6] Arm lowered — publishing /arm_placed → marker_navigator drives back.'
        )
        self._arm_placed_pub.publish(Bool(data=True))
        self._state = S.WAIT_FOR_BACK

    def _begin_settling(self):
        self.get_logger().info(f'[4/6] Box released — settling {T_SETTLE:.1f} s...')
        self._wait_until_ns = (
            self.get_clock().now().nanoseconds + int(T_SETTLE * 1e9)
        )
        self._state = S.SETTLING

    def _on_box_complete(self):
        self._box_count      += 1
        self._stack_z_offset += BOX_HEIGHT_STEP
        _save_state(self._box_count, self._stack_z_offset)

        msg      = String()
        msg.data = f'box_{self._box_count}'
        self._placed_pub.publish(msg)

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  [6/6] Box {self._box_count} placed!\n'
            f'  Stack count  : {self._box_count}\n'
            f'  Saved to     : {STATE_FILE}\n'
            f'\n'
            f'  Open gripper, place next box, then:\n'
            f"  ros2 topic pub --once /start_placing "
            f"std_msgs/msg/Bool '{{data: true}}'\n"
            f'{"="*55}'
        )
        self._state = S.DONE

    def _set(self, new_state: str):
        self.get_logger().info(f'  → {new_state}')
        self._state = new_state

    # ─────────────────────────────────────────────────────────────────────────
    # Arm motion
    # ─────────────────────────────────────────────────────────────────────────

    def _arm_go_multi(self, points: list, done):
        """Multi-point trajectory — controller interpolates between waypoints."""
        traj             = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        for positions, t_sec in points:
            pt                 = JointTrajectoryPoint()
            pt.positions       = [float(p) for p in positions]
            pt.velocities      = [0.0] * len(ARM_JOINTS)
            secs               = int(t_sec)
            nsecs              = int((t_sec - secs) * 1e9)
            pt.time_from_start = MsgDuration(sec=secs, nanosec=nsecs)
            traj.points.append(pt)
        self._arm_send_traj(traj, done)

    def _arm_go(self, positions: list, duration_sec: float, done):
        """Single-point trajectory."""
        pt                 = JointTrajectoryPoint()
        pt.positions       = [float(p) for p in positions]
        pt.velocities      = [0.0] * len(ARM_JOINTS)
        secs               = int(duration_sec)
        nsecs              = int((duration_sec - secs) * 1e9)
        pt.time_from_start = MsgDuration(sec=secs, nanosec=nsecs)

        traj             = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        traj.points      = [pt]
        self._arm_send_traj(traj, done)

    def _arm_send_traj(self, traj: JointTrajectory, done):
        goal                     = FollowJointTrajectory.Goal()
        goal.trajectory          = traj
        goal.goal_time_tolerance = MsgDuration(sec=3, nanosec=0)

        self._arm_busy = True

        def _goal_cb(future):
            try:
                gh = future.result()
            except Exception as e:
                self.get_logger().warn(f'Arm goal error: {e}')
                self._arm_busy = False
                if done:
                    done()
                return
            if not gh.accepted:
                self.get_logger().warn('Arm goal rejected — retrying next tick.')
                self._arm_busy = False
                return
            gh.get_result_async().add_done_callback(_result_cb)

        def _result_cb(future):
            try:
                future.result()
            except Exception as e:
                self.get_logger().warn(f'Arm result error: {e}')
            self._arm_busy = False
            if done:
                done()

        try:
            self._arm.send_goal_async(goal).add_done_callback(_goal_cb)
        except Exception as e:
            self.get_logger().error(f'send_goal_async failed: {e}')
            self._arm_busy = False
            if done:
                done()

    # ─────────────────────────────────────────────────────────────────────────
    # Gripper motion
    # ─────────────────────────────────────────────────────────────────────────

    def _grip_move(self, target: float, done=None):
        current   = self._joint_pos.get(GRIPPER_JOINT, 0.0)
        direction = 'CLOSING' if target > current else 'OPENING'
        self.get_logger().info(
            f'  Gripper {direction}: {current:.3f} → {target:.3f} rad '
            f'(wait {GRIP_DURATION:.1f} s)'
        )

        goal                    = GripperCommand.Goal()
        goal.command.position   = float(target)
        goal.command.max_effort = GRIP_EFFORT

        self._grip_busy      = True
        self._grip_finish_ns = (
            self.get_clock().now().nanoseconds + int(GRIP_DURATION * 1e9)
        )
        self._grip_finish_cb = done

        try:
            self._grip_client.send_goal_async(goal).add_done_callback(
                self._grip_goal_cb)
        except Exception as e:
            self.get_logger().warn(f'Gripper send_goal error: {e}')
            self._grip_busy = False
            if done:
                done()

    def _grip_goal_cb(self, future):
        try:
            gh = future.result()
            if not gh.accepted:
                self.get_logger().warn('Gripper goal rejected by controller.')
        except Exception as e:
            self.get_logger().warn(f'Gripper goal response: {e}')


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BoxPlacer()
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
