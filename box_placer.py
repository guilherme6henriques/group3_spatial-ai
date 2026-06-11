#!/usr/bin/env python3
"""
box_placer.py  —  Mirte Master box stacking (no IK, hardcoded joints).

TUNING CONSTANTS  (top of file)
────────────────────────────────
  POSE_CARRY_FALLBACK  — arm joints while carrying a box
  POSE_PLACE_FALLBACK  — arm joints at fully lowered (place) position
  WRIST_CARRY          — wrist angle while carrying / at place height (-0.3 rad)
  WRIST_BACK           — wrist angle at gripper release during drive-back (0.6 rad)
  T_WRIST_BACK         — duration of wrist sweep during drive-back (tune to match)
  GRIPPER_OPEN         — gripper open position  (-0.3 rad)
  GRIPPER_CLOSED       — gripper closed position (0.35 rad)
  GRIP_EFFORT          — gripper motor effort (increase if it won't close)

MOVEMENT SEQUENCE  (per box)
──────────────────────────────
  1. CLOSE_GRIP    — gripper closes
  2. PLACE_DOWN    — carry and place poses BOTH shifted up by the stack height
                     (3 cm per placed box, persisted across runs) — the descent
                     travel itself stays constant, only the gripping alignment
                     rises; publishes /arm_placed True
  3. WAIT_FOR_BACK — arm holds at POSE_PLACE; waits for /robot_backed_up
                     (failsafe: re-publishes /arm_placed, then continues anyway)
  4. REPOSITION    — wrist sweeps WRIST_PLACE → WRIST_BACK after backing up
  5. OPEN_GRIP     — gripper opens, box released
  5. SETTLING      — brief pause; box count + stack height saved here
  6. RETURN_HOME   — arm raises to the carry pose aligned for the NEXT box
  7. IDLE          — /box_placed published; ready for the next box

SIGNAL FLOW
───────────
  marker_navigator ──/robot_positioned──► box_placer  (auto-start, param auto_start)
  box_placer       ──/arm_placed────────► marker_navigator
  marker_navigator ──/robot_backed_up───► box_placer
  box_placer       ──/box_placed────────► marker_navigator  (turns 180°, hands off)
  box_placer       ──/place_failed──────► (supervisor failsafe on abort)
  (you, optional)  ──/start_placing─────► box_placer  (manual trigger)
  (you, optional)  ──/reset_stack───────► box_placer  (zero the stack counter)

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

  Step 4 — trigger (or let marker_navigator auto-start it via /robot_positioned):
      ros2 topic pub --once /start_placing std_msgs/msg/Bool '{data: true}'

  Repeat steps 2-4 for each box.
  Reset stack count:
      ros2 topic pub --once /reset_stack std_msgs/msg/Bool '{data: true}'
      (or: rm ~/.mirte_stack_state.json)
"""

import json
import os
import time
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
WRIST_CARRY = -0.3   # rad — wrist at carry height AND at place height (no sweep during lowering)
WRIST_PLACE = -0.3   # rad — same as carry: arm lowers with wrist held level
WRIST_BACK      = -0.6   # rad — wrist swept to during drive-back, reached at gripper release
ELBOW_BACK_DELTA =  0.15  # rad — elbow raised by this amount during drive-back (more turning clearance)
T_WRIST_BACK     = 10.0  # s  — tune to match typical drive-back duration

# Carry = arm raised, holding box forward.  Wrist already embedded.
POSE_CARRY = [0.0, -0.4329, -0.8916, WRIST_CARRY]

# Place = arm lowered to drop zone.  Tune shoulder_lift/elbow as needed.
POSE_PLACE = [0.0, -1.2, -0.8916, WRIST_PLACE]

# ─────────────────────────────────────────────────────────────────────────────
# Gripper
# ─────────────────────────────────────────────────────────────────────────────
GRIPPER_OPEN   = -0.3    # rad
GRIPPER_CLOSED =  0.35   # rad  — increase if grip is too weak
GRIPPER_JOINT  = 'gripper_joint'
GRIP_EFFORT    =  10.0   # N    — increase if gripper won't close
GRIP_DURATION  =   7.0   # s    — time to wait after sending gripper goal

# ─────────────────────────────────────────────────────────────────────────────
# Motion timing  (seconds)
# ─────────────────────────────────────────────────────────────────────────────
T_PLACE_DOWN = 12.0   # s — arm lowers (full descent duration)
T_RETURN     =  7.0   # s — arm raises back to carry height
T_SETTLE     =  1.5   # s — pause after gripper opens

# Height added per placed box — lifts the next place pose by this much
BOX_HEIGHT_STEP = 0.030   # m

# Physical vertical travel of the gripper between POSE_CARRY and POSE_PLACE.
# This is the metres→joint-space conversion scale for the stack offset.
# Measure on the robot: lower arm from carry to place and note how many cm
# the gripper descends.  Too small → stack shift raises the arm too much
# (gripper opens high); too large → shift is too small (barely moves up).
PLACE_TRAVEL_HEIGHT_M = 0.40   # m  ← tune to actual carry→place travel

MAX_STACK_BOXES = 5            # safety clamp on the stack height offset

# ─────────────────────────────────────────────────────────────────────────────
# Robustness timeouts / failsafes
# ─────────────────────────────────────────────────────────────────────────────
ARM_TIMEOUT_S       = 30.0    # arm action must report a result within this
ARM_MAX_REJECTS     = 5       # consecutive goal rejections before forcing on
WAIT_BACK_TIMEOUT_S = 120.0   # max wait for /robot_backed_up before releasing
STATE_TIMEOUT_S     = 180.0   # any active state stuck longer → abort to IDLE


# ─────────────────────────────────────────────────────────────────────────────
# State names
# ─────────────────────────────────────────────────────────────────────────────
class S:
    IDLE          = 'IDLE'
    CLOSE_GRIP    = 'CLOSE_GRIP'
    PLACE_DOWN    = 'PLACE_DOWN'
    PLACE_WAIT    = 'PLACE_WAIT'    # arm callback fired early — verifying joints
    WAIT_FOR_BACK = 'WAIT_FOR_BACK'
    REPOSITION    = 'REPOSITION'   # wrist sweeps after robot backed up
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

        # ── Parameters ────────────────────────────────────────────────────────
        # auto_start: begin placing as soon as marker_navigator publishes
        # /robot_positioned — no manual /start_placing needed.
        self._auto_start = bool(self.declare_parameter('auto_start', True).value)

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
        self._arm_placed_pub = self.create_publisher(Bool,   '/arm_placed',   latched)
        self._placed_pub     = self.create_publisher(String, '/box_placed',   10)
        self._failed_pub     = self.create_publisher(Bool,   '/place_failed', latched)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Bool,       '/start_placing',    self._on_start,           10)
        self.create_subscription(JointState, '/joint_states',     self._on_joint_states,    10)
        self.create_subscription(Bool,       '/robot_backed_up',  self._on_robot_backed_up, 10)
        # latched QoS so a /robot_positioned published before we start still arrives
        self.create_subscription(Bool,       '/robot_positioned', self._on_robot_positioned, latched)
        self.create_subscription(Bool,       '/reset_stack',      self._on_reset_stack,     10)

        # ── Internal state ────────────────────────────────────────────────────
        self._state          = S.IDLE
        self._state_t        = time.monotonic()
        self._arm_busy       = False
        self._arm_done_cb    = None
        self._arm_start_t    = 0.0
        self._arm_rejects    = 0
        self._grip_busy      = False
        self._grip_finish_ns = 0
        self._grip_finish_cb = None
        self._wait_until_ns  = 0
        self._place_wait_t   = 0.0   # monotonic time when PLACE_WAIT was entered
        self._back_nagged    = False
        self._joint_pos: dict = {}

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(0.05, self._tick)   # 20 Hz

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  BoxPlacer ready — box #{self._box_count + 1}  '
            f'(stack offset {self._stack_z_offset * 100:.1f} cm)\n'
            f'  Auto-start   : '
            f'{"ON — triggers on /robot_positioned" if self._auto_start else "OFF — send /start_placing"}\n'
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

    def _on_robot_positioned(self, msg: Bool):
        if not (msg.data and self._auto_start):
            return
        if self._state != S.IDLE:
            return
        self.get_logger().info(
            f'>>> /robot_positioned — auto-starting box #{self._box_count + 1} <<<')
        self._set(S.CLOSE_GRIP)

    def _on_reset_stack(self, msg: Bool):
        if not msg.data:
            return
        self._box_count      = 0
        self._stack_z_offset = 0.0
        _save_state(0, 0.0)
        self.get_logger().info('Stack counter reset — next box places at level 0.')

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
        self.get_logger().info(
            f'>>> /robot_backed_up — repositioning wrist '
            f'{WRIST_PLACE} → {WRIST_BACK} rad, then opening gripper <<<')
        wrist_target    = list(self._place_pose())
        wrist_target[2] += ELBOW_BACK_DELTA  # lift elbow for box turning clearance
        wrist_target[3] = WRIST_BACK
        self._arm_go(wrist_target, T_WRIST_BACK,
                     done=lambda: self._set(S.OPEN_GRIP))
        self._set(S.REPOSITION)

    # ─────────────────────────────────────────────────────────────────────────
    # State machine  (20 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self):
        # Arm watchdog — action result never arrived (controller hung/died)
        if self._arm_busy and \
                time.monotonic() - self._arm_start_t > ARM_TIMEOUT_S:
            self.get_logger().error(
                f'Arm action gave no result within {ARM_TIMEOUT_S:.0f} s — '
                f'forcing the sequence to continue.')
            self._arm_finish()

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

        # Global stuck-state failsafe
        if time.monotonic() - self._state_t > STATE_TIMEOUT_S:
            self._abort(f'state {s} stuck for {STATE_TIMEOUT_S:.0f} s')
            return

        elif s == S.CLOSE_GRIP:
            self.get_logger().info('[1/6] Closing gripper...')
            self._grip_move(GRIPPER_CLOSED, done=lambda: self._set(S.PLACE_DOWN))

        elif s == S.PLACE_DOWN:
            if not self._arm_busy:
                carry = self._carry_pose()
                place = self._place_pose()
                self.get_logger().info(
                    f'[2/6] Lowering arm to stack level {self._box_count} '
                    f'(grip alignment +{self._stack_z_offset * 100:.1f} cm, '
                    f'travel constant) — wrist held at {place[3]:.2f} rad throughout'
                )
                # Single-point: arm moves directly from its current position
                # (already at carry) straight down to place — no upward snap.
                # Wrist stays at WRIST_PLACE (-0.3) throughout; it sweeps
                # to WRIST_BACK (-0.6) only during drive-back (WAIT_FOR_BACK).
                self._arm_go(place, T_PLACE_DOWN, done=self._on_arm_placed_done)

        elif s == S.PLACE_WAIT:
            place = self._place_pose()
            if self._arm_at_target(place):
                self.get_logger().info('Arm confirmed at place position.')
                self._publish_arm_placed()
            elif not self._arm_busy:
                elapsed = time.monotonic() - self._place_wait_t
                if elapsed > T_PLACE_DOWN + 15.0:
                    self.get_logger().error(
                        'Cannot reach place position after retries — '
                        'publishing /arm_placed anyway.')
                    self._publish_arm_placed()
                else:
                    self.get_logger().warn(
                        'Arm not at target — re-sending place trajectory.',
                        throttle_duration_sec=3.0)
                    self._arm_go(place, T_PLACE_DOWN, done=lambda: None)
            else:
                j = {j: self._joint_pos.get(j, float('nan'))
                     for j in ARM_JOINTS}
                errs = [abs(self._joint_pos.get(jn, t) - t)
                        for jn, t in zip(ARM_JOINTS, place)]
                self.get_logger().info(
                    f'Moving to place — errors (rad): '
                    f'{[f"{e:.3f}" for e in errs]}',
                    throttle_duration_sec=1.0)

        elif s == S.WAIT_FOR_BACK:
            waited = time.monotonic() - self._state_t
            if waited > WAIT_BACK_TIMEOUT_S:
                self.get_logger().error(
                    f'No /robot_backed_up after {WAIT_BACK_TIMEOUT_S:.0f} s — '
                    f'failsafe: releasing the box anyway.')
                self._set(S.OPEN_GRIP)
            elif waited > WAIT_BACK_TIMEOUT_S / 2 and not self._back_nagged:
                self._back_nagged = True
                self.get_logger().warn(
                    'Still no /robot_backed_up — re-publishing /arm_placed.')
                self._arm_placed_pub.publish(Bool(data=True))
            else:
                self.get_logger().info(
                    'Waiting for /robot_backed_up from marker_navigator...',
                    throttle_duration_sec=5.0)

        elif s == S.REPOSITION:
            self.get_logger().info(
                f'Repositioning wrist {WRIST_PLACE} → {WRIST_BACK} rad...',
                throttle_duration_sec=2.0)
            # OPEN_GRIP is triggered by the done callback in _on_robot_backed_up.

        elif s == S.OPEN_GRIP:
            self.get_logger().info('[5/6] Opening gripper — releasing box...')
            self._grip_move(GRIPPER_OPEN, done=self._begin_settling)
            self._set(S.SETTLING)

        elif s == S.SETTLING:
            if self.get_clock().now().nanoseconds >= self._wait_until_ns:
                self._set(S.RETURN_HOME)

        elif s == S.RETURN_HOME:
            if not self._arm_busy:
                # Stack count was updated on release, so _carry_pose() is
                # already the gripping alignment for the NEXT box (+3 cm).
                carry = self._carry_pose()
                self.get_logger().info(
                    f'[5/6] Raising arm to next-box carry '
                    f'(grip alignment +{self._stack_z_offset * 100:.1f} cm) — '
                    f'wrist back to {carry[3]:.2f} rad'
                )
                self._arm_go(carry, T_RETURN, done=self._on_box_complete)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_arm_placed_done(self):
        self._place_wait_t = time.monotonic()
        self.get_logger().info(
            '[3/6] Arm action returned — verifying joint positions before /arm_placed.')
        self._set(S.PLACE_WAIT)

    def _arm_at_target(self, positions: List[float], tol: float = 0.12) -> bool:
        """True when every ARM_JOINT is within tol rad of the target."""
        for joint, target in zip(ARM_JOINTS, positions):
            current = self._joint_pos.get(joint)
            if current is None:
                return False
            if abs(current - target) > tol:
                return False
        return True

    def _publish_arm_placed(self):
        self.get_logger().info(
            '[3/6] Arm at target — publishing /arm_placed → marker_navigator drives back.\n'
            '      Arm holds at POSE_PLACE during drive-back; wrist sweeps after.'
        )
        self._arm_placed_pub.publish(Bool(data=True))
        self._back_nagged = False
        self._set(S.WAIT_FOR_BACK)

    def _begin_settling(self):
        # Box is on the stack now — update and persist the stack height so the
        # return motion (and every later run) uses the next box's alignment.
        self._box_count      += 1
        self._stack_z_offset += BOX_HEIGHT_STEP
        _save_state(self._box_count, self._stack_z_offset)
        self.get_logger().info(
            f'[4/6] Box {self._box_count} released — stack saved '
            f'(next grip alignment +{self._stack_z_offset * 100:.1f} cm), '
            f'settling {T_SETTLE:.1f} s...')
        self._wait_until_ns = (
            self.get_clock().now().nanoseconds + int(T_SETTLE * 1e9)
        )
        self._state = S.SETTLING

    def _on_box_complete(self):
        msg      = String()
        msg.data = f'box_{self._box_count}'
        self._placed_pub.publish(msg)

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  [6/6] Box {self._box_count} placed!\n'
            f'  Stack count  : {self._box_count}  '
            f'(next box +{self._stack_z_offset * 100:.1f} cm)\n'
            f'  Saved to     : {STATE_FILE}\n'
            f'\n'
            f'  /box_placed published — marker_navigator turns the robot around.\n'
            f'  Back to IDLE: next /robot_positioned or /start_placing begins\n'
            f'  box #{self._box_count + 1}.\n'
            f'{"="*55}'
        )
        self._set(S.IDLE)

    def _set(self, new_state: str):
        self.get_logger().info(f'  → {new_state}')
        self._state   = new_state
        self._state_t = time.monotonic()

    def _abort(self, reason: str):
        self.get_logger().error(
            f'\n{"="*55}\n'
            f'  PLACE SEQUENCE ABORTED: {reason}\n'
            f'  /place_failed published — resetting to IDLE.\n'
            f'  Box count NOT incremented.\n'
            f'{"="*55}'
        )
        self._failed_pub.publish(Bool(data=True))
        self._arm_busy       = False
        self._arm_done_cb    = None
        self._grip_busy      = False
        self._grip_finish_cb = None
        self._set(S.IDLE)

    def _stack_shift(self) -> List[float]:
        """Joint-space offset that raises the gripper by the persisted stack height.

        No IK on this arm: the carry→place joint delta corresponds to roughly
        PLACE_TRAVEL_HEIGHT_M of vertical gripper travel, so a fraction of that
        delta raises the gripper by the matching fraction in metres.  The shift
        is applied to BOTH carry and place poses — the descent travel stays
        constant, only the gripping alignment rises 3 cm per box (the boxes
        are incrementally smaller).  The wrist shifts along the same delta,
        which keeps the box level while the arm sits higher.
        """
        max_offset = (MAX_STACK_BOXES - 1) * BOX_HEIGHT_STEP
        offset     = self._stack_z_offset
        if offset > max_offset:
            self.get_logger().warn(
                f'Stack offset {offset * 100:.1f} cm exceeds the '
                f'{MAX_STACK_BOXES}-box limit — clamping to {max_offset * 100:.1f} cm. '
                f"Reset: ros2 topic pub --once /reset_stack std_msgs/msg/Bool '{{data: true}}'")
            offset = max_offset
        frac = max(0.0, min(offset / PLACE_TRAVEL_HEIGHT_M, 1.0))
        return [frac * (c - p) for p, c in zip(POSE_PLACE, POSE_CARRY)]

    def _carry_pose(self) -> List[float]:
        return [c + d for c, d in zip(POSE_CARRY, self._stack_shift())]

    def _place_pose(self) -> List[float]:
        return [p + d for p, d in zip(POSE_PLACE, self._stack_shift())]

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

    def _arm_finish(self):
        """Finish the current arm motion exactly once (result, error or watchdog)."""
        if not self._arm_busy:
            return
        self._arm_busy    = False
        cb                = self._arm_done_cb
        self._arm_done_cb = None
        if cb:
            cb()

    def _arm_send_traj(self, traj: JointTrajectory, done):
        if not self._arm.server_is_ready():
            # State machine re-sends next tick; STATE_TIMEOUT_S aborts eventually.
            self.get_logger().warn(
                'Arm action server not ready — retrying...',
                throttle_duration_sec=5.0)
            return

        goal                     = FollowJointTrajectory.Goal()
        goal.trajectory          = traj
        goal.goal_time_tolerance = MsgDuration(sec=3, nanosec=0)

        self._arm_busy    = True
        self._arm_done_cb = done
        self._arm_start_t = time.monotonic()

        def _goal_cb(future):
            try:
                gh = future.result()
            except Exception as e:
                self.get_logger().warn(f'Arm goal error: {e}')
                self._arm_finish()
                return
            if not gh.accepted:
                self._arm_rejects += 1
                if self._arm_rejects >= ARM_MAX_REJECTS:
                    self.get_logger().error(
                        f'Arm goal rejected {self._arm_rejects}× — forcing continue.')
                    self._arm_rejects = 0
                    self._arm_finish()
                else:
                    self.get_logger().warn('Arm goal rejected — retrying next tick.')
                    self._arm_busy    = False
                    self._arm_done_cb = None
                return
            self._arm_rejects = 0
            gh.get_result_async().add_done_callback(_result_cb)

        def _result_cb(future):
            try:
                future.result()
            except Exception as e:
                self.get_logger().warn(f'Arm result error: {e}')
            self._arm_finish()

        try:
            self._arm.send_goal_async(goal).add_done_callback(_goal_cb)
        except Exception as e:
            self.get_logger().error(f'send_goal_async failed: {e}')
            self._arm_finish()

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

        if not self._grip_client.server_is_ready():
            self.get_logger().warn(
                'Gripper action server not ready — sending anyway; the timed '
                'wait keeps the sequence moving.')

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
