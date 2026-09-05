#!/usr/bin/env python3
"""Solve IK on the host, so the joint command the arm follows is observable.

In vr_slave mode X5Controller takes the VR pose, runs IK inside a closed
library and drives the motors; the joint targets it computes never appear on
a topic, which is why collect.py has to record the arm's feedback as the
action. This node takes the same VR pose, solves IK with Placo on the vendor
URDF, and publishes the joint targets to the topic the arm listens on in
remote_slave mode. The target then exists as a message, and can be recorded.

The arm has to be started with v2_joint_control.launch.py, not the vr_slave
launcher; tools/08_teleop_ik.sh does that.

Engagement works the way the vendor app's "operating" state does, and for
the same reason. The VR stream is not always a hand: a controller that is
not tracking is published as a fixed pose that can sit a metre from the arm,
and even a tracking one lives in the headset's frame until the app zeroes
it. So the node never follows the VR pose as an absolute target by default.
It waits for an engage request, and at that moment records both where the
VR pose is and where the arm is; from then on the arm follows the VR pose's
motion relative to that moment, applied to the arm's pose at that moment.
Where the VR frame sits is irrelevant, and engaging on a frozen pose moves
nothing. The serial packet carries no button state, so the request is a ROS
service rather than a controller button:

    ros2 service call /vr_ik_r/engage std_srvs/srv/Trigger
    ros2 service call /vr_ik_r/disengage std_srvs/srv/Trigger
    ros2 service call /vr_ik_r/home std_srvs/srv/Trigger

--engage picks what starts and stops tracking:

  absolute  (default) the vendor's rule: follow the pose as sent, whenever
            the arm can reach it. X5Controller in vr_slave mode hands the
            VR pose straight to its solver; the one thing its closed solver
            adds is that it does not move for a target it cannot reach, so a
            parked controller's pose leaves the arm still. That is
            reproduced by converging a scratch solver on each target first.
            The app expresses the hand in an arm-aligned frame with a fixed
            origin, not zeroed onto the arm, so the operator has to bring
            the hand to where the arm is before holding the trigger - a hand
            at chest height maps 80 cm from the base and is out of reach.
            The app's trigger, release and reset all arrive as pose changes
            and work as they do with the vendor stack.
  motion    the trigger from the stream itself. The app streams the tracked
            pose only while the index trigger is held and repeats the last
            pose bit-for-bit once released, so the pose starting to change
            engages, relative to the arm's pose at that instant, and the
            pose freezing for --release-window disengages. Independent of
            where the app's frame sits.
  service   engage and disengage only on request (the services above).

The app's reset button reaches the robot only as a pose, so it works in
absolute mode; in the other modes use the /home service.

  WAITING   no arm feedback yet; nothing published
  ARMED     feedback present; holding, waiting to engage as --engage says
  TRACKING  solving and publishing. Each joint moves at most --max-velocity,
            using the measured message interval, so a far target is walked
            toward rather than lunged at. Two things end it: a target that
            moves more than --max-jump in a single message (a hand cannot;
            tracking was lost or the parking pose snapped in) holds at once,
            and a residual that stays above --max-residual for a whole
            --stall-window without shrinking means the target is out of
            reach. Either way publishing stops and the node returns to
            ARMED holding the last command.
  HOMING    the vendor app's "reset": walking every joint to zero at
            --max-velocity on a timer, ignoring the VR pose, then ARMED.
            /disengage aborts it and holds where it is.

The report line always says whether the VR pose has moved in the last
second and how long since it was last received, because "is the controller
tracking?" is the first question to answer when nothing happens.

Nothing is published without --execute.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x5_model import EE_FRAME, GRIPPER_SCALE, HOME_Q, JOINTS, kinematic_urdf, target_transform  # noqa: E402

WAITING, ARMED, TRACKING, HOMING = 'WAITING', 'ARMED', 'TRACKING', 'HOMING'
HOME_TOLERANCE = np.radians(0.5)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def build_node(args):
    import placo
    import rclpy
    from rclpy.node import Node
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotStatus
    from std_srvs.srv import Trigger

    class VrIkNode(Node):
        def __init__(self):
            super().__init__(args.node_name)
            urdf = kinematic_urdf(Path(args.urdf_cache))
            self.robot = placo.RobotWrapper(str(urdf), placo.Flags.ignore_collisions)
            self.solver = placo.KinematicsSolver(self.robot)
            self.solver.mask_fbase(True)
            self.solver.enable_joint_limits(True)
            self.task = self.solver.add_frame_task(EE_FRAME, np.eye(4))
            self.task.configure(EE_FRAME, 'soft', 1.0, 1.0)
            self.solver.add_regularization_task(1e-5)
            self.solver.dt = args.dt
            # A second model for FK of the feedback and for probing reachability,
            # so neither disturbs the tracking solver's state.
            self.fk_robot = placo.RobotWrapper(str(urdf), placo.Flags.ignore_collisions)
            self.probe_solver = placo.KinematicsSolver(self.fk_robot)
            self.probe_solver.mask_fbase(True)
            self.probe_solver.enable_joint_limits(True)
            self.probe_task = self.probe_solver.add_frame_task(EE_FRAME, np.eye(4))
            self.probe_task.configure(EE_FRAME, 'soft', 1.0, 1.0)
            self.probe_solver.add_regularization_task(1e-5)
            self.probe_solver.dt = 0.05

            self.state = WAITING
            self.q_feedback = None
            self.last_q = None
            self.t_prev_msg = None
            self.solved = self.clamped = 0
            self.residual = []
            self.res_hist = collections.deque(maxlen=args.stall_window)
            self.gate = None
            self.t_last_pose = None
            self.vr_p = self.vr_R = None                 # latest VR pose, base frame
            self.vr_recent = collections.deque()         # (t, p) over the last second
            self.origin = None                           # (p_arm0, R_arm0, p_vr0, R_vr0) set at engage
            self.p_target_prev = None                    # last target position, for the jump guard
            self.vr_prev = None                          # last raw pose tuple, to tell a frozen stream
            self.t_last_change = None                    # when the raw pose last differed from the one before
            self.last_gripper = 5.0 * GRIPPER_SCALE      # what the idle VR stream commands: open

            self.pub = self.create_publisher(RobotStatus, args.out_topic, 10) if args.execute else None
            self.create_subscription(RobotStatus, args.feedback_topic, self.on_feedback, 10)
            self.create_subscription(PosCmd, args.in_topic, self.on_pose, 10)
            self.create_service(Trigger, '~/engage', self.srv_engage)
            self.create_service(Trigger, '~/disengage', self.srv_disengage)
            self.create_service(Trigger, '~/home', self.srv_home)
            # Homing runs on its own clock: it must not depend on VR messages,
            # which may be exactly what has gone away.
            self.create_timer(args.dt, self.homing_tick)
            self.create_timer(args.report_period, self.report)
            mode = {
                'motion': f'engage when the VR pose starts moving, hold when it freezes for {args.release_window:.1f} s',
                'service': f'engage on request; call /{args.node_name}/engage',
                'absolute': 'follow the VR pose as sent whenever the arm can reach it (vendor behaviour)',
            }[args.engage]
            self.get_logger().info(
                f'{args.in_topic} -> IK -> {args.out_topic}  '
                f'{"PUBLISHING" if args.execute else "DRY-RUN, pass --execute to publish"}; {mode}; '
                f'max {args.max_velocity:.2f} rad/s')

        # --- inputs ------------------------------------------------------------
        def on_feedback(self, msg):
            self.q_feedback = np.array(msg.joint_pos[:6], dtype=float)
            if self.state == WAITING:
                self.set_state(ARMED, 'arm feedback arrived')

        def on_pose(self, msg):
            now = time.monotonic()
            self.t_last_pose = now
            T_vr = target_transform([msg.x, msg.y, msg.z], [msg.roll, msg.pitch, msg.yaw])
            self.vr_p, self.vr_R = T_vr[:3, 3].copy(), T_vr[:3, :3].copy()
            self.vr_recent.append((now, self.vr_p))
            while self.vr_recent and now - self.vr_recent[0][0] > 1.0:
                self.vr_recent.popleft()
            self.last_gripper = float(msg.gripper) * GRIPPER_SCALE
            # The app repeats the last pose bit-for-bit while the trigger is up, so
            # "changed since the previous message" is the trigger being down.
            raw = (msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw)
            changed = self.vr_prev is not None and raw != self.vr_prev
            self.vr_prev = raw
            if changed:
                self.t_last_change = now
            if self.state in (WAITING, HOMING):
                return

            if self.state == ARMED:
                if args.engage == 'service':
                    return
                if args.engage == 'motion':
                    if changed:
                        self.engage('VR pose started moving (trigger down)', now)
                    else:
                        return
            if self.state == ARMED:   # absolute: the vendor's rule, follow it if the arm can reach it
                ok, res = self.reachable(T_vr, self.q_feedback)
                self.gate = res
                if not ok:
                    return
                self.engage(f'target is reachable (probe residual {res * 1000:.0f} mm)', now)
            elif args.engage == 'absolute':
                ok, res = self.reachable(T_vr, self.last_q)
                if not ok:
                    self.gate = res
                    self.disengage(f'target went out of reach (probe residual {res * 1000:.0f} mm)')
                    return

            if args.engage == 'motion' and self.t_last_change is not None \
                    and now - self.t_last_change > args.release_window:
                self.disengage('VR pose froze (trigger up)')
                return

            if self.origin is None:          # absolute: follow the pose as sent
                T = T_vr
            else:                            # relative: VR motion since engage, applied to the arm's pose then
                p_arm0, R_arm0, p_vr0, R_vr0 = self.origin
                T = np.eye(4)
                T[:3, 3] = p_arm0 + (self.vr_p - p_vr0)
                T[:3, :3] = (self.vr_R @ R_vr0.T) @ R_arm0

            # A hand cannot move --max-jump in one message; a target that does has
            # lost tracking or snapped to the app's parking pose. Hold at once,
            # before the velocity cap starts walking the arm after it.
            if self.p_target_prev is not None:
                jump = float(np.linalg.norm(T[:3, 3] - self.p_target_prev))
                if jump > args.max_jump:
                    self.p_target_prev = None
                    self.disengage(f'target jumped {jump * 1000:.0f} mm in one message - tracking lost')
                    return
            self.p_target_prev = T[:3, 3].copy()

            dt = min(max(now - self.t_prev_msg, 1 / 500), 1 / 20)   # tolerate a stalled stream
            self.t_prev_msg = now
            self.task.T_world_frame = T
            self.solver.solve(True)
            self.robot.update_kinematics()
            q = self.current_q()

            step = q - self.last_q
            limit = args.max_velocity * dt
            if (np.abs(step) > limit).any():
                q = self.last_q + np.clip(step, -limit, limit)
                self.seed(q)
                self.clamped += 1
            self.last_q = q

            res = float(np.linalg.norm(self.robot.get_T_world_frame(EE_FRAME)[:3, 3] - T[:3, 3]))
            self.residual.append(res)
            self.res_hist.append(res)
            self.solved += 1
            # Out of reach means the solver has stalled: the residual has been
            # above the threshold for the whole window and is not shrinking.
            # Every sample in the window has to be above it - comparing only
            # against the oldest sample fired on the first message after any
            # target step, when the oldest sample was still near zero, before
            # the velocity cap had a chance to walk the arm in.
            if (len(self.res_hist) == self.res_hist.maxlen and min(self.res_hist) > args.max_residual
                    and res > 0.9 * self.res_hist[0]):
                self.disengage(f'residual {res * 1000:.0f} mm and not improving over '
                               f'{self.res_hist.maxlen} messages - target is out of reach')
                return

            self.publish(q)

        # --- homing ------------------------------------------------------------------
        def homing_tick(self):
            if self.state != HOMING:
                return
            limit = args.max_velocity * args.dt
            remaining = HOME_Q - self.last_q
            if np.abs(remaining).max() <= HOME_TOLERANCE:
                self.publish(HOME_Q)
                self.seed(HOME_Q)
                self.set_state(ARMED, 'at home')
                return
            self.last_q = self.last_q + np.clip(remaining, -limit, limit)
            self.publish(self.last_q)

        def srv_home(self, _req, resp):
            if self.state == WAITING:
                resp.success, resp.message = False, 'no arm feedback yet'
            elif self.state == HOMING:
                resp.success, resp.message = False, 'already homing'
            else:
                # Walk from where the arm actually is, not from the solver's last
                # command: while holding, the two can differ.
                self.origin = None
                self.last_q = self.q_feedback.copy()
                far = np.degrees(np.abs(self.q_feedback - HOME_Q).max())
                self.set_state(HOMING, f'homing by request, {far:.1f} deg to go')
                resp.success, resp.message = True, f'homing, {far:.1f} deg to go at {args.max_velocity:.2f} rad/s'
            return resp

        # --- engage / disengage --------------------------------------------------
        def engage(self, why, now=None):
            """Seed the solver from the arm and, unless auto-engaging, anchor the VR frame here."""
            now = now or time.monotonic()
            self.seed(self.q_feedback)
            self.t_prev_msg = now
            self.res_hist.clear()
            self.p_target_prev = None
            if args.engage != 'absolute':
                T_arm = self.fk_of(self.q_feedback)
                self.origin = (T_arm[:3, 3].copy(), T_arm[:3, :3].copy(), self.vr_p.copy(), self.vr_R.copy())
            self.set_state(TRACKING, why)

        def disengage(self, why):
            self.origin = None
            self.p_target_prev = None
            self.set_state(ARMED, why + '; holding')

        def srv_engage(self, _req, resp):
            if self.state == WAITING:
                resp.success, resp.message = False, 'no arm feedback yet'
            elif self.state == HOMING:
                resp.success, resp.message = False, 'homing; wait for it to finish or disengage to abort'
            elif self.state == TRACKING:
                resp.success, resp.message = False, 'already tracking; disengage first to re-anchor'
            elif self.t_last_pose is None or time.monotonic() - self.t_last_pose > 0.5:
                resp.success, resp.message = False, 'no recent VR pose; is serial_port_node running and the headset awake?'
            else:
                moving = self.vr_motion_mm()
                self.engage(f'engaged by request; VR pose {"moving" if moving > 1 else "NOT moving"} '
                            f'({moving:.0f} mm over the last second)')
                resp.success = True
                resp.message = (f'engaged at q={np.round(self.q_feedback, 3).tolist()}; '
                                + ('VR pose is moving' if moving > 1 else
                                   'WARNING: VR pose is not moving - the arm will not move until the controller tracks'))
            return resp

        def srv_disengage(self, _req, resp):
            if self.state == TRACKING:
                self.disengage('disengaged by request')
                resp.success, resp.message = True, 'holding'
            elif self.state == HOMING:
                self.seed(self.last_q)
                self.set_state(ARMED, 'homing aborted by request; holding')
                resp.success, resp.message = True, 'homing aborted; holding'
            else:
                resp.success, resp.message = False, f'nothing to disengage (state {self.state})'
            return resp

        def publish(self, q):
            if self.pub is None:
                return
            out = RobotStatus()
            out.header.stamp = self.get_clock().now().to_msg()
            out.joint_pos[:6] = np.asarray(q, dtype=float).tolist()
            out.joint_pos[6] = self.last_gripper
            self.pub.publish(out)

        # --- helpers -------------------------------------------------------------
        def set_state(self, state, why):
            if state != self.state:
                self.get_logger().info(f'{self.state} -> {state}: {why}')
                self.state = state

        def seed(self, q):
            for name, value in zip(JOINTS, q):
                self.robot.set_joint(name, float(value))
            self.robot.update_kinematics()
            self.last_q = np.asarray(q, dtype=float).copy()

        def fk_of(self, q):
            for name, value in zip(JOINTS, q):
                self.fk_robot.set_joint(name, float(value))
            self.fk_robot.update_kinematics()
            return np.array(self.fk_robot.get_T_world_frame(EE_FRAME))

        def reachable(self, T, q_seed):
            """Converge a scratch solver on T from q_seed; the vendor arm does not
            move for a target it cannot reach, and neither should this one -
            the tracking solver would otherwise walk the arm to its joint
            limits and leave it pinned there, as it did on the first dry run."""
            self.fk_of(q_seed)
            self.probe_task.T_world_frame = T
            for _ in range(args.probe_iterations):
                self.probe_solver.solve(True)
                self.fk_robot.update_kinematics()
            res = float(np.linalg.norm(np.array(self.fk_robot.get_T_world_frame(EE_FRAME))[:3, 3] - T[:3, 3]))
            return res <= args.max_residual, res

        def current_q(self):
            return np.array([self.robot.get_joint(n) for n in JOINTS])

        def vr_motion_mm(self):
            if len(self.vr_recent) < 2:
                return 0.0
            ps = np.array([p for _, p in self.vr_recent])
            return float(np.linalg.norm(ps.max(0) - ps.min(0)) * 1000)

        def report(self):
            stale = None if self.t_last_pose is None else time.monotonic() - self.t_last_pose
            if stale is not None and stale > 1.0:
                self.get_logger().warning(
                    f'{self.state}: no VR pose for {stale:.0f} s - is serial_port_node running, '
                    f'the headset awake, and the USB cable in?')
            elif self.state == WAITING:
                self.get_logger().info('WAITING: no arm feedback yet')
            elif self.state == HOMING:
                self.get_logger().info(f'HOMING: {np.degrees(np.abs(HOME_Q - self.last_q).max()):.1f} deg to go')
            elif self.state == ARMED:
                if self.t_last_pose is None:
                    self.get_logger().info('ARMED: no VR pose yet')
                else:
                    moving = self.vr_motion_mm()
                    vr = f'VR pose {"moving" if moving > 1 else "NOT moving"} ({moving:.0f} mm/s)'
                    if args.engage == 'absolute' and self.gate is not None:
                        self.get_logger().info(
                            f'ARMED: {vr}; target is OUT OF REACH (probe residual {self.gate * 1000:.0f} mm). '
                            f'Bring the hand to where the arm is, then hold the trigger.')
                    elif args.engage == 'motion':
                        self.get_logger().info(f'ARMED: {vr}; holding. Hold the trigger and move the hand to engage.')
                    else:
                        self.get_logger().info(
                            f'ARMED: {vr}; holding. Engage with: '
                            f'ros2 service call /{args.node_name}/engage std_srvs/srv/Trigger')
            elif self.residual:
                r = np.array(self.residual) * 1000
                self.get_logger().info(
                    f'TRACKING {self.solved / args.report_period:5.1f} Hz  VR {self.vr_motion_mm():.0f} mm/s  '
                    f'residual mean {r.mean():.2f} mm max {r.max():.2f} mm  '
                    f'clamped {self.clamped}/{self.solved}  q={np.round(self.last_q, 3)}')
            self.solved = self.clamped = 0
            self.residual.clear()

    return rclpy, VrIkNode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--side', choices=['left', 'right'], default='right',
                        help='sets the default topics; override individually below')
    parser.add_argument('--in-topic', help='VR pose, default /ARX_VR_{L,R}_filtered')
    parser.add_argument('--out-topic', help='joint targets, default /arm_master_{l,r}_status')
    parser.add_argument('--feedback-topic', help='arm feedback used to seed and gate, default /arm_slave_{l,r}_status')
    parser.add_argument('--node-name')
    parser.add_argument('--urdf-cache', default=str(Path(__file__).resolve().parent / 'x5_kin.urdf'))
    parser.add_argument('--dt', type=float, default=1 / 100.0,
                        help='solver integration step; roughly the VR message period')
    parser.add_argument('--engage', choices=['absolute', 'motion', 'service'], default='absolute',
                        help='what starts tracking: the pose being reachable (vendor behaviour), the pose starting '
                             'to move (trigger down), or a service call; see the module docstring')
    parser.add_argument('--release-window', type=float, default=0.3,
                        help='s; in motion mode, a pose frozen this long means the trigger is up: hold')
    parser.add_argument('--probe-iterations', type=int, default=20,
                        help='solver iterations used to decide whether a target is reachable')
    parser.add_argument('--max-velocity', type=float, default=1.5,
                        help='rad/s per joint, applied per message using the measured interval')
    parser.add_argument('--max-residual', type=float, default=0.03,
                        help='m; a residual above this that stops improving means the target is out of reach')
    parser.add_argument('--stall-window', type=int, default=30,
                        help='messages over which the residual must improve, else the target is out of reach')
    parser.add_argument('--max-jump', type=float, default=0.3,
                        help='m; a target that moves more than this in one message has lost tracking, hold at once')
    parser.add_argument('--report-period', type=float, default=2.0)
    parser.add_argument('--execute', action='store_true',
                        help='publish joint targets; default solves and reports only')
    args = parser.parse_args()

    s, S = ('l', 'L') if args.side == 'left' else ('r', 'R')
    args.in_topic = args.in_topic or f'/ARX_VR_{S}_filtered'
    args.out_topic = args.out_topic or f'/arm_master_{s}_status'
    args.feedback_topic = args.feedback_topic or f'/arm_slave_{s}_status'
    args.node_name = args.node_name or f'vr_ik_{s}'

    rclpy, VrIkNode = build_node(args)
    rclpy.init()
    node = VrIkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy's own SIGINT handler has usually shut the context down already.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
