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

Safety is a small state machine, because the VR stream is not always a hand:
when a controller is not tracking, the headset app publishes a fixed parking
pose some 30 cm from the arm's home with a resting-hand orientation, which is
outside the reachable workspace. Solving toward that pinned the solver against
its joint limits 236 mm from the target on the first dry run. So:

  WAITING   no arm feedback or no VR pose yet; nothing published
  ARMED     both present; the target is checked against where the arm
            actually is, and nothing is published until it comes within
            --engage-distance / --engage-angle of it. That is what the
            vendor app's "operating" state guarantees by construction: the
            hand starts where the arm is.
  TRACKING  solving and publishing. Each joint may move at most
            --max-velocity, using the measured message interval, so a far
            target is walked toward rather than lunged at. If the position
            residual exceeds --max-residual the target has left the
            workspace (or the controller dropped back to its parking pose);
            publishing stops and the node returns to ARMED, holding the
            last command, until the target comes back within reach.

Nothing is published without --execute.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x5_model import EE_FRAME, GRIPPER_SCALE, JOINTS, kinematic_urdf, target_transform  # noqa: E402

WAITING, ARMED, TRACKING = 'WAITING', 'ARMED', 'TRACKING'


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def build_node(args):
    import placo
    import rclpy
    from rclpy.node import Node
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotStatus

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
            # A second wrapper just for FK of the feedback, so the gate can be
            # evaluated without disturbing the solver's state.
            self.fk_robot = placo.RobotWrapper(str(urdf), placo.Flags.ignore_collisions)

            self.state = WAITING
            self.q_feedback = None
            self.last_q = None
            self.t_prev_msg = None
            self.solved = self.clamped = 0
            self.residual = []
            self.gate = None            # (distance m, angle deg) of the latest target vs the arm
            self.t_state_change = time.monotonic()

            self.pub = self.create_publisher(RobotStatus, args.out_topic, 10) if args.execute else None
            self.create_subscription(RobotStatus, args.feedback_topic, self.on_feedback, 10)
            self.create_subscription(PosCmd, args.in_topic, self.on_pose, 10)
            self.create_timer(args.report_period, self.report)
            self.get_logger().info(
                f'{args.in_topic} -> IK -> {args.out_topic}  '
                f'{"PUBLISHING" if args.execute else "DRY-RUN, pass --execute to publish"}; '
                f'engage within {args.engage_distance * 1000:.0f} mm / {args.engage_angle:.0f} deg, '
                f'max {args.max_velocity:.2f} rad/s, hold above {args.max_residual * 1000:.0f} mm residual')

        # --- inputs ------------------------------------------------------------
        def on_feedback(self, msg):
            self.q_feedback = np.array(msg.joint_pos[:6], dtype=float)
            if self.state == WAITING:
                self.set_state(ARMED, 'arm feedback arrived')

        def on_pose(self, msg):
            if self.state == WAITING:
                return
            T = target_transform([msg.x, msg.y, msg.z], [msg.roll, msg.pitch, msg.yaw])
            now = time.monotonic()

            if self.state == ARMED:
                # Gate against where the arm really is, not where the solver was.
                self.fk_set(self.q_feedback)
                T_arm = np.array(self.fk_robot.get_T_world_frame(EE_FRAME))
                self.gate = (float(np.linalg.norm(T[:3, 3] - T_arm[:3, 3])), rotation_angle_deg(T[:3, :3], T_arm[:3, :3]))
                if self.gate[0] > args.engage_distance or self.gate[1] > args.engage_angle:
                    return
                self.seed(self.q_feedback)
                self.t_prev_msg = now
                self.set_state(TRACKING, f'target within {self.gate[0] * 1000:.0f} mm / {self.gate[1]:.0f} deg of the arm')

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
            self.solved += 1
            if res > args.max_residual:
                self.set_state(ARMED, f'residual {res * 1000:.0f} mm - target left the workspace; holding')
                return

            if self.pub is not None:
                out = RobotStatus()
                out.header.stamp = self.get_clock().now().to_msg()
                out.joint_pos[:6] = q.tolist()
                out.joint_pos[6] = float(msg.gripper) * GRIPPER_SCALE
                self.pub.publish(out)

        # --- helpers -------------------------------------------------------------
        def set_state(self, state, why):
            if state != self.state:
                self.get_logger().info(f'{self.state} -> {state}: {why}')
                self.state = state
                self.t_state_change = time.monotonic()

        def seed(self, q):
            for name, value in zip(JOINTS, q):
                self.robot.set_joint(name, float(value))
            self.robot.update_kinematics()
            self.last_q = np.asarray(q, dtype=float).copy()

        def fk_set(self, q):
            for name, value in zip(JOINTS, q):
                self.fk_robot.set_joint(name, float(value))
            self.fk_robot.update_kinematics()

        def current_q(self):
            return np.array([self.robot.get_joint(n) for n in JOINTS])

        def report(self):
            if self.state == WAITING:
                self.get_logger().info('WAITING: no arm feedback yet')
            elif self.state == ARMED:
                if self.gate is None:
                    self.get_logger().info('ARMED: no VR pose yet')
                else:
                    self.get_logger().info(
                        f'ARMED: target is {self.gate[0] * 1000:.0f} mm / {self.gate[1]:.0f} deg from the arm; '
                        f'need < {args.engage_distance * 1000:.0f} mm / {args.engage_angle:.0f} deg. '
                        f'Is the controller tracking? Bring the hand to where the arm is.')
            elif self.residual:
                r = np.array(self.residual) * 1000
                self.get_logger().info(
                    f'TRACKING {self.solved / args.report_period:5.1f} Hz  residual mean {r.mean():.2f} mm max {r.max():.2f} mm  '
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
    parser.add_argument('--engage-distance', type=float, default=0.05,
                        help='m; the target must come this close to the arm before anything is published')
    parser.add_argument('--engage-angle', type=float, default=20.0, help='deg; orientation counterpart of --engage-distance')
    parser.add_argument('--max-velocity', type=float, default=1.5,
                        help='rad/s per joint, applied per message using the measured interval')
    parser.add_argument('--max-residual', type=float, default=0.03,
                        help='m; above this the target is treated as unreachable and publishing stops')
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
