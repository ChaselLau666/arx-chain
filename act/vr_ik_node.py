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

Engagement is deliberately gentle. The solver state is initialised from the
arm's actual joint feedback, so the first solves start from where the arm is
and walk toward the VR target at a bounded rate rather than commanding a jump
to it. Nothing is published until both a feedback message and a VR pose have
arrived, and never without --execute.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x5_model import EE_FRAME, GRIPPER_SCALE, JOINTS, kinematic_urdf, target_transform  # noqa: E402


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

            self.q_feedback = None      # latest joint_pos[0:6] from the arm
            self.engaged = False        # solver seeded from feedback and a pose has arrived
            self.last_q = None
            self.solved = self.clamped = 0
            self.residual = []
            self.t_last_pose = None

            self.pub = self.create_publisher(RobotStatus, args.out_topic, 10) if args.execute else None
            self.create_subscription(RobotStatus, args.feedback_topic, self.on_feedback, 10)
            self.create_subscription(PosCmd, args.in_topic, self.on_pose, 10)
            self.create_timer(args.report_period, self.report)
            self.get_logger().info(
                f'{args.in_topic} -> IK -> {args.out_topic}  '
                f'{"PUBLISHING" if args.execute else "DRY-RUN, pass --execute to publish"}; '
                f'max step {args.max_step:.3f} rad/tick, gripper scale {GRIPPER_SCALE:+.3f}')

        # --- inputs ------------------------------------------------------------
        def on_feedback(self, msg):
            self.q_feedback = np.array(msg.joint_pos[:6], dtype=float)

        def on_pose(self, msg):
            if self.q_feedback is None:
                return  # cannot engage without knowing where the arm is
            if not self.engaged:
                self.seed(self.q_feedback)
                self.engaged = True
                self.get_logger().info(f'engaged from feedback q={np.round(self.q_feedback, 3)}')

            self.task.T_world_frame = target_transform([msg.x, msg.y, msg.z], [msg.roll, msg.pitch, msg.yaw])
            self.solver.solve(True)
            self.robot.update_kinematics()
            q = self.current_q()

            # Belt and braces on top of the solver's own dt-bounded step: a
            # target that is far away, or a solver hiccup, must not become a
            # single-tick lunge. The clamp is per joint and logged.
            step = q - self.last_q
            over = np.abs(step) > args.max_step
            if over.any():
                q = self.last_q + np.clip(step, -args.max_step, args.max_step)
                self.seed(q)
                self.clamped += 1
            self.last_q = q

            T = self.robot.get_T_world_frame(EE_FRAME)
            self.residual.append(np.linalg.norm(T[:3, 3] - self.task.T_world_frame[:3, 3]))
            self.solved += 1
            self.t_last_pose = time.monotonic()

            if self.pub is not None:
                out = RobotStatus()
                out.header.stamp = self.get_clock().now().to_msg()
                out.joint_pos[:6] = q.tolist()
                out.joint_pos[6] = float(msg.gripper) * GRIPPER_SCALE
                self.pub.publish(out)

        # --- helpers -------------------------------------------------------------
        def seed(self, q):
            for name, value in zip(JOINTS, q):
                self.robot.set_joint(name, float(value))
            self.robot.update_kinematics()
            self.last_q = np.asarray(q, dtype=float).copy()

        def current_q(self):
            return np.array([self.robot.get_joint(n) for n in JOINTS])

        def report(self):
            if not self.engaged:
                why = 'no arm feedback yet' if self.q_feedback is None else 'no VR pose yet'
                self.get_logger().info(f'waiting: {why}')
                return
            if not self.residual:
                return
            r = np.array(self.residual) * 1000
            rate = self.solved / args.report_period
            self.get_logger().info(
                f'{rate:5.1f} Hz  residual mean {r.mean():.2f} mm max {r.max():.2f} mm  '
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
    parser.add_argument('--feedback-topic', help='arm feedback used to seed the solver, default /arm_slave_{l,r}_status')
    parser.add_argument('--node-name')
    parser.add_argument('--urdf-cache', default=str(Path(__file__).resolve().parent / 'x5_kin.urdf'))
    parser.add_argument('--dt', type=float, default=1 / 100.0,
                        help='solver integration step; roughly the VR message period')
    parser.add_argument('--max-step', type=float, default=0.06,
                        help='largest per-joint change allowed per message, rad')
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
