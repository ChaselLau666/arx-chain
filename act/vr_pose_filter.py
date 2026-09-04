# -- coding: UTF-8
"""Low-pass the VR pose stream before it reaches the arm's IK.

Sits between the VR serial node and the arm controller: it subscribes to the
raw pose stream, applies the one-pole filter teleop-app uses on its own teleop
input, and republishes. The arm is pointed at the filtered topic through its
arm_sub_topic_name parameter, so neither the SDK nor its configuration has to
change.

Filtering here rather than on the recorded joint angles is what teleop-app
does, and for the same reason: the pose is what feeds inverse kinematics, and a
jump in the pose becomes a jump in the joint solution.
"""

import os
import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import threading

import numpy as np

from utils.setup_loader import setup_loader


def build_node(args):
    import rclpy
    from rclpy.node import Node
    from arm_control.msg import PosCmd
    from scipy.spatial.transform import Rotation

    class VrPoseFilter(Node):
        def __init__(self):
            super().__init__(args.node_name)
            self.alpha = 1.0 - np.exp(-args.dt / max(args.tau, 1e-6))
            self.pos_prev = None
            self.rot_prev = None
            self.passed = 0
            self.pub = self.create_publisher(PosCmd, args.out_topic, 10)
            self.create_subscription(PosCmd, args.in_topic, self.on_pose, 10)
            self.create_timer(args.report_period, self.report)
            self.get_logger().info(
                f'{args.in_topic} -> {args.out_topic}  tau={args.tau:.3f}s '
                f'alpha={self.alpha:.4f} (cutoff {1 / (2 * np.pi * max(args.tau, 1e-9)):.2f} Hz)')

        def on_pose(self, msg):
            pos = np.array([msg.x, msg.y, msg.z], dtype=float)
            rot = Rotation.from_euler('xyz', [msg.roll, msg.pitch, msg.yaw])

            if self.pos_prev is None:
                self.pos_prev, self.rot_prev = pos, rot
            else:
                self.pos_prev = self.pos_prev + self.alpha * (pos - self.pos_prev)
                # Orientation is interpolated along the shortest arc, as slerp
                # would: scaling the relative rotation vector avoids the sign and
                # wrap problems of filtering Euler angles directly.
                relative = self.rot_prev.inv() * rot
                self.rot_prev = self.rot_prev * Rotation.from_rotvec(
                    relative.as_rotvec() * self.alpha)

            out = PosCmd()
            for field in ('gripper', 'chx', 'chy', 'chz', 'height', 'head_pit', 'head_yaw',
                          'mode1', 'mode2'):
                if hasattr(msg, field):
                    setattr(out, field, getattr(msg, field))
            if hasattr(msg, 'temp_float_data'):
                out.temp_float_data = msg.temp_float_data
            out.x, out.y, out.z = (float(v) for v in self.pos_prev)
            out.roll, out.pitch, out.yaw = (float(v) for v in self.rot_prev.as_euler('xyz'))
            self.pub.publish(out)
            self.passed += 1

        def report(self):
            self.get_logger().info(f'forwarded {self.passed} poses')

    return rclpy, VrPoseFilter()


def main(args):
    setup_loader(ROOT)
    rclpy, node = build_node(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-topic', default='/ARX_VR_L')
    parser.add_argument('--out-topic', default='/ARX_VR_L_filtered')
    parser.add_argument('--tau', type=float, default=0.05,
                        help='time constant in seconds; teleop-app uses 0.05')
    parser.add_argument('--dt', type=float, default=1 / 60.0,
                        help='expected input period, used to derive alpha')
    parser.add_argument('--node-name', default='vr_pose_filter')
    parser.add_argument('--report-period', type=float, default=2.0)

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    import rclpy
    rclpy.init()
    main(parse_args())
