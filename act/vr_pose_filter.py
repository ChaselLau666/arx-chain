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

It also carries the offset that makes the stream relative. The headset sends an
absolute pose and is never told that anything else moved the arm, so on the
first frame after the arm is parked it would command the arm to wherever the
hand happens to be. Anchoring on the arm once, when a /arx_joy mute ends, and
carrying the difference on everything after is the same trick Human DAgger plays
on every takeover (_rebase_one in human_dagger_core.py): a still hand holds the
arm where it is, a moving one carries on from there.
"""

import os
import sys
import time

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
    from arx5_arm_msg.msg import RobotStatus
    from std_msgs.msg import Int32MultiArray
    from scipy.spatial.transform import Rotation

    class VrPoseFilter(Node):
        def __init__(self):
            super().__init__(args.node_name)
            self.alpha = 1.0 - np.exp(-args.dt / max(args.tau, 1e-6))
            self.pos_prev = None
            self.rot_prev = None
            self.raw_pos_prev = None
            self.raw_rot_prev = None
            self.passed = 0
            self.rejected = 0
            self.pub = self.create_publisher(PosCmd, args.out_topic, 10)
            self.create_subscription(PosCmd, args.in_topic, self.on_pose, 10)
            # Stand aside while something is driving the arms home. data[1] == 1
            # puts X5Controller in GO_HOME, but VrCmdCallback restores
            # END_CONTROL on the very next pose published here, which cancels the
            # move within a frame. Every message refreshes the window, so a mute
            # lasts exactly as long as the sender keeps asking and no longer.
            self.mute_until = 0.0
            self.muted = False
            self.create_subscription(Int32MultiArray, '/arx_joy', self.on_joy, 10)
            # Where the arm actually is, which is the one thing the headset
            # cannot know: the serial link carries nothing back to it.
            self.arm_pos = None
            self.arm_rot = None
            self.arm_status_at = None
            self.offset_pos = np.zeros(3)
            self.offset_rot = Rotation.identity()
            self.initial_rebase_done = not args.rebase
            self.last_safety_warning_at = 0.0
            self.arm_status_timeout = max(args.arm_status_timeout, 0.01)
            self.max_position_jump = max(args.max_position_jump, 0.0)
            self.max_angle_jump = np.radians(max(args.max_angle_jump_deg, 0.0))
            if args.rebase:
                self.create_subscription(RobotStatus, args.arm_status_topic,
                                         self.on_arm_status, 10)
            self.create_timer(args.report_period, self.report)
            self.get_logger().info(
                f'{args.in_topic} -> {args.out_topic}  tau={args.tau:.3f}s '
                f'alpha={self.alpha:.4f} (cutoff {1 / (2 * np.pi * max(args.tau, 1e-9)):.2f} Hz)')

        def on_joy(self, msg):
            if len(msg.data) > 1 and msg.data[1] == 1:
                if time.monotonic() >= self.mute_until:
                    self.get_logger().info('/arx_joy GO_HOME: standing aside')
                self.mute_until = time.monotonic() + args.home_mute
                # Remember the transition even if no VR frame arrives during
                # the mute window. The first later frame must still rebase.
                self.muted = True

        def safety_warn(self, message):
            now = time.monotonic()
            if now - self.last_safety_warning_at >= 1.0:
                self.get_logger().warn(message)
                self.last_safety_warning_at = now

        def on_arm_status(self, msg):
            end_pos = np.array(msg.end_pos[:6], dtype=float)
            if end_pos.size != 6 or not np.all(np.isfinite(end_pos)):
                self.safety_warn(f'ignoring invalid pose on {args.arm_status_topic}')
                return
            self.arm_pos = end_pos[:3]
            self.arm_rot = Rotation.from_euler('xyz', end_pos[3:6])
            self.arm_status_at = time.monotonic()

        def rebase(self, reason):
            """Re-aim the stream at wherever the arm has just been left.

            The rotation offset is applied on the right so that a rotation of
            the hand becomes the same rotation of the tool: with
            f(X) = X * P^-1 * R, f(P) is R and f(D * X) is D * f(X). Composing
            on the left would satisfy the first and not the second.
            """
            if self.pos_prev is None:
                self.safety_warn('waiting for the first valid VR pose before publishing')
                return False
            if self.arm_pos is None or self.arm_status_at is None:
                self.safety_warn(
                    f'holding VR output: {args.arm_status_topic} has not published yet')
                return False
            age = time.monotonic() - self.arm_status_at
            if age > self.arm_status_timeout:
                self.safety_warn(
                    f'holding VR output: {args.arm_status_topic} is stale ({age:.2f}s)')
                return False
            self.offset_pos = self.arm_pos - self.pos_prev
            self.offset_rot = self.rot_prev.inv() * self.arm_rot
            self.get_logger().info(
                f'{reason}: re-aimed onto the arm: '
                f'{np.round(self.offset_pos * 1000, 1)} mm, '
                f'{np.degrees(self.offset_rot.magnitude()):.1f} deg')
            return True

        def on_pose(self, msg):
            values = np.array(
                [msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw], dtype=float)
            if not np.all(np.isfinite(values)):
                self.rejected += 1
                self.safety_warn('ignoring non-finite VR pose')
                return
            pos = values[:3]
            rot = Rotation.from_euler('xyz', values[3:])

            discontinuity = False
            position_jump = 0.0
            angle_jump = 0.0
            if self.raw_pos_prev is not None:
                position_jump = float(np.linalg.norm(pos - self.raw_pos_prev))
                angle_jump = float((self.raw_rot_prev.inv() * rot).magnitude())
                discontinuity = (
                    (self.max_position_jump > 0.0 and
                     position_jump > self.max_position_jump) or
                    (self.max_angle_jump > 0.0 and angle_jump > self.max_angle_jump)
                )
            self.raw_pos_prev = pos.copy()
            self.raw_rot_prev = rot

            if self.pos_prev is None or discontinuity:
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
            # Filter state above is kept current even while muted, so
            # publishing resumes from where the hand is now rather than from a
            # stale pose. The mute ending is what says the arm has been left
            # somewhere new, so that is where the stream is re-aimed.
            muted = time.monotonic() < self.mute_until
            rebase_reason = None
            if self.muted and not muted and args.rebase:
                rebase_reason = 'GO_HOME complete'
            self.muted = muted
            if muted:
                return

            if args.rebase and not self.initial_rebase_done:
                rebase_reason = 'initial startup'
            if discontinuity and args.rebase:
                self.rejected += 1
                self.initial_rebase_done = False
                rebase_reason = 'VR discontinuity'
                self.safety_warn(
                    f'VR pose jumped {position_jump * 1000:.1f} mm / '
                    f'{np.degrees(angle_jump):.1f} deg; rebasing before publishing')
            if rebase_reason is not None:
                if not self.rebase(rebase_reason):
                    return
                self.initial_rebase_done = True

            out.x, out.y, out.z = (float(v) for v in self.pos_prev + self.offset_pos)
            out.roll, out.pitch, out.yaw = (
                float(v) for v in (self.rot_prev * self.offset_rot).as_euler('xyz'))
            self.pub.publish(out)
            self.passed += 1

        def report(self):
            self.get_logger().info(
                f'forwarded {self.passed} poses, rejected/rebased {self.rejected}')

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
    parser.add_argument('--home-mute', type=float, default=0.5,
                        help='seconds to stop publishing after each /arx_joy GO_HOME message. '
                             'Every message refreshes it, so whoever is driving the arms home '
                             'sets the duration by how long it keeps asking')
    parser.add_argument('--arm-status-topic', default='/arm_l_status_full',
                        help='where the arm on this side reports its end-effector pose, read '
                             'to re-aim the stream after the arm is parked')
    parser.add_argument('--arm-status-timeout', type=float, default=0.5,
                        help='maximum age in seconds of arm feedback used for a rebase')
    parser.add_argument('--max-position-jump', type=float, default=0.08,
                        help='raw position step in metres that triggers a safe rebase; '
                             'zero disables the check')
    parser.add_argument('--max-angle-jump-deg', type=float, default=45.0,
                        help='raw orientation step in degrees that triggers a safe rebase; '
                             'zero disables the check')
    parser.add_argument('--no-rebase', dest='rebase', action='store_false',
                        help='forward poses as they come, without re-aiming them onto the '
                             'arm when a mute ends. The arm is then pulled back to wherever '
                             'the headset last commanded it')
    parser.add_argument('--report-period', type=float, default=2.0)

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    import rclpy
    rclpy.init()
    main(parse_args())
