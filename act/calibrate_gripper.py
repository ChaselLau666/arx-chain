# -- coding: UTF-8
"""Measure how gripper commands map to gripper feedback on this machine.

Replay commands the gripper with values recorded in an earlier session, but the
feedback read back differs from them by a large constant while the arm joints
agree to within 1-2%. This script varies only the gripper, holding every arm
joint at the pose it is already in, so the command-to-feedback relation can be
measured without moving the arm.
"""

import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import time
import argparse
import signal
import threading

import yaml
import rclpy
import numpy as np

from functools import partial

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader

np.set_printoptions(linewidth=200, suppress=True)

GRIPPER = 6
stop_requested = threading.Event()


def signal_handler(signum, frame, ros_operator):
    if stop_requested.is_set():
        print('\nSecond interrupt; exiting immediately.')
        sys.exit(1)
    print('\nCaught Ctrl+C; stopping after the current step.')
    stop_requested.set()
    ros_operator.request_arm_publish_stop()


def wait_for_feedback(ros_operator, timeout=5.0):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        left, right = ros_operator.follow_left_arm_deque, ros_operator.follow_right_arm_deque
        if left and right:
            return (list(left[-1].joint_pos)[:7], list(right[-1].joint_pos)[:7])
        time.sleep(0.02)
    raise RuntimeError('no arm feedback in %.1fs; is the arm stack running?' % timeout)


def sample_feedback(ros_operator, seconds, rate_hz):
    """Collect gripper feedback for both arms over a settle window."""
    samples = []
    rate = Rate(rate_hz)
    deadline = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < deadline and not stop_requested.is_set():
        left, right = ros_operator.follow_left_arm_deque, ros_operator.follow_right_arm_deque
        if left and right:
            samples.append((list(left[-1].joint_pos)[GRIPPER],
                            list(right[-1].joint_pos)[GRIPPER]))
        rate.sleep()
    if not samples:
        raise RuntimeError('no feedback collected while settling')

    return np.asarray(samples)


def ramp_gripper(ros_operator, hold_left, hold_right, arms, start, target, args):
    """Move the gripper from start to target in bounded steps, arms held still."""
    rate = Rate(args.frame_rate)
    steps = max(1, int(np.ceil(abs(target - start) / args.step)))
    for i in range(1, steps + 1):
        if stop_requested.is_set():
            return None
        value = start + (target - start) * i / steps
        left = list(hold_left)
        right = list(hold_right)
        if 'left' in arms:
            left[GRIPPER] = value
        if 'right' in arms:
            right[GRIPPER] = value
        ros_operator.follow_arm_publish(left, right)
        rate.sleep()

    return target


def main(args):
    arms = ('left', 'right') if args.arm == 'both' else (args.arm,)
    armed = bool(args.execute)
    if armed:
        print(f'This will move the {"/".join(arms)} gripper through: {args.values}')
        print('Arm joints are held at their current measured pose and will not be commanded to move.')
        if input('Type MOVE GRIPPER to continue: ') != 'MOVE GRIPPER':
            raise RuntimeError('cancelled; nothing was published')
    else:
        print('DRY-RUN: no publisher is created; pass --execute to move the gripper.')

    setup_loader(ROOT)
    rclpy.init()

    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    ros_operator = RosOperator(args, config, in_collect=False)
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()
    signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))

    try:
        hold_left, hold_right = wait_for_feedback(ros_operator, args.feedback_timeout)
        print(f'holding left  arm joints at {np.round(hold_left[:6], 4)}')
        print(f'holding right arm joints at {np.round(hold_right[:6], 4)}')
        print(f'gripper feedback right now : left {hold_left[GRIPPER]:.4f}, '
              f'right {hold_right[GRIPPER]:.4f}')

        rows = []
        current = hold_right[GRIPPER] if 'right' in arms else hold_left[GRIPPER]
        for target in args.values:
            if stop_requested.is_set():
                break
            print(f'\n--- commanding gripper {target:+.4f} ---')
            if armed:
                if ramp_gripper(ros_operator, hold_left, hold_right, arms,
                                current, target, args) is None:
                    break
                current = target
                measured = sample_feedback(ros_operator, args.settle, args.frame_rate)
                left_fb, right_fb = measured[:, 0], measured[:, 1]
                print(f'  feedback left  {left_fb.mean():+.4f} (spread {left_fb.ptp():.4f})')
                print(f'  feedback right {right_fb.mean():+.4f} (spread {right_fb.ptp():.4f})')
                rows.append((target, left_fb.mean(), right_fb.mean()))
                if args.prompt:
                    input('  look at the gripper, note the opening, then press Enter: ')
            else:
                print('  DRY-RUN: would ramp here and sample the feedback')
                rows.append((target, float('nan'), float('nan')))

        if rows:
            print('\n=== command vs feedback ===')
            print(f'{"command":>10} {"left fb":>10} {"right fb":>10} '
                  f'{"left off":>10} {"right off":>10}')
            for cmd, lf, rf in rows:
                print(f'{cmd:10.4f} {lf:10.4f} {rf:10.4f} {lf - cmd:10.4f} {rf - cmd:10.4f}')
            if len(rows) >= 2 and armed:
                cmds = np.array([r[0] for r in rows])
                for name, col in (('left', 1), ('right', 2)):
                    fb = np.array([r[col] for r in rows])
                    slope = np.polyfit(cmds, fb, 1)[0]
                    print(f'{name:>6} gripper: feedback = {slope:.4f} * command '
                          f'{np.mean(fb - slope * cmds):+.4f}')
                    print(f'         offset spread across points: '
                          f'{np.ptp(fb - cmds):.4f}  (0 means a pure constant offset)')
    finally:
        ros_operator.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['left', 'right', 'both'], default='right')
    parser.add_argument('--values', type=float, nargs='+',
                        default=[-3.39, -2.85, -2.30, -1.75, -1.28],
                        help='gripper commands to visit, in the recorded qpos scale')
    parser.add_argument('--execute', action='store_true',
                        help='publish gripper targets; default is dry-run')
    parser.add_argument('--step', type=float, default=0.05,
                        help='maximum gripper change per cycle while ramping')
    parser.add_argument('--settle', type=float, default=1.5,
                        help='seconds of feedback to average at each command')
    parser.add_argument('--prompt', action='store_true',
                        help='pause at each command so the opening can be observed')
    parser.add_argument('--frame-rate', dest='frame_rate', type=int, default=60)
    parser.add_argument('--feedback-timeout', type=float, default=5.0)
    parser.add_argument('--config', type=str, default=str(Path.joinpath(ROOT, 'data/config.yaml')))

    # RosOperator expects these; none of them are used by this script.
    parser.add_argument('--use_base', action='store_true')
    parser.add_argument('--use_depth_image', action='store_true')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        default=['head', 'left_wrist', 'right_wrist'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
