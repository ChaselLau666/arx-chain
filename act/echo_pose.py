# -- coding: UTF-8
"""Read the arm feedback once, publish it straight back, and see what moves.

If a joint's feedback and its command use the same scale, echoing the reading
back is a request to stay exactly where it is, and nothing should move. Any
joint that does move is one where the two scales differ.
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
import threading

import yaml
import rclpy
import numpy as np

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader

np.set_printoptions(linewidth=200, suppress=True, precision=4)

NAMES = [f'{side}_{name}' for side in ('L', 'R')
         for name in ('j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'grip')]


def read_pose(ros_operator, timeout=5.0):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        left, right = ros_operator.follow_left_arm_deque, ros_operator.follow_right_arm_deque
        if left and right:
            return (list(left[-1].joint_pos)[:7], list(right[-1].joint_pos)[:7])
        time.sleep(0.02)
    raise RuntimeError('no arm feedback in %.1fs; is the arm stack running?' % timeout)


def main(args):
    setup_loader(ROOT)
    rclpy.init()
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    ros_operator = RosOperator(args, config, in_collect=False)
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()

    try:
        left, right = read_pose(ros_operator, args.feedback_timeout)
        before = np.array(left + right)
        print('read back from the arms:')
        print(f'  left  {np.array(left)}')
        print(f'  right {np.array(right)}')

        if not args.execute:
            print('\nDRY-RUN: these exact values would be published back for '
                  f'{args.seconds:.1f}s. Pass --execute to do it.')

            return

        print(f'\nAbout to publish those same values back for {args.seconds:.1f}s.')
        print('Anything that moves is a joint whose command and feedback disagree.')
        print('WARNING: the gripper may move. Keep hands clear.')
        if input('Type ECHO POSE to continue: ') != 'ECHO POSE':
            raise RuntimeError('cancelled; nothing was published')

        rate = Rate(args.frame_rate)
        deadline = time.monotonic() + args.seconds
        sent = 0
        while rclpy.ok() and time.monotonic() < deadline:
            ros_operator.follow_arm_publish(left, right)
            sent += 1
            rate.sleep()

        after_left, after_right = read_pose(ros_operator, args.feedback_timeout)
        after = np.array(after_left + after_right)

        print(f'\npublished {sent} identical messages\n')
        print(f'{"joint":>7} {"sent":>10} {"before":>10} {"after":>10} {"moved":>10}')
        for i, name in enumerate(NAMES):
            moved = after[i] - before[i]
            flag = '  <-- moved' if abs(moved) > args.threshold else ''
            print(f'{name:>7} {before[i]:10.4f} {before[i]:10.4f} {after[i]:10.4f} '
                  f'{moved:10.4f}{flag}')
        print(f'\nthreshold for "moved" is {args.threshold} rad')
    finally:
        ros_operator.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='actually publish; default is dry-run')
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--frame-rate', dest='frame_rate', type=int, default=60)
    parser.add_argument('--feedback-timeout', type=float, default=5.0)
    parser.add_argument('--config', type=str, default=str(Path.joinpath(ROOT, 'data/config.yaml')))

    # RosOperator expects these; unused here.
    parser.add_argument('--use_base', action='store_true')
    parser.add_argument('--use_depth_image', action='store_true')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        default=['head', 'left_wrist', 'right_wrist'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
