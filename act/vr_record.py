#!/usr/bin/env python3
"""Record the VR stream and the arms together, to be read back afterwards.

Watching these live means coordinating "start now" between two people and a
terminal, which has cost this project several inconclusive sessions. Recording
instead lets the operator work the controllers at their own pace and the log be
read afterwards.

Every field of both controller messages is kept, plus both arms' joint
positions, so a session can answer questions that were not asked when it was
recorded: whether the headset was awake, whether a controller was tracking,
what the trigger did, and how the arms responded.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

VR_FIELDS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper',
             'height', 'head_pit', 'head_yaw', 'chx', 'chy', 'chz', 'mode1', 'mode2')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('seconds', nargs='?', type=float, default=30.0)
    parser.add_argument('--out-dir', default=str(Path.home() / 'vr_logs'))
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotStatus

    rclpy.init()
    node = Node('vr_record')
    data = {'vr_l': [], 'vr_r': [], 'arm_l': [], 'arm_r': []}

    def on_vr(msg, key):
        data[key].append([time.time()] + [float(getattr(msg, f)) for f in VR_FIELDS])

    def on_arm(msg, key):
        data[key].append([time.time()] + [float(v) for v in msg.joint_pos[:7]])

    node.create_subscription(PosCmd, '/ARX_VR_L', lambda m: on_vr(m, 'vr_l'), 50)
    node.create_subscription(PosCmd, '/ARX_VR_R', lambda m: on_vr(m, 'vr_r'), 50)
    node.create_subscription(RobotStatus, '/arm_l_status_full', lambda m: on_arm(m, 'arm_l'), 50)
    node.create_subscription(RobotStatus, '/arm_r_status_full', lambda m: on_arm(m, 'arm_r'), 50)

    print(f'Recording {args.seconds:.0f}s. Work the controllers now - hold the trigger and move,')
    print('release, hold again. What matters is that it happens; the log keeps the timing.\n')
    print(f'  {"sec":>4s} {"L grip":>7s} {"L moved":>9s} {"R grip":>7s} {"R moved":>9s} {"head":>7s}   arms')

    start = time.time()
    last = start
    try:
        while time.time() - start < args.seconds and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.time() - last < 1.0:
                continue
            last = time.time()

            def window(key, lo=1, hi=4):
                rows = [r for r in data[key] if r[0] > last - 1.0]
                if not rows:
                    return None, 0.0
                block = np.array(rows)
                return block[-1], float(np.linalg.norm(block[:, lo:hi].max(0) - block[:, lo:hi].min(0)) * 1000)

            lrow, lmm = window('vr_l')
            rrow, rmm = window('vr_r')
            head = '-'
            if lrow is not None:
                hb = np.array([r for r in data['vr_l'] if r[0] > last - 1.0])
                head = 'moving' if hb[:, 9:11].std(0).max() > 1e-4 else 'still'
            arms = []
            for key in ('arm_l', 'arm_r'):
                rows = [r for r in data[key] if r[0] > last - 1.0]
                arms.append('-' if not rows else f'{np.degrees(np.abs(np.array(rows)[-1, 1:7])).max():.0f}deg')
            print(f'  {int(last - start):4d} '
                  f'{"-" if lrow is None else f"{lrow[7]:.2f}":>7s} {lmm:7.0f}mm '
                  f'{"-" if rrow is None else f"{rrow[7]:.2f}":>7s} {rmm:7.0f}mm '
                  f'{head:>7s}   {" ".join(arms)}')
    except KeyboardInterrupt:
        print('\nStopped early.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'vr_{time.strftime("%Y%m%d-%H%M%S")}.npz'
    np.savez(path, vr_fields=np.array(VR_FIELDS),
             **{k: np.array(v) if v else np.zeros((0, 1)) for k, v in data.items()})
    counts = {k: len(v) for k, v in data.items()}
    print(f'\nSaved {path}')
    print(f'  samples: {counts}')
    if not counts['vr_r']:
        print('  /ARX_VR_R carried nothing: serial_port_node is not running, or the headset is asleep.')


if __name__ == '__main__':
    main()
