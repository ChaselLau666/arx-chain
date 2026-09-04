#!/usr/bin/env python3
"""Report whether teleop is actually going through the pose filter.

Watches the raw VR stream, the filtered stream and the arm's own feedback at
the same time, and prints what each one is doing. It answers two separate
questions that are easy to confuse: is the filter wired into the path the arms
listen to, and is it measurably removing anything.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))
os.chdir(str(ROOT / "act"))

from utils.setup_loader import setup_loader


def jitter(values):
    """Second difference: how abruptly the signal changes direction."""
    if len(values) < 3:
        return float("nan")

    return float(np.std(np.diff(np.asarray(values, dtype=float), n=2)))


def main(args):
    setup_loader(str(ROOT / "act"))
    import rclpy
    from rclpy.node import Node
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotStatus

    class Monitor(Node):
        def __init__(self):
            super().__init__("vr_filter_monitor")
            self.samples = {}
            self.counts = {}
            for topic in (args.raw, args.filtered):
                self.create_subscription(PosCmd, topic, self._pose(topic), 50)
            self.create_subscription(RobotStatus, args.feedback, self._joints(), 50)

        def _pose(self, topic):
            def callback(message):
                self.counts[topic] = self.counts.get(topic, 0) + 1
                self.samples.setdefault(topic, []).append(
                    [message.x, message.y, message.z])
            return callback

        def _joints(self):
            def callback(message):
                self.counts[args.feedback] = self.counts.get(args.feedback, 0) + 1
                self.samples.setdefault(args.feedback, []).append(
                    list(message.joint_pos)[:6])
            return callback

    rclpy.init()
    node = Monitor()
    print(f"watching for {args.seconds:.0f}s - move the VR controllers now\n", flush=True)
    deadline = time.monotonic() + args.seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    counts, samples = node.counts, node.samples
    node.destroy_node()
    rclpy.shutdown()

    print(f'{"topic":>26} {"msgs":>7} {"rate":>9} {"jitter":>11} {"travel":>9}')
    for topic in (args.raw, args.filtered, args.feedback):
        n = counts.get(topic, 0)
        data = np.asarray(samples.get(topic, []), dtype=float)
        if len(data) < 3:
            print(f"{topic:>26} {n:7d} {n / args.seconds:8.1f}Hz {'-':>11} {'-':>9}")
            continue
        per_column = [jitter(data[:, i]) for i in range(data.shape[1])]
        travel = float(np.max(data.max(axis=0) - data.min(axis=0)))
        print(f"{topic:>26} {n:7d} {n / args.seconds:8.1f}Hz "
              f"{np.mean(per_column):11.6f} {travel:9.4f}")

    raw, filtered = samples.get(args.raw, []), samples.get(args.filtered, [])
    print()
    if not raw:
        print("No raw VR poses arrived. The filter has nothing to work on, and this")
        print("says nothing about whether it is wired in - check the VR rig first.")

        return 1
    if not filtered:
        print("Raw poses are arriving but the filtered topic is silent: the filter")
        print("node is not running, and the arms are receiving nothing.")

        return 1

    raw, filtered = np.asarray(raw, dtype=float), np.asarray(filtered, dtype=float)
    raw_jitter = float(np.mean([jitter(raw[:, i]) for i in range(3)]))
    filtered_jitter = float(np.mean([jitter(filtered[:, i]) for i in range(3)]))
    raw_travel = float(np.max(raw.max(axis=0) - raw.min(axis=0)))
    filtered_travel = float(np.max(filtered.max(axis=0) - filtered.min(axis=0)))

    print(f"jitter removed : {(1 - filtered_jitter / max(raw_jitter, 1e-12)) * 100:.1f}%")
    print(f"motion kept    : {filtered_travel / max(raw_travel, 1e-12) * 100:.1f}%")
    if raw_travel < args.min_travel:
        print(f"\nThe controllers barely moved ({raw_travel:.4f} m), so these numbers are")
        print("noise on noise. Move them around and run this again.")

    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/ARX_VR_L")
    parser.add_argument("--filtered", default="/ARX_VR_L_filtered")
    parser.add_argument("--feedback", default="/arm_l_status_full")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--min-travel", type=float, default=0.01,
                        help="below this the controllers were basically still")

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
