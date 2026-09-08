"""Wait for the lift feedback to reach and remain near a requested height."""
from __future__ import annotations

import argparse
import collections
import time

import numpy as np
import rclpy
from rclpy.node import Node
from arm_control.msg import PosCmd


def main(args) -> int:
    node = Node("tau0vla_height_monitor")
    samples = collections.deque(maxlen=2000)
    node.create_subscription(
        PosCmd,
        "/body_information",
        lambda message: samples.append((time.monotonic(), float(message.height))),
        10,
    )
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if len(samples) < 2 or samples[-1][0] - samples[0][0] < args.window:
                continue
            cutoff = samples[-1][0] - args.window
            values = np.asarray([value for stamp, value in samples if stamp >= cutoff])
            if (
                len(values) >= 2
                and abs(float(values[-1]) - args.target) <= args.tolerance
                and float(np.ptp(values)) <= args.tolerance
            ):
                print(f"HEIGHT_STABLE target={args.target:.6f} feedback={values[-1]:.6f}")
                return 0
        print("REFUSED: lift did not reach a stable target before timeout")
        return 1
    finally:
        node.destroy_node()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    arguments = parser.parse_args()
    rclpy.init()
    try:
        raise SystemExit(main(arguments))
    finally:
        rclpy.shutdown()
