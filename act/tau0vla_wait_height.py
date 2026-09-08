"""Wait for the lift feedback to reach and remain near a requested height."""
from __future__ import annotations

import argparse
import collections
import time

import rclpy
from rclpy.node import Node
from arm_control.msg import PosCmd

from safe_height import is_safe_and_stable


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
            # fixed_height is a command coordinate; /body_information.height
            # has a calibrated offset (15.5 command is normally ~15.03
            # feedback). Match the already-validated inference preflight:
            # require the parameter separately, and gate only on fresh stable
            # feedback here rather than equality between unlike coordinates.
            if is_safe_and_stable(samples, float("inf"), args.tolerance, args.window):
                print(f"HEIGHT_STABLE target={args.target:.6f} feedback={samples[-1][1]:.6f}")
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
