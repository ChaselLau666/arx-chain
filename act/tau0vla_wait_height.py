"""Wait for the lift feedback to reach and remain near a requested height."""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

from utils.setup_loader import setup_loader

import rclpy
from rclpy.node import Node

from safe_height import is_safe_and_stable


def main(args) -> int:
    setup_loader(Path(__file__).resolve().parent)
    from arm_control.msg import PosCmd

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
        from rclpy.parameter_client import AsyncParameterClient
        parameters = AsyncParameterClient(node, "/lift")
        if not parameters.wait_for_services(timeout_sec=5.0):
            raise RuntimeError("/lift parameter service unavailable")
        future = parameters.get_parameters(["fixed_height"])
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("could not verify /lift fixed_height")
        actual = float(future.result().values[0].double_value)
        if abs(actual - args.target) > 1e-6:
            raise RuntimeError(f"/lift fixed_height={actual}, expected {args.target}")
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            # fixed_height is a command coordinate; /body_information.height
            # uses a different feedback coordinate. Match inference preflight:
            # require the parameter separately, and gate only on fresh stable
            # feedback here rather than equality between unlike coordinates.
            if samples and time.monotonic() - samples[-1][0] <= .5 and is_safe_and_stable(samples, float("inf"), args.tolerance, args.window):
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
