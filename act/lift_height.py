"""Explicit, operator-confirmed LIFT2s height control.

This tool never starts or restarts the body controller.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.setup_loader import setup_loader


def feedback_is_stable(values, tolerance: float) -> bool:
    return len(values) >= 2 and max(values) - min(values) <= tolerance


def wait_future(node, future, timeout: float):
    import rclpy

    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        raise RuntimeError("service call timed out")
    return future.result()


def main(args) -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", "62")
    setup_loader(ROOT)
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import SetBool
    from arx_lift_controller.srv import LiftHeightStatus, SetLiftHeight

    rclpy.init()
    node = Node("lift_height_operator")
    try:
        status_client = node.create_client(LiftHeightStatus, "/lift_height_status")
        if not status_client.wait_for_service(timeout_sec=args.timeout):
            raise RuntimeError("body is not running or /lift_height_status is unavailable")

        def status():
            response = wait_future(node, status_client.call_async(LiftHeightStatus.Request()), args.timeout)
            print(
                f"Current height: {response.current_height:.6f}\n"
                f"Commanded height: {response.commanded_height:.6f}\n"
                f"Locked: {response.locked}"
            )
            return response

        if args.command == "status":
            status()
            return
        if args.command in {"lock", "unlock"}:
            client = node.create_client(SetBool, "/lift_height_lock")
            if not client.wait_for_service(timeout_sec=args.timeout):
                raise RuntimeError("/lift_height_lock unavailable")
            request = SetBool.Request()
            request.data = args.command == "lock"
            response = wait_future(node, client.call_async(request), args.timeout)
            if not response.success:
                raise RuntimeError(response.message)
            print(response.message)
            status()
            return

        before = status()
        target = float(args.target)
        direction = (
            "UP" if target > before.commanded_height else "DOWN" if target < before.commanded_height else "HOLD"
        )
        print(
            f"Target command: {target:.6f}\nDirection: {direction}\n"
            f"Expected motion: platform moves {direction}; feedback may use a calibrated offset"
        )
        if not args.execute:
            print("Dry run only. Re-run with --execute to permit motion.")
            return
        confirmation = input(f"Type SET HEIGHT {target:.6f} to continue: ").strip()
        if confirmation != f"SET HEIGHT {target:.6f}":
            raise RuntimeError("confirmation did not match; no command sent")
        client = node.create_client(SetLiftHeight, "/lift_height_set")
        if not client.wait_for_service(timeout_sec=args.timeout):
            raise RuntimeError("/lift_height_set unavailable")
        request = SetLiftHeight.Request()
        request.target_height = target
        response = wait_future(node, client.call_async(request), args.timeout)
        if not response.success:
            raise RuntimeError(response.message)
        print(response.message)
        deadline = time.monotonic() + args.motion_timeout
        sample_count = max(2, math.ceil(args.settle_window / args.poll_interval) + 1)
        recent_heights = collections.deque(maxlen=sample_count)
        while time.monotonic() < deadline:
            current = status()
            recent_heights.append(float(current.current_height))
            command_matches = abs(current.commanded_height - target) <= 1e-6
            stable = len(recent_heights) == sample_count and feedback_is_stable(
                recent_heights, args.stability_tolerance
            )
            expected_matches = args.expected_feedback is None or abs(
                current.current_height - args.expected_feedback
            ) <= args.feedback_tolerance
            if command_matches and stable and expected_matches:
                print(
                    f"Height feedback settled at {current.current_height:.6f} "
                    f"for command {target:.6f}."
                )
                return
            time.sleep(args.poll_interval)
        raise RuntimeError("height feedback did not settle before timeout; body remains running")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "set", "lock", "unlock"])
    parser.add_argument("target", type=float, nargs="?")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--motion-timeout", type=float, default=60.0)
    parser.add_argument("--settle-window", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--stability-tolerance", type=float, default=0.01)
    parser.add_argument("--expected-feedback", type=float, default=None)
    parser.add_argument("--feedback-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "set" and args.target is None:
        parser.error("set requires a target height")
    if args.command != "set" and args.target is not None:
        parser.error("target is only valid with set")
    return args


if __name__ == "__main__":
    main(parse_args())
