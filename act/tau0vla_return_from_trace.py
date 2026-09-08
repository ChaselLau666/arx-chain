"""Guarded recovery to the initial pose recorded in a calibrated rollout trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
import yaml

from tau0vla_calibration import current_robot_identity, return_trajectory
from utils.setup_loader import setup_loader


ROOT = Path(__file__).resolve().parent
ARM = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER = np.asarray([6, 13])


def load_target(path: Path):
    metadata = None
    initial = None
    first_tick = None
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "metadata" and metadata is None:
            metadata = event
        if event.get("event") == "return_result" and event.get("status") == "initial_pose":
            initial = event.get("target")
        if event.get("event") == "tick" and first_tick is None:
            first_tick = event.get("feedback")
    target = initial if initial is not None else first_tick
    values = np.asarray(target, dtype=np.float32)
    if metadata is None or values.shape != (14,) or not np.isfinite(values).all():
        raise RuntimeError("trace has no valid metadata and initial 14D feedback")
    return values, metadata


class ReturnNode(Node):
    def __init__(self, config):
        super().__init__("tau0vla_return_from_trace")
        from arx5_arm_msg.msg import RobotStatus

        self._message_type = RobotStatus
        self._lock = threading.Lock()
        self._latest = {}
        self._command_publishers = {
            "left": self.create_publisher(
                RobotStatus, config["arm_config"]["follow_arm_left_cmd_topic"], 10
            ),
            "right": self.create_publisher(
                RobotStatus, config["arm_config"]["follow_arm_right_cmd_topic"], 10
            ),
        }
        self.create_subscription(
            RobotStatus,
            config["arm_config"]["follow_arm_left_feedback_topic"],
            lambda message: self._receive("left", message),
            20,
        )
        self.create_subscription(
            RobotStatus,
            config["arm_config"]["follow_arm_right_feedback_topic"],
            lambda message: self._receive("right", message),
            20,
        )

    def _receive(self, side, message):
        values = np.asarray(message.joint_pos, dtype=np.float32)
        if values.shape == (7,) and np.isfinite(values).all():
            with self._lock:
                self._latest[side] = values.copy()

    def current(self):
        with self._lock:
            if set(self._latest) != {"left", "right"}:
                raise RuntimeError("both arm feedback streams are required")
            return np.concatenate((self._latest["left"], self._latest["right"]))

    def publish(self, values):
        for side, part in (("left", values[:7]), ("right", values[7:])):
            message = self._message_type()
            message.joint_pos[:7] = np.asarray(part, dtype=float).tolist()
            self._command_publishers[side].publish(message)


def run(args):
    target, metadata = load_target(args.trace)
    hostname, domain, boot_id, controllers = current_robot_identity()
    saved = metadata.get("calibration") or {}
    if (
        saved.get("hostname") != hostname
        or saved.get("ros_domain_id") != domain
        or saved.get("boot_id") != boot_id
        or saved.get("controller_identity") != controllers
    ):
        raise RuntimeError("trace calibration identity does not match the live robot/controllers")
    age = time.time() - args.trace.stat().st_mtime
    if not 0 <= age <= args.max_trace_age_s:
        raise RuntimeError(f"trace age {age:.1f}s exceeds recovery limit")
    print(f"Recovery trace: {args.trace}")
    print(f"Initial target: {np.array2string(target, precision=4)}")
    if not args.execute:
        print("DRY-RUN: no publisher was created.")
        return
    if input("Type RETURN LAST TRACE TO INITIAL POSE to move: ") != "RETURN LAST TRACE TO INITIAL POSE":
        raise RuntimeError("return recovery cancelled")

    setup_loader(ROOT)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    node = ReturnNode(config)
    abort = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: abort.set())
    spin_stop = threading.Event()

    def spin():
        while rclpy.ok() and not spin_stop.is_set():
            rclpy.spin_once(node, timeout_sec=.01)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                current = node.current()
                break
            except RuntimeError:
                time.sleep(.05)
        else:
            raise RuntimeError("arm feedback unavailable")
        trajectory = return_trajectory(current, target)
        period = 1 / 30
        next_tick = time.monotonic()
        for command in trajectory:
            if abort.is_set():
                raise RuntimeError("return recovery interrupted; publication stopped")
            node.publish(command)
            next_tick += period
            time.sleep(max(0.0, next_tick-time.monotonic()))
        hold_until = time.monotonic() + 1.0
        while time.monotonic() < hold_until:
            node.publish(target)
            time.sleep(period)
        samples = []
        verify_until = time.monotonic() + 2.0
        while time.monotonic() < verify_until:
            samples.append(node.current())
            time.sleep(period)
        values = np.asarray(samples)
        result = {
            "arm_error_max": float(np.max(np.abs(values[-1, ARM]-target[ARM]))),
            "gripper_error_max": float(np.max(np.abs(values[-1, GRIPPER]-target[GRIPPER]))),
            "feedback_spread_max": float(np.max(np.ptp(values, axis=0))),
            "trajectory_steps": len(trajectory),
        }
        if result["arm_error_max"] > .05 or result["gripper_error_max"] > .1 or result["feedback_spread_max"] > .01:
            raise RuntimeError(f"return recovery verification failed: {result}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(f"RETURN TO INITIAL COMPLETE: {json.dumps(result, sort_keys=True)}")
        print(f"Recovery report: {args.report}")
    finally:
        spin_stop.set()
        thread.join(timeout=2.0)
        node.destroy_node()


def parse_args():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-trace-age-s", type=float, default=1800.0)
    parser.add_argument("--config", type=Path, default=ROOT/"data/config.yaml")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/home/arx/logs/tau0vla-calibrated")/f"return_{stamp}.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    rclpy.init()
    try:
        run(parse_args())
    finally:
        rclpy.shutdown()
