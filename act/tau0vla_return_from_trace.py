"""Guarded recovery to the fixed training initial pose for a calibrated trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
import yaml

from tau0vla_calibration import (
    artifact_from_dict,
    current_robot_identity,
    feedback_pose_to_command,
    load_artifact,
    load_training_ready_arms,
    return_trajectory,
    training_ready_targets,
    validate_artifact,
)
from utils.setup_loader import setup_loader


ROOT = Path(__file__).resolve().parent
ARM = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER = np.asarray([6, 13])


def load_target(path: Path, ready_pose_config: Path):
    metadata = None
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "metadata" and metadata is None:
            metadata = event
    if metadata is None or not isinstance(metadata.get("calibration"), dict):
        raise RuntimeError("trace has no calibrated-v3 metadata")
    calibration = artifact_from_dict(metadata["calibration"])
    if "fixed_initial_command" in metadata and "fixed_initial_feedback" in metadata:
        command = np.asarray(metadata["fixed_initial_command"], dtype=np.float32)
        feedback = np.asarray(metadata["fixed_initial_feedback"], dtype=np.float32)
    else:
        ready_arms = load_training_ready_arms(ready_pose_config)
        command, feedback = training_ready_targets(calibration, ready_arms)
    if (
        command.shape != (14,)
        or feedback.shape != (14,)
        or not np.isfinite(command).all()
        or not np.isfinite(feedback).all()
    ):
        raise RuntimeError("trace has no valid fixed 14D ready pose")
    return command, feedback, calibration, metadata


def load_calibrated_target(calibration_path: Path, ready_pose_config: Path):
    calibration = load_artifact(calibration_path)
    ready_arms = load_training_ready_arms(ready_pose_config)
    command, feedback = training_ready_targets(calibration, ready_arms)
    return command, feedback, calibration


def create_return_node(config):
    from rclpy.node import Node

    class ReturnNode(Node):
        def __init__(self):
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

    return ReturnNode()


def run(args):
    import rclpy

    hostname, domain, boot_id, controllers = current_robot_identity()
    if args.trace is not None:
        command_target, feedback_target, calibration, metadata = load_target(
            args.trace, args.ready_pose_config
        )
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
    else:
        command_target, feedback_target, calibration = load_calibrated_target(
            args.calibration_file, args.ready_pose_config
        )
        validate_artifact(
            calibration,
            hostname=hostname,
            ros_domain_id=domain,
            boot_id=boot_id,
            controller_identity=controllers,
            max_age_s=args.max_calibration_age_s,
        )
        print(f"Calibration: {args.calibration_file}")
    print(f"Fixed initial feedback target: {np.array2string(feedback_target, precision=4)}")
    if not args.execute:
        print("DRY-RUN: no publisher was created.")
        return
    confirmation_text = (
        "RETURN LAST TRACE TO FIXED INITIAL POSE"
        if args.trace is not None
        else "RETURN TO FIXED INITIAL POSE"
    )
    if input(f"Type {confirmation_text} to move: ") != confirmation_text:
        raise RuntimeError("return recovery cancelled")

    setup_loader(ROOT)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    node = create_return_node(config)
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
        current_command = feedback_pose_to_command(calibration, current)
        trajectory = return_trajectory(current_command, command_target)
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
            node.publish(command_target)
            time.sleep(period)
        samples = []
        verify_until = time.monotonic() + 2.0
        while time.monotonic() < verify_until:
            samples.append(node.current())
            time.sleep(period)
        values = np.asarray(samples)
        result = {
            "arm_error_max": float(np.max(np.abs(values[-1, ARM]-feedback_target[ARM]))),
            "gripper_error_max": float(np.max(np.abs(values[-1, GRIPPER]-feedback_target[GRIPPER]))),
            "feedback_spread_max": float(np.max(np.ptp(values, axis=0))),
            "trajectory_steps": len(trajectory),
        }
        if result["arm_error_max"] > .05 or result["gripper_error_max"] > .1 or result["feedback_spread_max"] > .01:
            raise RuntimeError(f"return recovery verification failed: {result}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(f"RETURN TO FIXED INITIAL COMPLETE: {json.dumps(result, sort_keys=True)}")
        print(f"Recovery report: {args.report}")
    finally:
        spin_stop.set()
        thread.join(timeout=2.0)
        node.destroy_node()


def parse_args():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--calibration-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-trace-age-s", type=float, default=1800.0)
    parser.add_argument("--max-calibration-age-s", type=float, default=900.0)
    parser.add_argument("--config", type=Path, default=ROOT/"data/config.yaml")
    parser.add_argument(
        "--ready-pose-config",
        type=Path,
        default=ROOT/"data/tau0vla_calibrated_ready.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/home/arx/logs/tau0vla-calibrated")/f"return_{stamp}.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import rclpy

    rclpy.init()
    try:
        run(parse_args())
    finally:
        rclpy.shutdown()
