"""Guarded full-range gripper calibration for one calibrated-v3 rollout."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import signal
import threading
import time
import uuid

import numpy as np
import yaml

from tau0vla_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_VERSION,
    COMMAND_POINTS,
    CalibrationArtifact,
    CalibrationError,
    current_robot_identity,
    fit_side,
    save_artifact,
)
from utils.setup_loader import setup_loader


ROOT = Path(__file__).resolve().parent
GRIPPER = 6


def create_node(config: dict):
    from rclpy.node import Node

    class CalibrationNode(Node):
        def __init__(self):
            super().__init__("tau0vla_calibrated_gripper")
            from arx5_arm_msg.msg import RobotStatus

            self._message_type = RobotStatus
            self._lock = threading.Lock()
            self._feedback = {"left": deque(maxlen=4000), "right": deque(maxlen=4000)}
            # Do not shadow rclpy.Node._publishers; destroy_node() owns that
            # internal list and expects integer indexing.
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
                50,
            )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["follow_arm_right_feedback_topic"],
                lambda message: self._receive("right", message),
                50,
            )

        def _receive(self, side: str, message) -> None:
            values = np.asarray(message.joint_pos, dtype=np.float64)
            if values.shape == (7,) and np.isfinite(values).all():
                with self._lock:
                    self._feedback[side].append((time.monotonic(), values.copy()))

        def current(self) -> dict[str, np.ndarray]:
            with self._lock:
                if any(not self._feedback[side] for side in ("left", "right")):
                    raise CalibrationError("both arm feedback streams are required")
                return {side: self._feedback[side][-1][1].copy() for side in ("left", "right")}

        def sample_grippers(self, seconds: float, rate_hz: float) -> np.ndarray:
            rows = []
            deadline = time.monotonic() + seconds
            period = 1.0 / rate_hz
            while time.monotonic() < deadline:
                current = self.current()
                rows.append([current["left"][GRIPPER], current["right"][GRIPPER]])
                time.sleep(period)
            return np.asarray(rows, dtype=np.float64)

        def publish(self, targets: dict[str, np.ndarray]) -> None:
            for side in ("left", "right"):
                message = self._message_type()
                message.joint_pos[:7] = [float(value) for value in targets[side]]
                self._command_publishers[side].publish(message)

    return CalibrationNode()


def wait_feedback(node, timeout: float = 5.0) -> dict[str, np.ndarray]:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            return node.current()
        except CalibrationError as error:
            last = error
            time.sleep(0.05)
    raise CalibrationError(f"arm feedback unavailable: {last}")


def ramp(node, targets, side: str, target: float, *, step: float, rate_hz: float, stop):
    start = float(targets[side][GRIPPER])
    count = max(1, int(np.ceil(abs(target - start) / step)))
    period = 1.0 / rate_hz
    for index in range(1, count + 1):
        if stop.is_set():
            raise CalibrationError("calibration interrupted")
        targets[side][GRIPPER] = start + (target - start) * index / count
        node.publish(targets)
        time.sleep(period)


def hold(node, targets, seconds: float, *, rate_hz: float, stop) -> None:
    """Keep publishing the reached target while hardware settles."""
    deadline = time.monotonic() + seconds
    period = 1.0 / rate_hz
    while time.monotonic() < deadline:
        if stop.is_set():
            raise CalibrationError("calibration interrupted")
        node.publish(targets)
        time.sleep(period)


def spin_node(node, stop) -> None:
    import rclpy

    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=.01)


def run(args) -> Path:
    if not args.execute:
        print("DRY-RUN: no publisher is created; pass --execute for the guarded calibration.")
        print(f"Would command each gripper through {COMMAND_POINTS.tolist()} and return open.")
        return args.output
    print("This moves one gripper at a time while holding all arm joints at current feedback.")
    print(f"Command points: {COMMAND_POINTS.tolist()}, then return to {COMMAND_POINTS[0]:.2f}.")
    if args.auto_confirm:
        print("AUTO-CONFIRM: calibrating both grippers.")
    elif input("Type CALIBRATE BOTH GRIPPERS to continue: ") != "CALIBRATE BOTH GRIPPERS":
        raise CalibrationError("calibration cancelled; no publisher was created")

    import rclpy

    setup_loader(ROOT)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    hostname, ros_domain_id, boot_id, controllers = current_robot_identity()
    rclpy.init()
    node = create_node(config)
    stop = threading.Event()
    spin = threading.Thread(target=spin_node, args=(node, stop), daemon=True)
    spin.start()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        targets = wait_feedback(node)
        feedback_windows: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for side in ("left", "right"):
            print(f"Calibrating {side} gripper; the other side remains held.")
            for command in COMMAND_POINTS:
                ramp(
                    node,
                    targets,
                    side,
                    float(command),
                    step=args.step,
                    rate_hz=args.rate_hz,
                    stop=stop,
                )
                hold(
                    node,
                    targets,
                    args.pre_settle_s,
                    rate_hz=args.rate_hz,
                    stop=stop,
                )
                window = node.sample_grippers(args.settle_s, args.rate_hz)
                side_index = 0 if side == "left" else 1
                feedback_windows[side].append(window[:, side_index])
                print(
                    f"  command={command:+.3f}, feedback={window[:, side_index].mean():+.4f}, "
                    f"p90-p10={np.percentile(window[:, side_index], 90)-np.percentile(window[:, side_index], 10):.5f}"
                )
            ramp(
                node,
                targets,
                side,
                float(COMMAND_POINTS[0]),
                step=args.step,
                rate_hz=args.rate_hz,
                stop=stop,
            )
        hold(
            node,
            targets,
            args.pre_settle_s,
            rate_hz=args.rate_hz,
            stop=stop,
        )
        final_open = node.sample_grippers(args.open_settle_s, args.rate_hz)
        left = fit_side(COMMAND_POINTS, feedback_windows["left"], final_open[:, 0])
        right = fit_side(COMMAND_POINTS, feedback_windows["right"], final_open[:, 1])
        artifact = CalibrationArtifact(
            schema_version=CALIBRATION_SCHEMA_VERSION,
            calibration_version=CALIBRATION_VERSION,
            calibration_id=uuid.uuid4().hex,
            hostname=hostname,
            ros_domain_id=ros_domain_id,
            boot_id=boot_id,
            created_unix_s=time.time(),
            created_monotonic_ns=time.monotonic_ns(),
            controller_identity=controllers,
            left=left,
            right=right,
        )
        save_artifact(args.output, artifact)
        print(f"Calibration saved: {args.output}")
        return args.output
    finally:
        stop.set()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


def parse_args():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--pre-settle-s", type=float, default=.5)
    parser.add_argument("--open-settle-s", type=float, default=2.0)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/arx/logs/tau0vla-calibrated") / f"calibration_{timestamp}.json",
    )
    args = parser.parse_args()
    if args.step <= 0 or args.step > 0.05:
        parser.error("--step must be in (0, 0.05]")
    if args.rate_hz != 60.0:
        parser.error("--rate-hz is fixed at 60 for reviewed calibration")
    if args.settle_s < 1.5 or args.open_settle_s < 2.0:
        parser.error("settle windows may not be shorter than the reviewed defaults")
    if args.pre_settle_s < .5:
        parser.error("--pre-settle-s may not be shorter than 0.5")
    return args


if __name__ == "__main__":
    run(parse_args())
