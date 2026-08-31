"""ARX HTTP inference client. Dry-run is the default and safest mode."""

from __future__ import annotations

import argparse
import collections
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset_v2 import HEIGHT_TOLERANCE, WHEEL_SPEED_TOLERANCE
from http_protocol import HttpInferenceClient
from pipeline_contract import ACTION_DIM, FPS
from utils.setup_loader import setup_loader


class SafetyGate:
    def __init__(self, limits_path: Path | None, max_joint_step: float, max_gripper_step: float):
        self.minimum = None
        self.maximum = None
        self.previous = None
        self.max_joint_step = max_joint_step
        self.max_gripper_step = max_gripper_step
        if limits_path is not None:
            with limits_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            self.minimum = np.asarray(config["minimum"], dtype=np.float64)
            self.maximum = np.asarray(config["maximum"], dtype=np.float64)
            if self.minimum.shape != (ACTION_DIM,) or self.maximum.shape != (ACTION_DIM,):
                raise ValueError("joint limits must contain 14-element minimum and maximum arrays")

    def require_for_execute(self) -> None:
        if self.minimum is None:
            raise RuntimeError("--execute requires a reviewed --joint-limits file")

    def reset(self, current_qpos: np.ndarray) -> None:
        self.previous = np.asarray(current_qpos, dtype=np.float64)

    def validate(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("invalid action vector")
        if self.minimum is not None and (np.any(action < self.minimum) or np.any(action > self.maximum)):
            raise RuntimeError("action exceeds reviewed joint limits")
        if self.previous is None:
            raise RuntimeError("safety gate has no current qpos")
        delta = np.abs(action - self.previous)
        joint_indices = [index for index in range(ACTION_DIM) if index not in (6, 13)]
        if float(np.max(delta[joint_indices])) > self.max_joint_step:
            raise RuntimeError("action joint step exceeds limit")
        if float(np.max(delta[[6, 13]])) > self.max_gripper_step:
            raise RuntimeError("action gripper step exceeds limit")
        self.previous = action.copy()


def spin_node(node):
    import rclpy

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)


def main(args) -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", "62")
    if args.execute and args.confirm_execute != "I_UNDERSTAND":
        raise RuntimeError("physical execution requires --confirm-execute I_UNDERSTAND")
    setup_loader(ROOT)
    import rclpy
    from collection_node import create_collection_node

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    gate = SafetyGate(args.joint_limits, args.max_joint_step, args.max_gripper_step)
    if args.execute:
        gate.require_for_execute()

    rclpy.init()
    node = create_collection_node(config, enable_arm_publishers=args.execute)
    spin_thread = threading.Thread(target=spin_node, args=(node,), daemon=True)
    spin_thread.start()
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    try:
        height = node.height_status()
        if not height["locked"]:
            raise RuntimeError("height must be locked before inference")
        expected_height = height["commanded_height"] if args.expected_height is None else args.expected_height
        if abs(height["current_height"] - expected_height) > args.height_tolerance:
            raise RuntimeError("current height does not match expected height")
        print(f"Height preflight PASS: {height}")

        client = HttpInferenceClient(args.server_url, args.max_response_age_ms)
        print(f"Server health: {client.health()}")
        print(f"Server schema: {client.schema()}")
        session_id = uuid.uuid4().hex
        client.reset(session_id, args.task_instruction)

        executor = ThreadPoolExecutor(max_workers=1)
        pending = None
        action_buffer = collections.deque()
        request_id = 0
        baseline_height = expected_height
        period = 1.0 / FPS
        deadline = time.monotonic()
        gate_initialized = False
        received_actions = False
        print("Inference client running in EXECUTE mode" if args.execute else "Inference client running in DRY-RUN mode")
        while rclpy.ok() and not stop_event.is_set():
            now = time.monotonic()
            if now < deadline:
                time.sleep(min(deadline - now, 0.005))
                continue
            deadline += period
            try:
                sample = node.snapshot()
            except RuntimeError as error:
                if args.execute:
                    raise RuntimeError(f"disarmed: {error}") from error
                print(f"Skipped observation: {error}")
                continue
            if abs(float(sample.body_information[0]) - baseline_height) > HEIGHT_TOLERANCE or float(
                np.max(np.abs(sample.wheel_velocity))
            ) > WHEEL_SPEED_TOLERANCE:
                raise RuntimeError("disarmed: body height or wheel velocity changed")
            if not gate_initialized:
                gate.reset(sample.qpos)
                gate_initialized = True

            if pending is not None and pending.done():
                result = pending.result()
                action_buffer.extend(result.actions)
                received_actions = True
                print(
                    f"Response {result.request_id}: H={len(result.actions)}, "
                    f"RTT={result.round_trip_ms:.1f} ms, model={result.model_id}"
                )
                pending = None
            if pending is None:
                request_id += 1
                pending = executor.submit(client.infer, sample, args.task_instruction, session_id, request_id)

            if args.execute:
                if not action_buffer:
                    if received_actions:
                        raise RuntimeError("disarmed: action buffer exhausted")
                    continue
                action = np.asarray(action_buffer.popleft(), dtype=np.float64)
                gate.validate(action)
                node.publish_action(action)
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://192.168.31.83:8000")
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config.yaml")
    parser.add_argument("--expected-height", type=float, default=None)
    parser.add_argument("--height-tolerance", type=float, default=0.05)
    parser.add_argument("--max-response-age-ms", type=float, default=500.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-execute", default="")
    parser.add_argument("--joint-limits", type=Path, default=None)
    parser.add_argument("--max-joint-step", type=float, default=0.10)
    parser.add_argument("--max-gripper-step", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
