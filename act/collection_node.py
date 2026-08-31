"""Observation-only ROS subscriptions for collection and remote inference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from dataset_v2 import EpisodeSample
from pipeline_contract import CAMERA_NAMES

MAX_MESSAGE_AGE_NS = 50_000_000
MAX_CAMERA_SKEW_NS = 20_000_000


@dataclass
class Received:
    message: object
    monotonic_ns: int


def _stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def create_collection_node(config: dict, enable_arm_publishers: bool = False):
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotStatus
    from arx_lift_controller.srv import LiftHeightStatus

    class CollectionNode(Node):
        def __init__(self):
            super().__init__("arx_pipeline_observer")
            self._lock = threading.Lock()
            self._latest: dict[str, Received] = {}
            self._last_camera_stamps: dict[str, int] = {}
            self._height_status_type = LiftHeightStatus
            cameras = {
                "head": config["camera_config"]["img_head_topic"],
                "left_wrist": config["camera_config"]["img_left_topic"],
                "right_wrist": config["camera_config"]["img_right_topic"],
            }
            for camera, topic in cameras.items():
                self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda message, key=camera: self._receive(f"camera:{key}", message),
                    10,
                )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["arm_feedback_left_topic"],
                lambda message: self._receive("arm:left", message),
                10,
            )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["arm_feedback_right_topic"],
                lambda message: self._receive("arm:right", message),
                10,
            )
            self.create_subscription(
                PosCmd,
                config["robot_base_config"]["robot_base_topic"],
                lambda message: self._receive("body", message),
                10,
            )
            self.left_publisher = None
            self.right_publisher = None
            if enable_arm_publishers:
                self.left_publisher = self.create_publisher(
                    RobotStatus, config["arm_config"]["follow_arm_left_cmd_topic"], 10
                )
                self.right_publisher = self.create_publisher(
                    RobotStatus, config["arm_config"]["follow_arm_right_cmd_topic"], 10
                )

        def _receive(self, key: str, message) -> None:
            with self._lock:
                self._latest[key] = Received(message=message, monotonic_ns=time.monotonic_ns())

        def height_status(self, timeout: float = 5.0) -> dict:
            client = self.create_client(self._height_status_type, "/lift_height_status")
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError("/lift_height_status unavailable; body must already be running")
            future = client.call_async(self._height_status_type.Request())
            deadline = time.monotonic() + timeout
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not future.done() or future.result() is None:
                raise RuntimeError("timed out reading /lift_height_status")
            response = future.result()
            return {
                "current_height": float(response.current_height),
                "commanded_height": float(response.commanded_height),
                "locked": bool(response.locked),
                "message": response.message,
            }

        def snapshot(self, commit_camera_stamps: bool = True) -> EpisodeSample:
            now_ns = time.monotonic_ns()
            required = [*(f"camera:{name}" for name in CAMERA_NAMES), "arm:left", "arm:right", "body"]
            with self._lock:
                missing = [key for key in required if key not in self._latest]
                if missing:
                    raise RuntimeError(f"missing ROS data: {', '.join(missing)}")
                received = {key: self._latest[key] for key in required}
            stale = [key for key, value in received.items() if now_ns - value.monotonic_ns > MAX_MESSAGE_AGE_NS]
            if stale:
                raise RuntimeError(f"stale ROS data: {', '.join(stale)}")

            camera_stamps = {
                camera: _stamp_ns(received[f"camera:{camera}"].message) for camera in CAMERA_NAMES
            }
            if max(camera_stamps.values()) - min(camera_stamps.values()) > MAX_CAMERA_SKEW_NS:
                raise RuntimeError("camera timestamp skew exceeds 20 ms")
            duplicates = [camera for camera, stamp in camera_stamps.items() if self._last_camera_stamps.get(camera) == stamp]
            if duplicates:
                raise RuntimeError(f"duplicate camera frames: {', '.join(duplicates)}")

            left = received["arm:left"].message
            right = received["arm:right"].message
            body = received["body"].message
            left_eef = np.concatenate((np.asarray(left.end_pos, dtype=np.float64), [left.joint_pos[-1]]))
            right_eef = np.concatenate((np.asarray(right.end_pos, dtype=np.float64), [right.joint_pos[-1]]))
            if commit_camera_stamps:
                self._last_camera_stamps = camera_stamps
            return EpisodeSample(
                qpos=np.concatenate((left.joint_pos, right.joint_pos)).astype(np.float64),
                qvel=np.concatenate((left.joint_vel, right.joint_vel)).astype(np.float64),
                effort=np.concatenate((left.joint_cur, right.joint_cur)).astype(np.float64),
                eef=np.concatenate((left_eef, right_eef)).astype(np.float64),
                images={camera: bytes(received[f"camera:{camera}"].message.data) for camera in CAMERA_NAMES},
                camera_timestamp_ns=camera_stamps,
                arm_timestamp_ns={"left": _stamp_ns(left), "right": _stamp_ns(right)},
                sample_monotonic_ns=now_ns,
                body_information=np.asarray(
                    [body.height, body.temp_float_data[0], body.head_yaw, body.head_pit], dtype=np.float64
                ),
                wheel_velocity=np.asarray(body.temp_float_data[1:5], dtype=np.float64),
            )

        def publish_action(self, action: np.ndarray) -> None:
            if self.left_publisher is None or self.right_publisher is None:
                raise RuntimeError("arm publishers are disabled")
            left = RobotStatus()
            right = RobotStatus()
            left.joint_pos[:7] = [float(value) for value in action[:7]]
            right.joint_pos[:7] = [float(value) for value in action[7:]]
            self.left_publisher.publish(left)
            self.right_publisher.publish(right)

    return CollectionNode()
