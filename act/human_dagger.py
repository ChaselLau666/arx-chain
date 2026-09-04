#!/usr/bin/env python3
"""Human-in-the-loop DAgger rollout and collection entry point.

Only the control child owns the arm command publishers.  The foreground process
owns the keyboard, while policy inference and HDF5 I/O live in separate workers.
This separation is the safety boundary that lets Space gate policy output even
when a CUDA forward pass or a disk flush is blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

FILE = Path(__file__).resolve()
ACT_ROOT = FILE.parent
REPO_ROOT = ACT_ROOT.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from collection_ui import TerminalKeyReader  # noqa: E402
from human_dagger_policy import (  # noqa: E402
    PolicyWorkerConfig,
    mock_policy_worker_main,
    policy_worker_main,
)


def monotonic_ns() -> int:
    return time.monotonic_ns()


def _retry_review_reset_after_close(
    core: Any,
    pending: bool,
    now_ns: int,
) -> tuple[bool, bool]:
    """Retry a safe REVIEW->MANUAL_RESET transition after recorder close.

    The recorder acknowledgement and fresh VR/arm samples are asynchronous.  A
    stale sample at the instant the acknowledgement arrives must keep the latch
    pending rather than permanently stranding the coordinator in REVIEW_HOLD.
    The return value is ``(still_pending, reset_completed)``.
    """

    if not pending:
        return False, False
    completed = bool(core.reset_after_review(now_ns))
    return not completed, completed


def _feedback_pair_acknowledges_hold(
    samples: Mapping[str, Any],
    hold_published_ns: int,
    feedback_pair_valid: bool,
) -> bool:
    """Accept external HOLD only from a valid post-publish feedback pair."""

    feedback = samples.get("feedback", {})
    return bool(
        feedback_pair_valid
        and hold_published_ns > 0
        and set(feedback) == {"left", "right"}
        and all(
            feedback[side][0].timestamp_ns >= hold_published_ns
            for side in ("left", "right")
        )
    )


def _single_valid_feedback_for_hold(
    samples: Mapping[str, Any],
    now_ns: int,
    feedback_timeout_ns: int,
    future_tolerance_ns: int,
) -> tuple[str, Any] | None:
    """Return the sole fresh/finite arm sample available for a partial HOLD."""

    valid: list[tuple[str, Any]] = []
    for side in ("left", "right"):
        entry = samples.get("feedback", {}).get(side)
        if entry is None:
            continue
        sample = entry[0]
        try:
            joint_pos = tuple(float(value) for value in sample.joint_pos)
            eef_pose = tuple(float(value) for value in sample.eef_pose)
            gripper = float(sample.gripper)
            timestamp_ns = sample.timestamp_ns
        except (AttributeError, TypeError, ValueError):
            continue
        if len(joint_pos) != 6 or len(eef_pose) != 6:
            continue
        if not all(math.isfinite(value) for value in (*joint_pos, *eef_pose, gripper)):
            continue
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            continue
        age_ns = now_ns - timestamp_ns
        if not (-future_tolerance_ns <= age_ns <= feedback_timeout_ns):
            continue
        valid.append((side, sample))
    return valid[0] if len(valid) == 1 else None


def _timeline_event_records(
    events: Any,
    handoff_pending: dict[str, Any] | None,
    frame: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Expand core events into lossless raw rows plus completed handoffs.

    Raw rows are emitted even when an episode ends midway through a handoff.  A
    completed activation additionally emits the aggregate HANDOFF row consumed
    by the schema-v2 validator.
    """

    records: list[dict[str, Any]] = []
    ack_event_names = {
        "HOLD_ACK",
        "POLICY_RESET_ACK",
        "HUMAN_ACTIVE",
        "POLICY_ACTIVE",
    }
    for event in events:
        event_name = getattr(event.name, "value", str(event.name))
        aggregate: dict[str, Any] | None = None

        if event_name in {"EPISODE_START_REQUEST", "POLICY_RESUME_REQUEST"}:
            handoff_pending = {
                "event": "HANDOFF_TO_POLICY",
                "request_ns": event.timestamp_ns,
                "gate_ns": None,
                "ack_ns": None,
                "frame": frame,
                "epoch": event.control_epoch,
                "detail": event.detail,
            }
        elif event_name == "TAKEOVER_REQUEST":
            handoff_pending = {
                "event": "HANDOFF_TO_HUMAN",
                "request_ns": event.timestamp_ns,
                "gate_ns": None,
                "ack_ns": None,
                "frame": frame,
                "epoch": event.control_epoch,
                "detail": event.detail,
            }
        elif event_name == "CONTROL_GATE" and handoff_pending is not None:
            handoff_pending["gate_ns"] = event.timestamp_ns
            handoff_pending["epoch"] = event.control_epoch

        activation = {
            "HUMAN_ACTIVE": "HANDOFF_TO_HUMAN",
            "POLICY_ACTIVE": "HANDOFF_TO_POLICY",
        }.get(event_name)
        if (
            activation is not None
            and handoff_pending is not None
            and handoff_pending["event"] == activation
        ):
            handoff_pending["ack_ns"] = event.timestamp_ns
            handoff_pending["frame"] = frame
            handoff_pending["epoch"] = event.control_epoch
            aggregate = dict(handoff_pending)
            handoff_pending = None

        raw_record = {
            "event": event_name,
            "request_ns": None,
            "gate_ns": None,
            "ack_ns": None,
            "frame": frame,
            # Request events are emitted before the core invalidates the old
            # source epoch, while their frame belongs to the ensuing handoff.
            # Keep that intentional cross-boundary row unbound from a frame
            # epoch; the aggregate HANDOFF row is bound to the active epoch.
            "epoch": (
                -1
                if event_name
                in {
                    "EPISODE_START_REQUEST",
                    "TAKEOVER_REQUEST",
                    "POLICY_RESUME_REQUEST",
                    "EPISODE_END_REQUEST",
                }
                else event.control_epoch
            ),
            "detail": event.detail,
        }
        if event_name == "CONTROL_GATE":
            raw_record["gate_ns"] = event.timestamp_ns
        elif event_name in ack_event_names:
            raw_record["ack_ns"] = event.timestamp_ns
        else:
            # Requests naturally use this field; every remaining event keeps at
            # least one timestamp instead of silently losing temporal context.
            raw_record["request_ns"] = event.timestamp_ns
        records.append(raw_record)
        if aggregate is not None:
            records.append(aggregate)

    return handoff_pending, records


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def next_episode_index(dataset_dir: str | os.PathLike[str], requested: int) -> int:
    if requested >= 0:
        return requested
    directory = Path(dataset_dir)
    found = []
    if directory.exists():
        for path in directory.glob("episode_*.hdf5"):
            try:
                found.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(found, default=-1) + 1


def _linux_proc_start_ticks(pid: int) -> str:
    """Return Linux /proc starttime so shutdown can reject PID reuse."""

    stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    _comm, separator, remainder = stat.rpartition(") ")
    if not separator:
        raise RuntimeError(f"cannot parse /proc/{pid}/stat")
    fields_from_three = remainder.split()
    if len(fields_from_three) < 20:
        raise RuntimeError(f"incomplete /proc/{pid}/stat")
    return fields_from_three[19]


def _append_session_process(manifest: str, label: str, pid: int) -> None:
    """Atomically append a worker identity to the shell-owned session file."""

    if not manifest:
        return
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"session manifest is unavailable: {path}")
    start_ticks = _linux_proc_start_ticks(pid)
    payload = f"{label}\t{int(pid)}\t{start_ticks}\n".encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short manifest write: {written}/{len(payload)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git_metadata() -> dict[str, str]:
    """Best-effort provenance without making collection depend on git."""

    import subprocess

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "git_dirty": str(bool(run("status", "--porcelain") not in ("", "unknown"))).lower(),
    }


def recorder_worker_main(command_queue: Any, status_queue: Any) -> None:
    from human_dagger_recorder import HumanDaggerRecorder

    recorder = None
    try:
        while True:
            command = command_queue.get()
            kind = command.get("kind")
            try:
                if kind == "stop":
                    if recorder is not None:
                        try:
                            recorder.quarantine("recorder stopped with an active episode")
                        except FileNotFoundError:
                            # finalize() may already have quarantined a rejected file.
                            pass
                        recorder = None
                    return
                if kind == "start":
                    if recorder is not None:
                        recorder.quarantine("new episode started before previous episode closed")
                    recorder = HumanDaggerRecorder(
                        output_dir=command["output_dir"],
                        episode_name=command["episode_name"],
                        camera_names=tuple(command["camera_names"]),
                        metadata=command["metadata"],
                        flush_every=int(command.get("flush_every", 30)),
                    )
                    status_queue.put(
                        {"kind": "recorder_started", "episode": command["episode_name"]}
                    )
                elif kind == "frame":
                    if recorder is None:
                        raise RuntimeError("frame received without an active recorder")
                    recorder.append_frame(**command["frame"])
                elif kind == "event":
                    if recorder is not None:
                        recorder.record_event(**command["event"])
                elif kind == "finalize":
                    if recorder is None:
                        raise RuntimeError("finalize received without an active recorder")
                    try:
                        path = recorder.finalize()
                    except Exception as exc:
                        quarantine_path = getattr(exc, "quarantine_path", None)
                        if quarantine_path is not None:
                            recorder = None
                            status_queue.put(
                                {"kind": "recorder_quarantined", "path": str(quarantine_path)}
                            )
                        raise
                    else:
                        recorder = None
                        status_queue.put({"kind": "recorder_finalized", "path": str(path)})
                elif kind == "discard":
                    if recorder is not None:
                        recorder.discard()
                        recorder = None
                    status_queue.put({"kind": "recorder_discarded"})
                elif kind == "quarantine":
                    if recorder is not None:
                        path = recorder.quarantine(command.get("reason", "control fault"))
                        recorder = None
                        status_queue.put({"kind": "recorder_quarantined", "path": str(path)})
            except Exception as exc:
                # A failed mutation makes the episode untrustworthy even when
                # append_frame() managed to roll its datasets back.  Poison the
                # worker-side recorder before touching the status queue so an
                # already queued finalize command can never publish this file.
                # Validation failures from finalize() already set recorder=None
                # and report their quarantine path in the branch above.
                if recorder is not None:
                    failed_recorder = recorder
                    recorder = None
                    try:
                        quarantine_path = failed_recorder.quarantine(
                            f"recorder {kind or 'unknown'} mutation failed: {exc!r}"
                        )
                    except Exception as quarantine_exc:
                        status_queue.put(
                            {
                                "kind": "recorder_error",
                                "error": (
                                    f"failed to quarantine recorder after {kind or 'unknown'} "
                                    f"mutation error {exc!r}: {quarantine_exc!r}"
                                ),
                            }
                        )
                    else:
                        status_queue.put(
                            {"kind": "recorder_quarantined", "path": str(quarantine_path)}
                        )
                status_queue.put({"kind": "recorder_error", "error": repr(exc)})
    except KeyboardInterrupt:
        # Safe-shutdown sends SIGINT first so the finally block can atomically
        # quarantine an open partial instead of leaving it in the dataset root.
        pass
    finally:
        if recorder is not None:
            try:
                recorder.quarantine("recorder worker terminated unexpectedly")
            except Exception:
                # Preserve the original failure; a stale partial is detected by
                # the next startup preflight and never enters the formal set.
                pass


def control_process_main(
    runtime_args: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    ui_command_queue: Any,
    ui_status_queue: Any,
    policy_control_queue: Any,
    policy_observation_queue: Any,
    policy_result_queue: Any,
    policy_status_queue: Any,
    recorder_command_queue: Any,
    recorder_status_queue: Any,
) -> None:
    """ROS/control process body. Heavy ROS imports stay out of the UI process."""

    # Implemented below the pure helpers to keep module importable in non-ROS tests.
    _run_ros_control(
        runtime_args,
        runtime_config,
        ui_command_queue,
        ui_status_queue,
        policy_control_queue,
        policy_observation_queue,
        policy_result_queue,
        policy_status_queue,
        recorder_command_queue,
        recorder_status_queue,
    )


def _run_ros_control(
    runtime_args: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    ui_command_queue: Any,
    ui_status_queue: Any,
    policy_control_queue: Any,
    policy_observation_queue: Any,
    policy_result_queue: Any,
    policy_status_queue: Any,
    recorder_command_queue: Any,
    recorder_status_queue: Any,
) -> None:
    from human_dagger_core import (
        ArmFeedback,
        CommandMode,
        CommandSource,
        ControlState,
        HumanDaggerConfig,
        HumanDaggerCore,
        PolicyActionPacket,
        VrPose,
    )
    from utils.setup_loader import setup_loader

    setup_loader(ACT_ROOT)

    import rclpy
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotCmd, RobotStatus
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.experimental.events_executor import EventsExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    args = SimpleRuntimeArgs(runtime_args)
    topics = runtime_config.get("topics", {})
    ros_config = runtime_config.get("ros", {})
    camera_config = runtime_config.get("cameras", {})
    control_config = runtime_config.get("control", {})
    episode_config = runtime_config.get("episode", {})
    timeouts = runtime_config.get("timeouts_ms", runtime_config.get("timeouts", {}))
    precheck_cfg = runtime_config.get("precheck", {})

    def topic(name: str, default: str) -> str:
        return str(topics.get(name, default))

    def timeout_ns(name: str, default_ms: float) -> int:
        aliases = {
            "feedback": "feedback_timeout_ms",
            "vr": "vr_timeout_ms",
            "image": "camera_timeout_ms",
            "body": "body_timeout_ms",
            "policy": "policy_timeout_ms",
            "ui_heartbeat": "ui_heartbeat_timeout_ms",
        }
        value = float(timeouts.get(name, control_config.get(aliases.get(name, ""), default_ms)))
        return int(value * 1_000_000)

    feedback_timeout_ns = timeout_ns("feedback", 100.0)
    vr_timeout_ns = timeout_ns("vr", 100.0)
    image_timeout_ns = timeout_ns("image", 100.0)
    body_timeout_ns = timeout_ns("body", 500.0)
    policy_timeout_ns = timeout_ns("policy", 250.0)
    ui_timeout_ns = timeout_ns("ui_heartbeat", 500.0)
    handoff_timeout_ns = int(
        float(control_config.get("handoff_timeout_s", 2.0)) * 1e9
        if "handoff" not in timeouts
        else timeout_ns("handoff", 2000.0)
    )
    policy_slew_ns = int(
        float(control_config.get("policy_slew_duration_s", 2.0)) * 1e9
        if "policy_slew" not in timeouts
        else timeout_ns("policy_slew", 2000.0)
    )
    policy_slew_steps = tuple(
        float(value)
        for value in control_config.get(
            "policy_slew_step_per_arm",
            (0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.2),
        )
    )
    future_timestamp_tolerance_ns = int(
        float(control_config.get("future_timestamp_tolerance_ms", 5.0)) * 1_000_000
    )
    gripper_delta_scale = float(control_config.get("gripper_delta_scale", -3.4 / 5.0))
    human_filter_min_cutoff_hz = float(
        control_config.get("human_filter_min_cutoff_hz", 1.0)
    )
    human_filter_beta = float(control_config.get("human_filter_beta", 0.15))
    human_filter_d_cutoff_hz = float(
        control_config.get("human_filter_d_cutoff_hz", 1.0)
    )
    if int(control_config.get("joint_mode", 5)) != 5:
        raise ValueError("Human DAgger HOLD/POLICY requires X5 POSITION_CONTROL mode 5")
    if int(control_config.get("eef_mode", 4)) != 4:
        raise ValueError("Human DAgger HUMAN control requires X5 END_CONTROL mode 4")
    precheck_timeout_ns = int(float(precheck_cfg.get("timeout_seconds", 120.0)) * 1e9)
    stable_height_ns = int(float(precheck_cfg.get("height_stable_seconds", 1.0)) * 1e9)
    height_tolerance = float(precheck_cfg.get("height_tolerance", 0.1))

    configured_camera_topics = camera_config.get("topics", {})
    camera_topics = {
        "head": topic(
            "camera_head",
            configured_camera_topics.get("camera_h", "/camera/camera_h/color/image_rect_raw/compressed"),
        ),
        "left_wrist": topic(
            "camera_left_wrist",
            configured_camera_topics.get("camera_l", "/camera/camera_l/color/image_rect_raw/compressed"),
        ),
        "right_wrist": topic(
            "camera_right_wrist",
            configured_camera_topics.get("camera_r", "/camera/camera_r/color/image_rect_raw/compressed"),
        ),
    }
    vr_topics = {
        "left": topic("vr_left_raw", ros_config.get("left", {}).get("vr_topic", "/human_dagger/vr/left_raw")),
        "right": topic("vr_right_raw", ros_config.get("right", {}).get("vr_topic", "/human_dagger/vr/right_raw")),
    }
    status_topics = {
        "left": topic("arm_left_status", ros_config.get("left", {}).get("status_topic", "/human_dagger/arm/left/status")),
        "right": topic("arm_right_status", ros_config.get("right", {}).get("status_topic", "/human_dagger/arm/right/status")),
    }
    command_topics = {
        "left": topic("arm_left_command", ros_config.get("left", {}).get("command_topic", "/human_dagger/arm/left/command")),
        "right": topic("arm_right_command", ros_config.get("right", {}).get("command_topic", "/human_dagger/arm/right/command")),
    }
    body_status_topic = topic(
        "body_status",
        ros_config.get("body_feedback_topic", "/body_information"),
    )
    isolated_input_topics = (
        "/human_dagger/body/control",
        "/human_dagger/isolated/body_vr_disabled",
        "/human_dagger/isolated/body_joy_disabled",
        "/human_dagger/isolated/arm/left/joy_disabled",
        "/human_dagger/isolated/arm/right/joy_disabled",
    )

    class HumanDaggerRosNode(Node):
        def __init__(self) -> None:
            super().__init__("human_dagger_control")
            self.sample_lock = threading.Lock()
            self.cameras: dict[str, tuple[bytes, int]] = {}
            self.feedback: dict[str, tuple[ArmFeedback, tuple[float, ...], tuple[float, ...]]] = {}
            self.vr: dict[str, VrPose] = {}
            # Read-only latency diagnostics: per-stream inter-message gap and
            # count, drained every few seconds by the control loop's [diag] line.
            self.diag_last_ns: dict[str, int] = {}
            self.diag_max_gap_ns: dict[str, int] = {}
            self.diag_count: dict[str, int] = {}
            self.body: tuple[float, int] | None = None
            self.external_hold_requested = threading.Event()
            self.external_hold_ack = threading.Event()
            self.external_hold_request_ns = 0
            self.external_hold_published_ns = 0
            self.io_callback_group = ReentrantCallbackGroup()
            self.service_callback_group = MutuallyExclusiveCallbackGroup()

            for name, topic_name in camera_topics.items():
                self.create_subscription(
                    CompressedImage,
                    topic_name,
                    lambda message, camera=name: self._camera_callback(camera, message),
                    qos_profile_sensor_data,
                    callback_group=self.io_callback_group,
                )
            for side, topic_name in vr_topics.items():
                self.create_subscription(
                    PosCmd,
                    topic_name,
                    lambda message, arm=side: self._vr_callback(arm, message),
                    qos_profile_sensor_data,
                    callback_group=self.io_callback_group,
                )
            for side, topic_name in status_topics.items():
                self.create_subscription(
                    RobotStatus,
                    topic_name,
                    lambda message, arm=side: self._feedback_callback(arm, message),
                    qos_profile_sensor_data,
                    callback_group=self.io_callback_group,
                )
            self.create_subscription(
                PosCmd,
                body_status_topic,
                self._body_callback,
                qos_profile_sensor_data,
                callback_group=self.io_callback_group,
            )
            self.command_publishers = {
                side: self.create_publisher(RobotCmd, topic_name, 10)
                for side, topic_name in command_topics.items()
            }
            state_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.state_publisher = self.create_publisher(
                String,
                topic("state", ros_config.get("state_topic", "/human_dagger/state")),
                state_qos,
            )
            self.create_service(
                Trigger,
                topic("request_hold_service", ros_config.get("hold_service", "/human_dagger/request_hold")),
                self._request_hold,
                callback_group=self.service_callback_group,
            )

        def _camera_callback(self, camera: str, message: Any) -> None:
            payload = bytes(message.data)
            now = monotonic_ns()
            with self.sample_lock:
                self.cameras[camera] = (payload, now)

        def _feedback_callback(self, side: str, message: Any) -> None:
            now = monotonic_ns()
            joint_pos = tuple(float(value) for value in message.joint_pos)
            qvel = tuple(float(value) for value in message.joint_vel)
            effort = tuple(float(value) for value in message.joint_cur)
            sample = ArmFeedback(
                joint_pos=joint_pos[:6],
                eef_pose=tuple(float(value) for value in message.end_pos),
                gripper=joint_pos[6],
                timestamp_ns=now,
            )
            with self.sample_lock:
                self.feedback[side] = (sample, qvel, effort)
                self._diag_note_locked(f"fb_{side}", now)

        def _vr_callback(self, side: str, message: Any) -> None:
            sample = VrPose(
                eef_pose=(
                    message.x,
                    message.y,
                    message.z,
                    message.roll,
                    message.pitch,
                    message.yaw,
                ),
                gripper=message.gripper,
                timestamp_ns=monotonic_ns(),
            )
            with self.sample_lock:
                self.vr[side] = sample
                self._diag_note_locked(f"vr_{side}", sample.timestamp_ns)

        def _body_callback(self, message: Any) -> None:
            with self.sample_lock:
                self.body = (float(message.height), monotonic_ns())

        def _request_hold(self, _request: Any, response: Any) -> Any:
            self.external_hold_ack.clear()
            self.external_hold_request_ns = monotonic_ns()
            self.external_hold_published_ns = 0
            self.external_hold_requested.set()
            acknowledged = self.external_hold_ack.wait(timeout=1.5)
            response.success = acknowledged
            response.message = (
                "both arm commands gated and HOLD published"
                if acknowledged
                else "timed out waiting for Human DAgger HOLD acknowledgement"
            )
            return response

        def _diag_note_locked(self, stream: str, now: int) -> None:
            # Caller holds sample_lock.
            last = self.diag_last_ns.get(stream)
            if last is not None and now - last > self.diag_max_gap_ns.get(stream, 0):
                self.diag_max_gap_ns[stream] = now - last
            self.diag_last_ns[stream] = now
            self.diag_count[stream] = self.diag_count.get(stream, 0) + 1

        def drain_diagnostics(self) -> dict[str, tuple[int, int]]:
            """Max inter-message gap and count per stream since the last drain."""
            with self.sample_lock:
                drained = {
                    stream: (self.diag_max_gap_ns.get(stream, 0), self.diag_count.get(stream, 0))
                    for stream in self.diag_last_ns
                }
                self.diag_max_gap_ns.clear()
                self.diag_count.clear()
                return drained

        def snapshot(self) -> dict[str, Any]:
            with self.sample_lock:
                return {
                    "cameras": dict(self.cameras),
                    "feedback": dict(self.feedback),
                    "vr": dict(self.vr),
                    "body": self.body,
                }

        def publish_command(self, command: Any) -> None:
            for side in ("left", "right"):
                arm = getattr(command, side)
                message = RobotCmd()
                message.header.stamp = self.get_clock().now().to_msg()
                message.end_pos = list(arm.end_pos)
                message.joint_pos = list(arm.joint_pos)
                message.gripper = float(arm.gripper)
                message.mode = int(arm.mode)
                self.command_publishers[side].publish(message)

        def publish_measured_hold(self, side: str, feedback: Any) -> None:
            """Publish a mode-5 HOLD for one arm before a complete pair exists."""

            message = RobotCmd()
            message.header.stamp = self.get_clock().now().to_msg()
            message.end_pos = list(feedback.eef_pose)
            message.joint_pos = list(feedback.joint_pos)
            message.gripper = float(feedback.gripper)
            message.mode = int(CommandMode.POSITION_CONTROL)
            self.command_publishers[side].publish(message)

        def publish_state(self, state: ControlState) -> None:
            message = String()
            message.data = state.value
            self.state_publisher.publish(message)

    core = HumanDaggerCore(
        HumanDaggerConfig(
            feedback_timeout_ns=feedback_timeout_ns,
            vr_timeout_ns=vr_timeout_ns,
            policy_timeout_ns=policy_timeout_ns,
            handoff_timeout_ns=handoff_timeout_ns,
            policy_slew_duration_ns=policy_slew_ns,
            policy_slew_step_per_arm=policy_slew_steps,
            future_timestamp_tolerance_ns=future_timestamp_tolerance_ns,
            gripper_delta_scale=gripper_delta_scale,
            human_filter_min_cutoff_hz=human_filter_min_cutoff_hz,
            human_filter_beta=human_filter_beta,
            human_filter_d_cutoff_hz=human_filter_d_cutoff_hz,
        )
    )
    rclpy.init(args=[])
    node = HumanDaggerRosNode()
    # DAGGER_EXECUTOR=classic restores the MultiThreadedExecutor if the
    # experimental one ever misbehaves; everything else is unchanged.
    if os.environ.get('DAGGER_EXECUTOR', 'events') == 'classic':
        executor = MultiThreadedExecutor(num_threads=4)
    else:
        executor = EventsExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="human-dagger-ros-spin", daemon=True)
    spin_thread.start()

    frame_period = 1.0 / float(args.frame_rate)
    start_ns = monotonic_ns()
    # Latency diagnostics: report tick health and per-stream gaps every 5 s.
    diag_interval_ns = 5_000_000_000
    last_diag_report_ns = start_ns
    tick_count = 0
    tick_overruns = 0
    last_ui_heartbeat_ns = start_ns
    height_stable_since_ns: int | None = None
    precheck_submitted = False
    policy_ready = False
    reset_request_epoch: int | None = None
    policy_observation_ready_epoch: int | None = None
    previous_state = core.state
    last_state_revision = -1
    observation_seq = 0
    episode_index = next_episode_index(args.datasets, int(args.episode_idx))
    recording_open = False
    recorded_frames = 0
    close_pending: str | None = None
    review_reset_pending = False
    quit_after_close = False
    fault_quarantine_sent = False
    shutdown_requested = False
    external_hold_seen = False
    last_precheck_report_ns = 0
    handoff_pending: dict[str, Any] | None = None
    last_graph_check_ns = 0
    cached_graph_error: str | None = "waiting for ROS graph discovery"

    checkpoint_path = Path(args.ckpt_dir) / args.ckpt_name
    checkpoint_sha = ("0" * 64) if args.mock_policy else sha256_file(checkpoint_path)

    def send_ui(kind: str, **values: Any) -> None:
        ui_status_queue.put({"kind": kind, **values})

    def recorder_send(message: dict[str, Any]) -> bool:
        try:
            recorder_command_queue.put_nowait(message)
            return True
        except queue.Full:
            core.request_fault("recorder queue is full", monotonic_ns())
            return False

    def policy_control_send(message: dict[str, Any]) -> bool:
        try:
            policy_control_queue.put_nowait(message)
            return True
        except (OSError, ValueError, queue.Full) as exc:
            core.request_fault(f"policy control queue failure: {exc}", monotonic_ns())
            return False

    def policy_observation_send(message: dict[str, Any]) -> bool:
        try:
            policy_observation_queue.put_nowait(message)
            return True
        except queue.Full:
            # Keep memory bounded while CUDA is busy. The worker rejects an old
            # queued observation by timestamp before starting another forward.
            return False
        except (OSError, ValueError) as exc:
            core.request_fault(f"policy observation queue failure: {exc}", monotonic_ns())
            return False

    def start_recording(now_ns: int) -> None:
        nonlocal recording_open, recorded_frames
        metadata: dict[str, Any] = {
            "schema_version": 2,
            "collection_mode": "human_dagger",
            "action_semantics": "current_measured_qpos",
            "training_action_offset_frames": 1,
            "task": args.task,
            "height_command": float(args.height),
            "nominal_fps": float(args.frame_rate),
            "dagger_round": int(args.dagger_round),
            "policy_checkpoint": str(checkpoint_path),
            "policy_checkpoint_sha256": checkpoint_sha,
            **git_metadata(),
        }
        queued = recorder_send(
            {
                "kind": "start",
                "output_dir": args.datasets,
                "episode_name": f"episode_{episode_index}",
                "camera_names": ("head", "left_wrist", "right_wrist"),
                "metadata": metadata,
                "flush_every": int(
                    runtime_config.get(
                        "flush_every_frames",
                        episode_config.get("writer_flush_frames", 30),
                    )
                ),
            }
        )
        if not queued:
            return
        recording_open = True
        recorded_frames = 0
        send_ui("info", message=f"episode_{episode_index} recording started")

    def build_observation(samples: Mapping[str, Any], now_ns: int) -> dict[str, Any] | None:
        feedback = samples["feedback"]
        cameras = samples["cameras"]
        vr = samples["vr"]
        if set(feedback) != {"left", "right"} or set(cameras) != set(camera_topics):
            return None
        left, left_qvel, left_effort = feedback["left"]
        right, right_qvel, right_effort = feedback["right"]
        qpos = np.asarray(
            (*left.joint_pos, left.gripper, *right.joint_pos, right.gripper),
            dtype=np.float64,
        )
        eef = np.asarray(
            (*left.eef_pose, left.gripper, *right.eef_pose, right.gripper),
            dtype=np.float64,
        )
        qvel = np.asarray((*left_qvel[:7], *right_qvel[:7]), dtype=np.float64)
        effort = np.asarray((*left_effort[:7], *right_effort[:7]), dtype=np.float64)
        if any(array.shape != (14,) or not np.all(np.isfinite(array)) for array in (qpos, eef, qvel, effort)):
            core.request_fault("invalid arm observation", now_ns)
            return None
        timestamps = {
            "observation_ns": now_ns,
            "control_ns": now_ns,
            "arm_left_ns": left.timestamp_ns,
            "arm_right_ns": right.timestamp_ns,
            "camera_head_ns": cameras["head"][1],
            "camera_left_wrist_ns": cameras["left_wrist"][1],
            "camera_right_wrist_ns": cameras["right_wrist"][1],
            "vr_left_ns": vr["left"].timestamp_ns if "left" in vr else -1,
            "vr_right_ns": vr["right"].timestamp_ns if "right" in vr else -1,
        }
        policy_basis_ns = min(
            left.timestamp_ns,
            right.timestamp_ns,
            *(cameras[name][1] for name in camera_topics),
        )
        return {
            "qpos": qpos,
            "eef": eef,
            "qvel": qvel,
            "effort": effort,
            "robot_base": np.zeros(6, dtype=np.float64),
            "base_velocity": np.zeros(4, dtype=np.float64),
            "images_jpeg": {name: cameras[name][0] for name in camera_topics},
            "expert_eef_raw": (
                np.asarray(
                    (
                        *vr["left"].eef_pose,
                        vr["left"].gripper,
                        *vr["right"].eef_pose,
                        vr["right"].gripper,
                    ),
                    dtype=np.float64,
                )
                if set(vr) == {"left", "right"}
                else None
            ),
            "policy_basis_ns": policy_basis_ns,
            "timestamps": timestamps,
        }

    def health_error(samples: Mapping[str, Any], now_ns: int, require_policy: bool) -> str | None:
        nonlocal last_graph_check_ns, cached_graph_error
        if now_ns - last_graph_check_ns >= 250_000_000:
            cached_graph_error = None
            graph_expectations = (
                *((name, 1, 1) for name in command_topics.values()),
                *((name, 1, None) for name in status_topics.values()),
                *((name, 1, None) for name in vr_topics.values()),
                *((name, 1, None) for name in camera_topics.values()),
                (body_status_topic, 1, None),
                *((name, 0, 1) for name in isolated_input_topics),
            )
            for topic_name, publishers, subscribers in graph_expectations:
                actual_publishers = node.count_publishers(topic_name)
                if actual_publishers != publishers:
                    cached_graph_error = (
                        f"ROS graph mismatch on {topic_name}: "
                        f"expected {publishers} publisher, found {actual_publishers}"
                    )
                    break
                if subscribers is not None:
                    actual_subscribers = node.count_subscribers(topic_name)
                    if actual_subscribers != subscribers:
                        cached_graph_error = (
                            f"ROS graph mismatch on {topic_name}: "
                            f"expected {subscribers} subscriber, found {actual_subscribers}"
                        )
                        break
            last_graph_check_ns = now_ns
        if cached_graph_error is not None:
            return cached_graph_error
        if set(samples["feedback"]) != {"left", "right"}:
            return "waiting for both arm feedback streams"
        if set(samples["vr"]) != {"left", "right"}:
            return "waiting for both raw VR streams"
        if set(samples["cameras"]) != set(camera_topics):
            return "waiting for all three RGB cameras"
        for side, (feedback, _qvel, _effort) in samples["feedback"].items():
            if now_ns - feedback.timestamp_ns > feedback_timeout_ns:
                return f"{side} arm feedback timeout"
        for side, vr_sample in samples["vr"].items():
            if now_ns - vr_sample.timestamp_ns > vr_timeout_ns:
                return f"{side} VR timeout"
        for name, (_payload, stamp_ns) in samples["cameras"].items():
            if now_ns - stamp_ns > image_timeout_ns:
                return f"{name} camera timeout"
        if samples["body"] is None:
            return "waiting for body height feedback"
        height, body_stamp_ns = samples["body"]
        if not math.isfinite(height):
            return "body height feedback is not finite"
        if now_ns - body_stamp_ns > body_timeout_ns:
            return "body height feedback timeout"
        if abs(height - float(args.height)) > height_tolerance:
            return f"fixed lift height mismatch ({height:.3f} vs {float(args.height):.3f})"
        if require_policy and not policy_ready:
            return "waiting for policy worker"
        return None

    def record_timeline_events(events: Any) -> None:
        nonlocal handoff_pending
        if not recording_open:
            return
        handoff_pending, records = _timeline_event_records(
            events,
            handoff_pending,
            recorded_frames,
        )
        for event_record in records:
            recorder_send({"kind": "event", "event": event_record})

    def frame_payload(observation: Mapping[str, Any], result: Any) -> dict[str, Any]:
        command = result.command
        state = result.snapshot.state
        policy_action = None
        if command is not None and command.source in (CommandSource.POLICY, CommandSource.POLICY_SLEW):
            policy_action = np.asarray(
                (*command.left.joint_pos, command.left.gripper, *command.right.joint_pos, command.right.gripper),
                dtype=np.float32,
            )
        raw_expert = None
        rebased_expert = None
        if state is ControlState.HUMAN and observation["expert_eef_raw"] is not None:
            raw_expert = np.asarray(observation["expert_eef_raw"], dtype=np.float32)
            if result.snapshot.latest_rebased_expert is not None:
                rebased_expert = np.asarray(result.snapshot.latest_rebased_expert, dtype=np.float32)

        legacy_action = np.asarray(observation["qpos"], dtype=np.float32).copy()
        legacy_eef = np.asarray(observation["eef"], dtype=np.float32).copy()
        for index in (6, 13):
            legacy_action[index] = 0.0 if legacy_action[index] > -2.1 else legacy_action[index]
            legacy_eef[index] = 0.0 if legacy_eef[index] > -2.1 else legacy_eef[index]
        return {
            "observation": {
                key: observation[key]
                for key in ("qpos", "eef", "qvel", "effort", "robot_base", "base_velocity")
            },
            "images_jpeg": observation["images_jpeg"],
            "action": legacy_action,
            "action_eef": legacy_eef,
            "control_mode": state.value,
            "observation_ns": observation["timestamps"]["observation_ns"],
            "control_ns": observation["timestamps"]["control_ns"],
            "source_timestamps": {
                key: value
                for key, value in observation["timestamps"].items()
                if key not in {"observation_ns", "control_ns"}
            },
            "policy_action_joint": policy_action,
            "expert_action_eef_raw": raw_expert,
            "expert_action_eef_rebased": rebased_expert,
            "control_epoch": result.snapshot.control_epoch,
            "action_seq": int(result.snapshot.latest_policy_sequence),
        }

    node.publish_state(ControlState.PRECHECK_HOLD)
    send_ui("state", state=ControlState.PRECHECK_HOLD.value, detail="waiting for policy and sensor health")
    next_tick = time.monotonic()
    try:
        while rclpy.ok():
            now_ns = monotonic_ns()

            while True:
                try:
                    message = ui_command_queue.get_nowait()
                except queue.Empty:
                    break
                kind = message.get("kind")
                if kind == "heartbeat":
                    last_ui_heartbeat_ns = int(message["time_ns"])
                elif kind in {"ui_eof", "worker_fault"}:
                    core.request_fault(
                        f"{kind}: {message.get('source', 'terminal')}",
                        int(message.get("time_ns", now_ns)),
                    )
                elif kind == "shutdown":
                    shutdown_requested = True
                    core.request_fault("operator shutdown", int(message.get("time_ns", now_ns)))
                elif kind == "key":
                    key = str(message.get("key", ""))
                    timestamp_ns = int(message.get("time_ns", now_ns))
                    if core.state is ControlState.REVIEW_HOLD:
                        if close_pending is None and key in {"s", "d", "q"}:
                            if key == "s":
                                recorder_send({"kind": "finalize"})
                                close_pending = "save"
                            else:
                                recorder_send({"kind": "discard"})
                                close_pending = "discard"
                                quit_after_close = key == "q"
                    elif core.state is ControlState.MANUAL_RESET and key == "q":
                        shutdown_requested = True
                        core.request_fault("operator shutdown", timestamp_ns)
                    else:
                        if key in {" ", "space"} and core.state is ControlState.POLICY:
                            send_ui("state", state="TAKEOVER_REQUESTED", detail="gating policy")
                        elif key == "p" and core.state is ControlState.HUMAN:
                            send_ui("state", state="POLICY_REQUESTED", detail="gating human control")
                        core.handle_key(key, timestamp_ns)

            # UI messages originate in another process and may have been stamped
            # just after this loop's first clock sample. Refresh before timeout,
            # packet-freshness, and state-machine decisions.
            now_ns = monotonic_ns()
            if now_ns - last_ui_heartbeat_ns > ui_timeout_ns:
                core.request_fault("operator UI heartbeat timeout", now_ns)

            while True:
                try:
                    message = policy_status_queue.get_nowait()
                except queue.Empty:
                    break
                if message.get("kind") == "policy_ready":
                    policy_ready = True
                elif message.get("kind") == "policy_reset_ack":
                    ack_epoch = int(message["control_epoch"])
                    if core.acknowledge_policy_reset(
                        ack_epoch,
                        int(message.get("time_ns", now_ns)),
                    ):
                        policy_observation_ready_epoch = ack_epoch
                elif message.get("kind") == "policy_observation_dropped":
                    dropped_epoch = int(message["control_epoch"])
                    if (
                        dropped_epoch == core.control_epoch
                        and core.state
                        in (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY)
                    ):
                        policy_observation_ready_epoch = dropped_epoch
                elif message.get("kind") == "policy_error":
                    error_epoch = message.get("control_epoch")
                    if error_epoch is None or (
                        int(error_epoch) == core.control_epoch
                        and core.state in (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY)
                    ):
                        core.request_fault(f"policy worker: {message.get('error')}", now_ns)

            while True:
                try:
                    message = policy_result_queue.get_nowait()
                except queue.Empty:
                    break
                if message.get("kind") != "policy_action":
                    continue
                if int(message.get("episode_id", -1)) != episode_index:
                    continue
                result_epoch = int(message["control_epoch"])
                # Discard an expired forward before touching its action array.
                # Even malformed/NaN data from a gated epoch must not fault the
                # currently active HUMAN or newer policy epoch.
                if result_epoch != core.control_epoch:
                    continue
                if (
                    core.state
                    in (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY)
                ):
                    # The worker has completed this forward and is now free to
                    # consume one newly captured observation. Keeping this
                    # handshake in the control process prevents a queue of old
                    # frames from forming behind CUDA inference.
                    policy_observation_ready_epoch = result_epoch
                packet = PolicyActionPacket(
                    control_epoch=result_epoch,
                    sequence=int(message["action_seq"]),
                    timestamp_ns=int(message["generated_ns"]),
                    action=tuple(np.asarray(message["action"], dtype=np.float64)),
                    observation_timestamp_ns=int(
                        message.get("policy_basis_ns", message["observation_ns"])
                    ),
                )
                core.submit_policy_action(packet, now_ns)

            while True:
                try:
                    message = recorder_status_queue.get_nowait()
                except queue.Empty:
                    break
                kind = message.get("kind")
                if kind == "recorder_error":
                    core.request_fault(f"recorder worker: {message.get('error')}", now_ns)
                elif kind == "recorder_finalized":
                    recording_open = False
                    send_ui("saved", message=f"saved {message['path']}")
                    episode_index += 1
                    if core.state is ControlState.REVIEW_HOLD:
                        review_reset_pending = True
                    else:
                        close_pending = None
                elif kind == "recorder_discarded":
                    recording_open = False
                    send_ui("discarded", message=f"episode_{episode_index} discarded; index will be reused")
                    if quit_after_close:
                        shutdown_requested = True
                    elif core.state is ControlState.REVIEW_HOLD:
                        review_reset_pending = True
                    else:
                        close_pending = None
                elif kind == "recorder_quarantined":
                    recording_open = False
                    send_ui("quarantined", message=f"fault data quarantined at {message['path']}")

            # Take one coherent source snapshot only after asynchronous UI and
            # worker acknowledgements have been drained. In particular, the
            # first post-reset policy observation must not be an older snapshot
            # captured before its reset acknowledgement arrived.
            samples = node.snapshot()
            now_ns = monotonic_ns()

            if node.external_hold_requested.is_set() and not external_hold_seen:
                external_hold_seen = True
                core.request_fault("external safe shutdown request", now_ns)

            feedback_pair_valid = False
            if set(samples["feedback"]) == {"left", "right"}:
                feedback_pair_valid = core.update_feedback(
                    samples["feedback"]["left"][0],
                    samples["feedback"]["right"][0],
                )
            if set(samples["vr"]) == {"left", "right"}:
                core.update_vr(samples["vr"]["left"], samples["vr"]["right"])

            if review_reset_pending:
                if core.state is ControlState.REVIEW_HOLD:
                    review_reset_pending, reset_completed = _retry_review_reset_after_close(
                        core,
                        review_reset_pending,
                        now_ns,
                    )
                    if reset_completed:
                        close_pending = None
                        quit_after_close = False
                else:
                    # A concurrent fault/shutdown supersedes review continuation.
                    review_reset_pending = False

            current_health_error = health_error(samples, now_ns, require_policy=True)
            if core.state is ControlState.PRECHECK_HOLD:
                if current_health_error is None:
                    height_stable_since_ns = height_stable_since_ns or now_ns
                    if not precheck_submitted and now_ns - height_stable_since_ns >= stable_height_ns:
                        core.mark_precheck_complete(now_ns)
                        precheck_submitted = True
                else:
                    height_stable_since_ns = None
                    if now_ns - last_precheck_report_ns >= 1_000_000_000:
                        send_ui("state", state=ControlState.PRECHECK_HOLD.value, detail=current_health_error)
                        last_precheck_report_ns = now_ns
                if now_ns - start_ns > precheck_timeout_ns:
                    send_ui("fatal", error=f"precheck timeout: {current_health_error}")
                    shutdown_requested = True
            elif core.state not in (ControlState.REVIEW_HOLD, ControlState.FAULT_HOLD):
                if current_health_error is not None:
                    core.request_fault(current_health_error, now_ns)

            if recorded_frames >= int(args.max_timesteps) and core.snapshot().episode_active:
                core.handle_key("e", now_ns)

            result = core.tick(now_ns)

            if result.snapshot.pending_policy_reset_epoch is not None:
                epoch = int(result.snapshot.pending_policy_reset_epoch)
                if reset_request_epoch != epoch:
                    if policy_control_send({"kind": "reset", "control_epoch": epoch}):
                        reset_request_epoch = epoch
                        policy_observation_ready_epoch = None
            elif result.snapshot.state not in (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY):
                if previous_state in (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY):
                    policy_control_send({"kind": "pause"})
                reset_request_epoch = None
                policy_observation_ready_epoch = None

            observation = build_observation(samples, now_ns)
            if (
                observation is not None
                and result.snapshot.state in (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY)
                and result.snapshot.policy_reset_acknowledged
                and policy_observation_ready_epoch == result.snapshot.control_epoch
            ):
                observation_seq += 1
                if policy_observation_send(
                    {
                        "kind": "observation",
                        "episode_id": episode_index,
                        "control_epoch": result.snapshot.control_epoch,
                        "observation_seq": observation_seq,
                        "observation": observation,
                    },
                ):
                    policy_observation_ready_epoch = None

            if result.command is not None:
                node.publish_command(result.command)
                if (
                    result.snapshot.state is ControlState.HANDOFF_TO_HUMAN
                    and result.command.source is CommandSource.HOLD
                ):
                    core.acknowledge_handoff_hold_published(
                        result.snapshot.control_epoch,
                        monotonic_ns(),
                    )
                if external_hold_seen and result.command.source is CommandSource.HOLD:
                    if node.external_hold_published_ns == 0:
                        # Only post-publish feedback may acknowledge the HOLD
                        # service. A callback received after the service request
                        # but before this publish does not prove that the driver
                        # has seen the HOLD command.
                        node.external_hold_published_ns = monotonic_ns()
                    feedback_after_hold = _feedback_pair_acknowledges_hold(
                        samples,
                        node.external_hold_published_ns,
                        feedback_pair_valid,
                    )
                    if feedback_after_hold:
                        node.external_hold_ack.set()
            elif result.snapshot.state in (
                ControlState.PRECHECK_HOLD,
                ControlState.FAULT_HOLD,
            ):
                partial_feedback = _single_valid_feedback_for_hold(
                    samples,
                    now_ns,
                    feedback_timeout_ns,
                    future_timestamp_tolerance_ns,
                )
                if partial_feedback is not None:
                    side, feedback = partial_feedback
                    node.publish_measured_hold(side, feedback)

            if (
                result.snapshot.episode_active
                and not recording_open
                and result.snapshot.state is ControlState.HANDOFF_TO_POLICY
            ):
                start_recording(now_ns)

            record_timeline_events(result.events)

            if recording_open and result.snapshot.episode_active and observation is not None:
                if recorder_send({"kind": "frame", "frame": frame_payload(observation, result)}):
                    recorded_frames += 1

            if (
                result.snapshot.state is ControlState.REVIEW_HOLD
                and recording_open
                and close_pending is None
            ):
                if not result.snapshot.intervention_occurred:
                    recorder_send({"kind": "finalize"})
                    close_pending = "auto_save"
                    send_ui("info", message="no-intervention rollout: auto-saving")

            if result.snapshot.state is ControlState.FAULT_HOLD and recording_open and not fault_quarantine_sent:
                if recorder_send(
                    {
                        "kind": "quarantine",
                        "reason": result.snapshot.fault_reason or "control fault",
                    }
                ):
                    fault_quarantine_sent = True

            if result.state_changed or result.snapshot.transition_revision != last_state_revision:
                node.publish_state(result.snapshot.state)
                if (
                    result.snapshot.state is ControlState.REVIEW_HOLD
                    and result.snapshot.intervention_occurred
                ):
                    detail = "[s]ave [d]iscard [q]discard+quit"
                else:
                    detail = result.snapshot.fault_reason or _state_hint(result.snapshot.state)
                send_ui("state", state=result.snapshot.state.value, detail=detail)
                last_state_revision = result.snapshot.transition_revision

            previous_state = result.snapshot.state

            tick_count += 1
            if now_ns - last_diag_report_ns >= diag_interval_ns:
                elapsed_s = (now_ns - last_diag_report_ns) / 1e9
                diag = node.drain_diagnostics()
                if result.snapshot.state is not ControlState.PRECHECK_HOLD:
                    def _diag_gap(stream: str) -> str:
                        entry = diag.get(stream)
                        return f"{entry[0] / 1e6:.0f}ms" if entry and entry[1] else "--"

                    # Reading the line: tick low / overrun high -> frontend starved;
                    # one arm's fb gap high -> that arm's USB-CAN link; VR gap high
                    # -> VR serial side.
                    send_ui(
                        "info",
                        message=(
                            f"[diag] tick {tick_count / elapsed_s:.1f}Hz overrun={tick_overruns}"
                            f" | L-fb gap {_diag_gap('fb_left')} | R-fb gap {_diag_gap('fb_right')}"
                            f" | VR-L gap {_diag_gap('vr_left')} | VR-R gap {_diag_gap('vr_right')}"
                        ),
                    )
                tick_count = 0
                tick_overruns = 0
                last_diag_report_ns = now_ns

            if shutdown_requested:
                if result.command is not None and result.command.source is CommandSource.HOLD:
                    if recording_open and not fault_quarantine_sent:
                        recorder_send({"kind": "quarantine", "reason": "operator shutdown"})
                    break
                if result.command is None and result.snapshot.state in (
                    ControlState.PRECHECK_HOLD,
                    ControlState.FAULT_HOLD,
                ):
                    break
                if result.snapshot.state is ControlState.MANUAL_RESET:
                    # MANUAL_RESET is EEF control; force an explicit position HOLD first.
                    core.request_fault("operator shutdown", now_ns)

            next_tick += frame_period
            sleep_seconds = next_tick - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            else:
                tick_overruns += 1
                next_tick = time.monotonic()
    except BaseException as exc:
        try:
            core.request_fault(f"control exception: {exc}", monotonic_ns())
            hold_now_ns = monotonic_ns()
            result = core.tick(hold_now_ns)
            if result.command is not None:
                node.publish_command(result.command)
            else:
                partial_feedback = _single_valid_feedback_for_hold(
                    node.snapshot(),
                    hold_now_ns,
                    feedback_timeout_ns,
                    future_timestamp_tolerance_ns,
                )
                if partial_feedback is not None:
                    side, feedback = partial_feedback
                    node.publish_measured_hold(side, feedback)
        except BaseException:
            pass
        send_ui("fatal", error=repr(exc))
    finally:
        send_ui("exit")
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        # The exception path above may already have shut the context down; a
        # second unconditional call raises "rcl_shutdown already called" and
        # turns a clean exit into a traceback.
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


class SimpleRuntimeArgs:
    """Pickle-friendly mapping-to-attributes adapter for spawned children."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.__dict__.update(values)


def _state_hint(state: Any) -> str:
    hints = {
        "PRECHECK_HOLD": "waiting for health checks",
        "MANUAL_RESET": "use VR to reset; [r] start, [q] quit",
        "HANDOFF_TO_POLICY": "HOLD; resetting policy",
        "POLICY": "[space] human takeover, [e] end",
        "HANDOFF_TO_HUMAN": "HOLD; waiting for post-key VR and feedback",
        "HUMAN": "human active; [p] resume policy, [e] end",
        "REVIEW_HOLD": "episode ended",
        "FAULT_HOLD": "fault latched; run safe shutdown",
    }
    key = getattr(state, "value", str(state))
    return hints.get(key, "")


def _print_status(message: Mapping[str, Any]) -> None:
    kind = message.get("kind")
    if kind == "state":
        suffix = f" - {message['detail']}" if message.get("detail") else ""
        print(f"\n[Human DAgger] {message['state']}{suffix}", flush=True)
    elif kind in {"info", "saved", "discarded", "quarantined"}:
        print(f"\n[Human DAgger] {message.get('message', message)}", flush=True)
    elif kind == "fatal":
        print(f"\n[Human DAgger ERROR] {message.get('error', message)}", file=sys.stderr, flush=True)


def run_supervisor(args: argparse.Namespace, runtime_config: dict[str, Any]) -> int:
    context = mp.get_context("spawn")
    ui_command_queue = context.Queue()
    ui_status_queue = context.Queue()
    # Control messages must never be overwritten. Observations are separately
    # bounded because each carries three JPEGs and only the freshest useful frame
    # should wait behind an in-flight CUDA forward.
    policy_control_queue = context.Queue()
    policy_observation_queue = context.Queue(maxsize=1)
    policy_result_queue = context.Queue()
    policy_status_queue = context.Queue()
    recorder_command_queue = context.Queue(maxsize=int(runtime_config.get("recorder_queue_size", 32)))
    recorder_status_queue = context.Queue()

    if args.mock_policy:
        policy_process = context.Process(
            name="human-dagger-policy-mock",
            target=mock_policy_worker_main,
            args=(
                policy_control_queue,
                policy_observation_queue,
                policy_result_queue,
                policy_status_queue,
                args.mock_policy_delay,
            ),
        )
    else:
        worker_config = PolicyWorkerConfig(
            ckpt_dir=args.ckpt_dir,
            ckpt_name=args.ckpt_name,
            stats_name=args.stats_name,
            args_name=args.policy_args_name,
            gripper_gate=args.gripper_gate,
            temporal_agg=args.temporal_agg,
            max_observation_age_ns=int(
                float(runtime_config.get("control", {}).get("policy_timeout_ms", 250.0))
                * 1_000_000
            ),
        )
        policy_process = context.Process(
            name="human-dagger-policy",
            target=policy_worker_main,
            args=(
                worker_config,
                policy_control_queue,
                policy_observation_queue,
                policy_result_queue,
                policy_status_queue,
            ),
        )

    recorder_process = context.Process(
        name="human-dagger-writer",
        target=recorder_worker_main,
        args=(recorder_command_queue, recorder_status_queue),
    )
    control_process = context.Process(
        name="human-dagger-control",
        target=control_process_main,
        args=(
            vars(args),
            runtime_config,
            ui_command_queue,
            ui_status_queue,
            policy_control_queue,
            policy_observation_queue,
            policy_result_queue,
            policy_status_queue,
            recorder_command_queue,
            recorder_status_queue,
        ),
    )

    started_processes = []
    try:
        recorder_process.start()
        started_processes.append(recorder_process)
        _append_session_process(args.session_manifest, "writer", recorder_process.pid)
        policy_process.start()
        started_processes.append(policy_process)
        _append_session_process(args.session_manifest, "policy", policy_process.pid)
        control_process.start()
        started_processes.append(control_process)
        _append_session_process(args.session_manifest, "coordinator", control_process.pid)
    except BaseException:
        for process in reversed(started_processes):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
        raise
    last_heartbeat = 0
    requested_exit = False
    exit_code = 0
    policy_fault_sent = False
    recorder_fault_sent = False

    try:
        with TerminalKeyReader() as key_reader:
            print("Human DAgger UI active. Waiting for PRECHECK...", flush=True)
            while control_process.is_alive():
                now_ns = monotonic_ns()
                if now_ns - last_heartbeat >= 50_000_000:
                    ui_command_queue.put({"kind": "heartbeat", "time_ns": now_ns})
                    last_heartbeat = now_ns

                while True:
                    try:
                        status = ui_status_queue.get_nowait()
                    except queue.Empty:
                        break
                    _print_status(status)
                    if status.get("kind") == "fatal":
                        exit_code = 1
                    if status.get("kind") == "exit":
                        requested_exit = True
                        break
                if requested_exit:
                    break

                key = key_reader.poll_key()
                if key == "":
                    ui_command_queue.put({"kind": "ui_eof", "time_ns": now_ns})
                    exit_code = 1
                    break
                if key is not None:
                    ui_command_queue.put({"kind": "key", "key": key.lower(), "time_ns": now_ns})

                if not policy_process.is_alive() and not policy_fault_sent:
                    ui_command_queue.put({"kind": "worker_fault", "source": "policy", "time_ns": now_ns})
                    policy_fault_sent = True
                if not recorder_process.is_alive() and not recorder_fault_sent:
                    ui_command_queue.put({"kind": "worker_fault", "source": "recorder", "time_ns": now_ns})
                    recorder_fault_sent = True
                time.sleep(0.01)
    except KeyboardInterrupt:
        ui_command_queue.put({"kind": "shutdown", "time_ns": monotonic_ns()})
    finally:
        ui_command_queue.put({"kind": "shutdown", "time_ns": monotonic_ns()})
        try:
            policy_control_queue.put_nowait({"kind": "stop"})
        except (OSError, ValueError, queue.Full):
            pass
        try:
            recorder_command_queue.put_nowait({"kind": "stop"})
        except queue.Full:
            pass
        deadline = time.monotonic() + 3.0
        for process in (control_process, policy_process, recorder_process):
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in (control_process, policy_process, recorder_process):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for process in (control_process, policy_process, recorder_process):
            if process.exitcode not in (0, None):
                exit_code = 1
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human DAgger dual-arm rollout collector")
    parser.add_argument("--config", default=str(ACT_ROOT / "data" / "human_dagger.yaml"))
    parser.add_argument("--datasets", default=str(ACT_ROOT / "dagger_datasets"))
    parser.add_argument("--episode-idx", type=int, default=-1)
    parser.add_argument("--task", required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--dagger-round", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=800)
    parser.add_argument("--frame-rate", type=float, default=60.0)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="policy_best.ckpt")
    parser.add_argument("--stats-name", default="dataset_stats.pkl")
    parser.add_argument("--policy-args-name", default="args.yaml")
    parser.add_argument("--gripper-gate", type=float, default=-1.0)
    parser.add_argument("--temporal-agg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock-policy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mock-policy-delay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--session-manifest", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def validate_startup_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.height <= 20.0:
        raise ValueError("--height must be within [0, 20]")
    if args.frame_rate <= 0 or args.max_timesteps < 2:
        raise ValueError("frame rate must be positive and max timesteps must be at least 2")
    if not args.mock_policy:
        ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
        for filename in (args.ckpt_name, args.stats_name):
            if not (ckpt_dir / filename).is_file():
                raise FileNotFoundError(ckpt_dir / filename)
        args.ckpt_dir = str(ckpt_dir)
    args.datasets = str(Path(args.datasets).expanduser().resolve())
    args.config = str(Path(args.config).expanduser().resolve())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_startup_args(args)
        runtime_config = load_yaml(args.config)
        return run_supervisor(args, runtime_config)
    except BaseException as exc:
        print(f"Human DAgger refused to start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
