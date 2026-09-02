"""Streaming HDF5 recorder for Human DAgger episodes.

This module deliberately has no ROS dependency.  The control process can hand
complete, synchronized frames to :class:`HumanDaggerRecorder`, or it can own
the recorder in a dedicated writer process.

The on-disk format is schema version 2.  Images are the original JPEG payloads
from ROS ``CompressedImage`` messages.  Each camera dataset is a zero-padded,
extendable uint8 matrix, and ``/compress_len`` stores the real byte length in
camera-major order.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import h5py
import numpy as np


SCHEMA_VERSION = 2
DEFAULT_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
SOURCE_TIMESTAMP_NAMES = (
    "arm_left_ns",
    "arm_right_ns",
    "camera_head_ns",
    "camera_left_wrist_ns",
    "camera_right_wrist_ns",
    "vr_left_ns",
    "vr_right_ns",
)


class ControlMode(IntEnum):
    """Values persisted in ``/dagger/control_mode``.

    Do not renumber these values: datasets use the integer representation as a
    stable public interface.
    """

    POLICY = 1
    HANDOFF_TO_HUMAN = 2
    HUMAN = 3
    HANDOFF_TO_POLICY = 4
    FAULT_HOLD = 5
    FAULT = 5  # Friendly alias used by some callers.


class EventType:
    """Canonical event names understood by the offline validator."""

    EPISODE_START = "EPISODE_START"
    HANDOFF_TO_HUMAN = "HANDOFF_TO_HUMAN"
    HANDOFF_TO_POLICY = "HANDOFF_TO_POLICY"
    EPISODE_END = "EPISODE_END"
    FAULT = "FAULT"


HANDOFF_EVENT_TYPES = {
    EventType.HANDOFF_TO_HUMAN,
    EventType.HANDOFF_TO_POLICY,
}

OBSERVATION_SPECS = {
    "qpos": 14,
    "qvel": 14,
    "effort": 14,
    "eef": 14,
    "robot_base": 6,
    "base_velocity": 4,
}

LEGACY_ACTION_SPECS = {
    "action": 14,
    "action_eef": 14,
    "action_base": 6,
    "action_velocity": 4,
}

DAGGER_VECTOR_SPECS = {
    "policy_action_joint": 14,
    "expert_action_eef_raw": 14,
    "expert_action_eef_rebased": 14,
}

REQUIRED_METADATA_DEFAULTS = {
    "task": "",
    "height_command": -1.0,
    "dagger_round": 0,
    "policy_checkpoint": "",
    "policy_checkpoint_sha256": "",
    "git_commit": "",
    "nominal_fps": 60.0,
}

_EVENT_DTYPE = np.dtype(
    [
        ("event", "S48"),
        ("request_ns", "<i8"),
        ("gate_ns", "<i8"),
        ("ack_ns", "<i8"),
        ("frame", "<i8"),
        ("epoch", "<i8"),
        ("detail", "S256"),
    ]
)


class EpisodeValidationError(RuntimeError):
    """Raised when finalize-time validation rejects an episode."""

    def __init__(self, errors: Sequence[str], quarantine_path: Optional[Path] = None):
        message = "Human DAgger episode validation failed: " + "; ".join(errors)
        if quarantine_path is not None:
            message += f" (quarantined at {quarantine_path})"
        super().__init__(message)
        self.errors = tuple(errors)
        self.quarantine_path = quarantine_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _normalise_episode_stem(episode_name: Union[str, os.PathLike[str]]) -> str:
    name = os.fspath(episode_name)
    if Path(name).name != name:
        raise ValueError("episode_name must be a file name, not a path")
    for suffix in (".partial.hdf5", ".hdf5"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name or name in {".", ".."}:
        raise ValueError("episode_name must not be empty")
    return name


def _encode_fixed_utf8(value: Any, limit: int, field: str) -> bytes:
    encoded = str(value).encode("utf-8")
    if len(encoded) > limit:
        raise ValueError(f"{field} exceeds {limit} UTF-8 bytes")
    return encoded


def _metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _vector(
    value: Any,
    width: int,
    name: str,
    *,
    default: Optional[np.ndarray] = None,
) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        result = np.asarray(default, dtype=np.float32).copy()
    else:
        result = np.asarray(value, dtype=np.float32)
    if result.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def normalise_control_mode(mode: Union[ControlMode, str, int]) -> ControlMode:
    if isinstance(mode, ControlMode):
        return mode
    if isinstance(mode, str):
        key = mode.strip().upper()
        if key == "FAULT":
            key = "FAULT_HOLD"
        try:
            return ControlMode[key]
        except KeyError as exc:
            raise ValueError(f"unknown control mode: {mode!r}") from exc
    try:
        return ControlMode(int(mode))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown control mode: {mode!r}") from exc


class HumanDaggerRecorder:
    """Append synchronized Human DAgger frames directly to an HDF5 partial.

    Parameters
    ----------
    output_dir:
        Directory containing final episodes and the ``quarantine`` directory.
    episode_name:
        A basename such as ``episode_12``.  ``.hdf5`` is accepted and stripped.
    camera_names:
        Stable camera ordering used by ``/compress_len``.
    metadata:
        Per-episode values.  At minimum callers should provide task, fixed
        height, checkpoint path/hash, git commit, DAgger round and nominal FPS.
    flush_every:
        Flush the HDF5 file after this many appended frames.
    image_capacity:
        Initial padded JPEG width.  It grows automatically when needed.
    """

    def __init__(
        self,
        output_dir: Union[str, os.PathLike[str]],
        episode_name: Union[str, os.PathLike[str]],
        *,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        metadata: Optional[Mapping[str, Any]] = None,
        flush_every: int = 30,
        image_capacity: int = 256 * 1024,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.episode_stem = _normalise_episode_stem(episode_name)
        self.final_path = self.output_dir / f"{self.episode_stem}.hdf5"
        self.partial_path = self.output_dir / f"{self.episode_stem}.partial.hdf5"
        self.camera_names = tuple(str(name) for name in camera_names)
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be non-empty and unique")
        if any(not name or "/" in name for name in self.camera_names):
            raise ValueError("camera names must be non-empty HDF5 path components")
        if int(flush_every) < 1:
            raise ValueError("flush_every must be at least 1")
        if int(image_capacity) < 1:
            raise ValueError("image_capacity must be positive")

        self.flush_every = int(flush_every)
        self._image_capacity_step = int(image_capacity)
        self._frame_count = 0
        self._frames_since_flush = 0
        self._last_observation_ns: Optional[int] = None
        self._last_control_ns: Optional[int] = None
        self._last_epoch = 0
        self._closed = False
        self._lock = threading.RLock()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(f"final episode already exists: {self.final_path}")
        if self.partial_path.exists():
            raise FileExistsError(f"partial episode already exists: {self.partial_path}")

        self._file = h5py.File(self.partial_path, "w", libver="latest")
        try:
            self._create_schema(dict(metadata or {}))
            self._file.flush()
        except Exception:
            self._file.close()
            self._closed = True
            try:
                self.partial_path.unlink()
            except FileNotFoundError:
                pass
            raise

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def event_count(self) -> int:
        return int(self._events.shape[0])

    def _create_schema(self, metadata: Mapping[str, Any]) -> None:
        attrs = self._file.attrs
        attrs["schema_version"] = SCHEMA_VERSION
        attrs["collection_mode"] = "human_dagger"
        attrs["sim"] = False
        attrs["compress"] = True
        attrs["image_encoding"] = "jpeg"
        attrs["action_semantics"] = "current_measured_qpos"
        attrs["training_action_offset_frames"] = 1
        attrs["camera_names"] = json.dumps(self.camera_names)
        attrs["control_mode_values"] = json.dumps(
            {mode.name: int(mode) for mode in ControlMode if mode.name != "FAULT"},
            sort_keys=True,
        )
        attrs["created_utc"] = _utc_now()
        attrs["finalized"] = False
        attrs["num_frames"] = 0

        merged_metadata = dict(REQUIRED_METADATA_DEFAULTS)
        merged_metadata.update(metadata)
        reserved = {
            "schema_version",
            "collection_mode",
            "sim",
            "compress",
            "image_encoding",
            "action_semantics",
            "training_action_offset_frames",
            "camera_names",
            "control_mode_values",
            "created_utc",
            "finalized",
            "num_frames",
        }
        fixed_values = {
            "schema_version": SCHEMA_VERSION,
            "collection_mode": "human_dagger",
            "sim": False,
            "compress": True,
            "image_encoding": "jpeg",
            "action_semantics": "current_measured_qpos",
            "training_action_offset_frames": 1,
        }
        conflict = reserved.intersection(metadata)
        for key in conflict:
            if key not in fixed_values or metadata[key] != fixed_values[key]:
                raise ValueError(f"metadata may not replace reserved attr {key!r}")
            merged_metadata.pop(key, None)
        for key, value in merged_metadata.items():
            if not isinstance(key, str) or not key or "/" in key:
                raise ValueError(f"invalid metadata key: {key!r}")
            attrs[key] = _metadata_value(value)

        observations = self._file.create_group("observations")
        for name, width in OBSERVATION_SPECS.items():
            self._create_frame_vector(observations, name, width)

        images = observations.create_group("images")
        self._image_datasets = {}
        for camera_name in self.camera_names:
            dataset = images.create_dataset(
                camera_name,
                shape=(0, self._image_capacity_step),
                maxshape=(None, None),
                chunks=(1, self._image_capacity_step),
                dtype=np.uint8,
                fillvalue=0,
            )
            dataset.attrs["encoding"] = "jpeg"
            self._image_datasets[camera_name] = dataset

        self._compress_len = self._file.create_dataset(
            "compress_len",
            shape=(len(self.camera_names), 0),
            maxshape=(len(self.camera_names), None),
            chunks=(len(self.camera_names), max(1, min(256, self.flush_every))),
            dtype=np.int32,
        )
        self._compress_len.attrs["axis_0_camera_names"] = json.dumps(self.camera_names)

        for name, width in LEGACY_ACTION_SPECS.items():
            self._create_frame_vector(self._file, name, width)

        dagger = self._file.create_group("dagger")
        for name, width in DAGGER_VECTOR_SPECS.items():
            self._create_frame_vector(dagger, name, width)
        self._create_frame_scalar(dagger, "control_mode", np.uint8)
        self._create_frame_scalar(dagger, "intervention_mask", np.bool_)
        self._create_frame_scalar(dagger, "supervision_valid", np.bool_)
        self._create_frame_scalar(dagger, "policy_action_valid", np.bool_)
        self._create_frame_scalar(dagger, "expert_action_valid", np.bool_)
        self._create_frame_scalar(dagger, "control_epoch", np.int64)
        self._create_frame_scalar(dagger, "action_seq", np.int64)
        self._events = dagger.create_dataset(
            "events", shape=(0,), maxshape=(None,), chunks=(64,), dtype=_EVENT_DTYPE
        )

        timestamps = self._file.create_group("timestamps")
        self._create_frame_scalar(timestamps, "observation_ns", np.int64)
        self._create_frame_scalar(timestamps, "control_ns", np.int64)
        for name in SOURCE_TIMESTAMP_NAMES:
            self._create_frame_scalar(timestamps, name, np.int64)

        self._frame_datasets = []
        for path in (
            *(f"/observations/{name}" for name in OBSERVATION_SPECS),
            *(f"/{name}" for name in LEGACY_ACTION_SPECS),
            *(f"/dagger/{name}" for name in DAGGER_VECTOR_SPECS),
            "/dagger/control_mode",
            "/dagger/intervention_mask",
            "/dagger/supervision_valid",
            "/dagger/policy_action_valid",
            "/dagger/expert_action_valid",
            "/dagger/control_epoch",
            "/dagger/action_seq",
            "/timestamps/observation_ns",
            "/timestamps/control_ns",
            *(f"/timestamps/{name}" for name in SOURCE_TIMESTAMP_NAMES),
        ):
            self._frame_datasets.append(self._file[path])
        self._frame_datasets.extend(self._image_datasets.values())

    def _create_frame_vector(self, group: h5py.Group, name: str, width: int) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=(0, width),
            maxshape=(None, width),
            chunks=(max(1, min(64, self.flush_every)), width),
            dtype=np.float32,
        )

    def _create_frame_scalar(self, group: h5py.Group, name: str, dtype: Any) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, min(256, self.flush_every)),),
            dtype=dtype,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")

    def append_frame(
        self,
        *,
        observation: Mapping[str, Any],
        images_jpeg: Mapping[str, Union[bytes, bytearray, memoryview, np.ndarray]],
        control_mode: Union[ControlMode, str, int],
        observation_ns: Optional[int] = None,
        control_ns: Optional[int] = None,
        timestamps: Optional[Mapping[str, int]] = None,
        source_timestamps: Optional[Mapping[str, int]] = None,
        action: Optional[Any] = None,
        action_eef: Optional[Any] = None,
        action_base: Optional[Any] = None,
        action_velocity: Optional[Any] = None,
        policy_action_joint: Optional[Any] = None,
        expert_action_eef_raw: Optional[Any] = None,
        expert_action_eef_rebased: Optional[Any] = None,
        control_epoch: int = 0,
        action_seq: int = -1,
    ) -> int:
        """Append one fully synchronized frame and return its zero-based index.

        Missing legacy action arrays default to the matching measured state.
        Policy/expert validity masks are derived from both ownership and whether
        the corresponding command was supplied; invalid command rows are zero.
        """

        with self._lock:
            self._ensure_open()
            mode = normalise_control_mode(control_mode)
            timestamp_values = dict(timestamps or {})
            unknown_timestamp_names = set(timestamp_values) - {
                "observation_ns",
                "control_ns",
                *SOURCE_TIMESTAMP_NAMES,
            }
            if unknown_timestamp_names:
                raise ValueError(
                    f"unknown timestamps: {sorted(unknown_timestamp_names)}"
                )
            if observation_ns is None:
                observation_ns = timestamp_values.pop("observation_ns", None)
            elif "observation_ns" in timestamp_values:
                if int(observation_ns) != int(timestamp_values.pop("observation_ns")):
                    raise ValueError("conflicting observation_ns values")
            if control_ns is None:
                control_ns = timestamp_values.pop("control_ns", None)
            elif "control_ns" in timestamp_values:
                if int(control_ns) != int(timestamp_values.pop("control_ns")):
                    raise ValueError("conflicting control_ns values")
            if observation_ns is None or control_ns is None:
                raise ValueError("observation_ns and control_ns are required")
            combined_source_timestamps = {
                name: int(timestamp_values.get(name, -1)) for name in SOURCE_TIMESTAMP_NAMES
            }
            if source_timestamps is not None:
                unknown_sources = set(source_timestamps) - set(SOURCE_TIMESTAMP_NAMES)
                if unknown_sources:
                    raise ValueError(f"unknown source_timestamps: {sorted(unknown_sources)}")
                for name, value in source_timestamps.items():
                    value = int(value)
                    existing = combined_source_timestamps[name]
                    if existing != -1 and existing != value:
                        raise ValueError(f"conflicting timestamp for {name}")
                    combined_source_timestamps[name] = value
            for name, value in combined_source_timestamps.items():
                if value == 0 or value < -1:
                    raise ValueError(f"{name} must be -1 (not applicable) or positive")
            observation_ns = int(observation_ns)
            control_ns = int(control_ns)
            epoch = int(control_epoch)
            seq = int(action_seq)
            if observation_ns <= 0 or control_ns <= 0:
                raise ValueError("observation_ns and control_ns must be positive")
            if self._last_observation_ns is not None and observation_ns <= self._last_observation_ns:
                raise ValueError("observation_ns must be strictly increasing")
            if self._last_control_ns is not None and control_ns <= self._last_control_ns:
                raise ValueError("control_ns must be strictly increasing")
            if epoch < 0:
                raise ValueError("control_epoch must be non-negative")
            if epoch < self._last_epoch:
                raise ValueError("control_epoch must not decrease")

            observation_values = {}
            for name, width in OBSERVATION_SPECS.items():
                if name in {"qvel", "effort", "robot_base", "base_velocity"}:
                    default = np.zeros(width, dtype=np.float32)
                else:
                    default = None
                observation_values[name] = _vector(
                    observation.get(name), width, f"observation[{name!r}]", default=default
                )

            legacy_values = {
                "action": _vector(
                    action,
                    14,
                    "action",
                    default=observation_values["qpos"],
                ),
                "action_eef": _vector(
                    action_eef,
                    14,
                    "action_eef",
                    default=observation_values["eef"],
                ),
                "action_base": _vector(
                    action_base,
                    6,
                    "action_base",
                    default=observation_values["robot_base"],
                ),
                "action_velocity": _vector(
                    action_velocity,
                    4,
                    "action_velocity",
                    default=observation_values["base_velocity"],
                ),
            }

            zero14 = np.zeros(14, dtype=np.float32)
            # During HANDOFF_TO_POLICY the arbiter may already be publishing a
            # rate-limited policy target.  Preserve that actually adopted joint
            # command as valid while allowing the earlier HOLD-only handoff rows
            # to remain invalid.
            policy_valid = (
                mode in (ControlMode.POLICY, ControlMode.HANDOFF_TO_POLICY)
                and policy_action_joint is not None
            )
            expert_inputs_present = (
                expert_action_eef_raw is not None and expert_action_eef_rebased is not None
            )
            expert_valid = mode == ControlMode.HUMAN and expert_inputs_present
            if mode == ControlMode.POLICY and not policy_valid:
                raise ValueError("POLICY frames require policy_action_joint")
            if mode == ControlMode.HUMAN and not expert_valid:
                raise ValueError(
                    "HUMAN frames require expert_action_eef_raw and expert_action_eef_rebased"
                )
            policy_value = _vector(
                policy_action_joint if policy_valid else None,
                14,
                "policy_action_joint",
                default=zero14,
            )
            expert_raw_value = _vector(
                expert_action_eef_raw if expert_valid else None,
                14,
                "expert_action_eef_raw",
                default=zero14,
            )
            expert_rebased_value = _vector(
                expert_action_eef_rebased if expert_valid else None,
                14,
                "expert_action_eef_rebased",
                default=zero14,
            )
            if policy_valid and seq < 0:
                raise ValueError("POLICY frames require a non-negative action_seq")

            jpeg_values = self._prepare_jpegs(images_jpeg)

            new_count = self._frame_count + 1
            old_image_widths = {
                name: int(dataset.shape[1]) for name, dataset in self._image_datasets.items()
            }
            required_width = max(len(payload) for payload in jpeg_values.values())
            current_width = max(old_image_widths.values())
            if required_width > current_width:
                new_width = (
                    (required_width + self._image_capacity_step - 1)
                    // self._image_capacity_step
                    * self._image_capacity_step
                )
            else:
                new_width = current_width

            try:
                for dataset in self._frame_datasets:
                    if dataset.name.startswith("/observations/images/"):
                        dataset.resize((new_count, new_width))
                    else:
                        dataset.resize((new_count,) + dataset.shape[1:])
                self._compress_len.resize((len(self.camera_names), new_count))

                row = self._frame_count
                for name, value in observation_values.items():
                    self._file[f"/observations/{name}"][row] = value
                for name, value in legacy_values.items():
                    self._file[f"/{name}"][row] = value

                dagger = self._file["/dagger"]
                dagger["policy_action_joint"][row] = policy_value
                dagger["expert_action_eef_raw"][row] = expert_raw_value
                dagger["expert_action_eef_rebased"][row] = expert_rebased_value
                dagger["control_mode"][row] = int(mode)
                dagger["intervention_mask"][row] = mode == ControlMode.HUMAN
                dagger["supervision_valid"][row] = False
                dagger["policy_action_valid"][row] = policy_valid
                dagger["expert_action_valid"][row] = expert_valid
                dagger["control_epoch"][row] = epoch
                dagger["action_seq"][row] = seq
                self._file["/timestamps/observation_ns"][row] = observation_ns
                self._file["/timestamps/control_ns"][row] = control_ns
                for name, value in combined_source_timestamps.items():
                    self._file[f"/timestamps/{name}"][row] = value

                for camera_index, camera_name in enumerate(self.camera_names):
                    payload = jpeg_values[camera_name]
                    dataset = self._image_datasets[camera_name]
                    dataset[row, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
                    self._compress_len[camera_index, row] = len(payload)
            except Exception:
                # Keep all time-indexed datasets at the last committed length.
                for dataset in self._frame_datasets:
                    if dataset.name.startswith("/observations/images/"):
                        camera_name = dataset.name.rsplit("/", 1)[-1]
                        old_width = old_image_widths[camera_name]
                        dataset.resize((self._frame_count, old_width))
                    else:
                        dataset.resize((self._frame_count,) + dataset.shape[1:])
                self._compress_len.resize((len(self.camera_names), self._frame_count))
                raise

            committed_index = self._frame_count
            self._frame_count = new_count
            self._last_observation_ns = observation_ns
            self._last_control_ns = control_ns
            self._last_epoch = epoch
            self._frames_since_flush += 1
            self._file.attrs["num_frames"] = self._frame_count
            if self._frames_since_flush >= self.flush_every:
                self.flush()
            return committed_index

    def _prepare_jpegs(
        self,
        images_jpeg: Mapping[str, Union[bytes, bytearray, memoryview, np.ndarray]],
    ) -> Mapping[str, bytes]:
        if set(images_jpeg) != set(self.camera_names):
            missing = sorted(set(self.camera_names) - set(images_jpeg))
            extra = sorted(set(images_jpeg) - set(self.camera_names))
            raise ValueError(f"images_jpeg camera mismatch; missing={missing}, extra={extra}")
        result = {}
        for camera_name in self.camera_names:
            value = images_jpeg[camera_name]
            if isinstance(value, np.ndarray):
                if value.dtype != np.uint8 or value.ndim != 1:
                    raise ValueError(f"JPEG for {camera_name} must be a 1-D uint8 array")
                payload = value.tobytes()
            elif isinstance(value, (bytes, bytearray, memoryview)):
                payload = bytes(value)
            else:
                raise TypeError(f"JPEG for {camera_name} must be bytes-like")
            if len(payload) < 4 or not payload.startswith(b"\xff\xd8") or b"\xff\xd9" not in payload[-16:]:
                raise ValueError(f"JPEG for {camera_name} lacks JPEG SOI/EOI markers")
            if len(payload) > np.iinfo(np.int32).max:
                raise ValueError(f"JPEG for {camera_name} is too large")
            result[camera_name] = payload
        return result

    def record_event(
        self,
        event: str,
        *,
        request_ns: Optional[int] = None,
        gate_ns: Optional[int] = None,
        ack_ns: Optional[int] = None,
        frame: Optional[int] = None,
        epoch: Optional[int] = None,
        detail: str = "",
    ) -> int:
        """Append an event row.

        For handoff events, pass all three timestamps.  ``frame`` is the first
        frame in the newly active mode, and ``epoch`` is that mode's epoch.
        Missing fields use ``-1`` for non-handoff events.
        """

        with self._lock:
            self._ensure_open()
            event_name = str(event).strip().upper()
            if not event_name:
                raise ValueError("event must not be empty")
            values = [request_ns, gate_ns, ack_ns]
            if event_name in HANDOFF_EVENT_TYPES and any(value is None for value in values):
                raise ValueError("handoff events require request_ns, gate_ns and ack_ns")
            normalized_times = [-1 if value is None else int(value) for value in values]
            present_times = [value for value in normalized_times if value >= 0]
            if any(value <= 0 for value in present_times):
                raise ValueError("event timestamps must be positive")
            if present_times != sorted(present_times):
                raise ValueError("event timestamps must satisfy request <= gate <= ack")
            frame_value = self._frame_count if frame is None else int(frame)
            epoch_value = self._last_epoch if epoch is None else int(epoch)
            if frame_value < -1:
                raise ValueError("event frame must be -1 or non-negative")
            if epoch_value < -1:
                raise ValueError("event epoch must be -1 or non-negative")

            row = np.zeros((), dtype=_EVENT_DTYPE)
            row["event"] = _encode_fixed_utf8(event_name, 48, "event")
            row["request_ns"] = normalized_times[0]
            row["gate_ns"] = normalized_times[1]
            row["ack_ns"] = normalized_times[2]
            row["frame"] = frame_value
            row["epoch"] = epoch_value
            row["detail"] = _encode_fixed_utf8(detail, 256, "detail")
            index = int(self._events.shape[0])
            self._events.resize((index + 1,))
            self._events[index] = row
            return index

    def flush(self) -> None:
        with self._lock:
            self._ensure_open()
            self._file.attrs["num_frames"] = self._frame_count
            self._file.flush()
            self._frames_since_flush = 0

    def finalize(self, *, validate: bool = True) -> Path:
        """Validate, close, and atomically publish the completed episode."""

        with self._lock:
            self._ensure_open()
            if self._frame_count == 0:
                raise ValueError("cannot finalize an empty episode")
            supervision = np.zeros(self._frame_count, dtype=np.bool_)
            if self._frame_count > 1:
                modes = self._file["/dagger/control_mode"][:]
                supervision[:-1] = modes[:-1] == int(ControlMode.HUMAN)
            self._file["/dagger/supervision_valid"][:] = supervision
            self._file.attrs["num_frames"] = self._frame_count
            self._file.attrs["finalized"] = True
            self._file.attrs["finalized_utc"] = _utc_now()
            self._file.flush()
            self._file.close()
            self._closed = True
            self._fsync_file(self.partial_path)

            if validate:
                try:
                    from .validate_dagger_episode import validate_episode
                except ImportError:
                    from validate_dagger_episode import validate_episode  # type: ignore

                result = validate_episode(self.partial_path, require_finalized=True)
                if not result.valid:
                    quarantine_path = self.quarantine("validation_failed")
                    raise EpisodeValidationError(result.errors, quarantine_path)

            if self.final_path.exists():
                raise FileExistsError(f"final episode already exists: {self.final_path}")
            os.replace(self.partial_path, self.final_path)
            self._fsync_directory(self.output_dir)
            return self.final_path

    def quarantine(self, reason: str) -> Path:
        """Close and atomically move a failed partial out of the dataset set."""

        with self._lock:
            reason_text = str(reason).strip() or "unknown"
            if not self._closed:
                self._file.attrs["finalized"] = False
                self._file.attrs["quarantine_reason"] = reason_text
                self._file.attrs["quarantined_utc"] = _utc_now()
                self._file.attrs["num_frames"] = self._frame_count
                self._file.flush()
                self._file.close()
                self._closed = True
            elif self.partial_path.exists():
                try:
                    with h5py.File(self.partial_path, "r+") as root:
                        root.attrs["finalized"] = False
                        root.attrs["quarantine_reason"] = reason_text
                        root.attrs["quarantined_utc"] = _utc_now()
                        root.flush()
                except OSError:
                    # Preserve even a structurally damaged partial for diagnosis.
                    pass

            if not self.partial_path.exists():
                raise FileNotFoundError(f"partial episode is unavailable: {self.partial_path}")
            safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason_text)[:64] or "unknown"
            quarantine_dir = self.output_dir / "quarantine"
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            destination = quarantine_dir / (
                f"{self.episode_stem}.{time.time_ns()}.{safe_reason}.partial.hdf5"
            )
            os.replace(self.partial_path, destination)
            self._fsync_directory(quarantine_dir)
            return destination

    def discard(self) -> None:
        """Close and permanently remove the current partial episode."""

        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True
            try:
                self.partial_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _fsync_file(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)


__all__ = [
    "ControlMode",
    "DEFAULT_CAMERA_NAMES",
    "EpisodeValidationError",
    "EventType",
    "HumanDaggerRecorder",
    "SCHEMA_VERSION",
    "SOURCE_TIMESTAMP_NAMES",
    "normalise_control_mode",
]
