"""Streaming HDF5 v2 writer and offline validation."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from pipeline_contract import (
    ACTION_DIM,
    ACTION_SEMANTICS,
    CAMERA_NAMES,
    FPS,
    JOINT_NAMES,
    SCHEMA_VERSION,
)


HEIGHT_TOLERANCE = 0.05
WHEEL_SPEED_TOLERANCE = 0.05
MAX_CAMERA_SKEW_NS = 20_000_000
MAX_DROP_RATIO = 0.01


class EpisodeValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeSample:
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    eef: np.ndarray
    images: Mapping[str, bytes]
    camera_timestamp_ns: Mapping[str, int]
    arm_timestamp_ns: Mapping[str, int]
    sample_monotonic_ns: int
    body_information: np.ndarray  # height, waist, head_yaw, head_pitch
    wheel_velocity: np.ndarray


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class EpisodeWriter:
    def __init__(
        self,
        dataset_dir: str | Path,
        task_name: str,
        task_instruction: str,
        expected_height: float,
        repo_root: str | Path,
    ) -> None:
        if not task_name.strip() or not task_instruction.strip():
            raise ValueError("task_name and task_instruction are required")
        self.dataset_dir = Path(dataset_dir)
        self.pending_dir = self.dataset_dir / ".pending"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.pending_dir / f"{uuid.uuid4().hex}.hdf5.partial"
        self._root = h5py.File(self.path, "w", libver="latest")
        self._size = 0
        self._closed = False
        self._attempted_ticks = 0
        self._dropped_ticks = 0
        self._create(task_name, task_instruction, expected_height, Path(repo_root))

    def _create(self, task_name: str, task_instruction: str, expected_height: float, repo_root: Path) -> None:
        root = self._root
        root.attrs.update(
            {
                "sim": False,
                "task": task_name,
                "task_name": task_name,
                "task_instruction": task_instruction,
                "schema_version": SCHEMA_VERSION,
                "fps": FPS,
                "frame_rate": FPS,
                "action_dim": ACTION_DIM,
                "action_semantics": ACTION_SEMANTICS,
                "action_offset_frames": 1,
                "joint_names": json.dumps(JOINT_NAMES),
                "joint_units": json.dumps(["rad"] * 12 + ["sdk_gripper_unit"] * 2),
                "camera_names": json.dumps(CAMERA_NAMES),
                "camera_storage": "ros_compressed_jpeg_bytes",
                "robot_type": "ARX_LIFT2s",
                "git_commit": _git_commit(repo_root),
                "height_target": float(expected_height),
                "height_locked": True,
            }
        )
        observations = root.create_group("observations")
        for name in ("qpos", "qvel", "effort", "eef"):
            observations.create_dataset(name, (0, ACTION_DIM), maxshape=(None, ACTION_DIM), dtype="f4")
        observations.create_dataset("robot_base", (0, 6), maxshape=(None, 6), dtype="f4")
        observations.create_dataset("base_velocity", (0, 4), maxshape=(None, 4), dtype="f4")
        images = observations.create_group("images")
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        for camera in CAMERA_NAMES:
            images.create_dataset(camera, (0,), maxshape=(None,), dtype=jpeg_dtype)

        root.create_dataset("action", (0, ACTION_DIM), maxshape=(None, ACTION_DIM), dtype="f4")
        root.create_dataset("action_eef", (0, ACTION_DIM), maxshape=(None, ACTION_DIM), dtype="f4")
        root.create_dataset("action_base", (0, 6), maxshape=(None, 6), dtype="f4")
        root.create_dataset("action_velocity", (0, 4), maxshape=(None, 4), dtype="f4")

        timestamps = root.create_group("timestamps")
        timestamps.create_dataset("sample_monotonic_ns", (0,), maxshape=(None,), dtype="i8")
        timestamps.create_dataset("action_sample_monotonic_ns", (0,), maxshape=(None,), dtype="i8")
        camera_group = timestamps.create_group("camera_ros_ns")
        for camera in CAMERA_NAMES:
            camera_group.create_dataset(camera, (0,), maxshape=(None,), dtype="i8")
        arm_group = timestamps.create_group("arm_ros_ns")
        for arm in ("left", "right"):
            arm_group.create_dataset(arm, (0,), maxshape=(None,), dtype="i8")

        diagnostics = root.create_group("diagnostics")
        diagnostics.create_dataset("body_information", (0, 4), maxshape=(None, 4), dtype="f4")
        diagnostics.create_dataset("wheel_velocity", (0, 4), maxshape=(None, 4), dtype="f4")

    def set_sampling_stats(self, attempted_ticks: int, dropped_ticks: int) -> None:
        self._attempted_ticks = int(attempted_ticks)
        self._dropped_ticks = int(dropped_ticks)

    @staticmethod
    def _append(dataset: h5py.Dataset, value) -> None:
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def append_transition(self, observation: EpisodeSample, next_sample: EpisodeSample) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        for value in (observation.qpos, observation.qvel, observation.effort, observation.eef, next_sample.qpos):
            if np.asarray(value).shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise EpisodeValidationError("state/action must be finite 14-vectors")

        root = self._root
        obs = root["observations"]
        for name in ("qpos", "qvel", "effort", "eef"):
            self._append(obs[name], np.asarray(getattr(observation, name), dtype=np.float32))
        body = np.asarray(observation.body_information, dtype=np.float32)
        wheels = np.asarray(observation.wheel_velocity, dtype=np.float32)
        robot_base = np.array([0.0, 0.0, 0.0, body[0], body[3], body[2]], dtype=np.float32)
        self._append(obs["robot_base"], robot_base)
        self._append(obs["base_velocity"], wheels)
        for camera in CAMERA_NAMES:
            self._append(obs["images"][camera], np.frombuffer(observation.images[camera], dtype=np.uint8))

        self._append(root["action"], np.asarray(next_sample.qpos, dtype=np.float32))
        self._append(root["action_eef"], np.asarray(next_sample.eef, dtype=np.float32))
        self._append(root["action_base"], np.zeros(6, dtype=np.float32))
        self._append(root["action_velocity"], np.zeros(4, dtype=np.float32))

        ts = root["timestamps"]
        self._append(ts["sample_monotonic_ns"], observation.sample_monotonic_ns)
        self._append(ts["action_sample_monotonic_ns"], next_sample.sample_monotonic_ns)
        for camera in CAMERA_NAMES:
            self._append(ts["camera_ros_ns"][camera], observation.camera_timestamp_ns[camera])
        for arm in ("left", "right"):
            self._append(ts["arm_ros_ns"][arm], observation.arm_timestamp_ns[arm])
        self._append(root["diagnostics/body_information"], body)
        self._append(root["diagnostics/wheel_velocity"], wheels)
        self._size += 1
        if self._size % 30 == 0:
            root.flush()

    def finalize(self) -> dict:
        if self._closed:
            raise RuntimeError("writer is already closed")
        self._root.attrs["attempted_ticks"] = self._attempted_ticks
        self._root.attrs["dropped_ticks"] = self._dropped_ticks
        self._root.flush()
        self._root.close()
        self._closed = True
        summary = validate_episode(self.path)
        with h5py.File(self.path, "r+") as root:
            root.attrs["validation_summary"] = json.dumps(summary, sort_keys=True)
            root.attrs["body_motion_valid"] = bool(summary["body_motion_valid"])
        return summary

    def save_as(self, final_path: str | Path) -> Path:
        if not self._closed:
            raise RuntimeError("finalize before save")
        final_path = Path(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise FileExistsError(final_path)
        os.replace(self.path, final_path)
        return final_path

    def discard(self) -> None:
        if not self._closed:
            self._root.close()
            self._closed = True
        self.path.unlink(missing_ok=True)


def validate_episode(path: str | Path) -> dict:
    errors: list[str] = []
    with h5py.File(path, "r") as root:
        if root.attrs.get("schema_version") != SCHEMA_VERSION:
            errors.append("wrong schema_version")
        if int(root.attrs.get("fps", 0)) != FPS:
            errors.append("wrong fps")
        if root.attrs.get("action_semantics") != ACTION_SEMANTICS:
            errors.append("wrong action_semantics")
        if not str(root.attrs.get("task_instruction", "")).strip():
            errors.append("empty task instruction")
        action = root["action"][()]
        qpos = root["observations/qpos"][()]
        frame_count = len(action)
        if frame_count < 2:
            errors.append("episode has fewer than 2 transitions")
        if action.shape != (frame_count, ACTION_DIM) or qpos.shape != (frame_count, ACTION_DIM):
            errors.append("state/action shape mismatch")
        if not np.isfinite(action).all() or not np.isfinite(qpos).all():
            errors.append("state/action contains NaN or Inf")
        lengths = [len(root["observations/images"][camera]) for camera in CAMERA_NAMES]
        if any(length != frame_count for length in lengths):
            errors.append("camera length mismatch")

        sample_ns = root["timestamps/sample_monotonic_ns"][()]
        observed_fps = 0.0
        if len(sample_ns) > 1 and sample_ns[-1] > sample_ns[0]:
            observed_fps = (len(sample_ns) - 1) * 1e9 / float(sample_ns[-1] - sample_ns[0])
            if not 29.5 <= observed_fps <= 30.5:
                errors.append(f"observed fps {observed_fps:.3f} outside [29.5, 30.5]")

        attempted = int(root.attrs.get("attempted_ticks", frame_count + 1))
        dropped = int(root.attrs.get("dropped_ticks", 0))
        drop_ratio = dropped / max(attempted, 1)
        if drop_ratio > MAX_DROP_RATIO:
            errors.append(f"drop ratio {drop_ratio:.3%} exceeds {MAX_DROP_RATIO:.1%}")

        camera_stamps = np.stack(
            [root[f"timestamps/camera_ros_ns/{camera}"][()] for camera in CAMERA_NAMES], axis=1
        )
        max_camera_skew_ns = int(np.max(np.ptp(camera_stamps, axis=1))) if frame_count else 0
        if max_camera_skew_ns > MAX_CAMERA_SKEW_NS:
            errors.append("camera skew exceeds 20 ms")
        for index, camera in enumerate(CAMERA_NAMES):
            if len(np.unique(camera_stamps[:, index])) != frame_count:
                errors.append(f"duplicate {camera} timestamps")

        body = root["diagnostics/body_information"][()]
        wheels = root["diagnostics/wheel_velocity"][()]
        height_delta = float(np.max(np.abs(body[:, 0] - body[0, 0]))) if frame_count else 0.0
        max_wheel_speed = float(np.max(np.abs(wheels))) if frame_count else 0.0
        body_motion_valid = height_delta <= HEIGHT_TOLERANCE and max_wheel_speed <= WHEEL_SPEED_TOLERANCE
        if not body_motion_valid:
            errors.append("body height or wheel velocity changed during episode")

    if errors:
        raise EpisodeValidationError("; ".join(errors))
    return {
        "frames": frame_count,
        "observed_fps": observed_fps,
        "drop_ratio": drop_ratio,
        "max_camera_skew_ms": max_camera_skew_ns / 1e6,
        "height_delta": height_delta,
        "max_wheel_speed": max_wheel_speed,
        "body_motion_valid": body_motion_valid,
    }


def next_episode_path(dataset_dir: str | Path, requested_index: int = -1) -> Path:
    dataset_dir = Path(dataset_dir)
    if requested_index >= 0:
        candidate = dataset_dir / f"episode_{requested_index}.hdf5"
        if candidate.exists():
            raise FileExistsError(candidate)
        return candidate
    indices = []
    for path in dataset_dir.glob("episode_*.hdf5"):
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return dataset_dir / f"episode_{max(indices, default=-1) + 1}.hdf5"
