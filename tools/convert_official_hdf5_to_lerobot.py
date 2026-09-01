#!/usr/bin/env python3
"""Convert official ROS2_LIFT_Play HDF5 episodes to LeRobot 0.4.3/v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)
ACTION_DIM = 14
ACTION_SEMANTICS = "official_current_qpos_with_gripper_threshold"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb(encoded) -> np.ndarray:
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def selected_paths(input_dir: Path, start: int, end: int) -> list[Path]:
    if start < 0 or end < start:
        raise ValueError("require 0 <= start <= end")
    paths = [input_dir / f"episode_{index}.hdf5" for index in range(start, end + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing selected episodes: {missing}")
    return paths


def inspect_episode(path: Path) -> dict:
    with h5py.File(path, "r") as root:
        required = [
            "observations/qpos",
            "observations/qvel",
            "observations/effort",
            "observations/eef",
            "observations/images",
            "action",
        ]
        missing = [key for key in required if key not in root]
        if missing:
            raise ValueError(f"{path.name}: missing keys {missing}")
        qpos = root["observations/qpos"]
        action = root["action"]
        frames = len(action)
        if qpos.shape != (frames, ACTION_DIM) or action.shape != (frames, ACTION_DIM):
            raise ValueError(f"{path.name}: expected qpos/action shape (T, 14)")
        if not np.isfinite(qpos[()]).all() or not np.isfinite(action[()]).all():
            raise ValueError(f"{path.name}: qpos/action contains NaN or Inf")
        image_shapes = {}
        for camera in CAMERA_NAMES:
            key = f"observations/images/{camera}"
            if key not in root or len(root[key]) != frames:
                raise ValueError(f"{path.name}: missing or misaligned camera {camera}")
            indices = sorted({0, frames // 2, frames - 1})
            decoded = [decode_rgb(root[key][index]) for index in indices]
            if len({image.shape for image in decoded}) != 1:
                raise ValueError(f"{path.name}: inconsistent decoded shape for {camera}")
            image_shapes[camera] = list(decoded[0].shape)
        base_max = 0.0
        for key in (
            "observations/robot_base",
            "observations/base_velocity",
            "action_base",
            "action_velocity",
        ):
            if key in root:
                base_max = max(base_max, float(np.max(np.abs(root[key][()]))))
        return {
            "file": path.name,
            "sha256": sha256(path),
            "frames": frames,
            "height_command": float(root.attrs["height_command"]) if "height_command" in root.attrs else None,
            "source_task": str(root.attrs.get("task", "")),
            "image_shapes": image_shapes,
            "base_max_abs": base_max,
        }


def validate_selection(paths: list[Path], fps: int, task: str) -> dict:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not task.strip():
        raise ValueError("--task is required because official HDF5 task is empty")
    episodes = [inspect_episode(path) for path in paths]
    shapes = {tuple(info["image_shapes"][camera]) for info in episodes for camera in CAMERA_NAMES}
    if len(shapes) != 1:
        raise ValueError(f"camera decoded shapes disagree: {sorted(shapes)}")
    return {
        "source_format": "official_ros2_lift_play_hdf5",
        "lerobot_version": "0.4.3",
        "lerobot_format": "v3",
        "fps": fps,
        "timestamps_available": False,
        "action_dim": ACTION_DIM,
        "action_semantics": ACTION_SEMANTICS,
        "joint_names": list(JOINT_NAMES),
        "camera_names": list(CAMERA_NAMES),
        "task": task,
        "total_episodes": len(episodes),
        "total_frames": sum(info["frames"] for info in episodes),
        "episodes": episodes,
    }


def convert(args) -> dict:
    paths = selected_paths(args.input, args.start, args.end)
    manifest = validate_selection(paths, args.fps, args.task)
    if args.validate_only:
        return manifest
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    image_shape = tuple(manifest["episodes"][0]["image_shapes"][CAMERA_NAMES[0]])
    features = {
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            }
            for camera in CAMERA_NAMES
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(JOINT_NAMES),
        },
    }

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=args.fps,
        robot_type="ARX_LIFT2s",
        features=features,
        use_videos=True,
    )
    for path in paths:
        with h5py.File(path, "r") as root:
            frames = len(root["action"])
            for index in range(frames):
                frame = {
                    "observation.state": np.asarray(root["observations/qpos"][index], dtype=np.float32),
                    "action": np.asarray(root["action"][index], dtype=np.float32),
                    "timestamp": float(index / args.fps),
                    "task": args.task,
                }
                for camera in CAMERA_NAMES:
                    frame[f"observation.images.{camera}"] = decode_rgb(
                        root[f"observations/images/{camera}"][index]
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
    dataset.finalize()

    sidecar = args.output / "meta" / "arx.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and (args.output is None or not args.repo_id):
        parser.error("conversion requires --output and --repo-id")
    return args


if __name__ == "__main__":
    print(json.dumps(convert(parse_args()), indent=2, ensure_ascii=False))
