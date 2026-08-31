#!/usr/bin/env python3
"""Convert ARX HDF5 v2 episodes into LeRobot 0.4.3/v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "act"
sys.path.insert(0, str(ACT_ROOT))

from pipeline_contract import ACTION_DIM, ACTION_SEMANTICS, CAMERA_NAMES, FPS, JOINT_NAMES
from pipeline_contract import validate_dataset_contract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb(encoded) -> np.ndarray:
    with Image.open(BytesIO(np.asarray(encoded, dtype=np.uint8).tobytes())) as image:
        return np.asarray(image.convert("RGB"))


def convert(args) -> Path:
    contract = validate_dataset_contract(args.input)
    paths = sorted(args.input.glob("episode_*.hdf5"))
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    with h5py.File(paths[0], "r") as root:
        first_image = decode_rgb(root[f"observations/images/{CAMERA_NAMES[0]}"][0])
    height, width, channels = first_image.shape
    features = {
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": (height, width, channels),
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
        fps=FPS,
        robot_type="ARX_LIFT2s",
        features=features,
        use_videos=True,
    )
    sources = []
    for path in paths:
        with h5py.File(path, "r") as root:
            task = str(root.attrs["task_instruction"])
            qpos = root["observations/qpos"]
            actions = root["action"]
            timestamps = root["timestamps/sample_monotonic_ns"][()]
            for index in range(len(actions)):
                frame = {
                    "observation.state": np.asarray(qpos[index], dtype=np.float32),
                    "action": np.asarray(actions[index], dtype=np.float32),
                    # LeRobot's canonical timestamp is the exact frame grid.
                    # The measured monotonic timestamps remain in HDF5/meta.
                    "timestamp": float(index / FPS),
                    "task": task,
                }
                for camera in CAMERA_NAMES:
                    frame[f"observation.images.{camera}"] = decode_rgb(
                        root[f"observations/images/{camera}"][index]
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
            sources.append(
                {
                    "file": path.name,
                    "sha256": sha256(path),
                    "frames": len(actions),
                    "first_state": np.asarray(qpos[0]).tolist(),
                    "last_action": np.asarray(actions[-1]).tolist(),
                }
            )
    dataset.finalize()
    arx_meta = args.output / "meta" / "arx.json"
    arx_meta.parent.mkdir(parents=True, exist_ok=True)
    arx_meta.write_text(
        json.dumps(
            {
                **contract,
                "lerobot_version": "0.4.3",
                "lerobot_format": "v3",
                "repo_id": args.repo_id,
                "sources": sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return args.output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(convert(parse_args()))
