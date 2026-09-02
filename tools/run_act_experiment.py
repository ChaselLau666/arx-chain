#!/usr/bin/env python3
"""Validate an inclusive HDF5 range and launch one reproducible ACT run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "act"
sys.path.insert(0, str(ACT_ROOT))

from act_contract import (  # noqa: E402
    CAMERA_NAMES,
    EXPECTED_HEIGHT_COMMAND,
    PHYSICAL_ACTION_DIM,
    SOURCE_ACTION_SEMANTICS,
    SOURCE_FPS,
    sha256,
)


def episode_path(source_dir: Path, episode: int) -> Path:
    return source_dir / f"episode_{episode}.hdf5"


def selected_episodes(start: int, end: int, eval_episode: int) -> list[int]:
    if start < 0 or end < start:
        raise ValueError("require 0 <= start <= end")
    selected = list(range(start, end + 1))
    if eval_episode in selected:
        raise ValueError("--eval-episode must not be inside the training range")
    return selected


def inspect_episode(path: Path, source_index: int, include_hash: bool = True) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as root:
        required = (
            "observations/qpos",
            "observations/qvel",
            "observations/effort",
            "observations/eef",
            "observations/robot_base",
            "observations/base_velocity",
            "action",
            "action_eef",
            "action_base",
            "action_velocity",
        )
        missing = [key for key in required if key not in root]
        if missing:
            raise ValueError(f"{path.name}: missing fields {missing}")
        qpos = np.asarray(root["observations/qpos"])
        action = np.asarray(root["action"])
        frames = len(qpos)
        if frames < 2 or qpos.shape != (frames, PHYSICAL_ACTION_DIM):
            raise ValueError(f"{path.name}: expected qpos shape (T, 14), got {qpos.shape}")
        if action.shape != qpos.shape:
            raise ValueError(f"{path.name}: action shape {action.shape} != qpos shape {qpos.shape}")
        if not np.isfinite(qpos).all() or not np.isfinite(action).all():
            raise ValueError(f"{path.name}: qpos/action contains NaN or Inf")
        joints = [index for index in range(14) if index not in (6, 13)]
        if not np.array_equal(action[:, joints], qpos[:, joints]):
            raise ValueError(f"{path.name}: source joint action is not current qpos")
        for gripper in (6, 13):
            expected = np.where(qpos[:, gripper] > -2.1, 0.0, qpos[:, gripper])
            if not np.array_equal(action[:, gripper], expected):
                raise ValueError(f"{path.name}: gripper threshold semantics mismatch")
        height = float(root.attrs.get("height_command", np.nan))
        if not np.isclose(height, EXPECTED_HEIGHT_COMMAND):
            raise ValueError(f"{path.name}: height_command={height}, expected {EXPECTED_HEIGHT_COMMAND}")
        lengths = {key: len(root[key]) for key in required}
        image_shapes = {}
        for camera in CAMERA_NAMES:
            key = f"observations/images/{camera}"
            if key not in root or len(root[key]) != frames:
                raise ValueError(f"{path.name}: missing or misaligned camera {camera}")
            image = cv2.imdecode(np.asarray(root[key][frames // 2], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"{path.name}: cannot decode {camera} image")
            image_shapes[camera] = list(image.shape)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name}: non-image field lengths disagree: {lengths}")
        base_max = max(
            float(np.max(np.abs(np.asarray(root[key]))))
            for key in (
                "observations/robot_base",
                "observations/base_velocity",
                "action_base",
                "action_velocity",
            )
        )
        if base_max != 0.0:
            raise ValueError(f"{path.name}: base data is not zero (max={base_max})")
        result = {
            "source_episode": source_index,
            "source_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "frames": frames,
            "task": str(root.attrs.get("task", "")),
            "height_command": height,
            "image_shapes": image_shapes,
            "source_action_semantics": SOURCE_ACTION_SEMANTICS,
        }
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def build_manifest(args, selected: list[int]) -> dict:
    episodes = []
    for local_index, source_index in enumerate(selected):
        info = inspect_episode(
            episode_path(args.source_dir, source_index), source_index, not args.skip_hash
        )
        info["local_episode"] = local_index
        episodes.append(info)
    evaluation = inspect_episode(
        episode_path(args.source_dir, args.eval_episode), args.eval_episode, not args.skip_hash
    )
    return {
        "manifest_version": "arx_act_range_v1",
        "repo_commit": git_commit(),
        "source_dir": str(args.source_dir.resolve()),
        "start": args.start,
        "end": args.end,
        "num_episodes": len(selected),
        "eval_episode": args.eval_episode,
        "source_fps": SOURCE_FPS,
        "height_command": EXPECTED_HEIGHT_COMMAND,
        "episodes": episodes,
        "evaluation": evaluation,
    }


def prepare_view(view_dir: Path, manifest: dict) -> Path:
    view_dir.mkdir(parents=True, exist_ok=True)
    for info in manifest["episodes"]:
        link = view_dir / f"episode_{info['local_episode']}.hdf5"
        source = Path(info["source_path"])
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source:
                raise FileExistsError(f"conflicting dataset view entry: {link}")
        else:
            link.symlink_to(source)
    manifest_path = view_dir / "split_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "repo_commit"}
        comparable_current = {key: value for key, value in manifest.items() if key != "repo_commit"}
        if comparable_existing != comparable_current:
            raise ValueError(f"existing manifest differs: {manifest_path}")
        if existing != manifest:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return manifest_path


def train_command(args, view_dir: Path, manifest_path: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ACT_ROOT / "train.py"),
        "--datasets", str(view_dir),
        "--ckpt_dir", str(run_dir),
        "--ckpt_name", "policy_best.ckpt",
        "--dataset_manifest", str(manifest_path),
        "--num_episodes", "-1",
        "--policy_class", "ACT",
        "--camera_names", *CAMERA_NAMES,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--seed", str(args.seed),
        "--lr", "4e-5",
        "--lr_backbone", "4e-5",
        "--weight_decay", "1e-4",
        "--loss_function", "l1",
        "--backbone", "resnet18",
        "--chunk_size", "30",
        "--hidden_dim", "512",
        "--enc_layers", "4",
        "--dec_layers", "7",
        "--nheads", "8",
        "--dropout", "0.1",
        "--kl_weight", "10",
        "--dim_feedforward", "3200",
        "--arm_delay_time", "0",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="inclusive source episode")
    parser.add_argument("--end", type=int, required=True, help="inclusive source episode")
    parser.add_argument("--eval-episode", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, default=ACT_ROOT / "datasets")
    parser.add_argument("--view-root", type=Path, default=ACT_ROOT / "dataset_views")
    parser.add_argument("--run-root", type=Path, default=ACT_ROOT / "runs")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-hash", action="store_true", help="tests only; omit SHA-256")
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.view_root = args.view_root.resolve()
    args.run_root = args.run_root.resolve()
    if args.epochs <= 0 or args.batch_size <= 0 or args.checkpoint_interval <= 0:
        parser.error("--epochs, --batch-size, and --checkpoint-interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    selected = selected_episodes(args.start, args.end, args.eval_episode)
    manifest = build_manifest(args, selected)
    range_name = f"ep{args.start:03d}_{args.end:03d}"
    view_dir = args.view_root / range_name
    manifest_path = prepare_view(view_dir, manifest)
    run_name = args.run_name or f"act_{range_name}_seed{args.seed}_epochs{args.epochs}"
    run_dir = args.run_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"refused to overwrite run directory: {run_dir}")
    command = train_command(args, view_dir, manifest_path, run_dir)
    print(json.dumps({
        "view_dir": str(view_dir),
        "run_dir": str(run_dir),
        "num_episodes": len(selected),
        "command": command,
    }, indent=2))
    if args.prepare_only:
        return 0
    subprocess.run(command, cwd=ACT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
