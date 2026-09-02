#!/usr/bin/env python3
"""Offline ACT replay on one held-out HDF5 episode; never imports ROS."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "act"
sys.path.insert(0, str(ACT_ROOT))

from act_contract import (  # noqa: E402
    CAMERA_NAMES,
    EFFECTIVE_ACTION_SEMANTICS,
    JOINT_NAMES,
    PHYSICAL_ACTION_DIM,
    SOURCE_FPS,
)
from utils.policy import ACTPolicy  # noqa: E402


GRIPPER_INDICES = (6, 13)
JOINT_INDICES = tuple(index for index in range(14) if index not in GRIPPER_INDICES)
GRIPPER_CLASS_THRESHOLD = -1.05


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=/path/to/run")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must be LABEL=/path/to/run")
    return label, Path(path).resolve()


def load_episode(path: Path) -> dict:
    with h5py.File(path, "r") as root:
        source_action = np.asarray(root["action"], dtype=np.float32)
        return {
            "qpos": np.asarray(root["observations/qpos"], dtype=np.float32),
            "qvel": np.asarray(root["observations/qvel"], dtype=np.float32),
            "effort": np.asarray(root["observations/effort"], dtype=np.float32),
            "robot_base": np.asarray(root["observations/robot_base"], dtype=np.float32),
            "base_velocity": np.asarray(root["observations/base_velocity"], dtype=np.float32),
            "target": source_action[1:],
            "encoded_images": {
                camera: np.asarray(root[f"observations/images/{camera}"])
                for camera in CAMERA_NAMES
            },
        }


def decoded_images(encoded_images: dict, timestep: int) -> tuple[np.ndarray, dict]:
    images = {}
    tensors = []
    for camera in CAMERA_NAMES:
        image = cv2.imdecode(
            np.asarray(encoded_images[camera][timestep], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError(f"cannot decode {camera} at frame {timestep}")
        images[camera] = image
        tensors.append(np.moveaxis(image, -1, 0))
    batch = torch.from_numpy(np.stack(tensors) / 255.0).float().unsqueeze(0)
    return batch, images


def normalized_state(episode: dict, stats: dict, timestep: int) -> torch.Tensor:
    state = episode["qpos"][timestep]
    normalized = (state - stats["left_states_mean"]) / stats["left_states_std"]
    return torch.from_numpy(normalized).float().unsqueeze(0)


def official_temporal_action(
    chunks: np.ndarray, valid: np.ndarray, timestep: int, decay: float = 0.01
) -> np.ndarray:
    candidates = chunks[valid[:, timestep], timestep]
    if not len(candidates):
        raise RuntimeError(f"no action candidate at timestep {timestep}")
    weights = np.exp(-decay * np.arange(len(candidates), dtype=np.float32))
    weights /= weights.sum()
    return np.sum(candidates * weights[:, None], axis=0)


def evaluate_run(
    label: str, run_dir: Path, episode: dict, episode_name: str, output_root: Path
) -> dict:
    contract_path = run_dir / "data_contract.yaml"
    stats_path = run_dir / "dataset_stats.pkl"
    checkpoint_path = run_dir / "policy_best.ckpt"
    for path in (contract_path, stats_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if contract["effective_action_semantics"] != EFFECTIVE_ACTION_SEMANTICS:
        raise ValueError(f"{label}: action semantics mismatch")
    if contract["physical_action_dim"] != PHYSICAL_ACTION_DIM:
        raise ValueError(f"{label}: physical action dimension mismatch")
    policy_config = contract["policy_config"]
    if int(policy_config["action_dim"]) != 28:
        raise ValueError(f"{label}: official ACT model action dimension must be 28")
    with stats_path.open("rb") as stream:
        stats = pickle.load(stream)
    if np.asarray(stats["action_mean"]).shape != (28,):
        raise ValueError(f"{label}: expected 28-D action normalization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACTPolicy(policy_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    timesteps = len(episode["target"])
    chunk_size = int(policy_config["chunk_size"])
    model_dim = int(policy_config["action_dim"])
    chunks = np.zeros((timesteps, timesteps + chunk_size, model_dim), dtype=np.float32)
    valid = np.zeros((timesteps, timesteps + chunk_size), dtype=bool)
    predictions = np.zeros((timesteps, PHYSICAL_ACTION_DIM), dtype=np.float32)
    run_output = output_root / label
    if run_output.exists():
        raise FileExistsError(f"refused to overwrite evaluation output: {run_output}")
    run_output.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(run_output / f"{episode_name}_openloop.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), SOURCE_FPS, (960, 240),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {episode_name}_openloop.mp4")

    with torch.inference_mode():
        for timestep in range(timesteps):
            image_batch, images = decoded_images(episode["encoded_images"], timestep)
            state = normalized_state(episode, stats, timestep).to(device)
            output = model(
                image_batch.to(device), None, state, state,
                robot_base=None, robot_head=None, base_velocity=None,
            )[0].detach().cpu().numpy()
            chunks[timestep, timestep:timestep + chunk_size] = output
            valid[timestep, timestep:timestep + chunk_size] = True
            normalized_action = official_temporal_action(chunks, valid, timestep)
            physical = normalized_action * stats["action_std"] + stats["action_mean"]
            predictions[timestep] = physical[:PHYSICAL_ACTION_DIM]

            panels = [cv2.resize(images[camera], (320, 240)) for camera in CAMERA_NAMES]
            frame = np.concatenate(panels, axis=1)
            cv2.rectangle(frame, (0, 0), (960, 28), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{label} {episode_name} frame {timestep}/{timesteps - 1}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            writer.write(frame)
    writer.release()

    target = episode["target"]
    error = predictions - target

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for axis, joint_index in zip(axes.flat, JOINT_INDICES):
        axis.plot(target[:, joint_index], label="target state(t+1)")
        axis.plot(predictions[:, joint_index], label="predicted action", alpha=0.8)
        axis.set_title(JOINT_NAMES[joint_index])
        axis.grid(True)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(run_output / f"{episode_name}_joint_overlay.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, gripper_index in zip(axes, GRIPPER_INDICES):
        axis.plot(target[:, gripper_index], label="target state(t+1)")
        axis.plot(predictions[:, gripper_index], label="predicted action", alpha=0.8)
        axis.axhline(GRIPPER_CLASS_THRESHOLD, linestyle="--", color="gray")
        axis.set_title(JOINT_NAMES[gripper_index])
        axis.grid(True)
        axis.legend()
    fig.tight_layout()
    fig.savefig(run_output / f"{episode_name}_gripper_overlay.png", dpi=160)
    plt.close(fig)

    gripper_accuracy = {
        JOINT_NAMES[index]: float(np.mean(
            (predictions[:, index] <= GRIPPER_CLASS_THRESHOLD)
            == (target[:, index] <= GRIPPER_CLASS_THRESHOLD)
        ))
        for index in GRIPPER_INDICES
    }
    metrics = {
        "label": label,
        "episode": episode_name,
        "frames": timesteps,
        "source_fps": SOURCE_FPS,
        "joint_mae": {JOINT_NAMES[i]: float(np.mean(np.abs(error[:, i]))) for i in JOINT_INDICES},
        "joint_rmse": {JOINT_NAMES[i]: float(np.sqrt(np.mean(error[:, i] ** 2))) for i in JOINT_INDICES},
        "joint_max_abs_error": {
            JOINT_NAMES[i]: float(np.max(np.abs(error[:, i]))) for i in JOINT_INDICES
        },
        "gripper_mae": {
            JOINT_NAMES[i]: float(np.mean(np.abs(error[:, i]))) for i in GRIPPER_INDICES
        },
        "gripper_class_accuracy": gripper_accuracy,
        "prediction_max_step": float(np.max(np.abs(np.diff(predictions, axis=0)))),
        "left_prediction_range_max": float(np.max(np.ptp(predictions[:, :7], axis=0))),
        "physical_action_dim": PHYSICAL_ACTION_DIM,
        "model_action_dim": model_dim,
        "effective_action_semantics": EFFECTIVE_ACTION_SEMANTICS,
    }
    (run_output / f"{episode_name}_error.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return metrics


def comparison_plot(metrics: list[dict], episode_name: str, output_root: Path) -> None:
    labels = [item["label"] for item in metrics]
    joint_mae = [float(np.mean(list(item["joint_mae"].values()))) for item in metrics]
    gripper_accuracy = [
        float(np.mean(list(item["gripper_class_accuracy"].values()))) for item in metrics
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, joint_mae)
    axes[0].set_title(f"{episode_name} mean joint MAE")
    axes[0].grid(True, axis="y")
    axes[1].bar(labels, gripper_accuracy)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{episode_name} mean gripper accuracy")
    axes[1].grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / f"{episode_name}_model_comparison.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise FileExistsError(f"refused to overwrite output directory: {output_root}")
    output_root.mkdir(parents=True)
    episode_path = args.episode.resolve()
    episode_name = episode_path.stem
    episode = load_episode(episode_path)
    metrics = [
        evaluate_run(label, path, episode, episode_name, output_root)
        for label, path in args.run
    ]
    comparison_plot(metrics, episode_name, output_root)
    (output_root / "summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
