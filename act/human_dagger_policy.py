"""Isolated ACT inference worker used by the Human DAgger controller.

The worker never owns a ROS publisher.  Every result is tagged with the episode,
control epoch and observation sequence that produced it, so the control process
can reject a result that completes after a takeover request.
"""

from __future__ import annotations

import argparse
import pickle
import queue
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class PolicyWorkerConfig:
    ckpt_dir: str
    ckpt_name: str = "policy_best.ckpt"
    stats_name: str = "dataset_stats.pkl"
    args_name: str = "args.yaml"
    gripper_gate: float = -1.0
    temporal_agg: bool = True
    max_observation_age_ns: int = 250_000_000


class TemporalAggregator:
    """Small, resettable equivalent of inference.py's all_time_actions array."""

    def __init__(self, chunk_size: int, decay: float = 0.01) -> None:
        self.chunk_size = int(chunk_size)
        self.decay = float(decay)
        self._chunks: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def add(self, timestep: int, chunk: np.ndarray) -> np.ndarray:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"policy chunk must be [steps, action_dim], got {values.shape}")
        self._chunks.append((int(timestep), values))
        while self._chunks and timestep - self._chunks[0][0] >= self.chunk_size:
            self._chunks.popleft()

        candidates = []
        for start, candidate_chunk in self._chunks:
            offset = timestep - start
            if 0 <= offset < candidate_chunk.shape[0]:
                candidates.append(candidate_chunk[offset])
        if not candidates:
            raise RuntimeError("temporal aggregator has no action for current timestep")

        stacked = np.stack(candidates, axis=0)
        weights = np.exp(-self.decay * np.arange(len(candidates), dtype=np.float32))
        weights /= weights.sum()
        return np.sum(stacked * weights[:, None], axis=0)


def _training_defaults() -> dict[str, Any]:
    return {
        "policy_class": "ACT",
        "lr": 4e-5,
        "lr_backbone": 4e-5,
        "weight_decay": 1e-4,
        "loss_function": "l1",
        "backbone": "resnet18",
        "chunk_size": 30,
        "hidden_dim": 512,
        "camera_names": list(CAMERA_NAMES),
        "position_embedding": "sine",
        "masks": False,
        "dilation": False,
        "use_base": False,
        "use_depth_image": False,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "dropout": 0.1,
        "pre_norm": False,
        "kl_weight": 10,
        "dim_feedforward": 3200,
        "use_qvel": False,
        "use_effort": False,
        "use_eef_states": False,
        "use_eef_action": False,
    }


def load_policy_args(ckpt_dir: str, args_name: str = "args.yaml") -> dict[str, Any]:
    result = _training_defaults()
    args_path = Path(ckpt_dir) / args_name
    if args_path.exists():
        with args_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{args_path} must contain a YAML mapping")
        result.update(loaded)

    if result.get("policy_class") != "ACT":
        raise ValueError("Human DAgger MVP supports ACT joint-space checkpoints only")
    if result.get("use_base"):
        raise ValueError("Human DAgger MVP refuses checkpoints that command the base")
    if result.get("use_depth_image"):
        raise ValueError("Human DAgger MVP uses the three RGB cameras only")
    if result.get("use_eef_states") or result.get("use_eef_action"):
        raise ValueError("the existing EEF training path is incomplete; use a joint-space checkpoint")
    if tuple(result.get("camera_names", ())) != CAMERA_NAMES:
        raise ValueError(f"checkpoint camera_names must be {list(CAMERA_NAMES)}")
    return result


def build_policy_config(training_args: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(training_args)
    cfg["states_dim"] = (7 + (7 if cfg["use_qvel"] else 0) + (1 if cfg["use_effort"] else 0)) * 2
    cfg["action_dim"] = 28
    return cfg


def _require_vector(stats: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    if key not in stats:
        raise KeyError(f"checkpoint statistics missing {key}")
    value = np.asarray(stats[key], dtype=np.float32).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"{key} must have shape {(size,)}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains non-finite values")
    return value


class ACTRuntime:
    def __init__(self, config: PolicyWorkerConfig) -> None:
        # Heavy imports are deliberately delayed until the child process starts.
        import torch
        from utils.policy import ACTPolicy

        self.torch = torch
        self.worker_config = config
        self.training_args = load_policy_args(config.ckpt_dir, config.args_name)
        self.policy_config = build_policy_config(self.training_args)
        self.policy = ACTPolicy(self.policy_config)

        ckpt_path = Path(config.ckpt_dir) / config.ckpt_name
        stats_path = Path(config.ckpt_dir) / config.stats_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        self.policy.load_state_dict(state)
        with stats_path.open("rb") as stream:
            self.stats = pickle.load(stream)

        state_size = int(self.policy_config["states_dim"])
        action_size = int(self.policy_config["action_dim"])
        self.left_mean = _require_vector(self.stats, "left_states_mean", state_size)
        self.left_std = _require_vector(self.stats, "left_states_std", state_size)
        self.right_mean = _require_vector(self.stats, "right_states_mean", state_size)
        self.right_std = _require_vector(self.stats, "right_states_std", state_size)
        self.action_mean = _require_vector(self.stats, "action_mean", action_size)
        self.action_std = _require_vector(self.stats, "action_std", action_size)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real ACT rollout")
        self.device = torch.device("cuda")
        self.policy.to(self.device)
        self.policy.eval()
        self.timestep = 0
        self.aggregator = TemporalAggregator(self.policy_config["chunk_size"])

    def reset(self) -> None:
        self.timestep = 0
        self.aggregator.reset()

    def _decode_images(self, encoded: Mapping[str, bytes]):
        images = []
        for camera_name in CAMERA_NAMES:
            payload = encoded.get(camera_name)
            if not payload:
                raise ValueError(f"missing JPEG payload for {camera_name}")
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"invalid JPEG payload for {camera_name}")
            images.append(np.moveaxis(image, -1, 0))
        array = np.stack(images, axis=0).astype(np.float32) / 255.0
        return self.torch.from_numpy(array).unsqueeze(0).to(self.device)

    def _states(self, observation: Mapping[str, Any]):
        source = np.asarray(observation["qpos"], dtype=np.float32).reshape(-1)
        qvel = np.asarray(observation["qvel"], dtype=np.float32).reshape(-1)
        effort = np.asarray(observation["effort"], dtype=np.float32).reshape(-1)
        if source.shape != (14,) or qvel.shape != (14,) or effort.shape != (14,):
            raise ValueError("qpos, qvel and effort must each contain 14 values")
        left = source[:7]
        right = source[7:14]
        if self.policy_config["use_qvel"]:
            left = np.concatenate((left, qvel[:7]))
            right = np.concatenate((right, qvel[7:14]))
        if self.policy_config["use_effort"]:
            left = np.concatenate((left, effort[6:7]))
            right = np.concatenate((right, effort[13:14]))
        combined = np.concatenate((left, right))
        left_norm = (combined - self.left_mean) / self.left_std
        right_norm = (combined - self.right_mean) / self.right_std
        left_tensor = self.torch.from_numpy(left_norm).float().unsqueeze(0).to(self.device)
        right_tensor = self.torch.from_numpy(right_norm).float().unsqueeze(0).to(self.device)
        return left_tensor, right_tensor

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        torch = self.torch
        image = self._decode_images(observation["images_jpeg"])
        left, right = self._states(observation)
        robot_base = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        robot_head = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        base_velocity = torch.zeros((1, 4), dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            chunk = self.policy(
                image,
                None,
                left,
                right,
                robot_base=robot_base,
                robot_head=robot_head,
                base_velocity=base_velocity,
            )[0].detach().cpu().numpy()

        if self.worker_config.temporal_agg:
            raw = self.aggregator.add(self.timestep, chunk)
        else:
            raw = chunk[self.timestep % len(chunk)]
        denormalized = raw * self.action_std + self.action_mean
        action = np.asarray(denormalized[:14], dtype=np.float64)
        if not np.all(np.isfinite(action)):
            raise ValueError("policy produced a non-finite action")
        gate = self.worker_config.gripper_gate
        if gate != -1:
            action[6] = 0.0 if action[6] < gate else 5.0
            action[13] = 0.0 if action[13] < gate else 5.0
        self.timestep += 1
        return action


def policy_worker_main(
    worker_config: PolicyWorkerConfig,
    control_queue: Any,
    observation_queue: Any,
    result_queue: Any,
    status_queue: Any,
) -> None:
    """Multiprocessing entry point. Messages are intentionally plain dicts."""

    try:
        runtime = ACTRuntime(worker_config)
        status_queue.put({"kind": "policy_ready", "time_ns": time.monotonic_ns()})
    except BaseException as exc:  # propagate initialization failures to the safety process
        status_queue.put({"kind": "policy_error", "error": repr(exc), "time_ns": time.monotonic_ns()})
        return

    active_epoch: int | None = None
    action_seq = 0
    while True:
        try:
            message = control_queue.get_nowait()
        except queue.Empty:
            try:
                message = observation_queue.get(timeout=0.02)
            except queue.Empty:
                continue
        kind = message.get("kind")
        if kind == "stop":
            return
        if kind == "pause":
            active_epoch = None
            continue
        if kind == "reset":
            runtime.reset()
            active_epoch = int(message["control_epoch"])
            status_queue.put(
                {
                    "kind": "policy_reset_ack",
                    "control_epoch": active_epoch,
                    "time_ns": time.monotonic_ns(),
                }
            )
            continue
        if kind != "observation" or active_epoch is None:
            continue
        if int(message["control_epoch"]) != active_epoch:
            continue

        observation_ns = int(message["observation"]["timestamps"]["observation_ns"])
        try:
            policy_basis_ns = int(message["observation"]["policy_basis_ns"])
        except KeyError:
            policy_basis_ns = observation_ns
        if time.monotonic_ns() - policy_basis_ns > worker_config.max_observation_age_ns:
            # A bounded queue may contain the frame that arrived while the prior
            # CUDA forward was running. Drop it and let the producer supply a
            # current frame instead of acting on stale robot state.
            status_queue.put(
                {
                    "kind": "policy_observation_dropped",
                    "control_epoch": active_epoch,
                    "time_ns": time.monotonic_ns(),
                }
            )
            continue

        try:
            action = runtime.infer(message["observation"])
            action_seq += 1
            result = {
                "kind": "policy_action",
                "episode_id": int(message["episode_id"]),
                "control_epoch": active_epoch,
                "observation_seq": int(message["observation_seq"]),
                "action_seq": action_seq,
                "generated_ns": time.monotonic_ns(),
                "observation_ns": observation_ns,
                "policy_basis_ns": policy_basis_ns,
                "action": action,
            }
            result_queue.put(result)
        except Exception as exc:
            status_queue.put(
                {
                    "kind": "policy_error",
                    "control_epoch": active_epoch,
                    "error": repr(exc),
                    "time_ns": time.monotonic_ns(),
                }
            )
            active_epoch = None


def mock_policy_worker_main(
    control_queue: Any,
    observation_queue: Any,
    result_queue: Any,
    status_queue: Any,
    delay_seconds: float = 0.0,
) -> None:
    """Hardware-free worker used by integration tests and topic dry-runs."""

    status_queue.put({"kind": "policy_ready", "mock": True, "time_ns": time.monotonic_ns()})
    active_epoch: int | None = None
    action_seq = 0
    while True:
        try:
            message = control_queue.get_nowait()
        except queue.Empty:
            try:
                message = observation_queue.get(timeout=0.02)
            except queue.Empty:
                continue
        kind = message.get("kind")
        if kind == "stop":
            return
        if kind == "pause":
            active_epoch = None
            continue
        if kind == "reset":
            active_epoch = int(message["control_epoch"])
            status_queue.put(
                {
                    "kind": "policy_reset_ack",
                    "control_epoch": active_epoch,
                    "time_ns": time.monotonic_ns(),
                }
            )
            continue
        if kind != "observation" or active_epoch is None:
            continue
        if int(message["control_epoch"]) != active_epoch:
            continue
        observation_ns = int(message["observation"]["timestamps"]["observation_ns"])
        try:
            policy_basis_ns = int(message["observation"]["policy_basis_ns"])
        except KeyError:
            policy_basis_ns = observation_ns
        if delay_seconds:
            time.sleep(delay_seconds)
        observation = message["observation"]
        action = np.asarray(observation["qpos"], dtype=np.float64).reshape(-1)
        if action.shape != (14,):
            status_queue.put(
                {
                    "kind": "policy_error",
                    "control_epoch": active_epoch,
                    "error": "mock qpos shape",
                    "time_ns": time.monotonic_ns(),
                }
            )
            active_epoch = None
            continue
        action_seq += 1
        result_queue.put(
            {
                "kind": "policy_action",
                "episode_id": int(message["episode_id"]),
                "control_epoch": active_epoch,
                "observation_seq": int(message["observation_seq"]),
                "action_seq": action_seq,
                "generated_ns": time.monotonic_ns(),
                "observation_ns": observation_ns,
                "policy_basis_ns": policy_basis_ns,
                "action": action,
            }
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an ACT checkpoint for Human DAgger")
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="policy_best.ckpt")
    parser.add_argument("--stats-name", default="dataset_stats.pkl")
    parser.add_argument("--args-name", default="args.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ACTRuntime(
        PolicyWorkerConfig(
            ckpt_dir=args.ckpt_dir,
            ckpt_name=args.ckpt_name,
            stats_name=args.stats_name,
            args_name=args.args_name,
        )
    )
    print("Human DAgger policy preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
