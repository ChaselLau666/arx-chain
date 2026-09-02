"""Shared ACT training/inference contract for the ARX LIFT2s."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)
PHYSICAL_STATE_DIM = 14
PHYSICAL_ACTION_DIM = 14
SOURCE_FPS = 60
ACTION_OFFSET_FRAMES = 1
SOURCE_ACTION_SEMANTICS = "official_current_qpos_with_gripper_threshold"
EFFECTIVE_ACTION_SEMANTICS = "state_t_plus_1_with_gripper_threshold"
AUXILIARY_ACTION_SEMANTICS = "official_auxiliary_zero_channels"
EXPECTED_HEIGHT_COMMAND = 15.5


def base_policy_config(args) -> dict:
    return {
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "loss_function": args.loss_function,
        "backbone": args.backbone,
        "chunk_size": args.chunk_size,
        "hidden_dim": args.hidden_dim,
        "camera_names": list(args.camera_names),
        "position_embedding": args.position_embedding,
        "masks": args.masks,
        "dilation": args.dilation,
        "use_base": args.use_base,
        "use_depth_image": args.use_depth_image,
    }


def build_act_policy_config(args) -> dict:
    physical_action_dim = PHYSICAL_ACTION_DIM + (10 if args.use_base else 0)
    per_arm_state_dim = 7
    if args.use_qvel:
        per_arm_state_dim += 7
    if args.use_effort:
        per_arm_state_dim += 1
    return {
        **base_policy_config(args),
        "policy_class": "ACT",
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "dropout": args.dropout,
        "pre_norm": args.pre_norm,
        "states_dim": per_arm_state_dim * 2,
        "physical_action_dim": physical_action_dim,
        "auxiliary_action_dim": physical_action_dim,
        "action_dim": physical_action_dim * 2,
        "kl_weight": args.kl_weight,
        "dim_feedforward": args.dim_feedforward,
        "use_qvel": args.use_qvel,
        "use_effort": args.use_effort,
        "use_eef_states": args.use_eef_states,
        "use_eef_action": getattr(args, "use_eef_action", False),
    }


def effective_actions(source_actions: np.ndarray, add_auxiliary: bool = True) -> np.ndarray:
    """Map source action(t) to target action(t)=source action(t+1)."""
    actions = np.asarray(source_actions)
    if actions.ndim != 2 or len(actions) <= ACTION_OFFSET_FRAMES:
        raise ValueError("an episode needs at least two 2-D action frames")
    shifted = np.asarray(actions[ACTION_OFFSET_FRAMES:], dtype=np.float32)
    if add_auxiliary:
        shifted = np.concatenate((shifted, np.zeros_like(shifted)), axis=1)
    return shifted


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_contract(policy_config: dict, dataset_manifest: dict | None = None) -> dict:
    result = {
        "contract_version": "arx_official_act_v1",
        "camera_names": list(CAMERA_NAMES),
        "joint_names": list(JOINT_NAMES),
        "physical_state_dim": PHYSICAL_STATE_DIM,
        "physical_action_dim": PHYSICAL_ACTION_DIM,
        "model_action_dim": int(policy_config["action_dim"]),
        "auxiliary_action_semantics": AUXILIARY_ACTION_SEMANTICS,
        "source_action_semantics": SOURCE_ACTION_SEMANTICS,
        "effective_action_semantics": EFFECTIVE_ACTION_SEMANTICS,
        "action_offset_frames": ACTION_OFFSET_FRAMES,
        "source_fps": SOURCE_FPS,
        "height_command": EXPECTED_HEIGHT_COMMAND,
        "use_base": bool(policy_config["use_base"]),
        "policy_config": policy_config,
    }
    if dataset_manifest is not None:
        result["dataset_manifest"] = dataset_manifest
    return result


def canonical_json_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
