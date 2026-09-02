"""Pure validation for guarded ARX ACT execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


ARCHITECTURE_KEYS = (
    "policy_class",
    "backbone",
    "chunk_size",
    "hidden_dim",
    "camera_names",
    "position_embedding",
    "masks",
    "dilation",
    "use_base",
    "use_depth_image",
    "enc_layers",
    "dec_layers",
    "nheads",
    "dropout",
    "pre_norm",
    "states_dim",
    "physical_action_dim",
    "auxiliary_action_dim",
    "action_dim",
    "kl_weight",
    "dim_feedforward",
    "use_qvel",
    "use_effort",
    "use_eef_states",
    "use_eef_action",
)


def validate_policy_contract(runtime: dict, checkpoint: dict) -> None:
    saved = checkpoint["policy_config"]
    mismatches = {
        key: (saved.get(key), runtime.get(key))
        for key in ARCHITECTURE_KEYS
        if saved.get(key) != runtime.get(key)
    }
    if mismatches:
        raise ValueError(f"checkpoint/runtime ACT config mismatch: {mismatches}")


def load_joint_limits(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    arrays = {}
    for key in ("lower", "upper", "max_step", "max_initial_delta"):
        value = np.asarray(data[key], dtype=np.float32)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError(f"joint limit {key} must contain 14 finite values")
        arrays[key] = value
    left_reset = np.asarray(data["left_reset"], dtype=np.float32)
    if left_reset.shape != (7,) or not np.isfinite(left_reset).all():
        raise ValueError("left_reset must contain 7 finite values")
    tolerance = float(data["left_reset_tolerance"])
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("left_reset_tolerance must be positive")
    if np.any(arrays["lower"] >= arrays["upper"]):
        raise ValueError("each lower limit must be less than its upper limit")
    if np.any(arrays["max_step"] <= 0) or np.any(arrays["max_initial_delta"] <= 0):
        raise ValueError("delta limits must be positive")
    return {**arrays, "left_reset": left_reset, "left_reset_tolerance": tolerance}


class ActionGuard:
    def __init__(self, limits: dict, initial_qpos):
        initial = np.asarray(initial_qpos, dtype=np.float32)
        if initial.shape != (14,) or not np.isfinite(initial).all():
            raise ValueError("initial qpos must contain 14 finite values")
        if np.max(np.abs(initial[:7] - limits["left_reset"])) > limits["left_reset_tolerance"]:
            raise ValueError("left arm is not within the reviewed reset tolerance")
        self.limits = limits
        self.previous = initial.copy()
        self.first = True

    def validate(self, model_action) -> np.ndarray:
        action = np.asarray(model_action, dtype=np.float32)
        if action.ndim != 1 or len(action) < 14 or not np.isfinite(action).all():
            raise ValueError("model action must be a finite vector with at least 14 values")
        physical = action[:14].copy()
        physical[:7] = self.limits["left_reset"]
        if np.any(physical < self.limits["lower"]) or np.any(physical > self.limits["upper"]):
            raise ValueError("action exceeds reviewed joint limits")
        delta_limit = self.limits["max_initial_delta"] if self.first else self.limits["max_step"]
        if np.any(np.abs(physical - self.previous) > delta_limit):
            raise ValueError("action delta exceeds reviewed safety limits")
        self.previous = physical.copy()
        self.first = False
        return physical
