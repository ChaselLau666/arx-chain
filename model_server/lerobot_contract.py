"""Strict LeRobot dataset boundary for TAU training adapters."""

from __future__ import annotations

import json
from pathlib import Path

from act.pipeline_contract import ACTION_DIM, ACTION_SEMANTICS, FPS, JOINT_NAMES


class LeRobotContractError(ValueError):
    pass


def validate_arx_lerobot(root: str | Path) -> dict:
    root = Path(root)
    meta_path = root / "meta" / "arx.json"
    if not meta_path.is_file():
        raise LeRobotContractError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "lerobot_version": "0.4.3",
        "lerobot_format": "v3",
        "fps": FPS,
        "action_dim": ACTION_DIM,
        "action_semantics": ACTION_SEMANTICS,
        "joint_names": list(JOINT_NAMES),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise LeRobotContractError(f"{key}={meta.get(key)!r}, expected {value!r}")
    return meta


def load_lerobot_dataset(root: str | Path, repo_id: str):
    validate_arx_lerobot(root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root)
