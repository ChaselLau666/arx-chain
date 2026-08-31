"""Versioned contracts shared by collection, training, conversion, and serving."""

from __future__ import annotations

import json
from pathlib import Path

import h5py


SCHEMA_VERSION = "arx_hdf5_v2"
HTTP_PROTOCOL_VERSION = "arx_http_v1"
ACTION_SEMANTICS = "state_t_plus_1"
ACTION_DIM = 14
FPS = 30
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)


class DatasetContractError(ValueError):
    pass


def read_contract(path: str | Path) -> dict:
    with h5py.File(path, "r") as root:
        return {
            "schema_version": str(root.attrs.get("schema_version", "legacy")),
            "fps": int(root.attrs.get("fps", root.attrs.get("frame_rate", 0))),
            "action_dim": int(root.attrs.get("action_dim", root["action"].shape[1])),
            "action_semantics": str(root.attrs.get("action_semantics", "legacy_state_shift")),
            "joint_names": json.loads(root.attrs.get("joint_names", "[]")),
        }


def validate_dataset_contract(
    dataset_dir: str | Path,
    expected_action_dim: int = ACTION_DIM,
    expected_action_semantics: str = ACTION_SEMANTICS,
    expected_fps: int = FPS,
) -> dict:
    paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if not paths:
        raise DatasetContractError(f"no episode_*.hdf5 files found in {dataset_dir}")

    contracts = [read_contract(path) for path in paths]
    for path, contract in zip(paths, contracts):
        expected = {
            "schema_version": SCHEMA_VERSION,
            "fps": expected_fps,
            "action_dim": expected_action_dim,
            "action_semantics": expected_action_semantics,
        }
        for key, value in expected.items():
            if contract[key] != value:
                raise DatasetContractError(
                    f"{path.name}: {key}={contract[key]!r}, expected {value!r}"
                )
        if tuple(contract["joint_names"]) != JOINT_NAMES:
            raise DatasetContractError(f"{path.name}: unexpected joint order")

    return {
        "schema_version": SCHEMA_VERSION,
        "fps": expected_fps,
        "action_dim": expected_action_dim,
        "action_semantics": expected_action_semantics,
        "joint_names": list(JOINT_NAMES),
        "episodes": len(paths),
    }
