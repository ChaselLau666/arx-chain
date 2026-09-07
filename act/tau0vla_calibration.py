"""Pure gripper calibration and calibrated-policy action mapping."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_VERSION = "arx-open-baseline-v1"
COMMAND_POINTS = np.asarray([-3.39, -2.55, -1.70, -0.85, 0.0], dtype=np.float64)
COMMAND_MARGIN = 0.05
GRIPPER_INDICES = (6, 13)
SIDES = ("left", "right")


class CalibrationError(RuntimeError):
    pass


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def current_controller_identity() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if "X5Controller" not in command or "v2_joint_control.yaml" not in command:
            continue
        side = next((name for name in SIDES if f"__node:=arm_slave_{name[0]}" in command), None)
        if side is None:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        result[side] = {"pid": int(entry.name), "start_ticks": int(fields[19])}
    if set(result) != set(SIDES):
        raise CalibrationError("expected exactly one live v2_joint_control process per side")
    return result


def current_robot_identity() -> tuple[str, int, str, dict[str, dict[str, int]]]:
    raw_domain = os.environ.get("ROS_DOMAIN_ID")
    if raw_domain is None:
        raise CalibrationError("ROS_DOMAIN_ID is not set")
    return (
        socket.gethostname(),
        int(raw_domain),
        current_boot_id(),
        current_controller_identity(),
    )


@dataclass(frozen=True)
class SideCalibration:
    slope: float
    intercept: float
    r_squared: float
    max_abs_residual: float
    command_points: list[float]
    feedback_means: list[float]
    feedback_spreads: list[float]
    open_feedback: float
    closed_feedback: float
    final_open_feedback: float
    final_open_spread: float
    open_drift: float


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: int
    calibration_version: str
    calibration_id: str
    hostname: str
    ros_domain_id: int
    boot_id: str
    created_unix_s: float
    created_monotonic_ns: int
    controller_identity: dict[str, dict[str, int]]
    left: SideCalibration
    right: SideCalibration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_side(
    command_points: np.ndarray,
    feedback_samples: list[np.ndarray],
    final_open_samples: np.ndarray,
) -> SideCalibration:
    commands = np.asarray(command_points, dtype=np.float64)
    if commands.shape != COMMAND_POINTS.shape or not np.allclose(commands, COMMAND_POINTS):
        raise CalibrationError(f"command points must be {COMMAND_POINTS.tolist()}")
    if len(feedback_samples) != len(commands):
        raise CalibrationError("one feedback sample window is required per command point")
    means: list[float] = []
    spreads: list[float] = []
    for index, raw in enumerate(feedback_samples):
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all():
            raise CalibrationError("each point requires at least 30 finite feedback samples")
        mean = float(np.mean(values))
        spread = float(np.percentile(values, 90) - np.percentile(values, 10))
        if spread > 0.01:
            raise CalibrationError(f"feedback spread {spread:.6f} exceeds 0.01 at point {index}")
        means.append(mean)
        spreads.append(spread)
    final_values = np.asarray(final_open_samples, dtype=np.float64)
    if final_values.ndim != 1 or len(final_values) < 60 or not np.isfinite(final_values).all():
        raise CalibrationError("final open baseline requires at least 60 finite samples")
    final_open = float(np.mean(final_values))
    final_spread = float(np.percentile(final_values, 90) - np.percentile(final_values, 10))
    if spreads[0] > 0.003 or final_spread > 0.003:
        raise CalibrationError("open feedback spread exceeds 0.003")
    open_drift = abs(final_open - means[0])
    if open_drift > 0.02:
        raise CalibrationError(f"open feedback drift {open_drift:.6f} exceeds 0.02")

    slope, intercept = np.polyfit(commands, means, 1)
    predicted = slope * commands + intercept
    residual = np.asarray(means) - predicted
    max_residual = float(np.max(np.abs(residual)))
    total = float(np.sum((np.asarray(means) - np.mean(means)) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - float(np.sum(residual**2)) / total
    feedback_span = float(means[-1] - means[0])
    if not 0.8 <= slope <= 1.2:
        raise CalibrationError(f"command-feedback slope {slope:.6f} is outside [0.8, 1.2]")
    if r_squared < 0.995:
        raise CalibrationError(f"command-feedback R^2 {r_squared:.6f} is below 0.995")
    if max_residual > 0.05:
        raise CalibrationError(f"fit residual {max_residual:.6f} exceeds 0.05")
    if not 2.5 <= feedback_span <= 4.0:
        raise CalibrationError(f"open-closed feedback span {feedback_span:.6f} is outside [2.5, 4.0]")
    return SideCalibration(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        max_abs_residual=max_residual,
        command_points=commands.tolist(),
        feedback_means=means,
        feedback_spreads=spreads,
        open_feedback=float(means[0]),
        closed_feedback=float(means[-1]),
        final_open_feedback=final_open,
        final_open_spread=final_spread,
        open_drift=open_drift,
    )


def save_artifact(path: Path, artifact: CalibrationArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_artifact(path: str | Path) -> CalibrationArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        left = SideCalibration(**payload.pop("left"))
        right = SideCalibration(**payload.pop("right"))
        artifact = CalibrationArtifact(left=left, right=right, **payload)
    except (KeyError, TypeError) as error:
        raise CalibrationError(f"invalid calibration artifact: {error}") from error
    if artifact.schema_version != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError("calibration schema version mismatch")
    if artifact.calibration_version != CALIBRATION_VERSION:
        raise CalibrationError("calibration algorithm version mismatch")
    return artifact


def validate_artifact(
    artifact: CalibrationArtifact,
    *,
    hostname: str,
    ros_domain_id: int,
    boot_id: str,
    controller_identity: dict[str, dict[str, int]],
    max_age_s: float = 900.0,
) -> None:
    if artifact.hostname != hostname or artifact.ros_domain_id != ros_domain_id:
        raise CalibrationError("calibration robot identity mismatch")
    if artifact.boot_id != boot_id:
        raise CalibrationError("calibration was created before the current boot")
    if artifact.controller_identity != controller_identity:
        raise CalibrationError("arm controller process identity changed after calibration")
    age = time.time() - artifact.created_unix_s
    if not 0.0 <= age <= max_age_s:
        raise CalibrationError(f"calibration age {age:.1f}s exceeds {max_age_s:.1f}s")


class CalibratedGripperMapper:
    def __init__(self, artifact: CalibrationArtifact, experiment: str):
        if experiment not in ("joint-feedback", "joint-vr"):
            raise CalibrationError(f"unsupported calibrated experiment: {experiment}")
        self.artifact = artifact
        self.experiment = experiment

    @property
    def open_baselines(self) -> dict[str, float]:
        return {
            "left": self.artifact.left.final_open_feedback,
            "right": self.artifact.right.final_open_feedback,
        }

    def map_chunk(self, calibrated_action: np.ndarray) -> np.ndarray:
        action = np.asarray(calibrated_action, dtype=np.float32)
        if action.shape != (30, 14) or not np.isfinite(action).all():
            raise CalibrationError("calibrated action chunk must be finite [30,14]")
        mapped = action.copy()
        for side, index in zip(SIDES, GRIPPER_INDICES, strict=True):
            fit: SideCalibration = getattr(self.artifact, side)
            values = action[:, index].astype(np.float64)
            if self.experiment == "joint-feedback":
                desired_feedback = fit.final_open_feedback + values
            else:
                if np.any(values < 0.0) or np.any(values > 1.0):
                    raise CalibrationError(f"{side} VR gripper intent is outside [0,1]")
                desired_feedback = fit.final_open_feedback + values * (
                    fit.closed_feedback - fit.final_open_feedback
                )
            command = (desired_feedback - fit.intercept) / fit.slope
            lower = min(fit.command_points) - COMMAND_MARGIN
            upper = max(fit.command_points) + COMMAND_MARGIN
            if np.any(command < lower) or np.any(command > upper):
                raise CalibrationError(
                    f"{side} mapped gripper command is outside calibrated envelope [{lower}, {upper}]"
                )
            mapped[:, index] = command.astype(np.float32)
        return mapped


def consumed_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_suffix(source.suffix + ".consumed")


def mark_consumed(path: str | Path, *, session_id: str) -> Path:
    target = consumed_path(path)
    target.write_text(
        json.dumps({"session_id": session_id, "consumed_unix_s": time.time()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def require_unconsumed(path: str | Path) -> None:
    target = consumed_path(path)
    if target.exists():
        raise CalibrationError(f"calibration has already been consumed: {target}")


def assert_current_open(artifact: CalibrationArtifact, samples: np.ndarray) -> None:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 60 or not np.isfinite(values).all():
        raise CalibrationError("open recheck requires at least 60 finite [left,right] samples")
    for side_index, side in enumerate(SIDES):
        fit: SideCalibration = getattr(artifact, side)
        column = values[:, side_index]
        spread = float(np.percentile(column, 90) - np.percentile(column, 10))
        drift = abs(float(np.mean(column)) - fit.final_open_feedback)
        if spread > 0.003 or drift > 0.02:
            raise CalibrationError(
                f"{side} open recheck failed: spread={spread:.6f}, drift={drift:.6f}"
            )


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_VERSION",
    "COMMAND_MARGIN",
    "COMMAND_POINTS",
    "CalibrationArtifact",
    "CalibrationError",
    "CalibratedGripperMapper",
    "SideCalibration",
    "assert_current_open",
    "current_boot_id",
    "current_controller_identity",
    "current_robot_identity",
    "consumed_path",
    "fit_side",
    "load_artifact",
    "mark_consumed",
    "require_unconsumed",
    "save_artifact",
    "validate_artifact",
]
