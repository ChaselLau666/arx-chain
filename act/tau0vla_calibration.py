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
import yaml


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_VERSION = "arx-open-baseline-v1"
COMMAND_POINTS = np.asarray([-3.39, -2.55, -1.70, -0.85, 0.0], dtype=np.float64)
COMMAND_MARGIN = 0.05
# Flow-policy samples can land just beyond a demonstrated gripper endpoint.
# Accept only a small semantic error, then saturate to the calibrated endpoint;
# never publish the extrapolated value. The hard limit remains close enough to
# the measured range to catch a wrong route, baseline, or mapping immediately.
COMMAND_SOFT_TOLERANCE = 0.10
# The two ends of the gripper envelope are not physically symmetric. Commands
# past the CLOSED end (upper, 0.0) only add grip force on a held object -- the
# 0908 model does this routinely (rollout max excess 0.045, a DAgger episode hit
# 0.118 mid-grasp), and the mapper clips before publishing. Commands past the
# OPEN end (lower, -3.39) have no such reading and keep the strict tolerance.
COMMAND_CLOSE_SOFT_TOLERANCE = 0.20
VR_INTENT_SOFT_TOLERANCE = 0.05
GRIPPER_INDICES = (6, 13)
ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
SIDES = ("left", "right")


class CalibrationError(RuntimeError):
    pass


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


# Each arm stack names its X5Controller nodes differently. The rollout uses
# v2_joint_control.yaml with arm_slave_l/r; Human DAgger passes parameters
# directly and names its nodes human_dagger_arm_left/right. What actually
# guarantees "the arms have not restarted since calibration" is the
# (pid, start_ticks) pair plus boot_id -- the naming is only how a process is
# located, so recognising both layouts does not weaken the check.
ARM_STACKS: dict[str, dict[str, str]] = {
    "rollout": {
        "marker": "v2_joint_control.yaml",
        "left": "__node:=arm_slave_l",
        "right": "__node:=arm_slave_r",
    },
    "dagger": {
        "marker": "arm_pub_topic_name:=/human_dagger/arm/",
        "left": "__node:=human_dagger_arm_left",
        "right": "__node:=human_dagger_arm_right",
    },
}


def current_controller_identity(stack: str | None = None) -> dict[str, dict[str, int]]:
    """Locate the live X5Controller process per side.

    stack selects an expected layout from ARM_STACKS; None auto-detects and
    requires exactly one layout to be present, so a half-torn-down rollout
    stack overlapping a dagger stack is an error rather than a coin flip.
    """
    if stack is not None and stack not in ARM_STACKS:
        raise CalibrationError(f"unknown arm stack: {stack}")
    candidates = {stack: ARM_STACKS[stack]} if stack else dict(ARM_STACKS)
    found: dict[str, dict[str, dict[str, int]]] = {name: {} for name in candidates}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if "X5Controller" not in command:
            continue
        for name, patterns in candidates.items():
            if patterns["marker"] not in command:
                continue
            side = next((s for s in SIDES if patterns[s] in command), None)
            if side is None:
                continue
            fields = stat[stat.rfind(")") + 2 :].split()
            found[name][side] = {"pid": int(entry.name), "start_ticks": int(fields[19])}
    complete = {name: sides for name, sides in found.items() if set(sides) == set(SIDES)}
    if not complete:
        expected = stack or " or ".join(sorted(ARM_STACKS))
        raise CalibrationError(
            f"expected exactly one live X5Controller per side for the {expected} arm stack"
        )
    if len(complete) > 1:
        raise CalibrationError(
            f"multiple arm stacks are live ({sorted(complete)}); stop one before calibrating"
        )
    return next(iter(complete.values()))


def current_robot_identity(
    stack: str | None = None,
) -> tuple[str, int, str, dict[str, dict[str, int]]]:
    raw_domain = os.environ.get("ROS_DOMAIN_ID")
    if raw_domain is None:
        raise CalibrationError("ROS_DOMAIN_ID is not set")
    return (
        socket.gethostname(),
        int(raw_domain),
        current_boot_id(),
        current_controller_identity(stack),
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
    return artifact_from_dict(payload)


def artifact_from_dict(payload: dict[str, Any]) -> CalibrationArtifact:
    payload = dict(payload)
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


def load_training_ready_arms(path: str | Path) -> np.ndarray:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise CalibrationError("training ready-pose config version mismatch")
    try:
        left = np.asarray(payload["left_arm"], dtype=np.float64)
        right = np.asarray(payload["right_arm"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError(f"invalid training ready-pose config: {error}") from error
    if left.shape != (6,) or right.shape != (6,) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise CalibrationError("training ready pose requires two finite 6D arm targets")
    return np.concatenate((left, right)).astype(np.float32)


def feedback_pose_to_command(
    artifact: CalibrationArtifact,
    feedback: np.ndarray,
) -> np.ndarray:
    """Convert a measured 14D pose to the command coordinates used by X5."""
    values = np.asarray(feedback, dtype=np.float64)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise CalibrationError("feedback pose must be a finite 14-vector")
    command = values.copy()
    for side, index in zip(SIDES, GRIPPER_INDICES, strict=True):
        fit: SideCalibration = getattr(artifact, side)
        value = (values[index] - fit.intercept) / fit.slope
        lower = min(fit.command_points) - COMMAND_MARGIN
        upper = max(fit.command_points) + COMMAND_MARGIN
        if not lower <= value <= upper:
            raise CalibrationError(
                f"{side} feedback maps outside calibrated command envelope [{lower}, {upper}]"
            )
        command[index] = value
    return command.astype(np.float32)


def training_ready_targets(
    artifact: CalibrationArtifact,
    ready_arms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return command and expected-feedback targets for the fixed training pose."""
    arms = np.asarray(ready_arms, dtype=np.float64)
    if arms.shape != (12,) or not np.isfinite(arms).all():
        raise CalibrationError("training ready arm target must be a finite 12-vector")
    expected = np.empty(14, dtype=np.float64)
    expected[np.asarray(ARM_INDICES)] = arms
    expected[6] = artifact.left.final_open_feedback
    expected[13] = artifact.right.final_open_feedback
    return feedback_pose_to_command(artifact, expected), expected.astype(np.float32)


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
        self.last_saturation = {
            "count": 0,
            "max_command_excess": 0.0,
            "intent_count": 0,
            "max_intent_excess": 0.0,
            "by_side": {},
        }

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
        saturation_count = 0
        saturation_max = 0.0
        intent_saturation_count = 0
        intent_saturation_max = 0.0
        saturation_by_side: dict[str, dict[str, float | int]] = {}
        for side, index in zip(SIDES, GRIPPER_INDICES, strict=True):
            fit: SideCalibration = getattr(self.artifact, side)
            values = action[:, index].astype(np.float64)
            if self.experiment == "joint-feedback":
                desired_feedback = fit.final_open_feedback + values
            else:
                intent_excess = np.maximum.reduce(
                    (-values, values - 1.0, np.zeros_like(values))
                )
                maximum_intent_excess = float(np.max(intent_excess))
                if maximum_intent_excess > VR_INTENT_SOFT_TOLERANCE:
                    raise CalibrationError(
                        f"{side} VR gripper intent range "
                        f"[{float(np.min(values)):.6f}, {float(np.max(values)):.6f}] "
                        f"exceeds [0,1] plus soft tolerance {VR_INTENT_SOFT_TOLERANCE:.3f}"
                    )
                intent_count = int(np.count_nonzero(intent_excess > 0.0))
                if intent_count:
                    intent_saturation_count += intent_count
                    intent_saturation_max = max(
                        intent_saturation_max, maximum_intent_excess
                    )
                    saturation_by_side[side] = {
                        "intent_count": intent_count,
                        "max_intent_excess": maximum_intent_excess,
                    }
                values = np.clip(values, 0.0, 1.0)
                desired_feedback = fit.final_open_feedback + values * (
                    fit.closed_feedback - fit.final_open_feedback
                )
            raw_command = (desired_feedback - fit.intercept) / fit.slope
            lower = min(fit.command_points)
            upper = max(fit.command_points)
            hard_lower = lower - COMMAND_SOFT_TOLERANCE
            hard_upper = upper + COMMAND_CLOSE_SOFT_TOLERANCE
            if np.any(raw_command < hard_lower) or np.any(raw_command > hard_upper):
                raise CalibrationError(
                    f"{side} mapped gripper command range "
                    f"[{float(np.min(raw_command)):.6f}, {float(np.max(raw_command)):.6f}] "
                    f"exceeds calibrated endpoints [{lower:.2f}, {upper:.2f}] plus "
                    f"soft tolerances (open {COMMAND_SOFT_TOLERANCE:.3f} / "
                    f"close {COMMAND_CLOSE_SOFT_TOLERANCE:.3f})"
                )
            command = np.clip(raw_command, lower, upper)
            excess = np.abs(raw_command - command)
            count = int(np.count_nonzero(excess > 0.0))
            maximum = float(np.max(excess))
            if count:
                saturation_count += count
                saturation_max = max(saturation_max, maximum)
                saturation_by_side.setdefault(side, {}).update(
                    {
                        "command_count": count,
                        "max_command_excess": maximum,
                    }
                )
            mapped[:, index] = command.astype(np.float32)
        self.last_saturation = {
            "count": saturation_count + intent_saturation_count,
            "max_command_excess": saturation_max,
            "intent_count": intent_saturation_count,
            "max_intent_excess": intent_saturation_max,
            "by_side": saturation_by_side,
        }
        return mapped


def return_trajectory(
    current: np.ndarray,
    target: np.ndarray,
    *,
    rate_hz: float = 30.0,
    minimum_duration_s: float = 5.0,
    max_arm_step: float = .02,
    max_gripper_step: float = .05,
) -> np.ndarray:
    """Generate a bounded smoothstep path back to a captured 14D pose."""
    start = np.asarray(current, dtype=np.float32)
    goal = np.asarray(target, dtype=np.float32)
    if start.shape != (14,) or goal.shape != (14,):
        raise CalibrationError("return poses must be 14-vectors")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise CalibrationError("return poses must be finite")
    arm = np.asarray(ARM_INDICES)
    gripper = np.asarray(GRIPPER_INDICES)
    arm_steps = int(
        np.ceil(1.5 * np.max(np.abs(goal[arm] - start[arm])) / max_arm_step)
    )
    gripper_steps = int(
        np.ceil(1.5 * np.max(np.abs(goal[gripper] - start[gripper])) / max_gripper_step)
    )
    steps = max(int(np.ceil(minimum_duration_s * rate_hz)), arm_steps, gripper_steps, 1)
    progress = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
    alpha = progress * progress * (3.0 - 2.0 * progress)
    return start[None, :] + alpha[:, None] * (goal - start)[None, :]


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
    "COMMAND_CLOSE_SOFT_TOLERANCE",
    "COMMAND_SOFT_TOLERANCE",
    "VR_INTENT_SOFT_TOLERANCE",
    "CalibrationArtifact",
    "CalibrationError",
    "CalibratedGripperMapper",
    "SideCalibration",
    "assert_current_open",
    "artifact_from_dict",
    "current_boot_id",
    "current_controller_identity",
    "current_robot_identity",
    "consumed_path",
    "fit_side",
    "feedback_pose_to_command",
    "load_artifact",
    "load_training_ready_arms",
    "mark_consumed",
    "require_unconsumed",
    "return_trajectory",
    "save_artifact",
    "training_ready_targets",
    "validate_artifact",
]
