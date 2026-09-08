"""JSONL trace and plots for calibrated-v3 rollout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tau0vla_calibrated_protocol import ARM_INDICES, GRIPPER_INDICES, JOINT_NAMES


class TraceWriter:
    def __init__(self, path: Path | None):
        self.path = path
        self._stream = None
        self._pending = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("w", encoding="utf-8")

    def _write(self, payload: dict[str, Any]) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        self._pending += 1
        if self._pending >= 30:
            self._stream.flush()
            self._pending = 0

    def metadata(self, **values: Any) -> None:
        self._write({"event": "metadata", **values})

    def adoption(self, request_id: int, info) -> None:
        self._write({
            "event": "adoption",
            "request_id": int(request_id),
            "arm_skipped": info.arm_skipped,
            "gripper_skipped": info.gripper_skipped,
            "blended_steps": info.blended_steps,
            "gripper_blended_steps": info.gripper_blended_steps,
            "age_ms": info.age_ms,
            "raw_boundary_jump_max": info.raw_boundary_jump_max,
            "blended_boundary_jump_max": info.blended_boundary_jump_max,
        })

    def tick(
        self,
        *,
        monotonic_ns: int,
        control_step: int,
        scheduled,
        command: np.ndarray,
        feedback: np.ndarray,
        execute: bool,
    ) -> None:
        self._write({
            "event": "tick",
            "monotonic_ns": int(monotonic_ns),
            "control_step": int(control_step),
            "execute": bool(execute),
            "request_id": scheduled.request_id,
            "arm_source_index": scheduled.arm_source_index,
            "gripper_source_index": scheduled.gripper_source_index,
            "arm_skipped": scheduled.arm_skipped,
            "gripper_skipped": scheduled.gripper_skipped,
            "blend_alpha": scheduled.blend_alpha,
            "gripper_blend_alpha": scheduled.gripper_blend_alpha,
            "round_trip_ms": scheduled.round_trip_ms,
            "model_calibrated_action": scheduled.calibrated_action.tolist(),
            "mapped_raw_action": scheduled.raw_action.tolist(),
            "scheduled_action": scheduled.action.tolist(),
            "command": np.asarray(command, dtype=np.float32).tolist(),
            "feedback": np.asarray(feedback, dtype=np.float32).tolist(),
        })

    def starvation(self, monotonic_ns: int, control_step: int) -> None:
        self._write({"event": "starvation", "monotonic_ns": monotonic_ns, "control_step": control_step})

    def return_tick(
        self,
        *,
        monotonic_ns: int,
        return_step: int,
        command: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        self._write({
            "event": "return_tick",
            "monotonic_ns": int(monotonic_ns),
            "return_step": int(return_step),
            "command": np.asarray(command, dtype=np.float32).tolist(),
            "feedback": np.asarray(feedback, dtype=np.float32).tolist(),
        })

    def return_result(self, *, status: str, target: np.ndarray, detail: dict[str, Any]) -> None:
        self._write({
            "event": "return_result",
            "status": status,
            "target": np.asarray(target, dtype=np.float32).tolist(),
            "detail": detail,
        })

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None


def analyze_trace(path: str | Path) -> dict[str, Any]:
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    ticks = [event for event in events if event.get("event") == "tick"]
    adoptions = [event for event in events if event.get("event") == "adoption"]
    starvation = sum(event.get("event") == "starvation" for event in events)
    if not ticks:
        return {"ticks": 0, "adoptions": len(adoptions), "starvation": starvation}
    command = np.asarray([row["command"] for row in ticks], dtype=np.float32)
    feedback = np.asarray([row["feedback"] for row in ticks], dtype=np.float32)
    mapped = np.asarray([row["mapped_raw_action"] for row in ticks], dtype=np.float32)
    request_ids = np.asarray([row["request_id"] for row in ticks])
    boundaries = np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1

    def metrics(values, indices):
        step = np.max(np.abs(np.diff(values[:, indices], axis=0)), axis=1) if len(values) > 1 else np.zeros(1)
        jerk = np.max(np.abs(np.diff(values[:, indices], n=2, axis=0)), axis=1) if len(values) > 2 else np.zeros(1)
        return {
            "step_p95": float(np.percentile(step, 95)),
            "step_max": float(np.max(step)),
            "jerk_p95": float(np.percentile(jerk, 95)),
            "jerk_max": float(np.max(jerk)),
        }

    tracking = np.max(np.abs(command - feedback), axis=1)
    boundary_step = (
        np.max(np.abs(command[boundaries] - command[boundaries - 1]), axis=1)
        if len(boundaries)
        else np.zeros(1)
    )
    return {
        "ticks": len(ticks),
        "adoptions": len(adoptions),
        "starvation": starvation,
        "boundary_count": int(len(boundaries)),
        "boundary_step_p95": float(np.percentile(boundary_step, 95)),
        "boundary_step_max": float(np.max(boundary_step)),
        "arm": metrics(command, ARM_INDICES),
        "gripper": metrics(command, GRIPPER_INDICES),
        "mapped_gripper_min": np.min(mapped[:, GRIPPER_INDICES], axis=0).tolist(),
        "mapped_gripper_max": np.max(mapped[:, GRIPPER_INDICES], axis=0).tolist(),
        "tracking_error_p95": float(np.percentile(tracking, 95)),
        "tracking_error_max": float(np.max(tracking)),
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    source = Path(path)
    target = source.with_name(f"{source.stem}_summary.json")
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def plot_trace(path: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    source = Path(path)
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    ticks = [event for event in events if event.get("event") == "tick"]
    if not ticks:
        return []
    metadata = next((event for event in events if event.get("event") == "metadata"), {})
    ns = np.asarray([row["monotonic_ns"] for row in ticks], dtype=np.int64)
    seconds = (ns - ns[0]) / 1e9
    mapped = np.asarray([row["mapped_raw_action"] for row in ticks])
    scheduled = np.asarray([row["scheduled_action"] for row in ticks])
    command = np.asarray([row["command"] for row in ticks])
    feedback = np.asarray([row["feedback"] for row in ticks])
    calibrated = np.asarray([row["model_calibrated_action"] for row in ticks])

    state_path = source.with_name(f"{source.stem}_state_action.png")
    figure, axes = plt.subplots(7, 2, figsize=(18, 22), sharex=True)
    for row in range(7):
        for column, index in enumerate((row, row + 7)):
            axis = axes[row, column]
            axis.plot(seconds, mapped[:, index], color="0.6", linewidth=.7, label="mapped raw")
            axis.plot(seconds, scheduled[:, index], color="#ff7f0e", linewidth=.8, label="scheduled")
            axis.plot(seconds, command[:, index], color="#d62728", linewidth=1.0, label="command")
            axis.plot(seconds, feedback[:, index], color="#1f77b4", linewidth=1.0, label="feedback")
            axis.set_title(JOINT_NAMES[index])
            axis.grid(alpha=.25)
    axes[-1, 0].set_xlabel("rollout time (s)")
    axes[-1, 1].set_xlabel("rollout time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.suptitle(
        f"Calibrated Tau0VLA state/action\n{metadata.get('model_id', '')} "
        f"{metadata.get('experiment', '')}", y=.995
    )
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(state_path, dpi=150)
    plt.close(figure)

    diag_path = source.with_name(f"{source.stem}_diagnostics.png")
    figure, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    arm_step = np.max(np.abs(np.diff(command[:, ARM_INDICES], axis=0, prepend=command[:1, ARM_INDICES])), axis=1)
    grip_step = np.max(np.abs(np.diff(command[:, GRIPPER_INDICES], axis=0, prepend=command[:1, GRIPPER_INDICES])), axis=1)
    axes[0].plot(seconds, arm_step, label="arm command step")
    axes[1].plot(seconds, grip_step, label="gripper command step")
    axes[2].plot(seconds, np.max(np.abs(command-feedback), axis=1), label="tracking error")
    axes[3].plot(seconds, [row["round_trip_ms"] for row in ticks], label="RTT ms")
    for side, index in zip(("left", "right"), GRIPPER_INDICES, strict=True):
        axes[1].plot(seconds, calibrated[:, index], alpha=.45, label=f"{side} calibrated output")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("rollout time (s)")
    figure.tight_layout()
    figure.savefig(diag_path, dpi=150)
    plt.close(figure)
    return [state_path, diag_path]


__all__ = ["TraceWriter", "analyze_trace", "plot_trace", "write_summary"]
