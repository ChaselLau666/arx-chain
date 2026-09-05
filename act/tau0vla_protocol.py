"""Pure HTTP and action-chunk contract for ARX LIFT2s Tau0VLA inference."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "arx_lift2s_http_v1"
FPS = 30
ACTION_DIM = 14
ACTION_HORIZON = 30
ACTION_SEMANTICS = "state_t_plus_1"
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    qpos: np.ndarray
    images: Mapping[str, bytes]
    sample_monotonic_ns: int


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    request_id: int
    sample_monotonic_ns: int
    round_trip_ms: float
    inference_ms: float
    model_id: str


@dataclass(frozen=True)
class AdoptionInfo:
    skipped: int
    blended_steps: int
    gripper_blended_steps: int
    age_ms: float
    raw_boundary_jump_max: float
    blended_boundary_jump_max: float


@dataclass(frozen=True)
class ScheduledAction:
    action: np.ndarray
    raw_action: np.ndarray
    request_id: int
    source_index: int
    skipped: int
    blend_alpha: float
    gripper_blend_alpha: float
    round_trip_ms: float


class Tau0VLAHttpClient:
    def __init__(self, base_url: str, *, request_timeout: float = 5.0, max_response_age_ms: float = 2000.0):
        import requests

        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        self.max_response_age_ms = float(max_response_age_ms)
        self.session = requests.Session()
        self.session_id: str | None = None
        self.model_id: str | None = None

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=(1.0, 3.0))
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok" or payload.get("ready") is not True:
            raise ProtocolError(f"server is not ready: {payload}")
        return payload

    def policy_contract(self) -> dict:
        response = self.session.get(
            f"{self.base_url}/api/v1/arx-lift2s/policy-contract", timeout=(1.0, 3.0)
        )
        response.raise_for_status()
        payload = response.json()
        expected = {
            "protocol_version": PROTOCOL_VERSION,
            "fps": FPS,
            "camera_names": list(CAMERA_NAMES),
            "state_dim": ACTION_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_semantics": ACTION_SEMANTICS,
            "joint_names": list(JOINT_NAMES),
        }
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches:
            raise ProtocolError(f"server policy contract mismatch: {mismatches}")
        self.model_id = str(payload.get("model_id", "unknown"))
        return payload

    def create_session(self, task_instruction: str) -> dict:
        instruction = task_instruction.strip()
        if not instruction:
            raise ProtocolError("task instruction must not be empty")
        response = self.session.post(
            f"{self.base_url}/api/v1/arx-lift2s/sessions",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "task_instruction": instruction,
                "client_name": "arx1",
            },
            timeout=(1.0, 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("protocol_version") != PROTOCOL_VERSION or not payload.get("session_id"):
            raise ProtocolError("invalid create-session response")
        if self.model_id is not None and payload.get("model_id") != self.model_id:
            raise ProtocolError("model changed between policy-contract and session creation")
        self.session_id = str(payload["session_id"])
        return payload

    def infer(self, observation: Observation, request_id: int) -> ActionChunk:
        if self.session_id is None:
            raise ProtocolError("create_session must be called before infer")
        state = np.asarray(observation.qpos, dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ProtocolError("observation qpos must be a finite 14-vector")
        missing = set(CAMERA_NAMES) - set(observation.images)
        if missing:
            raise ProtocolError(f"observation is missing cameras: {sorted(missing)}")
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": int(request_id),
            "sample_monotonic_ns": int(observation.sample_monotonic_ns),
            "observation_state": state.tolist(),
        }
        files = {
            camera: (f"{camera}.jpg", bytes(observation.images[camera]), "image/jpeg")
            for camera in CAMERA_NAMES
        }
        started = time.monotonic()
        response = self.session.post(
            f"{self.base_url}/api/v1/arx-lift2s/sessions/{self.session_id}/action-chunks",
            data={"metadata": json.dumps(metadata, separators=(",", ":"))},
            files=files,
            timeout=(1.0, self.request_timeout),
        )
        round_trip_ms = (time.monotonic() - started) * 1000.0
        response.raise_for_status()
        if round_trip_ms > self.max_response_age_ms:
            raise ProtocolError(f"response age {round_trip_ms:.1f} ms exceeds {self.max_response_age_ms:.1f} ms")
        payload = response.json()
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError("response protocol version mismatch")
        if payload.get("session_id") != self.session_id or int(payload.get("request_id", -1)) != request_id:
            raise ProtocolError("response session/request ID mismatch")
        if int(payload.get("sample_monotonic_ns", -1)) != observation.sample_monotonic_ns:
            raise ProtocolError("response observation timestamp mismatch")
        if payload.get("action_semantics") != ACTION_SEMANTICS:
            raise ProtocolError("response action semantics mismatch")
        if not math.isclose(float(payload.get("action_dt", 0.0)), 1.0 / FPS, abs_tol=1e-8):
            raise ProtocolError("response action_dt mismatch")
        if self.model_id is not None and payload.get("model_id") != self.model_id:
            raise ProtocolError("response model ID changed")
        actions = np.asarray(payload.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ProtocolError(f"invalid action chunk shape/content: {actions.shape}")
        inference_ms = float(payload.get("inference_ms", -1.0))
        if inference_ms < 0.0 or not math.isfinite(inference_ms):
            raise ProtocolError("invalid server inference time")
        return ActionChunk(
            actions=actions,
            request_id=request_id,
            sample_monotonic_ns=observation.sample_monotonic_ns,
            round_trip_ms=round_trip_ms,
            inference_ms=inference_ms,
            model_id=str(payload.get("model_id")),
        )


def recommended_replan_steps(
    round_trip_ms: Sequence[float], *, margin_ms: float = 100.0, maximum: int = 15
) -> tuple[int, float]:
    values = np.asarray(round_trip_ms, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("round-trip samples must be a non-empty finite non-negative sequence")
    p99_ms = float(np.percentile(values, 99))
    maximum_window_ms = (ACTION_HORIZON - 1) * 1000.0 / FPS
    if p99_ms + margin_ms >= maximum_window_ms:
        raise ProtocolError(
            f"p99 RTT {p99_ms:.1f} ms plus margin {margin_ms:.1f} ms cannot fit the action horizon"
        )
    available_steps = math.floor(ACTION_HORIZON - (p99_ms + margin_ms) * FPS / 1000.0)
    return max(1, min(int(maximum), available_steps)), p99_ms


class ChunkScheduler:
    """Single-request buffer with time alignment and smooth chunk handoff."""

    def __init__(
        self,
        replan_steps: int,
        blend_steps: int = 6,
        gripper_blend_steps: int | None = None,
    ):
        if not 1 <= int(replan_steps) < ACTION_HORIZON:
            raise ValueError(f"replan_steps must be in [1, {ACTION_HORIZON - 1}]")
        if not 0 <= int(blend_steps) < ACTION_HORIZON:
            raise ValueError(f"blend_steps must be in [0, {ACTION_HORIZON - 1}]")
        if gripper_blend_steps is not None and not 0 <= int(gripper_blend_steps) < ACTION_HORIZON:
            raise ValueError(f"gripper_blend_steps must be in [0, {ACTION_HORIZON - 1}]")
        self.replan_steps = int(replan_steps)
        self.blend_steps = int(blend_steps)
        self.gripper_blend_steps = (
            self.blend_steps if gripper_blend_steps is None else int(gripper_blend_steps)
        )
        self._steps: list[ScheduledAction] = []
        self._index = 0
        self._published_since_adopt = 0
        self._last_action: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return len(self._steps) - self._index

    def should_request(self, request_pending: bool) -> bool:
        return not request_pending and (
            self.remaining == 0
            or self._published_since_adopt >= self.replan_steps
            # A delayed response may skip most of its prefix. If the usable
            # suffix is already shorter than the normal replan interval,
            # prefetch its successor immediately instead of consuming the
            # short suffix first and guaranteeing starvation.
            or self.remaining <= self.replan_steps
        )

    def adopt(
        self,
        chunk: ActionChunk,
        *,
        initial: bool = False,
        arrival_monotonic_ns: int | None = None,
    ) -> AdoptionInfo:
        actions = np.asarray(chunk.actions, dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ProtocolError("cannot adopt an invalid action chunk")
        if arrival_monotonic_ns is None:
            age_ms = float(chunk.round_trip_ms)
        else:
            age_ms = max(0.0, (int(arrival_monotonic_ns) - int(chunk.sample_monotonic_ns)) / 1_000_000.0)
        # action[0] targets state(t+1), so it remains usable until one full
        # control period has elapsed. Floor avoids discarding a still-future
        # first target on low-latency Ethernet responses.
        skipped = 0 if initial else min(ACTION_HORIZON - 1, int(math.floor(age_ms * FPS / 1000.0)))
        fresh = actions[skipped:]
        old_steps = self._steps[self._index :]
        overlap = 0 if initial else min(self.blend_steps, len(old_steps), len(fresh))
        gripper_overlap = (
            0 if initial else min(self.gripper_blend_steps, len(old_steps), len(fresh))
        )
        scheduled: list[ScheduledAction] = []
        for offset, raw_action in enumerate(fresh):
            alpha = 1.0
            gripper_alpha = 1.0
            action = raw_action.copy()
            if offset < overlap:
                progress = (offset + 1) / overlap
                alpha = progress * progress * (3.0 - 2.0 * progress)
                action[ARM_INDICES] = (
                    (1.0 - alpha) * old_steps[offset].action[ARM_INDICES]
                    + alpha * raw_action[ARM_INDICES]
                )
            if offset < gripper_overlap:
                progress = (offset + 1) / gripper_overlap
                gripper_alpha = progress * progress * (3.0 - 2.0 * progress)
                action[GRIPPER_INDICES] = (
                    (1.0 - gripper_alpha) * old_steps[offset].action[GRIPPER_INDICES]
                    + gripper_alpha * raw_action[GRIPPER_INDICES]
                )
            scheduled.append(
                ScheduledAction(
                    action=np.asarray(action, dtype=np.float32),
                    raw_action=np.asarray(raw_action, dtype=np.float32).copy(),
                    request_id=int(chunk.request_id),
                    source_index=skipped + offset,
                    skipped=skipped,
                    blend_alpha=float(alpha),
                    gripper_blend_alpha=float(gripper_alpha),
                    round_trip_ms=float(chunk.round_trip_ms),
                )
            )
        raw_jump = 0.0
        blended_jump = 0.0
        if self._last_action is not None and len(fresh):
            raw_jump = float(np.max(np.abs(fresh[0] - self._last_action)))
            blended_jump = float(np.max(np.abs(scheduled[0].action - self._last_action)))
        self._steps = scheduled
        self._index = 0
        self._published_since_adopt = 0
        return AdoptionInfo(
            skipped=skipped,
            blended_steps=overlap,
            gripper_blended_steps=gripper_overlap,
            age_ms=age_ms,
            raw_boundary_jump_max=raw_jump,
            blended_boundary_jump_max=blended_jump,
        )

    def next_action(self) -> ScheduledAction:
        if self.remaining <= 0:
            raise BufferError("action chunk exhausted")
        step = self._steps[self._index]
        self._index += 1
        self._published_since_adopt += 1
        self._last_action = step.action.copy()
        return step


class BinaryGripperStabilizer:
    """Debounce binary-like gripper intent and emit canonical training endpoints.

    A value at or below ``low_threshold`` votes for the low endpoint; a value at
    or above ``high_threshold`` votes for the high endpoint. Values in the gap
    retain the current state. An opposite state must persist for
    ``confirm_frames`` consecutive control ticks before it is accepted.
    ``confirm_frames=0`` preserves the input exactly for backwards compatibility.
    """

    def __init__(
        self,
        confirm_frames: int = 0,
        *,
        low_threshold: float = -2.1,
        high_threshold: float = -1.05,
        low_value: float = -3.384,
        high_value: float = 0.0,
    ):
        if int(confirm_frames) < 0:
            raise ValueError("confirm_frames must be non-negative")
        values = np.asarray([low_threshold, high_threshold, low_value, high_value], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("gripper stabilizer thresholds/endpoints must be finite")
        if not float(low_threshold) < float(high_threshold):
            raise ValueError("low_threshold must be less than high_threshold")
        if not float(low_value) < float(high_value):
            raise ValueError("low_value must be less than high_value")
        self.confirm_frames = int(confirm_frames)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.low_value = float(low_value)
        self.high_value = float(high_value)
        self._states: list[str] | None = None
        self._candidates = ["", ""]
        self._candidate_counts = [0, 0]

    def _initial_state(self, value: float) -> str:
        if value <= self.low_threshold:
            return "low"
        if value >= self.high_threshold:
            return "high"
        return "low" if abs(value - self.low_value) <= abs(value - self.high_value) else "high"

    def reset(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ProtocolError("gripper stabilizer initial action must be a finite 14-vector")
        self._states = [self._initial_state(float(value[index])) for index in GRIPPER_INDICES]
        self._candidates = list(self._states)
        self._candidate_counts = [0, 0]

    def _desired_state(self, value: float, current: str) -> str:
        if value <= self.low_threshold:
            return "low"
        if value >= self.high_threshold:
            return "high"
        return current

    def apply(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ProtocolError("gripper stabilizer input must be a finite 14-vector")
        if self.confirm_frames == 0:
            return value.copy()
        if self._states is None:
            self.reset(value)
        assert self._states is not None
        output = value.copy()
        for slot, index in enumerate(GRIPPER_INDICES):
            current = self._states[slot]
            desired = self._desired_state(float(value[index]), current)
            if desired == current:
                self._candidates[slot] = current
                self._candidate_counts[slot] = 0
            else:
                if self._candidates[slot] == desired:
                    self._candidate_counts[slot] += 1
                else:
                    self._candidates[slot] = desired
                    self._candidate_counts[slot] = 1
                if self._candidate_counts[slot] >= self.confirm_frames:
                    self._states[slot] = desired
                    self._candidates[slot] = desired
                    self._candidate_counts[slot] = 0
            output[index] = self.low_value if self._states[slot] == "low" else self.high_value
        return output

    def snapshot(self) -> dict:
        return {
            "enabled": self.confirm_frames > 0,
            "states": list(self._states or ()),
            "candidate_states": list(self._candidates),
            "candidate_counts": list(self._candidate_counts),
        }


class ActionEMA:
    """Optional command EMA; alpha=1 keeps the scheduled action unchanged."""

    def __init__(self, arm_alpha: float = 1.0, gripper_alpha: float = 1.0):
        for name, value in (("arm_alpha", arm_alpha), ("gripper_alpha", gripper_alpha)):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self._alpha = np.full(ACTION_DIM, float(arm_alpha), dtype=np.float32)
        self._alpha[[6, 13]] = float(gripper_alpha)
        self._previous: np.ndarray | None = None

    def reset(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ProtocolError("EMA initial action must be a finite 14-vector")
        self._previous = value.copy()

    def apply(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ProtocolError("EMA input must be a finite 14-vector")
        if self._previous is None:
            filtered = value.copy()
        else:
            filtered = self._alpha * value + (1.0 - self._alpha) * self._previous
        self._previous = filtered.copy()
        return filtered


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_SEMANTICS",
    "ActionEMA",
    "ActionChunk",
    "AdoptionInfo",
    "ARM_INDICES",
    "BinaryGripperStabilizer",
    "CAMERA_NAMES",
    "ChunkScheduler",
    "FPS",
    "GRIPPER_INDICES",
    "JOINT_NAMES",
    "Observation",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "ScheduledAction",
    "Tau0VLAHttpClient",
    "recommended_replan_steps",
]
