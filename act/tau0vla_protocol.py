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
    """Single-request double buffer with RTT-based action-prefix skipping."""

    def __init__(self, replan_steps: int):
        if not 1 <= int(replan_steps) < ACTION_HORIZON:
            raise ValueError(f"replan_steps must be in [1, {ACTION_HORIZON - 1}]")
        self.replan_steps = int(replan_steps)
        self._actions = np.empty((0, ACTION_DIM), dtype=np.float32)
        self._index = 0
        self._published_since_adopt = 0

    @property
    def remaining(self) -> int:
        return len(self._actions) - self._index

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

    def adopt(self, chunk: ActionChunk, *, initial: bool = False) -> int:
        actions = np.asarray(chunk.actions, dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ProtocolError("cannot adopt an invalid action chunk")
        skipped = 0 if initial else min(
            ACTION_HORIZON - 1,
            int(math.ceil(chunk.round_trip_ms * FPS / 1000.0)),
        )
        self._actions = actions[skipped:].copy()
        self._index = 0
        self._published_since_adopt = 0
        return skipped

    def next_action(self) -> np.ndarray:
        if self.remaining <= 0:
            raise BufferError("action chunk exhausted")
        action = self._actions[self._index].copy()
        self._index += 1
        self._published_since_adopt += 1
        return action


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_SEMANTICS",
    "ActionChunk",
    "CAMERA_NAMES",
    "ChunkScheduler",
    "FPS",
    "JOINT_NAMES",
    "Observation",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "Tau0VLAHttpClient",
    "recommended_replan_steps",
]
