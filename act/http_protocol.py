"""HTTP v1 client and response validation for remote inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from pipeline_contract import ACTION_DIM, ACTION_SEMANTICS, CAMERA_NAMES, FPS, HTTP_PROTOCOL_VERSION


class ProtocolError(ValueError):
    pass


@dataclass
class InferenceResult:
    actions: np.ndarray
    request_id: int
    session_id: str
    round_trip_ms: float
    model_id: str


class HttpInferenceClient:
    def __init__(self, base_url: str, max_response_age_ms: float = 500.0):
        import requests

        self.base_url = base_url.rstrip("/")
        self.max_response_age_ms = float(max_response_age_ms)
        self.session = requests.Session()

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/healthz", timeout=(1.0, 2.0))
        response.raise_for_status()
        return response.json()

    def schema(self) -> dict:
        response = self.session.get(f"{self.base_url}/v1/schema", timeout=(1.0, 2.0))
        response.raise_for_status()
        payload = response.json()
        expected = {
            "protocol_version": HTTP_PROTOCOL_VERSION,
            "fps": FPS,
            "action_dim": ACTION_DIM,
            "action_semantics": ACTION_SEMANTICS,
            "camera_names": list(CAMERA_NAMES),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ProtocolError(f"server {key}={payload.get(key)!r}, expected {value!r}")
        return payload

    def reset(self, session_id: str, task_instruction: str) -> dict:
        response = self.session.post(
            f"{self.base_url}/v1/reset",
            json={
                "protocol_version": HTTP_PROTOCOL_VERSION,
                "session_id": session_id,
                "task_instruction": task_instruction,
            },
            timeout=(1.0, 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("session_id") != session_id:
            raise ProtocolError("reset response session_id mismatch")
        return payload

    def infer(self, sample, task_instruction: str, session_id: str, request_id: int) -> InferenceResult:
        metadata = {
            "protocol_version": HTTP_PROTOCOL_VERSION,
            "session_id": session_id,
            "request_id": int(request_id),
            "sample_monotonic_ns": int(sample.sample_monotonic_ns),
            "task_instruction": task_instruction,
            "observation_state": np.asarray(sample.qpos, dtype=np.float32).tolist(),
        }
        files = {
            camera: (f"{camera}.jpg", sample.images[camera], "image/jpeg") for camera in CAMERA_NAMES
        }
        started = time.monotonic()
        response = self.session.post(
            f"{self.base_url}/v1/infer",
            data={"metadata": json.dumps(metadata)},
            files=files,
            timeout=(1.0, 30.0),
        )
        round_trip_ms = (time.monotonic() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()
        if round_trip_ms > self.max_response_age_ms:
            raise ProtocolError(f"stale response: {round_trip_ms:.1f} ms")
        if payload.get("protocol_version") != HTTP_PROTOCOL_VERSION:
            raise ProtocolError("protocol version mismatch")
        if payload.get("session_id") != session_id or int(payload.get("request_id", -1)) != request_id:
            raise ProtocolError("session/request id mismatch")
        if payload.get("action_semantics") != ACTION_SEMANTICS:
            raise ProtocolError("action semantics mismatch")
        if abs(float(payload.get("action_dt", 0.0)) - 1.0 / FPS) > 1e-6:
            raise ProtocolError("action_dt mismatch")
        actions = np.asarray(payload.get("actions"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not 1 <= actions.shape[0] <= 64:
            raise ProtocolError(f"invalid action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise ProtocolError("actions contain NaN or Inf")
        return InferenceResult(
            actions=actions,
            request_id=request_id,
            session_id=session_id,
            round_trip_ms=round_trip_ms,
            model_id=str(payload.get("model_id", "unknown")),
        )
