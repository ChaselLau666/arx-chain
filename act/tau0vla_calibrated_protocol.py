"""Pure calibrated-v3 HTTP, gripper mapping, and asynchronous chunk scheduling."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Mapping

import numpy as np

from tau0vla_calibration import (
    CALIBRATION_VERSION,
    CalibrationArtifact,
    CalibrationError,
    CalibratedGripperMapper,
)
from tau0vla_protocol import ActionEMA, recommended_replan_steps, resolve_replan_steps


PROTOCOL_VERSION = "arx-calibrated-v3"
FEEDBACK_PROTOCOL_VERSION = "arx-feedback-v4"
FEEDBACK_CALIBRATION_VERSION = "arx-feedback-open-v1"
CLIENT_ADAPTER_VERSION = "arx-calibrated-client-v1"
FPS = 30
SOURCE_FPS = 60
ACTION_DIM = 14
ACTION_HORIZON = 30
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)
EXPERIMENTS = ("joint-feedback", "joint-vr")


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    qpos: np.ndarray
    eef: np.ndarray | None
    images: Mapping[str, bytes]
    sample_monotonic_ns: int


@dataclass(frozen=True)
class CalibratedActionChunk:
    calibrated_actions: np.ndarray
    native_actions: np.ndarray
    request_id: int
    sample_monotonic_ns: int
    round_trip_ms: float
    inference_ms: float
    model_id: str
    experiment: str
    arm_offset_steps: int
    gripper_offset_steps: int
    gripper_saturation_count: int = 0
    gripper_saturation_max: float = 0.0
    gripper_intent_saturation_count: int = 0
    gripper_intent_saturation_max: float = 0.0


@dataclass(frozen=True)
class AdoptionInfo:
    arm_skipped: int
    gripper_skipped: int
    blended_steps: int
    gripper_blended_steps: int
    age_ms: float
    raw_boundary_jump_max: float
    blended_boundary_jump_max: float
    gripper_saturation_count: int = 0
    gripper_saturation_max: float = 0.0
    gripper_intent_saturation_count: int = 0
    gripper_intent_saturation_max: float = 0.0


@dataclass(frozen=True)
class ScheduledAction:
    action: np.ndarray
    raw_action: np.ndarray
    calibrated_action: np.ndarray
    request_id: int
    arm_source_index: int
    gripper_source_index: int
    arm_skipped: int
    gripper_skipped: int
    blend_alpha: float
    gripper_blend_alpha: float
    round_trip_ms: float


class CalibratedHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        experiment: str,
        calibration: CalibrationArtifact | None,
        robot_id: str,
        protocol_version: str = PROTOCOL_VERSION,
        request_timeout: float = 5.0,
        max_response_age_ms: float = 500.0,
    ):
        import requests

        if experiment not in EXPERIMENTS:
            raise ProtocolError(f"unsupported experiment: {experiment}")
        if protocol_version not in (PROTOCOL_VERSION, FEEDBACK_PROTOCOL_VERSION):
            raise ProtocolError(f"unsupported calibrated protocol: {protocol_version}")
        if protocol_version == FEEDBACK_PROTOCOL_VERSION and experiment != "joint-feedback":
            raise ProtocolError("arx-feedback-v4 only supports joint-feedback")
        self.base_url = base_url.rstrip("/")
        self.experiment = experiment
        self.protocol_version = protocol_version
        self.calibration_version = (
            CALIBRATION_VERSION
            if protocol_version == PROTOCOL_VERSION
            else FEEDBACK_CALIBRATION_VERSION
        )
        self.api_prefix = "/arx/v3" if protocol_version == PROTOCOL_VERSION else "/arx/v4"
        self.requires_eef = protocol_version == PROTOCOL_VERSION
        self.calibration = calibration
        self.mapper = CalibratedGripperMapper(calibration, experiment) if calibration is not None else None
        self.robot_id = robot_id
        self.request_timeout = float(request_timeout)
        self.max_response_age_ms = float(max_response_age_ms)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session_id: str | None = None
        self.model_id: str | None = None
        self.arm_offset_steps: int | None = None
        self.gripper_offset_steps: int | None = None

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=(1.0, 3.0))
        response.raise_for_status()
        payload = response.json()
        expected = {
            "status": "ok",
            "ready": True,
            "protocol_version": self.protocol_version,
            "experiment": self.experiment,
            "required_client_adapter_version": CLIENT_ADAPTER_VERSION,
        }
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches:
            raise ProtocolError(f"calibrated server health mismatch: {mismatches}")
        return payload

    def policy_contract(self) -> dict:
        response = self.session.get(
            f"{self.base_url}{self.api_prefix}/policy-contract", timeout=(1.0, 3.0)
        )
        response.raise_for_status()
        payload = response.json()
        expected = {
            "protocol_version": self.protocol_version,
            "calibration_version": self.calibration_version,
            "required_client_adapter_version": CLIENT_ADAPTER_VERSION,
            "experiment": self.experiment,
            "fps": FPS,
            "camera_names": list(CAMERA_NAMES),
            "state_dim": ACTION_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "joint_names": list(JOINT_NAMES),
            "wire_action_field": "calibrated_action_chunk",
            "wire_action_is_robot_command": False,
        }
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches:
            raise ProtocolError(f"calibrated policy contract mismatch: {mismatches}")
        offsets = payload.get("component_source_offsets")
        if not isinstance(offsets, dict):
            raise ProtocolError("contract is missing component_source_offsets")
        if self.protocol_version == PROTOCOL_VERSION:
            extra_expected = {"source_fps": SOURCE_FPS, "temporal_stride": 2}
            expected_offsets = {
                "arm_action": 2,
                "gripper_action": 2 if self.experiment == "joint-feedback" else 0,
            }
            offset_divisor = 2
        else:
            extra_expected = {"offset_unit": "uploaded_30fps_frame"}
            expected_offsets = {"arm_action": 1, "gripper_action": 1}
            offset_divisor = 1
        mismatches.update(
            {
                key: (payload.get(key), value)
                for key, value in extra_expected.items()
                if payload.get(key) != value
            }
        )
        if mismatches:
            raise ProtocolError(f"calibrated policy contract mismatch: {mismatches}")
        for key, expected_value in expected_offsets.items():
            if offsets.get(key) != expected_value:
                raise ProtocolError(f"contract {key}={offsets.get(key)!r}, expected {expected_value}")
        self.arm_offset_steps = expected_offsets["arm_action"] // offset_divisor
        self.gripper_offset_steps = expected_offsets["gripper_action"] // offset_divisor
        self.model_id = str(payload.get("model_id", ""))
        if not self.model_id:
            raise ProtocolError("contract has no model_id")
        return payload

    def create_session(self, task_instruction: str) -> dict:
        if self.calibration is None or self.mapper is None:
            raise ProtocolError("a real robot calibration is required to create a session")
        if self.model_id is None or self.arm_offset_steps is None or self.gripper_offset_steps is None:
            raise ProtocolError("policy_contract must be called before create_session")
        response = self.session.post(
            f"{self.base_url}{self.api_prefix}/sessions",
            json={
                "protocol_version": self.protocol_version,
                "calibration_version": self.calibration_version,
                "client_adapter_version": CLIENT_ADAPTER_VERSION,
                "experiment": self.experiment,
                "task_instruction": task_instruction.strip(),
                "client_name": "arx-calibrated-standalone",
                "robot_id": self.robot_id,
                "calibration_id": self.calibration.calibration_id,
                "open_baselines": self.mapper.open_baselines,
            },
            timeout=(1.0, 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        if (
            payload.get("protocol_version") != self.protocol_version
            or payload.get("model_id") != self.model_id
            or payload.get("experiment") != self.experiment
            or payload.get("calibration_id") != self.calibration.calibration_id
            or not payload.get("session_id")
        ):
            raise ProtocolError(f"invalid calibrated session response: {payload}")
        self.session_id = str(payload["session_id"])
        return payload

    def infer(self, observation: Observation, request_id: int) -> CalibratedActionChunk:
        from requests import HTTPError

        if self.session_id is None or self.model_id is None:
            raise ProtocolError("create_session must be called before infer")
        qpos = np.asarray(observation.qpos, dtype=np.float32)
        if qpos.shape != (ACTION_DIM,) or not np.isfinite(qpos).all():
            raise ProtocolError("joint feedback must be a finite 14-vector")
        if self.requires_eef:
            eef = np.asarray(observation.eef, dtype=np.float32)
            if eef.shape != (ACTION_DIM,) or not np.isfinite(eef).all():
                raise ProtocolError("EEF feedback must be a finite 14-vector for v3")
        if set(observation.images) != set(CAMERA_NAMES):
            raise ProtocolError("observation must contain exactly three calibrated cameras")
        metadata = {
            "protocol_version": self.protocol_version,
            "request_id": int(request_id),
            "sample_monotonic_ns": int(observation.sample_monotonic_ns),
            "raw_joint_feedback": qpos.tolist(),
        }
        if self.requires_eef:
            metadata["raw_eef_feedback"] = eef.tolist()
        files = {
            camera: (f"{camera}.jpg", bytes(observation.images[camera]), "image/jpeg")
            for camera in CAMERA_NAMES
        }
        started = time.monotonic()
        response = self.session.post(
            f"{self.base_url}{self.api_prefix}/sessions/{self.session_id}/action-chunks",
            data={"metadata": json.dumps(metadata, separators=(",", ":"))},
            files=files,
            timeout=(1.0, self.request_timeout),
        )
        round_trip_ms = (time.monotonic() - started) * 1000.0
        try:
            response.raise_for_status()
        except HTTPError as error:
            raise ProtocolError(
                f"calibrated action HTTP {response.status_code}; session={self.session_id}; "
                f"request={request_id}; body={response.text[:2000]!r}"
            ) from error
        if round_trip_ms > self.max_response_age_ms:
            raise ProtocolError(
                f"response age {round_trip_ms:.1f} ms exceeds {self.max_response_age_ms:.1f} ms"
            )
        payload = response.json()
        expected = {
            "protocol_version": self.protocol_version,
            "calibration_version": self.calibration_version,
            "required_client_adapter_version": CLIENT_ADAPTER_VERSION,
            "session_id": self.session_id,
            "request_id": request_id,
            "sample_monotonic_ns": observation.sample_monotonic_ns,
            "model_id": self.model_id,
            "experiment": self.experiment,
            "wire_action_is_robot_command": False,
        }
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches:
            raise ProtocolError(f"calibrated action response mismatch: {mismatches}")
        calibrated = np.asarray(payload.get("calibrated_action_chunk"), dtype=np.float32)
        if calibrated.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(calibrated).all():
            raise ProtocolError(f"invalid calibrated action chunk: {calibrated.shape}")
        try:
            native = self.mapper.map_chunk(calibrated)
        except CalibrationError as error:
            raise ProtocolError(str(error)) from error
        inference_ms = float(payload.get("inference_ms", -1.0))
        if inference_ms < 0 or not math.isfinite(inference_ms):
            raise ProtocolError("invalid server inference time")
        return CalibratedActionChunk(
            calibrated_actions=calibrated,
            native_actions=native,
            request_id=request_id,
            sample_monotonic_ns=observation.sample_monotonic_ns,
            round_trip_ms=round_trip_ms,
            inference_ms=inference_ms,
            model_id=self.model_id,
            experiment=self.experiment,
            arm_offset_steps=int(self.arm_offset_steps),
            gripper_offset_steps=int(self.gripper_offset_steps),
            gripper_saturation_count=int(self.mapper.last_saturation["count"]),
            gripper_saturation_max=float(self.mapper.last_saturation["max_command_excess"]),
            gripper_intent_saturation_count=int(
                self.mapper.last_saturation["intent_count"]
            ),
            gripper_intent_saturation_max=float(
                self.mapper.last_saturation["max_intent_excess"]
            ),
        )


class CalibratedChunkScheduler:
    def __init__(self, replan_steps: int, *, blend_steps: int = 6, gripper_blend_steps: int = 6, max_response_age_ms: float = 500.0):
        if not 1 <= replan_steps < ACTION_HORIZON:
            raise ValueError("replan_steps must be in [1,29]")
        if not 0 <= blend_steps < ACTION_HORIZON or not 0 <= gripper_blend_steps < ACTION_HORIZON:
            raise ValueError("blend steps must be in [0,29]")
        if not math.isfinite(max_response_age_ms) or max_response_age_ms <= 0:
            raise ValueError("max_response_age_ms must be finite and positive")
        self.max_response_age_ms = float(max_response_age_ms)
        self.replan_steps = int(replan_steps)
        self.blend_steps = int(blend_steps)
        self.gripper_blend_steps = int(gripper_blend_steps)
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
            or self.remaining <= self.replan_steps
        )

    @staticmethod
    def _skip(age_frames: float, offset_steps: int) -> int:
        return max(0, int(math.ceil(age_frames - offset_steps - 1e-9)))

    def adopt(
        self,
        chunk: CalibratedActionChunk,
        *,
        initial: bool = False,
        arrival_monotonic_ns: int | None = None,
    ) -> AdoptionInfo:
        if arrival_monotonic_ns is None:
            age_ms = chunk.round_trip_ms
        else:
            age_ms = max(0.0, (arrival_monotonic_ns - chunk.sample_monotonic_ns) / 1e6)
        if not math.isfinite(age_ms) or age_ms > self.max_response_age_ms:
            raise ProtocolError(f"observation-to-adoption age {age_ms:.1f} ms exceeds {self.max_response_age_ms:.1f} ms")
        age_frames = age_ms * FPS / 1000.0
        arm_skip = self._skip(age_frames, chunk.arm_offset_steps)
        grip_skip = self._skip(age_frames, chunk.gripper_offset_steps)
        length = min(ACTION_HORIZON - arm_skip, ACTION_HORIZON - grip_skip)
        if length <= 0:
            raise ProtocolError("calibrated chunk has no aligned future actions")
        native: list[np.ndarray] = []
        calibrated: list[np.ndarray] = []
        for offset in range(length):
            native_row = np.empty(ACTION_DIM, dtype=np.float32)
            calibrated_row = np.empty(ACTION_DIM, dtype=np.float32)
            native_row[ARM_INDICES] = chunk.native_actions[arm_skip + offset, ARM_INDICES]
            native_row[GRIPPER_INDICES] = chunk.native_actions[grip_skip + offset, GRIPPER_INDICES]
            calibrated_row[ARM_INDICES] = chunk.calibrated_actions[arm_skip + offset, ARM_INDICES]
            calibrated_row[GRIPPER_INDICES] = chunk.calibrated_actions[grip_skip + offset, GRIPPER_INDICES]
            native.append(native_row)
            calibrated.append(calibrated_row)

        old = self._steps[self._index :]
        arm_overlap = 0 if initial else min(self.blend_steps, len(old), length)
        grip_overlap = 0 if initial else min(self.gripper_blend_steps, len(old), length)
        scheduled: list[ScheduledAction] = []
        for offset, raw in enumerate(native):
            action = raw.copy()
            arm_alpha = 1.0
            grip_alpha = 1.0
            if offset < arm_overlap:
                progress = (offset + 1) / arm_overlap
                arm_alpha = progress * progress * (3.0 - 2.0 * progress)
                action[ARM_INDICES] = (
                    (1.0 - arm_alpha) * old[offset].action[ARM_INDICES]
                    + arm_alpha * raw[ARM_INDICES]
                )
            if offset < grip_overlap:
                progress = (offset + 1) / grip_overlap
                grip_alpha = progress * progress * (3.0 - 2.0 * progress)
                action[GRIPPER_INDICES] = (
                    (1.0 - grip_alpha) * old[offset].action[GRIPPER_INDICES]
                    + grip_alpha * raw[GRIPPER_INDICES]
                )
            scheduled.append(
                ScheduledAction(
                    action=action,
                    raw_action=raw.copy(),
                    calibrated_action=calibrated[offset].copy(),
                    request_id=chunk.request_id,
                    arm_source_index=arm_skip + offset,
                    gripper_source_index=grip_skip + offset,
                    arm_skipped=arm_skip,
                    gripper_skipped=grip_skip,
                    blend_alpha=float(arm_alpha),
                    gripper_blend_alpha=float(grip_alpha),
                    round_trip_ms=chunk.round_trip_ms,
                )
            )
        raw_jump = 0.0
        blended_jump = 0.0
        if self._last_action is not None:
            raw_jump = float(np.max(np.abs(native[0] - self._last_action)))
            blended_jump = float(np.max(np.abs(scheduled[0].action - self._last_action)))
        self._steps = scheduled
        self._index = 0
        self._published_since_adopt = 0
        return AdoptionInfo(
            arm_skipped=arm_skip,
            gripper_skipped=grip_skip,
            blended_steps=arm_overlap,
            gripper_blended_steps=grip_overlap,
            age_ms=age_ms,
            raw_boundary_jump_max=raw_jump,
            blended_boundary_jump_max=blended_jump,
            gripper_saturation_count=chunk.gripper_saturation_count,
            gripper_saturation_max=chunk.gripper_saturation_max,
            gripper_intent_saturation_count=chunk.gripper_intent_saturation_count,
            gripper_intent_saturation_max=chunk.gripper_intent_saturation_max,
        )

    def next_action(self) -> ScheduledAction:
        if self.remaining <= 0:
            raise BufferError("calibrated action chunk exhausted")
        value = self._steps[self._index]
        self._index += 1
        self._published_since_adopt += 1
        self._last_action = value.action.copy()
        return value


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ARM_INDICES",
    "ActionEMA",
    "AdoptionInfo",
    "CAMERA_NAMES",
    "CLIENT_ADAPTER_VERSION",
    "CalibratedActionChunk",
    "CalibratedChunkScheduler",
    "CalibratedHttpClient",
    "EXPERIMENTS",
    "FEEDBACK_CALIBRATION_VERSION",
    "FEEDBACK_PROTOCOL_VERSION",
    "FPS",
    "GRIPPER_INDICES",
    "JOINT_NAMES",
    "Observation",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "ScheduledAction",
    "recommended_replan_steps",
    "resolve_replan_steps",
]
