"""FastAPI HTTP v1 server with a replaceable PolicyAdapter."""

from __future__ import annotations

import json
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from act.pipeline_contract import ACTION_DIM, ACTION_SEMANTICS, CAMERA_NAMES, FPS, HTTP_PROTOCOL_VERSION
from model_server.policy_adapter import MockPolicy, PolicyAdapter


class ResetRequest(BaseModel):
    protocol_version: str
    session_id: str
    task_instruction: str


def create_app(policy: PolicyAdapter | None = None) -> FastAPI:
    adapter = policy or MockPolicy()
    adapter.load()
    app = FastAPI(title="ARX LIFT2s Policy Server", version=HTTP_PROTOCOL_VERSION)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model_id": adapter.model_id}

    @app.get("/v1/schema")
    def schema():
        return {
            "protocol_version": HTTP_PROTOCOL_VERSION,
            "fps": FPS,
            "camera_names": list(CAMERA_NAMES),
            "state_dim": ACTION_DIM,
            "action_dim": ACTION_DIM,
            "action_semantics": ACTION_SEMANTICS,
            "max_action_horizon": 64,
            "model_id": adapter.model_id,
        }

    @app.post("/v1/reset")
    def reset(request: ResetRequest):
        if request.protocol_version != HTTP_PROTOCOL_VERSION or not request.task_instruction.strip():
            raise HTTPException(status_code=409, detail="protocol mismatch or empty task")
        adapter.reset(request.session_id, request.task_instruction)
        return {"ok": True, "session_id": request.session_id, "model_id": adapter.model_id}

    @app.post("/v1/infer")
    async def infer(
        metadata: str = Form(...),
        head: UploadFile = File(...),
        left_wrist: UploadFile = File(...),
        right_wrist: UploadFile = File(...),
    ):
        try:
            request = json.loads(metadata)
            if request["protocol_version"] != HTTP_PROTOCOL_VERSION:
                raise ValueError("protocol mismatch")
            state = np.asarray(request["observation_state"], dtype=np.float32)
            if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
                raise ValueError("observation_state must be a finite 14-vector")
            if not str(request["task_instruction"]).strip():
                raise ValueError("task_instruction is empty")
            images = {
                "head": await head.read(),
                "left_wrist": await left_wrist.read(),
                "right_wrist": await right_wrist.read(),
            }
            if any(not value for value in images.values()):
                raise ValueError("empty JPEG payload")
            started = time.monotonic()
            actions = np.asarray(adapter.infer(state, images, request["task_instruction"]), dtype=np.float32)
            inference_ms = (time.monotonic() - started) * 1000.0
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not 1 <= actions.shape[0] <= 64:
                raise ValueError(f"policy returned invalid shape {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("policy returned NaN or Inf")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "protocol_version": HTTP_PROTOCOL_VERSION,
            "session_id": request["session_id"],
            "request_id": request["request_id"],
            "actions": actions.tolist(),
            "action_dt": 1.0 / FPS,
            "action_semantics": ACTION_SEMANTICS,
            "model_id": adapter.model_id,
            "inference_ms": inference_ms,
        }

    return app


app = create_app()
