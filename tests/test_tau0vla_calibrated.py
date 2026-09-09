from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from tau0vla_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_VERSION,
    COMMAND_POINTS,
    CalibrationArtifact,
    CalibrationError,
    CalibratedGripperMapper,
    SideCalibration,
    feedback_pose_to_command,
    fit_side,
    load_training_ready_arms,
    return_trajectory,
    training_ready_targets,
    validate_artifact,
)
from tau0vla_calibrated_protocol import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    CalibratedActionChunk,
    CalibratedChunkScheduler,
    CalibratedHttpClient,
    FEEDBACK_CALIBRATION_VERSION,
    FEEDBACK_PROTOCOL_VERSION,
    Observation,
    PROTOCOL_VERSION,
)


def _fit(offset=0.05):
    samples = [np.full(90, command + offset) for command in COMMAND_POINTS]
    return fit_side(COMMAND_POINTS, samples, np.full(120, COMMAND_POINTS[0] + offset))


def _artifact():
    side = _fit()
    return CalibrationArtifact(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_version=CALIBRATION_VERSION,
        calibration_id="calibration",
        hostname="ark-2",
        ros_domain_id=63,
        boot_id="boot",
        created_unix_s=time.time(),
        created_monotonic_ns=1,
        controller_identity={
            "left": {"pid": 1, "start_ticks": 2},
            "right": {"pid": 3, "start_ticks": 4},
        },
        left=side,
        right=side,
    )


def _chunk(experiment: str, age_ms: float) -> CalibratedActionChunk:
    calibrated = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    native = np.zeros_like(calibrated)
    for index in range(ACTION_HORIZON):
        calibrated[index] = index
        native[index] = index
    return CalibratedActionChunk(
        calibrated_actions=calibrated,
        native_actions=native,
        request_id=1,
        sample_monotonic_ns=1_000_000_000,
        round_trip_ms=age_ms,
        inference_ms=40.0,
        model_id="model",
        experiment=experiment,
        arm_offset_steps=1,
        gripper_offset_steps=1 if experiment == "joint-feedback" else 0,
    )


def test_full_range_fit_and_identity_validation():
    fit = _fit()
    assert fit.slope == pytest.approx(1.0)
    assert fit.intercept == pytest.approx(0.05)
    assert fit.r_squared == pytest.approx(1.0)
    validate_artifact(
        _artifact(),
        hostname="ark-2",
        ros_domain_id=63,
        boot_id="boot",
        controller_identity={
            "left": {"pid": 1, "start_ticks": 2},
            "right": {"pid": 3, "start_ticks": 4},
        },
    )


def test_unstable_open_and_bad_fit_are_rejected():
    samples = [np.full(90, command) for command in COMMAND_POINTS]
    samples[0] = np.linspace(-3.4, -3.0, 90)
    with pytest.raises(CalibrationError, match="spread"):
        fit_side(COMMAND_POINTS, samples, np.full(120, COMMAND_POINTS[0]))
    flat = [np.full(90, 0.0) for _ in COMMAND_POINTS]
    with pytest.raises(CalibrationError, match="slope"):
        fit_side(COMMAND_POINTS, flat, np.zeros(120))


def test_joint_feedback_gripper_mapping_uses_feedback_fit():
    values = np.zeros((30, 14), dtype=np.float32)
    values[:, 6] = np.linspace(0.0, 3.28, 30)
    values[:, 13] = np.linspace(0.0, 3.28, 30)
    mapped = CalibratedGripperMapper(_artifact(), "joint-feedback").map_chunk(values)
    assert mapped[0, 6] == pytest.approx(-3.39, abs=1e-5)
    assert mapped[-1, 6] == pytest.approx(-0.11, abs=1e-5)


def test_joint_feedback_small_endpoint_error_saturates_but_large_error_rejects():
    values = np.zeros((30, 14), dtype=np.float32)
    values[:, 6] = -0.06
    values[:, 13] = 3.45
    mapper = CalibratedGripperMapper(_artifact(), "joint-feedback")
    mapped = mapper.map_chunk(values)
    np.testing.assert_allclose(mapped[:, 6], -3.39)
    np.testing.assert_allclose(mapped[:, 13], 0.0)
    assert mapper.last_saturation["count"] == 60
    assert mapper.last_saturation["max_command_excess"] == pytest.approx(0.06, abs=1e-5)

    # Covers the observed T-feedback worst case (0.07739 beyond the closed
    # endpoint) while retaining a hard rejection outside the reviewed 0.10.
    values[0, 13] = 3.46739
    mapped = mapper.map_chunk(values)
    assert mapped[0, 13] == pytest.approx(0.0)
    values[0, 6] = -0.11
    with pytest.raises(CalibrationError, match="soft tolerance"):
        mapper.map_chunk(values)


def test_joint_vr_gripper_mapping_and_range_rejection():
    values = np.zeros((30, 14), dtype=np.float32)
    values[:, [6, 13]] = np.linspace(0.0, 1.0, 30)[:, None]
    mapped = CalibratedGripperMapper(_artifact(), "joint-vr").map_chunk(values)
    assert mapped[0, 6] == pytest.approx(-3.39, abs=1e-5)
    assert mapped[-1, 6] == pytest.approx(0.0, abs=1e-5)
    values[5, 6] = 1.01
    values[6, 13] = -0.02
    mapped = CalibratedGripperMapper(_artifact(), "joint-vr")
    result = mapped.map_chunk(values)
    assert result[5, 6] == pytest.approx(0.0, abs=1e-5)
    assert result[6, 13] == pytest.approx(-3.39, abs=1e-5)
    assert mapped.last_saturation["intent_count"] == 2
    assert mapped.last_saturation["max_intent_excess"] == pytest.approx(0.02)
    values[5, 6] = 1.051
    with pytest.raises(CalibrationError, match="soft tolerance"):
        CalibratedGripperMapper(_artifact(), "joint-vr").map_chunk(values)


def test_feedback_and_vr_scheduler_use_component_offsets():
    arrival = 1_050_000_000  # 1.5 control frames after observation.
    feedback = CalibratedChunkScheduler(15, blend_steps=0, gripper_blend_steps=0)
    info = feedback.adopt(
        _chunk("joint-feedback", 50.0), arrival_monotonic_ns=arrival
    )
    assert (info.arm_skipped, info.gripper_skipped) == (1, 1)
    action = feedback.next_action()
    assert action.arm_source_index == action.gripper_source_index == 1

    vr = CalibratedChunkScheduler(15, blend_steps=0, gripper_blend_steps=0)
    info = vr.adopt(_chunk("joint-vr", 50.0), arrival_monotonic_ns=arrival)
    assert (info.arm_skipped, info.gripper_skipped) == (1, 2)
    action = vr.next_action()
    np.testing.assert_array_equal(action.action[:6], np.ones(6))
    assert action.action[6] == 2.0


def test_calibrated_scheduler_blends_after_mapping():
    scheduler = CalibratedChunkScheduler(5, blend_steps=2, gripper_blend_steps=2)
    scheduler.adopt(_chunk("joint-feedback", 0.0), initial=True, arrival_monotonic_ns=1_000_000_000)
    for _ in range(5):
        scheduler.next_action()
    replacement = _chunk("joint-feedback", 0.0)
    shifted = replacement.native_actions + 100
    replacement = CalibratedActionChunk(**{**replacement.__dict__, "native_actions": shifted, "request_id": 2})
    info = scheduler.adopt(replacement, arrival_monotonic_ns=1_000_000_000)
    assert info.blended_steps == info.gripper_blended_steps == 2
    first = scheduler.next_action()
    assert np.all(first.action > first.raw_action - 100)
    assert np.all(first.action < first.raw_action)


def test_http_client_validates_contract_and_maps_grippers():
    class Response:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def get(self, url, timeout):
            if url.endswith("/health"):
                return Response({
                    "status": "ok",
                    "ready": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "experiment": "joint-feedback",
                    "required_client_adapter_version": "arx-calibrated-client-v1",
                })
            return Response({
                "protocol_version": PROTOCOL_VERSION,
                "calibration_version": CALIBRATION_VERSION,
                "required_client_adapter_version": "arx-calibrated-client-v1",
                "experiment": "joint-feedback",
                "fps": 30,
                "source_fps": 60,
                "temporal_stride": 2,
                "camera_names": ["head", "left_wrist", "right_wrist"],
                "state_dim": 14,
                "action_dim": 14,
                "action_horizon": 30,
                "joint_names": [
                    *[f"left_j{i}" for i in range(6)], "left_gripper",
                    *[f"right_j{i}" for i in range(6)], "right_gripper",
                ],
                "wire_action_field": "calibrated_action_chunk",
                "wire_action_is_robot_command": False,
                "component_source_offsets": {"state": 0, "arm_action": 2, "gripper_action": 2},
                "model_id": "model",
            })

        def post(self, url, **kwargs):
            if url.endswith("/sessions"):
                return Response({
                    "protocol_version": PROTOCOL_VERSION,
                    "session_id": "session",
                    "model_id": "model",
                    "experiment": "joint-feedback",
                    "calibration_id": "calibration",
                })
            metadata = json.loads(kwargs["data"]["metadata"])
            return Response({
                "protocol_version": PROTOCOL_VERSION,
                "calibration_version": CALIBRATION_VERSION,
                "required_client_adapter_version": "arx-calibrated-client-v1",
                "session_id": "session",
                "request_id": metadata["request_id"],
                "sample_monotonic_ns": metadata["sample_monotonic_ns"],
                "model_id": "model",
                "experiment": "joint-feedback",
                "wire_action_is_robot_command": False,
                "calibrated_action_chunk": np.zeros((30, 14)).tolist(),
                "inference_ms": 40.0,
            })

    client = CalibratedHttpClient(
        "http://server",
        experiment="joint-feedback",
        calibration=_artifact(),
        robot_id="ark-2",
    )
    client.session = Session()
    client.health()
    client.policy_contract()
    client.create_session("pick")
    observation = Observation(
        qpos=np.zeros(14, dtype=np.float32),
        eef=np.zeros(14, dtype=np.float32),
        images={name: b"jpeg" for name in ("head", "left_wrist", "right_wrist")},
        sample_monotonic_ns=123,
    )
    result = client.infer(observation, 1)
    assert result.native_actions.shape == (30, 14)
    assert result.native_actions[0, 6] == pytest.approx(-3.39, abs=1e-5)


@pytest.mark.parametrize("eef", [None, np.full(14, np.nan), np.empty(0)])
def test_feedback_v4_client_uses_v4_session_path_and_omits_eef(eef):
    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.urls = []
            self.action_metadata = None

        def get(self, url, timeout):
            self.urls.append(url)
            if url.endswith("/health"):
                return Response({
                    "status": "ok",
                    "ready": True,
                    "protocol_version": FEEDBACK_PROTOCOL_VERSION,
                    "experiment": "joint-feedback",
                    "required_client_adapter_version": "arx-calibrated-client-v1",
                })
            return Response({
                "protocol_version": FEEDBACK_PROTOCOL_VERSION,
                "calibration_version": FEEDBACK_CALIBRATION_VERSION,
                "required_client_adapter_version": "arx-calibrated-client-v1",
                "experiment": "joint-feedback",
                "fps": 30,
                "offset_unit": "uploaded_30fps_frame",
                "camera_names": ["head", "left_wrist", "right_wrist"],
                "state_dim": 14,
                "action_dim": 14,
                "action_horizon": 30,
                "joint_names": [
                    *[f"left_j{i}" for i in range(6)], "left_gripper",
                    *[f"right_j{i}" for i in range(6)], "right_gripper",
                ],
                "wire_action_field": "calibrated_action_chunk",
                "wire_action_is_robot_command": False,
                "component_source_offsets": {"state": 0, "arm_action": 1, "gripper_action": 1},
                "model_id": "all-30k",
            })

        def post(self, url, **kwargs):
            self.urls.append(url)
            if url.endswith("/sessions"):
                return Response({
                    "protocol_version": FEEDBACK_PROTOCOL_VERSION,
                    "session_id": "v4-session",
                    "model_id": "all-30k",
                    "experiment": "joint-feedback",
                    "calibration_id": "calibration",
                })
            self.action_metadata = json.loads(kwargs["data"]["metadata"])
            return Response({
                "protocol_version": FEEDBACK_PROTOCOL_VERSION,
                "calibration_version": FEEDBACK_CALIBRATION_VERSION,
                "required_client_adapter_version": "arx-calibrated-client-v1",
                "session_id": "v4-session",
                "request_id": self.action_metadata["request_id"],
                "sample_monotonic_ns": self.action_metadata["sample_monotonic_ns"],
                "model_id": "all-30k",
                "experiment": "joint-feedback",
                "wire_action_is_robot_command": False,
                "calibrated_action_chunk": np.zeros((30, 14)).tolist(),
                "inference_ms": 40.0,
            })

    client = CalibratedHttpClient(
        "http://server",
        experiment="joint-feedback",
        calibration=_artifact(),
        robot_id="ark-2",
        protocol_version=FEEDBACK_PROTOCOL_VERSION,
    )
    session = Session()
    client.session = session
    client.health()
    client.policy_contract()
    client.create_session("pick")
    result = client.infer(
        Observation(
            qpos=np.zeros(14, dtype=np.float32),
            eef=eef,
            images={name: b"jpeg" for name in ("head", "left_wrist", "right_wrist")},
            sample_monotonic_ns=123,
        ),
        1,
    )
    assert any("/arx/v4/policy-contract" in url for url in session.urls)
    assert any("/arx/v4/sessions/v4-session/action-chunks" in url for url in session.urls)
    assert "raw_eef_feedback" not in session.action_metadata
    assert (result.arm_offset_steps, result.gripper_offset_steps) == (1, 1)


def test_ros_calibration_does_not_shadow_node_publishers_and_pre_settles():
    source = (ROOT / "act/tau0vla_calibrate_gripper.py").read_text(encoding="utf-8")
    assert "self._publishers =" not in source
    assert "self._command_publishers" in source
    assert "args.pre_settle_s" in source
    assert source.index("args.pre_settle_s") < source.index("node.sample_grippers(args.settle_s")


def test_return_trajectory_is_smooth_bounded_and_reaches_initial_pose():
    current = np.zeros(14, dtype=np.float32)
    target = np.ones(14, dtype=np.float32)
    trajectory = return_trajectory(
        current,
        target,
        rate_hz=30,
        minimum_duration_s=5.0,
        max_arm_step=.02,
        max_gripper_step=.05,
    )
    assert len(trajectory) >= 150
    np.testing.assert_allclose(trajectory[-1], target)
    arm_step = np.max(np.abs(np.diff(trajectory[:, [0, 1, 7, 8]], axis=0)))
    gripper_step = np.max(np.abs(np.diff(trajectory[:, [6, 13]], axis=0)))
    assert arm_step <= .02
    assert gripper_step <= .05
    assert np.all(np.diff(trajectory[:, 0]) >= 0)


def test_fixed_training_ready_pose_separates_gripper_command_and_feedback(tmp_path):
    config = tmp_path / "ready.yaml"
    config.write_text(
        "format_version: 1\n"
        "left_arm: [0, 1, 2, 3, 4, 5]\n"
        "right_arm: [6, 7, 8, 9, 10, 11]\n",
        encoding="utf-8",
    )
    arms = load_training_ready_arms(config)
    command, feedback = training_ready_targets(_artifact(), arms)
    np.testing.assert_allclose(command[[0, 1, 2, 3, 4, 5]], np.arange(6))
    np.testing.assert_allclose(command[[7, 8, 9, 10, 11, 12]], np.arange(6, 12))
    assert command[6] == pytest.approx(-3.39)
    assert feedback[6] == pytest.approx(-3.34)
    np.testing.assert_allclose(feedback_pose_to_command(_artifact(), feedback), command)


def test_one_command_rollout_orders_stack_calibration_policy_and_return():
    rollout = (ROOT / "tools/05_tau0vla_calibrated_rollout.sh").read_text(encoding="utf-8")
    client = (ROOT / "act/tau0vla_calibrated_client.py").read_text(encoding="utf-8")
    assert rollout.index("00_tau0vla_calibrated_up.sh") < rollout.index(
        "tau0vla_calibrate_gripper.py"
    ) < rollout.index("tau0vla_calibrated_client.py")
    profiles = (ROOT / "tools/tau0vla_robot_profile.sh").read_text()
    assert "expected_route=arx-lift2s-0907-blue-joint-feedback-ft" in profiles
    assert rollout.index("check_tau0vla_server") < rollout.index("00_tau0vla_calibrated_up.sh")
    assert "MOVE TO FIXED INITIAL POSE" in client
    assert "RETURN TO INITIAL POSE" in client
    assert "fixed-pose verification failed" in client
    assert "fixed_initial_feedback" in client
    assert "--auto-confirm" in rollout
    assert "--no-return-to-initial" not in rollout
    assert '00_tau0vla_calibrated_up.sh" --auto-confirm' in rollout
    assert "tau0vla_calibrate_gripper.py --execute --auto-confirm" in rollout
    assert ': "${LIFT_HEIGHT:=12.5}"' in rollout
    assert '--expected-height "${LIFT_HEIGHT}"' in rollout
    standalone = (ROOT / "tools/06_tau0vla_return_fixed.sh").read_text(encoding="utf-8")
    assert "--calibration-file" in standalone
    assert "--auto-confirm" in standalone
    assert "trace_" not in standalone
    assert "[t]au0vla_.*client.py" in standalone
    assert 'exec "${TAU0VLA_PYTHON}" tau0vla_calibrated_client.py' in rollout
    assert '2>&1 | tee -a "${client_log}"' not in rollout


def test_one_click_bringup_uses_profile_and_checks_reused_cameras():
    source = (ROOT / "tools/00_tau0vla_calibrated_up.sh").read_text(encoding="utf-8")
    assert "load_tau0vla_robot_profile" in source
    assert source.index("camera_h:${CAMERA_H_SERIAL}") < source.index(
        "camera_l:${CAMERA_L_SERIAL}"
    ) < source.index("camera_r:${CAMERA_R_SERIAL}")
    assert 'ros2 param get "/camera/${name}" serial_no' in source
    assert source.index("check_tau0vla_server") < source.index("00_can_up.sh")
    assert "tau0vla_calibrate_gripper" not in source
    assert "--auto-confirm" in source


def test_height_waiter_checks_stability_not_command_feedback_equality():
    source = (ROOT / "act/tau0vla_wait_height.py").read_text(encoding="utf-8")
    assert "is_safe_and_stable" in source
    assert "abs(float(values[-1]) - args.target)" not in source


def test_return_recovery_derives_fixed_pose_from_legacy_trace_metadata(tmp_path):
    from tau0vla_return_from_trace import load_target

    path = tmp_path / "trace.jsonl"
    ready = tmp_path / "ready.yaml"
    ready.write_text(
        "format_version: 1\n"
        "left_arm: [0, 1, 2, 3, 4, 5]\n"
        "right_arm: [6, 7, 8, 9, 10, 11]\n",
        encoding="utf-8",
    )
    path.write_text(
        '\n'.join([
            json.dumps({"event": "metadata", "calibration": _artifact().to_dict()}),
            json.dumps({"event": "tick", "feedback": [1.0]*14}),
            json.dumps({"event": "return_result", "status": "initial_pose", "target": [2.0]*14}),
        ]) + '\n',
        encoding="utf-8",
    )
    command, feedback, calibration, metadata = load_target(path, ready)
    assert calibration.calibration_id == "calibration"
    assert metadata["event"] == "metadata"
    np.testing.assert_allclose(command[[0, 1, 2, 3, 4, 5]], np.arange(6))
    np.testing.assert_allclose(command[[7, 8, 9, 10, 11, 12]], np.arange(6, 12))
    assert command[6] == pytest.approx(-3.39)
    assert feedback[6] == pytest.approx(-3.34)


@pytest.mark.parametrize("bad_eef", [None, [], [float("nan")] * 6])
def test_feedback_extraction_requires_eef_only_for_v3(bad_eef):
    from types import SimpleNamespace
    from tau0vla_calibrated_client import arm_feedback_vectors
    from tau0vla_calibrated_protocol import ProtocolError

    fields = {"joint_pos": list(range(7))}
    if bad_eef is not None:
        fields["end_pos"] = bad_eef
    left = SimpleNamespace(**fields)
    right = SimpleNamespace(**fields)
    joints, eef = arm_feedback_vectors(left, right, requires_eef=False)
    assert joints.shape == (14,)
    assert eef is None
    with pytest.raises(ProtocolError, match="EEF"):
        arm_feedback_vectors(left, right, requires_eef=True)
    left.joint_pos = [0.] * 6
    with pytest.raises(ProtocolError, match="joint feedback"):
        arm_feedback_vectors(left, right, requires_eef=False)


def test_v3_http_still_rejects_missing_eef():
    from tau0vla_calibrated_protocol import ProtocolError

    client = CalibratedHttpClient("http://server", experiment="joint-feedback",
                                  calibration=_artifact(), robot_id="ark-1")
    client.session_id, client.model_id = "session", "model"
    observation = Observation(qpos=np.zeros(14), eef=None,
                              images={}, sample_monotonic_ns=1)
    with pytest.raises(ProtocolError, match="EEF"):
        client.infer(observation, 1)


def test_preflight_only_client_cannot_create_session_without_calibration():
    from tau0vla_calibrated_protocol import ProtocolError

    client = CalibratedHttpClient("http://server", experiment="joint-feedback",
                                  calibration=None, robot_id="preflight")
    with pytest.raises(ProtocolError, match="real robot calibration"):
        client.create_session("pick")


@pytest.mark.parametrize("mode,exit_code", [("--check", 0), ("--dry-run", 1), ("--execute", 9)])
def test_rollout_no_motion_paths_and_failed_preflight(tmp_path, mode, exit_code):
    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hostname = fake_bin / "hostname"
    hostname.write_text('#!/bin/bash\necho ark-1\n')
    hostname.chmod(0o755)
    calls = tmp_path / "calls"
    python = fake_bin / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_CALLS"\nexit "$TEST_PREFLIGHT_EXIT"\n')
    python.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
           "ROS_DOMAIN_ID": "62", "TAU0VLA_PYTHON": str(python),
           "TEST_CALLS": str(calls), "TEST_PREFLIGHT_EXIT": "9" if mode == "--execute" else "0",
           "CALIBRATION_FILE": "", "MODEL_PROFILE": "all-blue-feedback"}
    for key in ("CAMERA_H_SERIAL", "CAMERA_L_SERIAL", "CAMERA_R_SERIAL", "LIFT_HEIGHT"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(ROOT / "tools/05_tau0vla_calibrated_rollout.sh"), mode],
                             env=env, text=True, capture_output=True)
    assert result.returncode == exit_code, result.stderr
    assert calls.read_text().count("tau0vla_server_preflight.py") == 1
    assert "tau0vla_calibrate_gripper.py" not in calls.read_text()
    assert "tau0vla_calibrated_client.py" not in calls.read_text()
    assert "/opt/ros" not in result.stderr  # These paths exit even before ROS setup.
    if mode == "--check":
        assert "lift=12.5" in result.stdout
        assert "head=260522272299, left=260422271992, right=260522274175" in result.stdout
    elif mode == "--dry-run":
        assert "requires CALIBRATION_FILE" in result.stderr


@pytest.mark.parametrize("host,domain,expected", [("ark-1", "62", 0), ("ark-2", "63", 0),
                                                  ("ark-1", "63", 1), ("new-arx", "62", 1)])
def test_robot_profile_rejects_wrong_domain_and_unknown_robot(host, domain, expected):
    import os
    import shlex
    import subprocess

    env = {**os.environ, "ROS_DOMAIN_ID": domain}
    script = (f"set -eu; hostname() {{ echo {shlex.quote(host)}; }}; "
              f"repo_root={shlex.quote(str(ROOT))}; "
              'source "$repo_root/tools/tau0vla_robot_profile.sh"; load_tau0vla_robot_profile')
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    assert result.returncode == expected, result.stderr


def test_scheduler_rejects_chunk_delayed_after_fast_http_response():
    from tau0vla_calibrated_protocol import ProtocolError

    scheduler = CalibratedChunkScheduler(15, max_response_age_ms=500)
    chunk = _chunk("joint-feedback", 40.0)
    # A fast HTTP response may still sit in a future/UI prompt for too long.
    with pytest.raises(ProtocolError, match="observation-to-adoption age"):
        scheduler.adopt(chunk, arrival_monotonic_ns=chunk.sample_monotonic_ns + 501_000_000)
    assert scheduler.remaining == 0
    permissive = CalibratedChunkScheduler(15, max_response_age_ms=2000)
    with pytest.raises(ProtocolError, match="no aligned future actions"):
        permissive.adopt(chunk, arrival_monotonic_ns=chunk.sample_monotonic_ns + 1_500_000_000)


def test_initial_action_is_requested_after_manual_confirmation_and_buffer_failure_stops():
    source = (ROOT / "act/tau0vla_calibrated_client.py").read_text()
    assert source.index('"Type EXECUTE CALIBRATED TAU0VLA') < source.index("first = client.infer")
    assert 'raise ProtocolError("BUFFER STARVED:' in source


@pytest.mark.parametrize("mutation", [None, "route", "protocol_version", "calibration_version", "camera_names", "model_id", "checkpoint_sha256", "rtc_enabled"])
def test_server_preflight_validates_contract_without_posting(monkeypatch, mutation):
    from tau0vla_calibrated_protocol import JOINT_NAMES, ProtocolError
    from tau0vla_server_preflight import check_server
    import requests

    health = {"status": "ok", "ready": True, "route": "all-route", "model_id": "model",
              "checkpoint_sha256": "sha", "protocol_version": FEEDBACK_PROTOCOL_VERSION,
              "experiment": "joint-feedback", "required_client_adapter_version": "arx-calibrated-client-v1"}
    contract = {**health, "calibration_version": FEEDBACK_CALIBRATION_VERSION,
                "fps": 30, "camera_names": ["head", "left_wrist", "right_wrist"],
                "state_dim": 14, "action_dim": 14, "action_horizon": 30,
                "joint_names": list(JOINT_NAMES), "wire_action_field": "calibrated_action_chunk",
                "wire_action_is_robot_command": False, "offset_unit": "uploaded_30fps_frame",
                "component_source_offsets": {"arm_action": 1, "gripper_action": 1}, "rtc_enabled": False}
    if mutation == "route":
        health[mutation] = "wrong"
    elif mutation is not None:
        contract[mutation] = True if mutation == "rtc_enabled" else "wrong"
    calls = []

    class Response:
        def __init__(self, value): self.value = value
        def raise_for_status(self): pass
        def json(self): return self.value

    class Session:
        def get(self, url, timeout):
            calls.append(url)
            return Response(health if url.endswith("/health") else contract)
        def close(self): pass
        def post(self, *args, **kwargs): pytest.fail("preflight must never create a session")

    monkeypatch.setattr(requests, "Session", Session)
    if mutation is None:
        result = check_server("http://server", route="all-route", experiment="joint-feedback",
                              protocol_version=FEEDBACK_PROTOCOL_VERSION)
        assert result["health"]["route"] == "all-route"
        assert calls == ["http://server/health", "http://server/arx/v4/policy-contract"]
    else:
        with pytest.raises(ProtocolError):
            check_server("http://server", route="all-route", experiment="joint-feedback",
                         protocol_version=FEEDBACK_PROTOCOL_VERSION)


@pytest.mark.parametrize("guard", ["lock", "process"])
def test_rollout_conflicting_execution_guard_precedes_hardware(tmp_path, guard):
    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, content in {
        "hostname": "echo ark-1",
        "python": "exit 0",
        "flock": "exit 1" if guard == "lock" else "exit 0",
        "pgrep": "exit 0",
    }.items():
        path = fake_bin / name
        path.write_text(f"#!/bin/bash\n{content}\n")
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "ROS_DOMAIN_ID": "62",
           "TAU0VLA_PYTHON": str(fake_bin / "python"), "MODEL_PROFILE": "all-blue-feedback"}
    result = subprocess.run(["bash", str(ROOT / "tools/05_tau0vla_calibrated_rollout.sh"), "--execute"],
                             env=env, text=True, capture_output=True)
    assert result.returncode == 1
    assert ("another rollout" if guard == "lock" else "process is still active") in result.stderr
    assert "/opt/ros" not in result.stderr
