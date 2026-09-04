from __future__ import annotations

import queue
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

import human_dagger_tau0vla_policy as worker_module  # noqa: E402
from human_dagger_tau0vla_policy import (  # noqa: E402
    Tau0VLAWorkerConfig,
    tau0vla_policy_worker_main,
)
from tau0vla_protocol import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ActionChunk,
    Observation,
    ProtocolError,
)

CAMERAS = ("head", "left_wrist", "right_wrist")


class FakeClient:
    """Stands in for Tau0VLAHttpClient: no network, scripted failures."""

    def __init__(self, base_url, *, request_timeout=5.0, max_response_age_ms=2000.0):
        self.base_url = base_url
        self.model_id = "fake-model"
        self.session_id = None
        self.infer_calls = 0
        self.fail_health = False
        self.fail_infer_after: int | None = None

    def health(self):
        if self.fail_health:
            raise ProtocolError("server not ready")
        return {"status": "ok", "ready": True}

    def policy_contract(self):
        return {}

    def create_session(self, task_instruction):
        self.session_id = "session"
        return {"session_id": "session"}

    def infer(self, observation, request_id):
        self.infer_calls += 1
        if self.fail_infer_after is not None and self.infer_calls > self.fail_infer_after:
            raise ProtocolError("scripted inference failure")
        actions = np.full((ACTION_HORIZON, ACTION_DIM), float(request_id), dtype=np.float32)
        return ActionChunk(
            actions=actions,
            request_id=request_id,
            sample_monotonic_ns=observation.sample_monotonic_ns,
            round_trip_ms=50.0,
            inference_ms=40.0,
            model_id=self.model_id,
        )


def _calibration_stub() -> Observation:
    return Observation(
        qpos=np.zeros(ACTION_DIM, dtype=np.float32),
        images={name: b"jpeg" for name in CAMERAS},
        sample_monotonic_ns=time.monotonic_ns(),
    )


def _config(**overrides) -> Tau0VLAWorkerConfig:
    values = {
        "server_url": "http://fake",
        "task_instruction": "pick the handle",
        "benchmark_warmup": 1,
        "benchmark_requests": 2,
    }
    values.update(overrides)
    return Tau0VLAWorkerConfig(**values)


def _observation_message(epoch: int, seq: int, *, basis_ns: int | None = None) -> dict:
    now = time.monotonic_ns()
    basis = now if basis_ns is None else basis_ns
    return {
        "kind": "observation",
        "episode_id": 7,
        "control_epoch": epoch,
        "observation_seq": seq,
        "observation": {
            "qpos": [0.0] * ACTION_DIM,
            "images_jpeg": {name: b"jpeg" for name in CAMERAS},
            "policy_basis_ns": basis,
            "timestamps": {"observation_ns": now},
        },
    }


def _run_worker(config, control_items, observation_items):
    """Feed scripted queues through the worker until the trailing stop."""

    control_queue: queue.Queue = queue.Queue()
    observation_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    status_queue: queue.Queue = queue.Queue()
    for item in control_items:
        control_queue.put(item)
    for item in observation_items:
        observation_queue.put(item)
    with mock.patch.object(worker_module, "Tau0VLAHttpClient", FakeClient), \
            mock.patch.object(worker_module, "_calibration_observation", _calibration_stub):
        tau0vla_policy_worker_main(
            config, control_queue, observation_queue, result_queue, status_queue
        )
    statuses = []
    while not status_queue.empty():
        statuses.append(status_queue.get_nowait())
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    return statuses, results


class Tau0VLAWorkerTests(unittest.TestCase):
    def test_init_failure_reports_policy_error_without_epoch(self):
        with mock.patch.object(worker_module, "Tau0VLAHttpClient") as factory:
            factory.side_effect = ProtocolError("server not ready")
            control_queue: queue.Queue = queue.Queue()
            observation_queue: queue.Queue = queue.Queue()
            result_queue: queue.Queue = queue.Queue()
            status_queue: queue.Queue = queue.Queue()
            tau0vla_policy_worker_main(
                _config(), control_queue, observation_queue, result_queue, status_queue
            )
        status = status_queue.get_nowait()
        self.assertEqual(status["kind"], "policy_error")
        self.assertNotIn("control_epoch", status)

    def test_reset_ack_then_observation_yields_action(self):
        statuses, results = _run_worker(
            _config(),
            control_items=[{"kind": "reset", "control_epoch": 3}],
            observation_items=[_observation_message(3, 1), {"kind": "stop"}],
        )
        kinds = [status["kind"] for status in statuses]
        self.assertEqual(kinds[0], "policy_ready")
        self.assertIn("policy_reset_ack", kinds)
        ack = next(s for s in statuses if s["kind"] == "policy_reset_ack")
        self.assertEqual(ack["control_epoch"], 3)
        self.assertEqual(len(results), 1)
        action_message = results[0]
        self.assertEqual(action_message["kind"], "policy_action")
        self.assertEqual(action_message["control_epoch"], 3)
        self.assertEqual(action_message["observation_seq"], 1)
        self.assertEqual(action_message["episode_id"], 7)
        self.assertEqual(action_message["action_seq"], 1)
        action = np.asarray(action_message["action"])
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_action_seq_is_monotonic_across_resets(self):
        statuses, results = _run_worker(
            _config(),
            control_items=[{"kind": "reset", "control_epoch": 1}],
            observation_items=[
                _observation_message(1, 1),
                {"kind": "reset", "control_epoch": 2},
                _observation_message(2, 2),
                {"kind": "stop"},
            ],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["action_seq"], 1)
        self.assertEqual(results[1]["action_seq"], 2)
        self.assertEqual(results[1]["control_epoch"], 2)

    def test_stale_observation_is_dropped(self):
        stale_ns = time.monotonic_ns() - 1_000_000_000
        statuses, results = _run_worker(
            _config(),
            control_items=[{"kind": "reset", "control_epoch": 1}],
            observation_items=[
                _observation_message(1, 1, basis_ns=stale_ns),
                {"kind": "stop"},
            ],
        )
        self.assertEqual(results, [])
        dropped = [s for s in statuses if s["kind"] == "policy_observation_dropped"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["control_epoch"], 1)

    def test_epoch_mismatch_observation_is_silently_ignored(self):
        statuses, results = _run_worker(
            _config(),
            control_items=[{"kind": "reset", "control_epoch": 5}],
            observation_items=[_observation_message(4, 1), {"kind": "stop"}],
        )
        self.assertEqual(results, [])
        kinds = {status["kind"] for status in statuses}
        self.assertNotIn("policy_observation_dropped", kinds)
        self.assertNotIn("policy_error", kinds)

    def test_pause_gates_subsequent_observations(self):
        statuses, results = _run_worker(
            _config(),
            control_items=[],
            observation_items=[
                {"kind": "reset", "control_epoch": 1},
                _observation_message(1, 1),
                {"kind": "pause"},
                _observation_message(1, 2),
                {"kind": "stop"},
            ],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["observation_seq"], 1)

    def test_inference_failure_reports_policy_error_and_gates_epoch(self):
        def scripted_client(*args, **kwargs):
            client = FakeClient(*args, **kwargs)
            client.fail_infer_after = 3  # calibration (1 warmup + 2 measured) passes
            return client

        control_queue: queue.Queue = queue.Queue()
        observation_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        status_queue: queue.Queue = queue.Queue()
        control_queue.put({"kind": "reset", "control_epoch": 1})
        observation_queue.put(_observation_message(1, 1))
        observation_queue.put(_observation_message(1, 2))
        observation_queue.put({"kind": "stop"})
        with mock.patch.object(worker_module, "Tau0VLAHttpClient", scripted_client), \
                mock.patch.object(worker_module, "_calibration_observation", _calibration_stub):
            tau0vla_policy_worker_main(
                _config(), control_queue, observation_queue, result_queue, status_queue
            )
        statuses = []
        while not status_queue.empty():
            statuses.append(status_queue.get_nowait())
        errors = [s for s in statuses if s["kind"] == "policy_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["control_epoch"], 1)
        # After the error the epoch is gated: the second observation must not
        # produce an action.
        self.assertTrue(result_queue.empty())


if __name__ == "__main__":
    unittest.main()
