from __future__ import annotations

import queue
import sys
import time
import unittest
from concurrent.futures import Future
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


def _run_worker(config, control_items, observation_items, *, clock=None):
    """Feed scripted queues through the worker until the trailing stop."""

    control_queue: queue.Queue = queue.Queue()
    class ClockedQueue(queue.Queue):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            if clock is not None:
                clock[0] = item.get("test_time_ns", clock[0])
            return item

    observation_queue: queue.Queue = ClockedQueue()
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
    def test_new_session_starts_at_one_but_policy_reset_keeps_request_sequence(self):
        class SequentialClient(FakeClient):
            def create_session(self, task_instruction):
                self.last_request_id = 0
                return super().create_session(task_instruction)

            def infer(self, observation, request_id):
                if request_id != self.last_request_id + 1:
                    raise ProtocolError(f"request_id {request_id} does not follow {self.last_request_id}")
                self.last_request_id = request_id
                return super().infer(observation, request_id)

        with mock.patch.object(worker_module, "Tau0VLAHttpClient", SequentialClient), \
                mock.patch.object(worker_module, "_calibration_observation", _calibration_stub):
            runtime = worker_module._Tau0VLARuntime(_config())
            self.assertEqual(runtime.client.infer_calls, 3)
            for expected in (1, 2, 3):
                runtime.reset()
                request_id = runtime.next_request_id()
                self.assertEqual(request_id, expected)
                runtime.client.infer(_calibration_stub(), request_id)

    @staticmethod
    def timed_observation(epoch, seq, elapsed_ns):
        stamp = 1_000_000_000 + elapsed_ns
        message = _observation_message(epoch, seq, basis_ns=stamp)
        message["observation"]["timestamps"]["observation_ns"] = stamp
        message["test_time_ns"] = stamp
        return message

    def test_sixty_hz_observations_only_advance_thirty_hz_actions(self):
        class ImmediateExecutor:
            def __init__(self, **kwargs): pass
            def submit(self, fn, *args):
                result = Future()
                result.set_result(fn(*args))
                return result
            def shutdown(self, **kwargs): pass

        clock = [1_000_000_000]
        observations = [
            self.timed_observation(1, i + 1, (i * 1_000_000_000 + 59) // 60)
            for i in range(61)
        ]
        with mock.patch.object(worker_module.time, "monotonic_ns", lambda: clock[0]), \
                mock.patch.object(worker_module, "ThreadPoolExecutor", ImmediateExecutor):
            statuses, results = _run_worker(
                _config(), [{"kind": "reset", "control_epoch": 1}],
                observations + [{"kind": "stop"}], clock=clock,
            )
        self.assertFalse([s for s in statuses if s["kind"] == "policy_error"])
        self.assertEqual([r["observation_seq"] for r in results], list(range(1, 62, 2)))
        self.assertEqual(sum(s["kind"] == "policy_observation_dropped" for s in statuses), 30)

    def test_pending_failure_is_ignored_only_when_its_epoch_is_old(self):
        class FailedRequestExecutor:
            def __init__(self, **kwargs): pass
            def submit(self, *args):
                result = Future()
                result.set_exception(ProtocolError("HTTP request failed"))
                return result
            def shutdown(self, **kwargs): pass

        for reset in (False, True):
            with self.subTest(reset=reset):
                clock = [1_000_000_000]
                messages = [
                    self.timed_observation(1, i + 1, (i * 1_000_000_000 + 29) // 30)
                    for i in range(16)
                ]
                if reset:
                    messages += [{"kind": "pause"}, {"kind": "reset", "control_epoch": 2}]
                messages += [self.timed_observation(2 if reset else 1, 17, 600_000_000), {"kind": "stop"}]
                with mock.patch.object(worker_module.time, "monotonic_ns", lambda: clock[0]), \
                        mock.patch.object(worker_module, "ThreadPoolExecutor", FailedRequestExecutor):
                    statuses, results = _run_worker(
                        _config(), [{"kind": "reset", "control_epoch": 1}], messages, clock=clock,
                    )
                errors = [s for s in statuses if s["kind"] == "policy_error"]
                if reset:
                    self.assertEqual(errors, [])
                    self.assertEqual(results[-1]["control_epoch"], 2)
                else:
                    self.assertEqual(len(errors), 1)
                    self.assertEqual(errors[0]["control_epoch"], 1)

    def test_replan_budget_matches_standalone(self):
        from tau0vla_client import _resolve_replan_steps
        expected_steps = {"auto": 15, "1": 1, "5": 5, "15": 15, "20": 20, "25": 25,
                          "26": None, "29": None, "0": None, "30": None}
        for value, expected in expected_steps.items():
            with self.subTest(value=value):
                if expected is None:
                    with self.assertRaises((ProtocolError, ValueError)):
                        _resolve_replan_steps(value, [50.0, 50.0], 100.0)
                else:
                    self.assertEqual(_resolve_replan_steps(value, [50.0, 50.0], 100.0), (expected, 50.0))
                with mock.patch.object(worker_module, "Tau0VLAHttpClient", FakeClient), \
                        mock.patch.object(worker_module, "_calibration_observation", _calibration_stub):
                    if expected is None:
                        with self.assertRaises((ProtocolError, ValueError)):
                            worker_module._Tau0VLARuntime(_config(replan_steps=value))
                    else:
                        runtime = worker_module._Tau0VLARuntime(_config(replan_steps=value))
                        self.assertEqual(runtime.replan_steps, expected)

    def test_late_observations_do_not_burst_catch_up(self):
        clock = [1_000_000_000]
        messages = [
            self.timed_observation(1, i + 1, elapsed_ms * 1_000_000)
            for i, elapsed_ms in enumerate((0, 16, 34, 35, 101, 102, 134))
        ]
        with mock.patch.object(worker_module.time, "monotonic_ns", lambda: clock[0]):
            statuses, results = _run_worker(
                _config(), [{"kind": "reset", "control_epoch": 1}],
                messages + [{"kind": "stop"}], clock=clock,
            )
        self.assertFalse([s for s in statuses if s["kind"] == "policy_error"])
        self.assertEqual([r["observation_seq"] for r in results], [1, 3, 5, 7])

    def test_gripper_pipeline_matches_standalone_and_resets(self):
        from tau0vla_protocol import ActionEMA, BinaryGripperStabilizer
        clock = [1_000_000_000]
        config = _config(gripper_debounce_frames=3, gripper_blend_steps=0,
                         arm_ema_alpha=0.6, gripper_ema_alpha=0.6)
        messages = [self.timed_observation(1, i + 1, i * 40_000_000) for i in range(4)]
        for message in messages:
            message["observation"]["qpos"][6] = -6.77
            message["observation"]["qpos"][13] = -6.77
        # A reset must seed the filters from the NEW measured pose, not the old EMA.
        messages += [{"kind": "reset", "control_epoch": 2}, self.timed_observation(2, 5, 121_000_000), {"kind": "stop"}]
        with mock.patch.object(worker_module.time, "monotonic_ns", lambda: clock[0]):
            statuses, results = _run_worker(config, [{"kind": "reset", "control_epoch": 1}], messages, clock=clock)
        self.assertFalse([s for s in statuses if s["kind"] == "policy_error"])
        self.assertEqual(len(results), 5)
        stabilizer = BinaryGripperStabilizer(3)
        ema = ActionEMA(0.6, 0.6)
        initial = np.asarray(messages[0]["observation"]["qpos"])
        stabilizer.reset(initial)
        ema.reset(stabilizer.apply(initial))
        for result in results[:4]:
            expected = ema.apply(stabilizer.apply(np.full(14, 1.0)))
            np.testing.assert_allclose(result["action"], expected)
        np.testing.assert_allclose(results[-1]["action"][[6, 13]], [0.0, 0.0])

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
