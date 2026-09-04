from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from tau0vla_protocol import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    PROTOCOL_VERSION,
    ActionEMA,
    ActionChunk,
    ChunkScheduler,
    Observation,
    ProtocolError,
    Tau0VLAHttpClient,
    recommended_replan_steps,
)
from tau0vla_trace import TraceWriter, analyze_trace  # noqa: E402


def _chunk(round_trip_ms: float, base: float = 0.0) -> ActionChunk:
    actions = np.arange(ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(ACTION_HORIZON, ACTION_DIM)
    return ActionChunk(
        actions=actions + base,
        request_id=1,
        sample_monotonic_ns=1,
        round_trip_ms=round_trip_ms,
        inference_ms=round_trip_ms - 10.0,
        model_id="test",
    )


class ChunkSchedulerTest(unittest.TestCase):
    def test_initial_chunk_is_not_skipped_and_replan_is_due(self):
        scheduler = ChunkScheduler(replan_steps=4)
        self.assertEqual(scheduler.adopt(_chunk(500.0), initial=True).skipped, 0)
        self.assertEqual(scheduler.remaining, ACTION_HORIZON)
        for _ in range(4):
            scheduler.next_action()
        self.assertTrue(scheduler.should_request(request_pending=False))
        self.assertFalse(scheduler.should_request(request_pending=True))

    def test_replacement_skips_rtt_equivalent_prefix(self):
        scheduler = ChunkScheduler(replan_steps=10, blend_steps=0)
        scheduler.adopt(_chunk(0.0), initial=True)
        for _ in range(10):
            scheduler.next_action()
        replacement = _chunk(100.0, base=1000.0)
        info = scheduler.adopt(replacement)
        self.assertEqual(info.skipped, 3)
        np.testing.assert_array_equal(scheduler.next_action().action, replacement.actions[3])

    def test_time_alignment_keeps_first_target_before_one_period(self):
        chunk = _chunk(20.0)
        object.__setattr__(chunk, "sample_monotonic_ns", 1_000_000_000)
        scheduler = ChunkScheduler(replan_steps=10, blend_steps=0)
        info = scheduler.adopt(chunk, arrival_monotonic_ns=1_020_000_000)
        self.assertEqual(info.skipped, 0)
        info = scheduler.adopt(chunk, arrival_monotonic_ns=1_040_000_000)
        self.assertEqual(info.skipped, 1)

    def test_smoothstep_blends_aligned_chunk_boundary(self):
        scheduler = ChunkScheduler(replan_steps=10, blend_steps=6)
        first = _chunk(0.0)
        scheduler.adopt(first, initial=True)
        for _ in range(10):
            scheduler.next_action()
        replacement = _chunk(100.0, base=1000.0)
        info = scheduler.adopt(replacement)
        self.assertEqual(info.skipped, 3)
        self.assertEqual(info.blended_steps, 6)
        alpha = (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
        expected = (1.0 - alpha) * first.actions[10] + alpha * replacement.actions[3]
        step = scheduler.next_action()
        np.testing.assert_allclose(step.action, expected, atol=1e-5)
        np.testing.assert_array_equal(step.raw_action, replacement.actions[3])
        self.assertLess(info.blended_boundary_jump_max, info.raw_boundary_jump_max)

    def test_delayed_short_replacement_prefetches_immediately(self):
        scheduler = ChunkScheduler(replan_steps=15)
        scheduler.adopt(_chunk(0.0), initial=True)
        replacement = _chunk(700.0, base=1000.0)
        info = scheduler.adopt(replacement)
        self.assertEqual(info.skipped, 21)
        self.assertEqual(scheduler.remaining, 9)
        self.assertTrue(scheduler.should_request(request_pending=False))
        self.assertFalse(scheduler.should_request(request_pending=True))

    def test_invalid_chunk_and_starvation_fail(self):
        scheduler = ChunkScheduler(replan_steps=10)
        with self.assertRaises(BufferError):
            scheduler.next_action()
        invalid = _chunk(1.0)
        object.__setattr__(invalid, "actions", np.zeros((2, ACTION_DIM), dtype=np.float32))
        with self.assertRaises(ProtocolError):
            scheduler.adopt(invalid)

    def test_auto_replan_and_latency_rejection(self):
        steps, p99 = recommended_replan_steps([300.0] * 30)
        self.assertEqual(steps, 15)
        self.assertEqual(p99, 300.0)
        with self.assertRaises(ProtocolError):
            recommended_replan_steps([900.0] * 30, margin_ms=100.0)

    def test_optional_ema_is_identity_by_default_and_smooths_arms_only(self):
        action0 = np.zeros(ACTION_DIM, dtype=np.float32)
        action1 = np.ones(ACTION_DIM, dtype=np.float32)
        identity = ActionEMA()
        identity.apply(action0)
        np.testing.assert_array_equal(identity.apply(action1), action1)

        filtered = ActionEMA(arm_alpha=0.4, gripper_alpha=1.0)
        filtered.reset(action0)
        result = filtered.apply(action1)
        np.testing.assert_allclose(result[[0, 5, 7, 12]], 0.4)
        np.testing.assert_allclose(result[[6, 13]], 1.0)

    def test_trace_reports_boundary_and_tracking_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            writer = TraceWriter(path)
            scheduler = ChunkScheduler(replan_steps=4, blend_steps=0)
            first = _chunk(0.0)
            info = scheduler.adopt(first, initial=True)
            writer.adoption(first.request_id, info)
            for control_step in range(3):
                scheduled = scheduler.next_action()
                writer.tick(
                    monotonic_ns=control_step + 1,
                    control_step=control_step,
                    scheduled=scheduled,
                    command=scheduled.action,
                    feedback=scheduled.action + 0.1,
                    execute=False,
                )
            writer.close()
            summary = analyze_trace(path)
        self.assertEqual(summary["ticks"], 3)
        self.assertAlmostEqual(summary["tracking_error_max"], 0.1, places=5)


class HttpClientTest(unittest.TestCase):
    def test_health_contract_session_and_chunk(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return

            def json(self):
                return self.payload

        class Session:
            def get(self, url, timeout):
                if url.endswith("/health"):
                    return Response({"status": "ok", "ready": True})
                return Response(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "fps": 30,
                        "camera_names": ["head", "left_wrist", "right_wrist"],
                        "state_dim": 14,
                        "action_dim": 14,
                        "action_horizon": 30,
                        "action_semantics": "state_t_plus_1",
                        "joint_names": [
                            *[f"left_j{i}" for i in range(6)], "left_gripper",
                            *[f"right_j{i}" for i in range(6)], "right_gripper",
                        ],
                        "model_id": "test-model",
                    }
                )

            def post(self, url, **kwargs):
                if url.endswith("/sessions"):
                    return Response(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "session_id": "session",
                            "model_id": "test-model",
                        }
                    )
                metadata = json.loads(kwargs["data"]["metadata"])
                return Response(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "session_id": "session",
                        "request_id": metadata["request_id"],
                        "sample_monotonic_ns": metadata["sample_monotonic_ns"],
                        "actions": np.zeros((30, 14), dtype=np.float32).tolist(),
                        "action_dt": 1.0 / 30.0,
                        "action_semantics": "state_t_plus_1",
                        "inference_ms": 10.0,
                        "model_id": "test-model",
                    }
                )

        client = Tau0VLAHttpClient("http://server")
        client.session = Session()
        client.health()
        client.policy_contract()
        client.create_session("pick")
        observation = Observation(
            qpos=np.zeros(14, dtype=np.float32),
            images={name: b"jpeg" for name in ("head", "left_wrist", "right_wrist")},
            sample_monotonic_ns=123,
        )
        chunk = client.infer(observation, 1)
        self.assertEqual(chunk.actions.shape, (30, 14))


if __name__ == "__main__":
    unittest.main()
