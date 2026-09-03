from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from tau0vla_protocol import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ActionChunk,
    ChunkScheduler,
    ProtocolError,
    recommended_replan_steps,
)


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
        self.assertEqual(scheduler.adopt(_chunk(500.0), initial=True), 0)
        self.assertEqual(scheduler.remaining, ACTION_HORIZON)
        for _ in range(4):
            scheduler.next_action()
        self.assertTrue(scheduler.should_request(request_pending=False))
        self.assertFalse(scheduler.should_request(request_pending=True))

    def test_replacement_skips_rtt_equivalent_prefix(self):
        scheduler = ChunkScheduler(replan_steps=10)
        scheduler.adopt(_chunk(0.0), initial=True)
        for _ in range(10):
            scheduler.next_action()
        replacement = _chunk(100.0, base=1000.0)
        skipped = scheduler.adopt(replacement)
        self.assertEqual(skipped, 3)
        np.testing.assert_array_equal(scheduler.next_action(), replacement.actions[3])

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


if __name__ == "__main__":
    unittest.main()
