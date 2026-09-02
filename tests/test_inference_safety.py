from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from inference_safety import ActionGuard, SingleStepGuard, validate_policy_contract


class InferenceSafetyTests(unittest.TestCase):
    def test_contract_rejects_model_dimension_mismatch(self):
        runtime = {"action_dim": 28, "camera_names": ["head", "left_wrist", "right_wrist"]}
        saved = {"policy_config": dict(runtime)}
        validate_policy_contract(runtime, saved)
        saved["policy_config"]["action_dim"] = 14
        with self.assertRaises(ValueError):
            validate_policy_contract(runtime, saved)

    def test_guard_holds_left_and_rejects_large_step(self):
        limits = {
            "lower": np.full(14, -5.0, dtype=np.float32),
            "upper": np.full(14, 5.0, dtype=np.float32),
            "max_step": np.full(14, 0.1, dtype=np.float32),
            "max_initial_delta": np.full(14, 0.5, dtype=np.float32),
            "left_reset": np.zeros(7, dtype=np.float32),
            "left_reset_tolerance": 0.01,
        }
        guard = ActionGuard(limits, np.zeros(14, dtype=np.float32))
        first = np.full(28, 0.05, dtype=np.float32)
        guarded = guard.validate(first)
        np.testing.assert_array_equal(guarded[:7], np.zeros(7))
        second = first.copy()
        second[7] = 1.0
        with self.assertRaises(ValueError):
            guard.validate(second)

    def test_single_step_guard_holds_left_and_allows_only_one_small_action(self):
        initial = np.zeros(14, dtype=np.float32)
        guard = SingleStepGuard(initial, joint_delta=0.02, gripper_delta=0.2)
        model_action = np.zeros(28, dtype=np.float32)
        model_action[:7] = 1.0
        model_action[7:13] = 0.01
        model_action[13] = 0.1
        guarded = guard.validate(model_action)
        np.testing.assert_array_equal(guarded[:7], initial[:7])
        with self.assertRaises(ValueError):
            guard.validate(model_action)

    def test_single_step_guard_rejects_large_right_joint_or_gripper_delta(self):
        initial = np.zeros(14, dtype=np.float32)
        action = np.zeros(28, dtype=np.float32)
        action[7] = 0.03
        with self.assertRaises(ValueError):
            SingleStepGuard(initial).validate(action)
        action[7] = 0.0
        action[13] = 0.3
        with self.assertRaises(ValueError):
            SingleStepGuard(initial).validate(action)


if __name__ == "__main__":
    unittest.main()
