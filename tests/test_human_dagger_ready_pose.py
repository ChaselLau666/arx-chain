"""How the ready-pose config section reaches the control core."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from human_dagger import _ready_pose_settings, _state_hint  # noqa: E402
from human_dagger_core import ControlState, HumanDaggerConfig  # noqa: E402

CONFIG = ROOT / "act" / "data" / "human_dagger.yaml"
ENTRYPOINT = ROOT / "act" / "human_dagger.py"
START = ROOT / "tools" / "05_human_dagger.sh"

# Measured on ark-1 and carried by tools/08_collect_ready_pose.sh.
MEASURED_LEFT = (-0.0002, 0.9447, 0.8597, -0.5755, 0.0006, -0.0013)
MEASURED_RIGHT = (-0.0002, 0.9466, 0.8604, -0.5724, -0.0002, -0.0006)
MEASURED_GRIPPER = (-2.9717, -2.9675)

SECTION = {
    "ready_pose": {
        "enabled": True,
        "left": list(MEASURED_LEFT),
        "right": list(MEASURED_RIGHT),
        "gripper": list(MEASURED_GRIPPER),
        "step_per_arm": [0.025, 0.025, 0.015, 0.025, 0.025, 0.025, 0.1],
        "timeout_s": 15.0,
        "arrival_tolerance": 0.05,
    }
}


class ReadyPoseSettingsTests(unittest.TestCase):
    def test_configured_section_produces_a_core_that_parks(self):
        config = HumanDaggerConfig(**_ready_pose_settings(SECTION))
        self.assertTrue(config.ready_move_enabled)
        self.assertEqual(config.ready_pose_left, MEASURED_LEFT)
        self.assertEqual(config.ready_pose_right, MEASURED_RIGHT)
        self.assertEqual(config.ready_gripper, MEASURED_GRIPPER)

    def test_timeout_is_read_in_seconds(self):
        config = HumanDaggerConfig(**_ready_pose_settings(SECTION))
        self.assertEqual(config.ready_move_timeout_ns, 15_000_000_000)

    def test_absent_section_leaves_the_move_off(self):
        self.assertEqual(_ready_pose_settings({}), {})
        self.assertFalse(HumanDaggerConfig().ready_move_enabled)

    def test_disabled_section_leaves_the_move_off(self):
        disabled = {"ready_pose": dict(SECTION["ready_pose"], enabled=False)}
        self.assertEqual(_ready_pose_settings(disabled), {})

    def test_operator_override_turns_a_configured_move_off(self):
        # A feature that drives the arms by itself has to be switchable at the
        # console without editing a file.
        self.assertEqual(_ready_pose_settings(SECTION, enabled_override=False), {})


class ShippedConfigTests(unittest.TestCase):
    def test_shipped_config_parks_at_the_measured_ready_pose(self):
        section = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        config = HumanDaggerConfig(**_ready_pose_settings(section))
        self.assertTrue(config.ready_move_enabled)
        self.assertEqual(config.ready_pose_left, MEASURED_LEFT)
        self.assertEqual(config.ready_pose_right, MEASURED_RIGHT)
        self.assertEqual(config.ready_gripper, MEASURED_GRIPPER)


class OperatorSurfaceTests(unittest.TestCase):
    def test_ready_move_tells_the_operator_the_arms_are_moving(self):
        hint = _state_hint(ControlState.READY_MOVE)
        self.assertIn("stand clear", hint.lower())

    def test_manual_reset_hint_says_the_record_key_moves_the_arms(self):
        self.assertIn("ready pose", _state_hint(ControlState.MANUAL_RESET).lower())

    def test_ctrl_c_during_the_ready_move_latches_a_hold(self):
        # The shutdown branch only breaks out of the control loop on a HOLD
        # command.  READY_MOVE emits its own source, so without an explicit
        # branch Ctrl-C would leave the arms walking.
        text = ENTRYPOINT.read_text(encoding="utf-8")
        branch = text[text.index("if shutdown_requested:"):]
        branch = branch[: branch.index("next_tick += frame_period")]
        self.assertIn("ControlState.READY_MOVE", branch)

    def test_launcher_exposes_a_ready_pose_switch(self):
        self.assertIn("READY_POSE_MOVE", START.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
