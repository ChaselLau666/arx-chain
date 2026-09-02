from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from human_dagger_recorder import (  # noqa: E402
    ControlMode,
    EpisodeValidationError,
    EventType,
    HumanDaggerRecorder,
    SOURCE_TIMESTAMP_NAMES,
)
from validate_dagger_episode import validate_episode  # noqa: E402


JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQ"
    "FxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMK"
    "ChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
    "KCj/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL"
    "/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS"
    "0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlq"
    "c3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJ"
    "ytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAA"
    "AAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMi"
    "MoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
    "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0"
    "tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIR"
    "AxEAPwD58ooor0TgP//Z"
)

CAMERAS = ("head", "left_wrist", "right_wrist")


def observation(seed: float = 0.0):
    return {
        "qpos": np.arange(14, dtype=np.float32) + seed,
        "qvel": np.full(14, seed, dtype=np.float32),
        "effort": np.full(14, seed + 0.5, dtype=np.float32),
        "eef": np.arange(14, dtype=np.float32) / 10 + seed,
        "robot_base": np.zeros(6, dtype=np.float32),
        "base_velocity": np.zeros(4, dtype=np.float32),
    }


def images():
    return {camera: JPEG for camera in CAMERAS}


def append_mode(recorder, frame, mode, epoch, seq=-1):
    kwargs = {}
    if mode == ControlMode.POLICY or (
        mode == ControlMode.HANDOFF_TO_POLICY and seq >= 0
    ):
        kwargs["policy_action_joint"] = np.full(14, frame + 0.25, dtype=np.float32)
    if mode == ControlMode.HUMAN:
        kwargs["expert_action_eef_raw"] = np.full(14, frame + 0.5, dtype=np.float32)
        kwargs["expert_action_eef_rebased"] = np.full(14, frame + 0.75, dtype=np.float32)
    return recorder.append_frame(
        observation=observation(frame),
        images_jpeg=images(),
        control_mode=mode,
        timestamps={
            "observation_ns": 1_000_000 + frame * 20_000,
            "control_ns": 1_001_000 + frame * 20_000,
            **{
                name: 900_000 + frame * 20_000 + index
                for index, name in enumerate(SOURCE_TIMESTAMP_NAMES)
            },
        },
        control_epoch=epoch,
        action_seq=seq,
        **kwargs,
    )


class HumanDaggerRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_recorder(self, name="episode_0"):
        return HumanDaggerRecorder(
            self.output_dir,
            name,
            camera_names=CAMERAS,
            image_capacity=64,
            flush_every=2,
            metadata={
                # The runtime also supplies these fixed values. Matching values
                # are accepted but cannot override the schema contract.
                "schema_version": 2,
                "collection_mode": "human_dagger",
                "action_semantics": "current_measured_qpos",
                "training_action_offset_frames": 1,
                "task": "insert_test",
                "height_command": 3.5,
                "dagger_round": 2,
                "policy_checkpoint": "/tmp/policy_best.ckpt",
                "policy_checkpoint_sha256": "a" * 64,
                "git_commit": "deadbeef",
                "nominal_fps": 60.0,
            },
        )

    def build_valid_episode(self, name="episode_0"):
        recorder = self.make_recorder(name)
        append_mode(recorder, 0, ControlMode.POLICY, 0, seq=0)
        append_mode(recorder, 1, ControlMode.HANDOFF_TO_HUMAN, 1)
        recorder.record_event(
            EventType.HANDOFF_TO_HUMAN,
            request_ns=1_010_000,
            gate_ns=1_011_000,
            ack_ns=1_012_000,
            frame=2,
            epoch=1,
        )
        append_mode(recorder, 2, ControlMode.HUMAN, 1)
        append_mode(recorder, 3, ControlMode.HUMAN, 1)
        append_mode(recorder, 4, ControlMode.HANDOFF_TO_POLICY, 2, seq=1)
        recorder.record_event(
            EventType.HANDOFF_TO_POLICY,
            request_ns=1_070_000,
            gate_ns=1_071_000,
            ack_ns=1_072_000,
            frame=5,
            epoch=2,
        )
        append_mode(recorder, 5, ControlMode.POLICY, 2, seq=2)
        return recorder.finalize()

    def test_streams_schema_v2_and_finalizes_atomically(self):
        final_path = self.build_valid_episode()
        self.assertEqual(final_path, self.output_dir.resolve() / "episode_0.hdf5")
        self.assertTrue(final_path.is_file())
        self.assertFalse((self.output_dir / "episode_0.partial.hdf5").exists())

        result = validate_episode(final_path)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.num_frames, 6)

        with h5py.File(final_path, "r") as root:
            self.assertEqual(root.attrs["schema_version"], 2)
            self.assertEqual(root.attrs["action_semantics"], "current_measured_qpos")
            self.assertEqual(root.attrs["training_action_offset_frames"], 1)
            self.assertEqual(root["/compress_len"].shape, (3, 6))
            for timestamp_name in SOURCE_TIMESTAMP_NAMES:
                self.assertEqual(root[f"/timestamps/{timestamp_name}"].shape, (6,))
                self.assertTrue(np.all(root[f"/timestamps/{timestamp_name}"][:] > 0))
            self.assertGreaterEqual(root["/observations/images/head"].shape[1], len(JPEG))
            length = int(root["/compress_len"][0, 0])
            self.assertEqual(length, len(JPEG))
            self.assertEqual(root["/observations/images/head"][0, :length].tobytes(), JPEG)
            self.assertFalse(np.any(root["/observations/images/head"][0, length:]))
            np.testing.assert_array_equal(
                root["/dagger/intervention_mask"][:],
                [False, False, True, True, False, False],
            )
            np.testing.assert_array_equal(
                root["/dagger/supervision_valid"][:],
                [False, False, True, True, False, False],
            )
            np.testing.assert_array_equal(
                root["/dagger/policy_action_valid"][:],
                [True, False, False, False, True, True],
            )
            np.testing.assert_array_equal(
                root["/dagger/expert_action_valid"][:],
                [False, False, True, True, False, False],
            )
            self.assertFalse(np.any(root["/dagger/policy_action_joint"][1]))
            self.assertFalse(np.any(root["/dagger/expert_action_eef_raw"][0]))
            self.assertEqual(root["/dagger/events"].shape, (2,))

    def test_append_rejects_stale_timestamps_and_missing_owned_action(self):
        recorder = self.make_recorder()
        append_mode(recorder, 0, ControlMode.POLICY, 0, seq=0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            recorder.append_frame(
                observation=observation(1),
                images_jpeg=images(),
                control_mode=ControlMode.POLICY,
                observation_ns=1_000_000,
                control_ns=1_021_000,
                policy_action_joint=np.zeros(14),
                action_seq=1,
            )
        with self.assertRaisesRegex(ValueError, "require expert"):
            recorder.append_frame(
                observation=observation(1),
                images_jpeg=images(),
                control_mode=ControlMode.HUMAN,
                observation_ns=1_020_000,
                control_ns=1_021_000,
                control_epoch=1,
            )
        self.assertEqual(recorder.frame_count, 1)
        recorder.discard()

    def test_quarantine_preserves_partial_and_reason(self):
        recorder = self.make_recorder()
        append_mode(recorder, 0, ControlMode.POLICY, 0, seq=0)
        quarantine_path = recorder.quarantine("camera/timeout")
        self.assertTrue(quarantine_path.is_file())
        self.assertIn("camera_timeout", quarantine_path.name)
        self.assertFalse(recorder.partial_path.exists())
        with h5py.File(quarantine_path, "r") as root:
            self.assertFalse(bool(root.attrs["finalized"]))
            self.assertEqual(root.attrs["quarantine_reason"], "camera/timeout")

    def test_finalize_validation_failure_is_quarantined(self):
        recorder = self.make_recorder()
        append_mode(recorder, 0, ControlMode.POLICY, 0, seq=0)
        # This direct POLICY -> HUMAN jump violates the ownership state machine
        # and intentionally has no matching handoff event.
        append_mode(recorder, 1, ControlMode.HUMAN, 1)
        with self.assertRaises(EpisodeValidationError) as context:
            recorder.finalize()
        self.assertIsNotNone(context.exception.quarantine_path)
        self.assertTrue(context.exception.quarantine_path.is_file())
        self.assertFalse(recorder.final_path.exists())

    def test_validator_detects_timestamp_event_and_jpeg_corruption(self):
        final_path = self.build_valid_episode()
        with h5py.File(final_path, "r+") as root:
            root["/timestamps/control_ns"][3] = root["/timestamps/control_ns"][2]
            event = root["/dagger/events"][0]
            event["frame"] = 3
            root["/dagger/events"][0] = event
            length = int(root["/compress_len"][0, 1])
            corrupt = np.zeros(length, dtype=np.uint8)
            corrupt[:2] = [0xFF, 0xD8]
            corrupt[-2:] = [0xFF, 0xD9]
            root["/observations/images/head"][1, :length] = corrupt

        result = validate_episode(final_path)
        self.assertFalse(result.valid)
        joined = "\n".join(result.errors)
        self.assertIn("not strictly increasing", joined)
        self.assertIn("cannot be decoded", joined)
        self.assertIn("missing HANDOFF_TO_HUMAN", joined)

    def test_validator_reports_length_mismatch_without_crashing(self):
        final_path = self.build_valid_episode()
        with h5py.File(final_path, "r+") as root:
            root["/observations/qvel"].resize((5, 14))
            root["/dagger/intervention_mask"].resize((5,))

        result = validate_episode(final_path, decode_images=False)
        self.assertFalse(result.valid)
        joined = "\n".join(result.errors)
        self.assertIn("/observations/qvel has shape (5, 14)", joined)
        self.assertIn("/dagger/intervention_mask has shape (5,)", joined)


if __name__ == "__main__":
    unittest.main()
