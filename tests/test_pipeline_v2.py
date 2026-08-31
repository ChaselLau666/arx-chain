from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from io import BytesIO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "act"))
sys.path.insert(0, str(REPO_ROOT))

from dataset_v2 import EpisodeSample, EpisodeValidationError, EpisodeWriter, next_episode_path, validate_episode
from pipeline_contract import ACTION_DIM, ACTION_SEMANTICS, FPS, validate_dataset_contract
from model_server.policy_adapter import MockPolicy
from model_server.lerobot_contract import validate_arx_lerobot
from http_protocol import HttpInferenceClient


def jpeg_bytes(value: int) -> bytes:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="JPEG")
    return stream.getvalue()


def sample(index: int, height: float = 15.0, wheel: float = 0.0) -> EpisodeSample:
    base_stamp = 1_000_000_000 + index * int(1e9 / FPS)
    return EpisodeSample(
        qpos=np.arange(ACTION_DIM, dtype=np.float64) + index,
        qvel=np.full(ACTION_DIM, index, dtype=np.float64),
        effort=np.zeros(ACTION_DIM, dtype=np.float64),
        eef=np.arange(ACTION_DIM, dtype=np.float64) * 0.1 + index,
        images={name: jpeg_bytes(index * 10) for name in ("head", "left_wrist", "right_wrist")},
        camera_timestamp_ns={
            "head": base_stamp,
            "left_wrist": base_stamp + 1_000_000,
            "right_wrist": base_stamp + 2_000_000,
        },
        arm_timestamp_ns={"left": base_stamp, "right": base_stamp + 500_000},
        sample_monotonic_ns=base_stamp,
        body_information=np.array([height, 0.0, 0.0, 0.0]),
        wheel_velocity=np.full(4, wheel),
    )


def write_valid_episode(directory: Path, name: str = "episode_0.hdf5") -> Path:
    writer = EpisodeWriter(directory, "task", "Move the object into the bowl.", 15.0, REPO_ROOT)
    samples = [sample(index) for index in range(5)]
    for current, following in zip(samples, samples[1:]):
        writer.append_transition(current, following)
    writer.set_sampling_stats(5, 0)
    summary = writer.finalize()
    assert summary["frames"] == 4
    return writer.save_as(directory / name)


class PipelineV2Tests(unittest.TestCase):
    def test_writer_stores_state_t_plus_1_and_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = write_valid_episode(directory)
            with h5py.File(path, "r") as root:
                np.testing.assert_allclose(root["action"][0], root["observations/qpos"][0] + 1)
                self.assertEqual(root.attrs["action_semantics"], ACTION_SEMANTICS)
                self.assertEqual(int(root.attrs["fps"]), FPS)
                self.assertTrue(bytes(root["observations/images/head"][0]).startswith(b"\xff\xd8"))
            self.assertTrue(validate_episode(path)["body_motion_valid"])
            self.assertEqual(validate_dataset_contract(directory)["episodes"], 1)

    def test_body_motion_refuses_finalize(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            writer = EpisodeWriter(directory, "task", "Move the object.", 15.0, REPO_ROOT)
            writer.append_transition(sample(0), sample(1))
            writer.append_transition(sample(1, height=15.2), sample(2, height=15.2))
            writer.set_sampling_stats(3, 0)
            with self.assertRaisesRegex(EpisodeValidationError, "body height"):
                writer.finalize()
            writer.discard()

    def test_next_episode_only_counts_saved_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_valid_episode(directory, "episode_0.hdf5")
            (directory / ".pending").mkdir(exist_ok=True)
            (directory / ".pending/test.hdf5.partial").touch()
            self.assertEqual(next_episode_path(directory).name, "episode_1.hdf5")

    def test_mock_policy_returns_hold_chunk(self):
        policy = MockPolicy(horizon=3)
        state = np.arange(ACTION_DIM, dtype=np.float32)
        actions = policy.infer(state, {"head": b"jpeg"}, "task")
        self.assertEqual(actions.shape, (3, ACTION_DIM))
        np.testing.assert_array_equal(actions[0], state)

    def test_http_client_validates_schema_and_action_chunk(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return

            def json(self):
                return self.payload

        class Session:
            def get(self, url, timeout):
                if url.endswith("/healthz"):
                    return Response({"ok": True})
                return Response(
                    {
                        "protocol_version": "arx_http_v1",
                        "fps": 30,
                        "camera_names": ["head", "left_wrist", "right_wrist"],
                        "action_dim": 14,
                        "action_semantics": "state_t_plus_1",
                    }
                )

            def post(self, url, **kwargs):
                if url.endswith("/v1/reset"):
                    return Response({"session_id": "session"})
                return Response(
                    {
                        "protocol_version": "arx_http_v1",
                        "session_id": "session",
                        "request_id": 1,
                        "actions": [np.arange(14).tolist()] * 2,
                        "action_dt": 1 / 30,
                        "action_semantics": "state_t_plus_1",
                        "model_id": "mock",
                    }
                )

        client = HttpInferenceClient("http://server")
        client.session = Session()
        self.assertEqual(client.schema()["action_dim"], 14)
        result = client.infer(sample(0), "task", "session", 1)
        self.assertEqual(result.actions.shape, (2, 14))

    def test_lerobot_sidecar_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta").mkdir()
            (root / "meta/arx.json").write_text(
                json.dumps(
                    {
                        "lerobot_version": "0.4.3",
                        "lerobot_format": "v3",
                        "fps": 30,
                        "action_dim": 14,
                        "action_semantics": "state_t_plus_1",
                        "joint_names": [
                            *[f"left_j{i}" for i in range(6)], "left_gripper",
                            *[f"right_j{i}" for i in range(6)], "right_gripper",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_arx_lerobot(root)["action_dim"], 14)


if __name__ == "__main__":
    unittest.main()
