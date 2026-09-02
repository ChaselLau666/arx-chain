import pathlib
import sys
import tempfile
import unittest

import numpy as np


ACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "act"
sys.path.insert(0, str(ACT_DIR))

from human_dagger_policy import (  # noqa: E402
    CAMERA_NAMES,
    TemporalAggregator,
    build_policy_config,
    load_policy_args,
)


class PolicyHelpersTest(unittest.TestCase):
    def test_temporal_aggregator_reset_removes_old_chunks(self):
        agg = TemporalAggregator(chunk_size=3, decay=0.01)
        np.testing.assert_allclose(agg.add(0, np.ones((3, 2))), [1.0, 1.0])
        mixed = agg.add(1, np.full((3, 2), 3.0))
        self.assertTrue(np.all(mixed > 1.0))
        self.assertTrue(np.all(mixed < 3.0))
        agg.reset()
        np.testing.assert_allclose(agg.add(0, np.full((3, 2), 7.0)), [7.0, 7.0])

    def test_training_args_defaults_build_legacy_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = load_policy_args(tmp)
        self.assertEqual(tuple(args["camera_names"]), CAMERA_NAMES)
        config = build_policy_config(args)
        self.assertEqual(config["states_dim"], 14)
        self.assertEqual(config["action_dim"], 28)

    def test_rejects_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "args.yaml").write_text(
                "policy_class: ACT\nuse_base: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base"):
                load_policy_args(tmp)


if __name__ == "__main__":
    unittest.main()
