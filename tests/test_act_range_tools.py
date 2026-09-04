from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from eval_act_openloop import official_temporal_action
from run_act_experiment import prepare_view, selected_episodes


class ActRangeToolTests(unittest.TestCase):
    def test_inclusive_range_and_held_out_episode(self):
        self.assertEqual(selected_episodes(25, 49, 50), list(range(25, 50)))
        with self.assertRaises(ValueError):
            selected_episodes(0, 50, 50)

    def test_view_renumbers_source_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for source_index in (25, 26):
                source = root / f"source_{source_index}.hdf5"
                source.touch()
                sources.append(source)
            manifest = {
                "episodes": [
                    {
                        "local_episode": local,
                        "source_path": str(source.resolve()),
                    }
                    for local, source in enumerate(sources)
                ]
            }
            manifest_path = prepare_view(root / "view", manifest)
            self.assertEqual((root / "view" / "episode_0.hdf5").resolve(), sources[0])
            self.assertTrue(manifest_path.is_file())

    def test_view_allows_code_commit_refresh_without_data_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.hdf5"
            source.touch()
            base = {
                "repo_commit": "old",
                "episodes": [{"local_episode": 0, "source_path": str(source.resolve())}],
            }
            manifest_path = prepare_view(root / "view", base)
            updated = {**base, "repo_commit": "new"}
            prepare_view(root / "view", updated)
            self.assertIn('"new"', manifest_path.read_text())

    def test_temporal_aggregation_uses_only_valid_chunks(self):
        chunks = np.zeros((3, 5, 2), dtype=np.float32)
        valid = np.zeros((3, 5), dtype=bool)
        chunks[0, 1] = [1.0, 2.0]
        chunks[1, 1] = [3.0, 4.0]
        valid[0, 1] = True
        valid[1, 1] = True
        result = official_temporal_action(chunks, valid, 1)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.all(result > [1.0, 2.0]))
        self.assertTrue(np.all(result < [3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
