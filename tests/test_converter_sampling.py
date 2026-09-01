from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/convert_official_hdf5_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("official_converter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConverterSamplingTests(unittest.TestCase):
    def test_60_to_30_uses_stride_two(self):
        self.assertEqual(MODULE.temporal_stride(60, 30), 2)
        self.assertEqual(list(MODULE.selected_frame_indices(6, 2)), [0, 2, 4])

    def test_odd_episode_preserves_last_even_sample(self):
        indices = list(MODULE.selected_frame_indices(601, 2))
        self.assertEqual(len(indices), 301)
        self.assertEqual(indices[-1], 600)

    def test_non_integer_ratio_is_refused(self):
        with self.assertRaises(ValueError):
            MODULE.temporal_stride(60, 25)


if __name__ == "__main__":
    unittest.main()
