from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from replay_support import episode_start_pose, resolve_replay_height


class ReplayStartPoseTests(unittest.TestCase):
    def test_splits_first_frame_into_two_arms(self):
        trajectory = np.arange(28, dtype=np.float32).reshape(2, 14)
        left, right = episode_start_pose(trajectory)
        self.assertEqual(left, [float(v) for v in range(7)])
        self.assertEqual(right, [float(v) for v in range(7, 14)])

    def test_keeps_recorded_gripper_instead_of_a_constant(self):
        trajectory = np.zeros((1, 14), dtype=np.float32)
        trajectory[0, 6] = -3.3715
        trajectory[0, 13] = -0.0633
        left, right = episode_start_pose(trajectory)
        self.assertAlmostEqual(left[6], -3.3715, places=4)
        self.assertAlmostEqual(right[6], -0.0633, places=4)
        for value in (left[6], right[6]):
            self.assertGreaterEqual(value, -3.4)
            self.assertLessEqual(value, 0.0)

    def test_empty_episode_is_refused(self):
        with self.assertRaises(ValueError):
            episode_start_pose(np.zeros((0, 14), dtype=np.float32))

    def test_wrong_width_is_refused(self):
        with self.assertRaises(ValueError):
            episode_start_pose(np.zeros((3, 7), dtype=np.float32))


class ReplayHeightTests(unittest.TestCase):
    def test_recorded_height_is_used_by_default(self):
        self.assertEqual(resolve_replay_height(15.5, None), 15.5)

    def test_matching_request_is_accepted(self):
        self.assertEqual(resolve_replay_height(15.5, 15.5), 15.5)

    def test_conflicting_request_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_replay_height(15.5, 12.0)

    def test_missing_attribute_without_request_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_replay_height(None, None)

    def test_missing_attribute_with_explicit_request_is_allowed(self):
        self.assertEqual(resolve_replay_height(None, 15.5), 15.5)

    def test_numpy_scalar_from_hdf5_attrs_is_accepted(self):
        self.assertEqual(resolve_replay_height(np.float64(15.5), None), 15.5)


if __name__ == '__main__':
    unittest.main()
