from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from collect import feedback_is_stable


class FixedHeightTests(unittest.TestCase):
    def test_feedback_requires_full_stable_window(self):
        self.assertFalse(feedback_is_stable([(0.0, 1.0), (1.0, 1.0)]))
        self.assertTrue(feedback_is_stable([(0.0, 1.0), (1.0, 1.005), (2.1, 1.004)]))
        self.assertFalse(feedback_is_stable([(0.0, 1.0), (1.0, 1.2), (2.1, 1.1)]))


if __name__ == '__main__':
    unittest.main()
