from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from safe_height import is_safe_and_stable


class SafeHeightTests(unittest.TestCase):
    def test_requires_low_and_stable_window(self):
        self.assertTrue(is_safe_and_stable([(0.0, 0.50), (1.0, 0.49), (2.1, 0.50)], 1.0, 0.02, 2.0))
        self.assertFalse(is_safe_and_stable([(0.0, 14.0), (1.0, 10.0), (2.1, 0.5)], 1.0, 0.02, 2.0))
        self.assertFalse(is_safe_and_stable([(0.0, 0.5), (2.1, 0.8)], 1.0, 0.02, 2.0))


if __name__ == '__main__':
    unittest.main()
