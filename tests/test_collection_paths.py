from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from collection_paths import normalize_task_name, task_dataset_dir


class CollectionPathTests(unittest.TestCase):
    def test_task_name_selects_its_own_dataset_directory(self):
        self.assertEqual(
            task_dataset_dir('/data/datasets', 'pickplace_zjy_20270901_2010'),
            Path('/data/datasets/pickplace_zjy_20270901_2010'),
        )

    def test_task_name_is_trimmed(self):
        self.assertEqual(normalize_task_name('  pickplace  '), 'pickplace')
        self.assertEqual(
            task_dataset_dir('/data/datasets', '  pickplace  '),
            Path('/data/datasets/pickplace'),
        )

    def test_empty_or_path_like_task_names_are_rejected(self):
        for task_name in ('', '   ', '.', '..', 'group/task', '/tmp/task'):
            with self.subTest(task_name=task_name):
                with self.assertRaises(ValueError):
                    normalize_task_name(task_name)


if __name__ == '__main__':
    unittest.main()
