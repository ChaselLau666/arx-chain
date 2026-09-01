from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from collection_ui import prompt_episode_decision, prompt_next_decision


class CollectionReviewTests(unittest.TestCase):
    def test_invalid_choice_reprompts_then_saves(self):
        answers = iter(['x', 'S'])
        self.assertEqual(prompt_episode_decision(lambda _: next(answers)), 's')

    def test_discard_and_quit(self):
        self.assertEqual(prompt_episode_decision(lambda _: 'd'), 'd')
        self.assertEqual(prompt_episode_decision(lambda _: 'q'), 'q')

    def test_next_requires_explicit_n_or_q(self):
        answers = iter(['x', 'N'])
        self.assertEqual(prompt_next_decision(9, lambda _: next(answers)), 'n')
        self.assertEqual(prompt_next_decision(10, lambda _: 'q'), 'q')


if __name__ == '__main__':
    unittest.main()
