from __future__ import annotations

import os
import sys
import termios
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from collection_ui import TerminalKeyReader, prompt_episode_decision, prompt_start_decision


class CollectionReviewTests(unittest.TestCase):
    def test_invalid_choice_reprompts_then_saves(self):
        answers = iter(['x', 'S'])
        self.assertEqual(prompt_episode_decision(lambda _: next(answers)), 's')

    def test_discard_and_quit(self):
        self.assertEqual(prompt_episode_decision(lambda _: 'd'), 'd')
        self.assertEqual(prompt_episode_decision(lambda _: 'q'), 'q')

    def test_start_requires_explicit_r_or_q(self):
        answers = iter(['x', 'R'])
        self.assertEqual(prompt_start_decision(9, lambda _: next(answers)), 'r')
        self.assertEqual(prompt_start_decision(10, lambda _: 'q'), 'q')

    def test_terminal_reader_accepts_one_key_without_enter_and_restores_terminal(self):
        master_fd, slave_fd = os.openpty()
        stream = os.fdopen(slave_fd, 'r')
        original = termios.tcgetattr(slave_fd)
        try:
            with TerminalKeyReader(stream) as reader:
                writer = threading.Thread(
                    target=lambda: (time.sleep(0.05), os.write(master_fd, b'R'))
                )
                writer.start()
                self.assertEqual(reader.read_key('test: '), 'r')
                writer.join()
            restored = termios.tcgetattr(slave_fd)
            for flag in (termios.ECHO, termios.ICANON, termios.ISIG):
                self.assertEqual(restored[3] & flag, original[3] & flag)
        finally:
            stream.close()
            os.close(master_fd)

    def test_terminal_reader_polls_recording_end_key_without_enter(self):
        master_fd, slave_fd = os.openpty()
        stream = os.fdopen(slave_fd, 'r')
        try:
            with TerminalKeyReader(stream) as reader:
                os.write(master_fd, b'E')
                self.assertEqual(reader.poll_key(), 'e')
        finally:
            stream.close()
            os.close(master_fd)


if __name__ == '__main__':
    unittest.main()
