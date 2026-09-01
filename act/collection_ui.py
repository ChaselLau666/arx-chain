"""Single-key terminal controls for the collection state machine."""

from __future__ import annotations

import select
import os
import sys
import termios
import tty


class TerminalKeyReader:
    """Read keys immediately while restoring terminal settings on every exit."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self.fd = None
        self.original_settings = None

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError('collection keyboard control requires an interactive terminal')
        self.fd = self.stream.fileno()
        self.original_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_settings)
        self.fd = None
        self.original_settings = None

    def read_key(self, prompt):
        termios.tcflush(self.fd, termios.TCIFLUSH)
        print(prompt, end='', flush=True)
        key = os.read(self.fd, 1).decode(errors='ignore').lower()
        print(key)
        return key

    def poll_key(self):
        readable, _, _ = select.select([self.fd], [], [], 0.0)
        if not readable:
            return None
        return os.read(self.fd, 1).decode(errors='ignore').lower()


def _prompt_choice(prompt, choices, read_key_fn):
    while True:
        decision = read_key_fn(prompt).lower()
        if decision in choices:
            return decision
        print(f"Invalid key. Press one of: {', '.join(sorted(choices))}.")


def prompt_start_decision(episode, read_key_fn):
    return _prompt_choice(
        f'Ready for episode {episode}: [r]ecord, [q]uit: ',
        {'r', 'q'},
        read_key_fn,
    )


def prompt_episode_decision(read_key_fn, allow_save=True):
    if not allow_save:
        print('Episode has no valid synchronized frames and cannot be saved.')
        return _prompt_choice(
            'Invalid episode: [d]iscard, [q]discard and quit: ',
            {'d', 'q'},
            read_key_fn,
        )
    return _prompt_choice(
        'Episode ended: [s]ave, [d]iscard, [q]discard and quit: ',
        {'s', 'd', 'q'},
        read_key_fn,
    )
