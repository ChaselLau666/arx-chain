"""Pure helpers shared by the replay entry point and offline tests."""

from __future__ import annotations

import numpy as np


def episode_start_pose(trajectory):
    """Split the first recorded frame into per-arm 7-DoF start targets.

    The start pose comes from the episode itself rather than a hardcoded
    constant, so every commanded value - the gripper included - stays in the
    same units the arm controller reports back.
    """
    if len(trajectory) == 0:
        raise ValueError('episode contains no frames; nothing to replay')

    first = np.asarray(trajectory[0], dtype=float)
    if first.shape != (14,):
        raise ValueError(f'expected a 14-D recorded frame, got shape {first.shape}')

    return first[:7].tolist(), first[7:14].tolist()


def resolve_replay_height(recorded, requested):
    """Pick the lift command for a replay, refusing ambiguous combinations.

    The episode records the height it was collected at. A mismatch between
    that and an explicit --height is a physical hazard, not a preference, so
    it is refused rather than silently resolved in either direction.
    """
    if requested is None:
        if recorded is None:
            raise ValueError('episode has no height_command; pass --height explicitly')
        return float(recorded)

    requested = float(requested)
    if recorded is not None and abs(float(recorded) - requested) > 1e-6:
        raise ValueError(
            f'--height {requested} conflicts with recorded height_command {float(recorded)}')

    return requested
