"""Pure safety checks shared by the shutdown monitor and offline tests."""

from __future__ import annotations


def is_safe_and_stable(samples, safe_max: float, tolerance: float, window_seconds: float) -> bool:
    if len(samples) < 2:
        return False
    newest_time = samples[-1][0]
    cutoff = newest_time - window_seconds
    start = 0
    for index, sample in enumerate(samples):
        if sample[0] <= cutoff:
            start = index
        else:
            break
    recent = list(samples)[start:]
    if recent[-1][0] - recent[0][0] < window_seconds:
        return False
    values = [value for _, value in recent]
    return max(values) <= safe_max and max(values) - min(values) <= tolerance
