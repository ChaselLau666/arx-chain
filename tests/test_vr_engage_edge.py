"""Rising-edge semantics of the VR hold-to-engage takeover channel.

The edge detector lives inside the control loop in ``human_dagger.py``, which
cannot be imported without ROS. This mirrors that logic exactly so the safety
properties it has to hold are pinned by a test that runs anywhere.
"""

from __future__ import annotations

import unittest


class VrEngageEdgeDetector:
    """Mirror of the control-loop edge detector in human_dagger.py."""

    def __init__(self, active_value: int = 1) -> None:
        self.active_value = active_value
        self.previous = None

    def feed(self, level):
        """Return True when this sample should raise TAKEOVER."""
        if level is None:
            return False
        engaged = level == self.active_value
        was_engaged = (
            None if self.previous is None else self.previous == self.active_value
        )
        fired = engaged and was_engaged is False
        self.previous = level
        return fired


class VrEngageEdgeTests(unittest.TestCase):
    def test_rising_edge_fires_once(self):
        detector = VrEngageEdgeDetector()
        self.assertFalse(detector.feed(0))
        self.assertTrue(detector.feed(1))
        # Held down: must not re-fire on every 60 Hz tick.
        self.assertFalse(detector.feed(1))
        self.assertFalse(detector.feed(1))

    def test_button_already_held_at_startup_does_not_fire(self):
        """A button held before the loop starts must not take over by itself."""
        detector = VrEngageEdgeDetector()
        self.assertFalse(detector.feed(1))
        self.assertFalse(detector.feed(1))

    def test_release_then_press_fires_again(self):
        detector = VrEngageEdgeDetector()
        detector.feed(0)
        self.assertTrue(detector.feed(1))
        self.assertFalse(detector.feed(0))
        self.assertTrue(detector.feed(1))

    def test_release_alone_never_fires(self):
        """Only takeover is automated; releasing must not resume the policy."""
        detector = VrEngageEdgeDetector()
        detector.feed(1)
        for _ in range(5):
            self.assertFalse(detector.feed(0))

    def test_missing_samples_are_ignored_and_do_not_reset_state(self):
        detector = VrEngageEdgeDetector()
        detector.feed(0)
        self.assertFalse(detector.feed(None))
        self.assertTrue(detector.feed(1))
        self.assertFalse(detector.feed(None))
        # Still held as far as the detector knows, so no re-fire.
        self.assertFalse(detector.feed(1))

    def test_non_binary_levels_are_compared_against_active_value(self):
        detector = VrEngageEdgeDetector(active_value=2)
        self.assertFalse(detector.feed(0))
        self.assertFalse(detector.feed(1))
        self.assertTrue(detector.feed(2))
        self.assertFalse(detector.feed(2))
        self.assertFalse(detector.feed(1))
        self.assertTrue(detector.feed(2))


if __name__ == '__main__':
    unittest.main()
