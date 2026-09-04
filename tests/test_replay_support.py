from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from replay_support import (episode_start_pose, resolve_replay_height,
                            best_lag, tracking_report, ARM_INDICES,
                            GRIPPER_INDICES, ema_alpha, smooth_causal)


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


class TrackingReportTests(unittest.TestCase):
    def _ramp(self, frames=200):
        t = np.linspace(0, 4 * np.pi, frames)
        return np.stack([np.sin(t + j * 0.3) for j in range(14)], axis=1)

    def test_recovers_a_known_lag_from_the_arm_joints(self):
        command = self._ramp()
        actual = np.roll(command, 4, axis=0)
        report = tracking_report(command, actual)
        self.assertEqual(report['lag_frames'], 4)
        self.assertLess(report['arm_rmse'], 1e-9)

    def test_perfect_tracking_has_zero_lag_and_error(self):
        command = self._ramp()
        report = tracking_report(command, command.copy())
        self.assertEqual(report['lag_frames'], 0)
        self.assertLess(report['arm_rmse'], 1e-12)
        self.assertTrue(np.all(report['gripper_turns'] == 0))

    def test_gripper_whole_turn_offset_is_removed(self):
        command = self._ramp()
        actual = command.copy()
        actual[:, 6] -= 2 * np.pi          # 左夹爪低一整圈
        actual[:, 13] += 4 * np.pi         # 右夹爪高两整圈
        report = tracking_report(command, actual)
        self.assertEqual(list(report['gripper_turns']), [-1, 2])
        self.assertLess(float(np.max(report['gripper_rmse'])), 1e-9)
        self.assertLess(report['arm_rmse'], 1e-12)

    def test_gripper_residual_survives_the_snap(self):
        command = self._ramp()
        actual = command.copy()
        actual[:, 13] += 2 * np.pi + 0.3   # 一整圈 + 真实的 0.3 误差
        report = tracking_report(command, actual)
        self.assertEqual(report['gripper_turns'][1], 1)
        self.assertAlmostEqual(float(report['gripper_rmse'][1]), 0.3, places=6)

    def test_per_joint_isolates_one_bad_arm_joint(self):
        command = self._ramp()
        actual = command.copy()
        actual[:, 9] += 0.5
        report = tracking_report(command, actual)
        idx = ARM_INDICES.index(9)
        self.assertAlmostEqual(report['arm_per_joint_rmse'][idx], 0.5, places=6)

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            tracking_report(np.zeros((10, 14)), np.zeros((10, 7)))

    def test_too_few_frames_is_refused(self):
        with self.assertRaises(ValueError):
            tracking_report(np.zeros((1, 14)), np.zeros((1, 14)))

    def test_best_lag_is_bounded_by_max_lag(self):
        command = self._ramp()
        actual = np.roll(command, 10, axis=0)
        lag, _ = best_lag(command, actual, max_lag=3)
        self.assertLessEqual(lag, 3)


class CausalSmoothingTests(unittest.TestCase):
    """The filter teleop-app uses, ported for offline replay."""

    def test_alpha_matches_the_teleop_app_formula(self):
        for tau, dt in ((0.05, 1 / 60), (0.1, 1 / 30), (0.02, 1 / 100)):
            self.assertAlmostEqual(ema_alpha(tau, dt), 1.0 - np.exp(-dt / tau), places=12)

    def test_zero_tau_leaves_the_trajectory_alone(self):
        traj = np.random.default_rng(0).normal(size=(50, 14))
        self.assertTrue(np.array_equal(smooth_causal(traj, 0.0, 1 / 60), traj))

    def test_step_reaches_63_percent_after_one_tau(self):
        dt, tau = 1 / 60, 0.05
        traj = np.ones((400, 14))
        traj[0] = 0.0
        out = smooth_causal(traj, tau, dt)
        at_tau = out[int(round(tau / dt)), ARM_INDICES[0]]
        self.assertAlmostEqual(at_tau, 1 - np.exp(-1.0), places=2)

    def test_grippers_are_never_touched(self):
        rng = np.random.default_rng(1)
        traj = rng.normal(size=(120, 14))
        out = smooth_causal(traj, 0.05, 1 / 60)
        for column in GRIPPER_INDICES:
            self.assertTrue(np.array_equal(out[:, column], traj[:, column]))
        for column in ARM_INDICES:
            self.assertFalse(np.array_equal(out[:, column], traj[:, column]))

    def test_high_frequency_ripple_is_attenuated(self):
        """Energy above the cutoff should drop; energy below it should survive."""
        dt, n = 1 / 60, 1200
        t = np.arange(n) * dt
        traj = np.zeros((n, 14))
        for column in ARM_INDICES:
            traj[:, column] = np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.sin(2 * np.pi * 12 * t)
        out = smooth_causal(traj, 0.05, dt)

        def band(x, lo, hi):
            spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
            freq = np.fft.rfftfreq(len(x), dt)
            return spectrum[(freq >= lo) & (freq < hi)].sum()

        col = ARM_INDICES[0]
        before_hi, after_hi = band(traj[:, col], 8, 30), band(out[:, col], 8, 30)
        before_lo, after_lo = band(traj[:, col], 0.1, 2), band(out[:, col], 0.1, 2)
        self.assertLess(after_hi, before_hi * 0.1)      # 12 Hz 被压掉一个数量级以上
        self.assertGreater(after_lo, before_lo * 0.5)   # 0.5 Hz 基本保留

    def test_output_lags_the_input(self):
        dt, n = 1 / 60, 400
        t = np.arange(n) * dt
        traj = np.zeros((n, 14))
        for column in ARM_INDICES:
            traj[:, column] = np.sin(2 * np.pi * 0.5 * t)
        out = smooth_causal(traj, 0.05, dt)
        lag, _ = best_lag(traj[:, ARM_INDICES], out[:, ARM_INDICES], 30)
        self.assertGreater(lag, 0)          # 因果滤波必然引入延迟

    def test_non_2d_input_is_refused(self):
        with self.assertRaises(ValueError):
            smooth_causal(np.zeros(14), 0.05, 1 / 60)


if __name__ == '__main__':
    unittest.main()
