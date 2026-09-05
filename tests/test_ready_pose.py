"""Exercise the ready-pose decisions and the launcher, no hardware involved."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

from ready_pose import (GO_HOME_JOY, READY_GRIPPER, VR_GRIPPER_SCALE, arms_have_arrived,
                        joints_are_still, vr_gripper_command)

LAUNCHER = ROOT / 'tools' / '08_collect_ready_pose.sh'
FILTER = ROOT / 'act' / 'vr_pose_filter.py'
COLLECT = ROOT / 'act' / 'collect.py'


def normalise_pose(pose):
    """Run the launcher's own pose normaliser, extracted rather than restated."""
    text = LAUNCHER.read_text()
    begin = text.index('die() {')
    fragment = text[begin:text.index('\nREADY_POSE_L=$(normalise_pose', begin)]
    return subprocess.run(
        ['bash', '-c', f'set -Eeuo pipefail\n{fragment}\nnormalise_pose TEST "$1"', '_', pose],
        text=True, capture_output=True,
    )


class GoHomeMessageTests(unittest.TestCase):
    def test_joy_message_carries_both_elements(self):
        # arxJoyCB reads data[0] and data[1] without checking the length, so a
        # one-element array is an out-of-bounds read on the robot.
        self.assertEqual(len(GO_HOME_JOY), 2)

    def test_joy_message_selects_home_and_not_gravity_compensation(self):
        # data[0] == 1 would put the arm in G_COMPENSATION and let it fall.
        self.assertEqual(GO_HOME_JOY[0], 0)
        self.assertEqual(GO_HOME_JOY[1], 1)


class GripperTests(unittest.TestCase):
    def test_command_survives_the_controller_scaling(self):
        for gripper in READY_GRIPPER:
            with self.subTest(gripper=gripper):
                commanded = vr_gripper_command(gripper)
                self.assertAlmostEqual(commanded * VR_GRIPPER_SCALE, gripper, places=9)


class StillnessTests(unittest.TestCase):
    def sample(self, times, value=0.0):
        return [(t, np.full(12, value + t * 0.0)) for t in times]

    def test_a_window_shorter_than_asked_for_is_not_still(self):
        # The first samples after a move starts are trivially identical; calling
        # that settled parks nothing and reports success.
        self.assertFalse(joints_are_still(self.sample([0.0, 0.05])))

    def test_a_single_sample_is_not_still(self):
        self.assertFalse(joints_are_still(self.sample([0.0])))

    def test_a_full_quiet_window_is_still(self):
        self.assertTrue(joints_are_still(self.sample([0.0, 0.2, 0.4])))

    def test_movement_across_a_full_window_is_not_still(self):
        samples = [(0.0, np.zeros(12)), (0.2, np.zeros(12)), (0.4, np.full(12, 0.01))]
        self.assertFalse(joints_are_still(samples))


class ArrivalTests(unittest.TestCase):
    def test_stopped_short_of_the_target_is_not_arrival(self):
        # GO_HOME and END_CONTROL fighting to a standstill partway is the failure
        # this exists to catch: still, but not where it was asked to go.
        target = np.zeros(12)
        current = np.concatenate([np.full(6, 0.3), np.zeros(6)])
        self.assertTrue(joints_are_still([(0.0, current), (0.2, current), (0.4, current)]))
        self.assertFalse(arms_have_arrived(current, target))

    def test_within_tolerance_is_arrival(self):
        target = np.zeros(12)
        self.assertTrue(arms_have_arrived(np.full(12, 0.04), target))

    def test_no_target_cannot_contradict_stillness(self):
        self.assertTrue(arms_have_arrived(np.full(12, 9.0), None))


class PoseNormalisationTests(unittest.TestCase):
    def test_measured_pose_passes_through_unchanged(self):
        pose = '[-0.0002, 0.9447, 0.8597, -0.5755, 0.0006, -0.0013]'
        result = normalise_pose(pose)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, pose)

    def test_integers_are_widened_to_doubles(self):
        # go_home_position is declared DOUBLE_ARRAY and rclpy rejects an
        # INTEGER_ARRAY override outright rather than widening it, so an
        # all-integer pose would stop the arm from starting at all.
        result = normalise_pose('[0, 1, 0, 0, 0, 0]')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]')
        self.assertNotIn('[0,', result.stdout)

    def test_short_and_non_numeric_poses_are_refused(self):
        for pose in ('[0, 1, 2]', 'not-a-list', '[a, b, c, d, e, f]', '[]'):
            with self.subTest(pose=pose):
                self.assertNotEqual(normalise_pose(pose).returncode, 0)


class LauncherWiringTests(unittest.TestCase):
    def setUp(self):
        self.text = LAUNCHER.read_text()

    def test_both_arms_are_given_a_ready_pose(self):
        # Without this the arm homes to the SDK default of all zeros.
        for side in ('READY_POSE_L', 'READY_POSE_R'):
            self.assertIn(f"go_home_position:='${{{side}}}'", self.text)

    def test_arms_and_collector_agree_on_the_pose_topic(self):
        # Publishing anywhere other than where the arm subscribes commands
        # nothing at all, and the failure is silent.
        self.assertIn('-p arm_sub_topic_name:=${ARM_POSE_L#/}', self.text)
        self.assertIn('-p arm_sub_topic_name:=${ARM_POSE_R#/}', self.text)
        self.assertIn('--ready_pose_topics ${ARM_POSE_L} ${ARM_POSE_R}', self.text)

    def test_each_filter_reads_the_arm_on_its_own_side(self):
        # The offset a filter carries is that arm's; crossing them re-aims the
        # left arm onto the right one's pose.
        self.assertIn('--out-topic ${ARM_POSE_L} --node-name vr_pose_filter_l '
                      '--arm-status-topic /arm_l_status_full', self.text)
        self.assertIn('--out-topic ${ARM_POSE_R} --node-name vr_pose_filter_r '
                      '--arm-status-topic /arm_r_status_full', self.text)

    def test_skipping_the_filter_also_skips_the_parking(self):
        # No filter means nothing to mute, so a parked arm is pulled straight
        # back by the first VR frame. Claiming to park would be a lie.
        self.assertIn('(( SKIP_FILTER )) && ready_args=""', self.text)

    def test_vendor_baseline_is_left_alone(self):
        baseline = (ROOT / 'tools' / '01_collect.sh').read_text()
        self.assertNotIn('go_home_position', baseline)
        self.assertNotIn('ready_pose', baseline)


class FilterRebaseTests(unittest.TestCase):
    def setUp(self):
        self.text = FILTER.read_text()

    def test_mute_is_driven_by_the_go_home_bit(self):
        self.assertIn("self.create_subscription(Int32MultiArray, '/arx_joy', self.on_joy, 10)",
                      self.text)
        self.assertIn('if len(msg.data) > 1 and msg.data[1] == 1:', self.text)

    def test_rebase_runs_when_the_mute_ends_not_when_it_starts(self):
        # The arm has been left somewhere new by then; measuring at the start
        # would anchor on where it was before it moved.
        self.assertIn('if self.muted and not muted and args.rebase:', self.text)

    def test_offset_is_carried_on_every_published_pose(self):
        self.assertIn('self.pos_prev + self.offset_pos', self.text)
        self.assertIn('(self.rot_prev * self.offset_rot).as_euler', self.text)

    def test_rotation_offset_composes_on_the_right(self):
        # f(X) = X * P^-1 * R makes a rotation of the hand the same rotation of
        # the tool. On the left it would not.
        self.assertIn('self.offset_rot = self.rot_prev.inv() * self.arm_rot', self.text)

    def test_filter_state_advances_while_muted(self):
        # Publishing has to resume from where the hand is now, not from the pose
        # it held when the mute began.
        body = self.text[self.text.index('def on_pose'):]
        self.assertLess(body.index('self.pos_prev = self.pos_prev + self.alpha'),
                        body.index('muted = time.monotonic() < self.mute_until'))


class CollectorWiringTests(unittest.TestCase):
    def test_parking_is_off_by_default(self):
        # 01_collect.sh starts the arms through the vendor launch file, which
        # carries no go_home_position, so parking there would home them to zero.
        text = COLLECT.read_text()
        self.assertIn("'--ready_pose', action='store_true'", text)
        self.assertIn('if args.ready_pose:', text)

    def test_parking_happens_before_recording_starts(self):
        text = COLLECT.read_text()
        loop = text[text.index('start_decision = prompt_start_decision'):]
        self.assertLess(loop.index('return_to_ready(ros_operator)'),
                        loop.index('Start recording episode'))


if __name__ == '__main__':
    unittest.main()
