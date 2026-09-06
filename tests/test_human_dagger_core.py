from __future__ import annotations

import math
import sys
from dataclasses import replace
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from human_dagger_core import (
    ArmFeedback,
    CommandMode,
    CommandSource,
    ControlEvent,
    ControlState,
    HumanDaggerConfig,
    HumanDaggerCore,
    PolicyActionPacket,
    TimelineEventName,
    VrPose,
)

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - dependency error is clearer than bad math
    Rotation = None


MS = 1_000_000
SECOND = 1_000_000_000


class FakeClock:
    def __init__(self, now_ns: int = SECOND) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def arm_feedback(
    timestamp_ns: int,
    offset: float = 0.0,
    eef_pose=(0.4, -0.1, 0.3, 0.2, -0.3, 0.4),
    gripper: float = -1.2,
) -> ArmFeedback:
    return ArmFeedback(
        joint_pos=tuple(offset + index / 10.0 for index in range(6)),
        eef_pose=eef_pose,
        gripper=gripper,
        timestamp_ns=timestamp_ns,
    )


def vr_pose(
    timestamp_ns: int,
    eef_pose=(1.0, 2.0, 3.0, -0.1, 0.15, 0.25),
    gripper: float = 2.0,
) -> VrPose:
    return VrPose(eef_pose=eef_pose, gripper=gripper, timestamp_ns=timestamp_ns)


def policy_action(delta: float = 1.0):
    return tuple(delta + index / 20.0 for index in range(14))


def measured_action():
    left = arm_feedback(0, offset=0.0, gripper=-1.2)
    right = arm_feedback(
        0,
        offset=1.0,
        eef_pose=(-0.4, 0.1, 0.35, -0.2, 0.1, -0.4),
        gripper=-2.2,
    )
    return (*left.joint_pos, left.gripper, *right.joint_pos, right.gripper)


class CoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.config = HumanDaggerConfig(
            feedback_timeout_ns=100 * MS,
            vr_timeout_ns=100 * MS,
            policy_timeout_ns=250 * MS,
            handoff_timeout_ns=2 * SECOND,
            policy_slew_duration_ns=2 * SECOND,
        )
        self.core = HumanDaggerCore(self.config, clock_ns=self.clock)

    def install_inputs(self, timestamp_ns=None) -> None:
        stamp = self.clock.now_ns if timestamp_ns is None else timestamp_ns
        self.assertTrue(
            self.core.update_feedback(
                arm_feedback(stamp, offset=0.0, gripper=-1.2),
                arm_feedback(
                    stamp,
                    offset=1.0,
                    eef_pose=(-0.4, 0.1, 0.35, -0.2, 0.1, -0.4),
                    gripper=-2.2,
                ),
            )
        )
        self.assertTrue(
            self.core.update_vr(
                vr_pose(stamp, gripper=2.0),
                vr_pose(
                    stamp,
                    eef_pose=(-1.0, -2.0, 2.5, 0.1, -0.15, -0.2),
                    gripper=3.0,
                ),
            )
        )

    def enter_manual_reset(self) -> None:
        self.install_inputs()
        self.core.mark_precheck_complete(self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.MANUAL_RESET)

    def begin_policy_handoff(self) -> int:
        self.enter_manual_reset()
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)
        self.assertTrue(result.snapshot.episode_active)
        epoch = result.snapshot.control_epoch
        self.assertEqual(result.snapshot.pending_policy_reset_epoch, epoch)
        return epoch

    def enter_policy(self) -> int:
        epoch = self.begin_policy_handoff()
        self.clock.now_ns += MS
        self.install_inputs()
        self.assertTrue(self.core.acknowledge_policy_reset(epoch, self.clock.now_ns))
        target = policy_action()
        self.assertTrue(
            self.core.submit_policy_action(
                PolicyActionPacket(epoch, 0, self.clock.now_ns, target),
                self.clock.now_ns,
            )
        )
        result = self.core.tick(self.clock.now_ns)
        self.assertIn(
            result.command.source,
            (CommandSource.POLICY_SLEW, CommandSource.POLICY),
        )

        # The handoff completes as soon as the slew reaches the first target;
        # two seconds is a hard upper bound, not a mandatory delay.
        for frame in range(1, 120):
            if result.snapshot.state is ControlState.POLICY:
                return epoch
            self.clock.now_ns += 2 * SECOND // 120
            self.install_inputs()
            self.assertTrue(
                self.core.submit_policy_action(
                    PolicyActionPacket(epoch, frame, self.clock.now_ns, target),
                    self.clock.now_ns,
                )
            )
            result = self.core.tick(self.clock.now_ns)
            self.assertIn(
                result.snapshot.state,
                (ControlState.HANDOFF_TO_POLICY, ControlState.POLICY),
            )
        self.fail("policy handoff did not converge before its two-second deadline")


class StateMachineTests(CoreFixture):
    def test_space_cancels_cold_start_and_moving_policy_handoff(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                self.setUp()
                epoch = self.begin_policy_handoff()
                if moving:
                    self.clock.now_ns += MS
                    self.install_inputs()
                    self.core.acknowledge_policy_reset(epoch, self.clock.now_ns)
                    self.core.submit_policy_action(
                        PolicyActionPacket(epoch, 0, self.clock.now_ns, policy_action(5)),
                        self.clock.now_ns,
                    )
                    self.assertEqual(self.core.tick().command.source, CommandSource.POLICY_SLEW)
                self.core.handle_key(" ", self.clock.now_ns)
                held = self.core.tick()
                self.assertEqual(held.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
                self.assertEqual(held.command.source, CommandSource.HOLD)
                self.assertGreater(held.snapshot.control_epoch, epoch)
                self.assertIsNone(held.snapshot.pending_policy_reset_epoch)
                self.assertFalse(self.core.acknowledge_policy_reset(epoch, self.clock.now_ns))
                self.assertFalse(self.core.submit_policy_action(
                    PolicyActionPacket(epoch, 1, self.clock.now_ns, (float("nan"),) * 14),
                    self.clock.now_ns,
                ))
                self.core.handle_key(" ", self.clock.now_ns)
                self.assertEqual(self.core.tick().command.source, CommandSource.HOLD)
                self.core.acknowledge_handoff_hold_published(held.snapshot.control_epoch, self.clock.now_ns)
                self.clock.now_ns += MS
                self.install_inputs()
                human = self.core.tick()
                self.assertEqual(human.snapshot.state, ControlState.HUMAN)
                self.assertEqual(human.command.left.end_pos, arm_feedback(self.clock.now_ns).eef_pose)
                # Absolute binary gripper: the left trigger is at 2.0, on the
                # open threshold, so the jaw opens regardless of the -1.2 the
                # feedback carried.
                self.assertEqual(
                    human.command.left.gripper, self.config.gripper_open_value
                )
                self.assertTrue(human.snapshot.intervention_occurred)

    def test_all_required_states_are_explicit(self):
        self.assertEqual(
            {state.value for state in ControlState},
            {
                "PRECHECK_HOLD",
                "MANUAL_RESET",
                "HANDOFF_TO_POLICY",
                "POLICY",
                "HANDOFF_TO_HUMAN",
                "HUMAN",
                "REVIEW_HOLD",
                "FAULT_HOLD",
                "READY_MOVE",
            },
        )

    def test_precheck_requires_fresh_bimanual_feedback_and_vr(self):
        self.core.update_feedback(
            arm_feedback(self.clock.now_ns), arm_feedback(self.clock.now_ns, 1.0)
        )
        self.core.mark_precheck_complete(self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.PRECHECK_HOLD)
        self.assertEqual(result.command.source, CommandSource.HOLD)

        self.install_inputs()
        self.core.mark_precheck_complete(self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.MANUAL_RESET)
        self.assertTrue(result.state_changed)

    def test_fixed_key_mapping_and_state_specific_transitions(self):
        self.enter_manual_reset()
        self.assertFalse(self.core.handle_key("q", self.clock.now_ns))
        self.assertTrue(self.core.handle_key(" ", self.clock.now_ns))
        self.core.tick(self.clock.now_ns)
        self.assertEqual(self.core.state, ControlState.MANUAL_RESET)

        self.core.handle_key("r", self.clock.now_ns)
        self.core.tick(self.clock.now_ns)
        self.assertEqual(self.core.state, ControlState.HANDOFF_TO_POLICY)
        self.assertTrue(self.core.handle_key("e", self.clock.now_ns))
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.REVIEW_HOLD)
        self.assertFalse(result.snapshot.episode_active)

    def test_fault_beats_end_and_takeover_when_threads_race(self):
        self.enter_policy()
        barrier = threading.Barrier(4)

        def submit(event, detail=""):
            barrier.wait()
            self.core.submit_event(event, self.clock.now_ns, detail)

        threads = [
            threading.Thread(target=submit, args=(ControlEvent.TAKEOVER,)),
            threading.Thread(target=submit, args=(ControlEvent.END_EPISODE,)),
            threading.Thread(target=submit, args=(ControlEvent.FAULT, "camera dead")),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertEqual(result.snapshot.fault_reason, "camera dead")
        self.assertEqual(result.command.source, CommandSource.HOLD)

    def test_end_beats_takeover_in_same_cycle(self):
        self.enter_policy()
        self.core.handle_key(" ", self.clock.now_ns)
        self.core.handle_key("e", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.REVIEW_HOLD)
        self.assertFalse(result.snapshot.intervention_occurred)

    def test_future_queued_key_advances_the_whole_control_tick(self):
        self.enter_policy()
        queued_ns = self.clock.now_ns + 10
        self.core.handle_key(" ", queued_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.assertEqual(result.command.timestamp_ns, queued_ns)
        request = next(
            event for event in result.events
            if event.name is TimelineEventName.TAKEOVER_REQUEST
        )
        gate = next(
            event for event in result.events
            if event.name is TimelineEventName.CONTROL_GATE
        )
        self.assertLessEqual(request.timestamp_ns, gate.timestamp_ns)


class PolicyHandshakeAndPacketTests(CoreFixture):
    def test_r_requests_cold_start_and_rejects_pre_ack_action(self):
        epoch = self.begin_policy_handoff()
        action = PolicyActionPacket(epoch, 0, self.clock.now_ns, policy_action())
        rejected = self.core.submit_policy_action(action, self.clock.now_ns)
        self.assertFalse(rejected)
        self.assertIn("reset", rejected.reason)

        self.assertFalse(self.core.acknowledge_policy_reset(epoch - 1, self.clock.now_ns))
        self.assertTrue(self.core.acknowledge_policy_reset(epoch, self.clock.now_ns))
        accepted = self.core.submit_policy_action(action, self.clock.now_ns)
        self.assertTrue(accepted)

    def test_epoch_sequence_and_freshness_checks(self):
        epoch = self.begin_policy_handoff()
        self.assertTrue(self.core.acknowledge_policy_reset(epoch, self.clock.now_ns))

        wrong_epoch = PolicyActionPacket(
            epoch - 1, 0, self.clock.now_ns, policy_action()
        )
        self.assertFalse(self.core.submit_policy_action(wrong_epoch, self.clock.now_ns))

        stale = PolicyActionPacket(
            epoch, 0, self.clock.now_ns - 251 * MS, policy_action()
        )
        self.assertFalse(self.core.submit_policy_action(stale, self.clock.now_ns))

        good = PolicyActionPacket(epoch, 3, self.clock.now_ns, policy_action())
        self.assertTrue(self.core.submit_policy_action(good, self.clock.now_ns))
        self.assertEqual(self.core.snapshot().latest_policy_sequence, 3)
        duplicate = PolicyActionPacket(epoch, 3, self.clock.now_ns, policy_action(2.0))
        self.assertFalse(self.core.submit_policy_action(duplicate, self.clock.now_ns))

    def test_policy_result_and_source_observation_have_independent_freshness(self):
        epoch = self.begin_policy_handoff()
        self.clock.now_ns += MS
        self.install_inputs()
        ack_ns = self.clock.now_ns
        self.assertTrue(self.core.acknowledge_policy_reset(epoch, ack_ns))

        # A result generated now is still rejected when its source snapshot is
        # older than the end-to-end policy deadline.
        self.clock.now_ns += 251 * MS
        self.install_inputs()
        stale_source = PolicyActionPacket(
            epoch,
            0,
            self.clock.now_ns,
            measured_action(),
            observation_timestamp_ns=ack_ns,
        )
        rejected = self.core.submit_policy_action(stale_source, self.clock.now_ns)
        self.assertFalse(rejected)
        self.assertIn("source observation", rejected.reason)

        # A source up to 200 ms old can safely complete, and once adopted the
        # command watchdog measures the result generation time rather than
        # charging the inference latency a second time.
        source_ns = self.clock.now_ns
        self.clock.now_ns += 200 * MS
        self.install_inputs()
        accepted = PolicyActionPacket(
            epoch,
            1,
            self.clock.now_ns,
            measured_action(),
            observation_timestamp_ns=source_ns,
        )
        self.assertTrue(self.core.submit_policy_action(accepted, self.clock.now_ns))
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.POLICY)

        self.clock.now_ns += 240 * MS
        self.install_inputs()
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.POLICY)

        self.clock.now_ns += 11 * MS
        self.install_inputs()
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)

    def test_nan_policy_action_faults_both_arms(self):
        epoch = self.begin_policy_handoff()
        self.core.acknowledge_policy_reset(epoch, self.clock.now_ns)
        invalid = list(policy_action())
        invalid[8] = math.nan
        accepted = self.core.submit_policy_action(
            PolicyActionPacket(epoch, 0, self.clock.now_ns, invalid),
            self.clock.now_ns,
        )
        self.assertFalse(accepted)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertEqual(result.command.source, CommandSource.HOLD)
        self.assertEqual(result.command.left.mode, int(CommandMode.POSITION_CONTROL))
        self.assertEqual(result.command.right.mode, int(CommandMode.POSITION_CONTROL))

    def test_two_second_slew_caps_every_emitted_step(self):
        epoch = self.begin_policy_handoff()
        self.clock.now_ns += MS
        self.install_inputs()
        self.core.acknowledge_policy_reset(epoch, self.clock.now_ns)
        target = policy_action(2.0)
        self.assertTrue(
            self.core.submit_policy_action(
                PolicyActionPacket(epoch, 0, self.clock.now_ns, target),
                self.clock.now_ns,
            )
        )
        result = self.core.tick(self.clock.now_ns)
        measured_left = arm_feedback(self.clock.now_ns).joint_pos
        self.assertEqual(result.command.source, CommandSource.POLICY_SLEW)
        previous = (*measured_left, -1.2)
        emitted = (*result.command.left.joint_pos, result.command.left.gripper)
        previous_right = (*arm_feedback(self.clock.now_ns, 1.0).joint_pos, -2.2)
        emitted_right = (*result.command.right.joint_pos, result.command.right.gripper)
        step_caps = self.config.policy_slew_step_per_arm
        for old, new, cap in zip(previous, emitted, step_caps):
            self.assertLessEqual(abs(new - old), cap + 1e-12)
        for old, new, cap in zip(previous_right, emitted_right, step_caps):
            self.assertLessEqual(abs(new - old), cap + 1e-12)

        completed_frame = None
        for frame in range(1, 120):
            self.clock.now_ns += 2 * SECOND // 120
            self.install_inputs()
            self.assertTrue(
                self.core.submit_policy_action(
                    PolicyActionPacket(epoch, frame, self.clock.now_ns, target),
                    self.clock.now_ns,
                )
            )
            complete = self.core.tick(self.clock.now_ns)
            current = (*complete.command.left.joint_pos, complete.command.left.gripper)
            current_right = (
                *complete.command.right.joint_pos,
                complete.command.right.gripper,
            )
            for old, new, cap in zip(emitted, current, step_caps):
                self.assertLessEqual(abs(new - old), cap + 1e-12)
            for old, new, cap in zip(emitted_right, current_right, step_caps):
                self.assertLessEqual(abs(new - old), cap + 1e-12)
            emitted, emitted_right = current, current_right
            if complete.snapshot.state is ControlState.POLICY:
                completed_frame = frame
                break

        self.assertIsNotNone(completed_frame)
        self.assertLess(completed_frame, 120)
        self.assertEqual(complete.snapshot.state, ControlState.POLICY)
        self.assertEqual(complete.command.left.joint_pos, target[:6])

    def test_slew_that_cannot_converge_by_deadline_faults(self):
        epoch = self.begin_policy_handoff()
        self.clock.now_ns += MS
        self.install_inputs()
        self.core.acknowledge_policy_reset(epoch, self.clock.now_ns)
        target = (100.0,) * 14
        self.core.submit_policy_action(
            PolicyActionPacket(epoch, 0, self.clock.now_ns, target),
            self.clock.now_ns,
        )
        slew_start = self.clock.now_ns
        self.core.tick(self.clock.now_ns)
        for frame in range(1, 120):
            self.clock.now_ns = slew_start + frame * (2 * SECOND // 120)
            self.install_inputs()
            self.core.submit_policy_action(
                PolicyActionPacket(epoch, frame, self.clock.now_ns, target),
                self.clock.now_ns,
            )
            self.core.tick(self.clock.now_ns)

        self.clock.now_ns = slew_start + 2 * SECOND
        self.install_inputs()
        self.core.submit_policy_action(
            PolicyActionPacket(epoch, 120, self.clock.now_ns, target),
            self.clock.now_ns,
        )
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertIn("failed to converge", result.snapshot.fault_reason)
        self.assertEqual(result.command.source, CommandSource.HOLD)

    def test_reachable_target_is_not_accepted_after_handoff_deadline(self):
        epoch = self.begin_policy_handoff()
        self.core.acknowledge_policy_reset(epoch, self.clock.now_ns)

        # Arriving at the target on the deadline is still a missed end-to-end
        # handoff budget.  The old ordering checked reachability first and
        # could incorrectly enter POLICY after the two-second cutoff.
        self.clock.now_ns += self.config.handoff_timeout_ns
        self.install_inputs()
        target = measured_action()
        self.assertTrue(
            self.core.submit_policy_action(
                PolicyActionPacket(epoch, 0, self.clock.now_ns, target),
                self.clock.now_ns,
            )
        )
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertIn("within 2s", result.snapshot.fault_reason)
        self.assertEqual(result.command.source, CommandSource.HOLD)

    def test_p_invalidates_old_epoch_and_requires_another_cold_start(self):
        old_epoch = self.enter_policy()
        self.clock.now_ns += MS
        self.install_inputs()
        self.core.handle_key(" ", self.clock.now_ns)
        first = self.core.tick(self.clock.now_ns)
        self.assertEqual(first.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.assertEqual(first.command.source, CommandSource.HOLD)
        self.clock.now_ns += 1
        self.assertTrue(
            self.core.acknowledge_handoff_hold_published(
                first.snapshot.control_epoch,
                self.clock.now_ns,
            )
        )
        self.install_inputs()
        self.core.tick(self.clock.now_ns)
        self.assertEqual(self.core.state, ControlState.HUMAN)

        self.clock.now_ns += MS
        self.install_inputs()
        self.core.handle_key("p", self.clock.now_ns)
        handoff = self.core.tick(self.clock.now_ns)
        new_epoch = handoff.snapshot.control_epoch
        self.assertGreater(new_epoch, old_epoch)
        self.assertEqual(handoff.snapshot.pending_policy_reset_epoch, new_epoch)
        self.assertFalse(
            self.core.submit_policy_action(
                PolicyActionPacket(old_epoch, 99, self.clock.now_ns, policy_action()),
                self.clock.now_ns,
            )
        )
        expired_invalid = list(policy_action())
        expired_invalid[0] = math.nan
        rejected = self.core.submit_policy_action(
            PolicyActionPacket(old_epoch, 100, self.clock.now_ns, expired_invalid),
            self.clock.now_ns,
        )
        self.assertFalse(rejected)
        self.assertEqual(rejected.reason, "control_epoch mismatch")
        self.assertEqual(self.core.state, ControlState.HANDOFF_TO_POLICY)
        self.assertTrue(self.core.acknowledge_policy_reset(new_epoch, self.clock.now_ns))
        packet_before_ack = PolicyActionPacket(
            new_epoch, 0, self.clock.now_ns - 1, policy_action()
        )
        self.assertFalse(self.core.submit_policy_action(packet_before_ack, self.clock.now_ns))

    def test_policy_timeout_enters_fault_hold(self):
        self.enter_policy()
        self.clock.now_ns += 251 * MS
        self.core.update_feedback(
            arm_feedback(self.clock.now_ns), arm_feedback(self.clock.now_ns, 1.0)
        )
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertIn("policy action timeout", result.snapshot.fault_reason)


@unittest.skipIf(Rotation is None, "scipy is required by the Human DAgger core")
class HumanRebaseTests(CoreFixture):
    def test_space_requires_post_gate_feedback_and_vr(self):
        old_epoch = self.enter_policy()
        self.clock.now_ns += MS
        gate_time = self.clock.now_ns
        self.core.handle_key(" ", gate_time)
        result = self.core.tick(gate_time)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.assertEqual(result.command.source, CommandSource.HOLD)
        self.assertEqual(result.command.left.mode, int(CommandMode.POSITION_CONTROL))
        self.assertEqual(result.command.right.mode, int(CommandMode.POSITION_CONTROL))
        self.assertGreater(result.snapshot.control_epoch, old_epoch)

        hold_publish_time = gate_time + 1
        self.assertTrue(
            self.core.acknowledge_handoff_hold_published(
                result.snapshot.control_epoch,
                hold_publish_time,
            )
        )

        self.core.update_feedback(
            arm_feedback(gate_time), arm_feedback(gate_time, 1.0)
        )
        result = self.core.tick(gate_time)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_HUMAN)

        self.core.update_vr(vr_pose(gate_time), vr_pose(gate_time))
        result = self.core.tick(gate_time)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.assertEqual(result.command.source, CommandSource.HOLD)

        next_tick = hold_publish_time
        self.clock.now_ns = next_tick
        self.core.update_feedback(
            arm_feedback(next_tick), arm_feedback(next_tick, 1.0)
        )
        self.core.update_vr(vr_pose(next_tick), vr_pose(next_tick))
        result = self.core.tick(next_tick)
        self.assertEqual(result.snapshot.state, ControlState.HUMAN)
        names = {event.name for event in self.core.timeline()}
        self.assertIn(TimelineEventName.TAKEOVER_REQUEST, names)
        self.assertIn(TimelineEventName.CONTROL_GATE, names)
        self.assertIn(TimelineEventName.HOLD_ACK, names)
        self.assertIn(TimelineEventName.HUMAN_ACTIVE, names)

    def test_first_human_frame_is_exactly_continuous(self):
        self.enter_policy()
        self.clock.now_ns += MS
        stamp = self.clock.now_ns
        left_feedback = arm_feedback(stamp, gripper=-1.7)
        right_feedback = arm_feedback(
            stamp,
            1.0,
            eef_pose=(-0.4, 0.2, 0.5, 2.8, -0.2, -2.7),
            gripper=-2.4,
        )
        left_vr = vr_pose(
            stamp,
            eef_pose=(1.2, 2.1, 3.3, -2.9, 0.3, 2.7),
            gripper=3.3,
        )
        right_vr = vr_pose(stamp, gripper=4.1)
        self.core.update_feedback(left_feedback, right_feedback)
        self.core.update_vr(left_vr, right_vr)
        self.core.handle_key(" ", stamp)
        result = self.core.tick(stamp)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.assertEqual(result.command.source, CommandSource.HOLD)

        self.clock.now_ns = stamp + 1
        self.assertTrue(
            self.core.acknowledge_handoff_hold_published(
                result.snapshot.control_epoch,
                self.clock.now_ns,
            )
        )
        left_feedback = ArmFeedback(
            left_feedback.joint_pos,
            left_feedback.eef_pose,
            left_feedback.gripper,
            self.clock.now_ns,
        )
        right_feedback = ArmFeedback(
            right_feedback.joint_pos,
            right_feedback.eef_pose,
            right_feedback.gripper,
            self.clock.now_ns,
        )
        left_vr = VrPose(left_vr.eef_pose, left_vr.gripper, self.clock.now_ns)
        right_vr = VrPose(right_vr.eef_pose, right_vr.gripper, self.clock.now_ns)
        self.core.update_feedback(left_feedback, right_feedback)
        self.core.update_vr(left_vr, right_vr)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.HUMAN)
        self.assertEqual(result.command.left.end_pos, left_feedback.eef_pose)
        self.assertEqual(result.command.right.end_pos, right_feedback.eef_pose)
        # The pose is exactly continuous, but the gripper is absolute: both
        # triggers (3.3 and 4.1) are above the close threshold, so both jaws
        # close instead of inheriting the -1.7 / -2.4 feedback values.
        self.assertEqual(
            result.command.left.gripper, self.config.gripper_closed_value
        )
        self.assertEqual(
            result.command.right.gripper, self.config.gripper_closed_value
        )

    def test_se3_delta_and_binary_gripper_are_applied(self):
        self.enter_policy()
        self.clock.now_ns += MS
        anchor_time = self.clock.now_ns
        left_feedback = arm_feedback(anchor_time, gripper=-1.0)
        right_feedback = arm_feedback(anchor_time, 1.0, gripper=-2.0)
        left_vr_anchor = vr_pose(anchor_time, gripper=2.0)
        right_vr_anchor = vr_pose(anchor_time, gripper=3.0)
        self.core.update_feedback(left_feedback, right_feedback)
        self.core.update_vr(left_vr_anchor, right_vr_anchor)
        self.core.handle_key(" ", anchor_time)
        first = self.core.tick(anchor_time)
        self.assertEqual(first.command.source, CommandSource.HOLD)

        anchor_time += 1
        self.clock.now_ns = anchor_time
        self.assertTrue(
            self.core.acknowledge_handoff_hold_published(
                first.snapshot.control_epoch,
                anchor_time,
            )
        )
        left_feedback = ArmFeedback(
            left_feedback.joint_pos,
            left_feedback.eef_pose,
            left_feedback.gripper,
            anchor_time,
        )
        right_feedback = ArmFeedback(
            right_feedback.joint_pos,
            right_feedback.eef_pose,
            right_feedback.gripper,
            anchor_time,
        )
        left_vr_anchor = VrPose(
            left_vr_anchor.eef_pose,
            left_vr_anchor.gripper,
            anchor_time,
        )
        right_vr_anchor = VrPose(
            right_vr_anchor.eef_pose,
            right_vr_anchor.gripper,
            anchor_time,
        )
        self.core.update_feedback(left_feedback, right_feedback)
        self.core.update_vr(left_vr_anchor, right_vr_anchor)
        active = self.core.tick(anchor_time)
        self.assertEqual(active.snapshot.state, ControlState.HUMAN)

        self.clock.now_ns += 10 * MS
        current_time = self.clock.now_ns
        left_current_rpy = (0.15, -0.25, 0.35)
        left_current = vr_pose(
            current_time,
            eef_pose=(1.1, 1.8, 3.3, *left_current_rpy),
            gripper=3.0,
        )
        right_current = vr_pose(
            current_time,
            eef_pose=(-0.9, -1.8, 2.7, 0.2, -0.1, -0.1),
            gripper=2.5,
        )
        self.core.update_feedback(
            arm_feedback(current_time, gripper=-1.0),
            arm_feedback(current_time, 1.0, gripper=-2.0),
        )
        self.core.update_vr(left_current, right_current)
        result = self.core.tick(current_time)

        self.assertEqual(result.command.source, CommandSource.HUMAN)
        for actual, wanted in zip(result.command.left.end_pos[:3], (0.5, -0.3, 0.6)):
            self.assertAlmostEqual(actual, wanted)

        robot_rotation = Rotation.from_euler("xyz", left_feedback.eef_pose[3:])
        anchor_rotation = Rotation.from_euler("xyz", left_vr_anchor.eef_pose[3:])
        current_rotation = Rotation.from_euler("xyz", left_current_rpy)
        expected_rotation = robot_rotation * (anchor_rotation.inv() * current_rotation)
        actual_rotation = Rotation.from_euler("xyz", result.command.left.end_pos[3:])
        angular_error = (expected_rotation.inv() * actual_rotation).magnitude()
        self.assertLess(angular_error, 1e-10)
        # Gripper is absolute and binary, independent of the anchor: the left
        # trigger at 3.0 is on the close threshold, the right at 2.5 sits in
        # the hysteresis band and so holds its current closed endpoint.
        self.assertAlmostEqual(
            result.command.left.gripper, self.config.gripper_closed_value
        )
        self.assertAlmostEqual(
            result.command.right.gripper, self.config.gripper_closed_value
        )
        self.assertEqual(result.snapshot.latest_rebased_expert, (
            *result.command.left.end_pos,
            result.command.left.gripper,
            *result.command.right.end_pos,
            result.command.right.gripper,
        ))


    def test_released_trigger_opens_a_jaw_the_policy_had_closed(self):
        """The regression this binary mapping exists for.

        Under the old anchor-relative delta, taking over from a policy that had
        closed the gripper left both the jaw and the trigger on their own lower
        endpoints, so no trigger travel could reopen the jaw.
        """
        self.enter_policy()
        self.clock.now_ns += MS
        stamp = self.clock.now_ns
        closed = self.config.gripper_closed_value
        # Policy has driven both jaws fully closed.
        self.core.update_feedback(
            arm_feedback(stamp, offset=0.0, gripper=closed),
            arm_feedback(stamp, offset=1.0, gripper=closed),
        )
        # The operator's hands are resting, both triggers fully released.
        self.core.update_vr(
            vr_pose(stamp, gripper=0.0),
            vr_pose(stamp, gripper=0.0),
        )
        self.core.handle_key(" ", stamp)
        held = self.core.tick(stamp)
        self.assertEqual(held.command.source, CommandSource.HOLD)
        self.clock.now_ns += MS
        self.assertTrue(
            self.core.acknowledge_handoff_hold_published(
                held.snapshot.control_epoch,
                self.clock.now_ns,
            )
        )
        stamp = self.clock.now_ns
        self.core.update_feedback(
            arm_feedback(stamp, offset=0.0, gripper=closed),
            arm_feedback(stamp, offset=1.0, gripper=closed),
        )
        self.core.update_vr(
            vr_pose(stamp, gripper=0.0),
            vr_pose(stamp, gripper=0.0),
        )
        result = self.core.tick(stamp)
        self.assertEqual(result.snapshot.state, ControlState.HUMAN)
        self.assertEqual(
            result.command.left.gripper, self.config.gripper_open_value
        )
        self.assertEqual(
            result.command.right.gripper, self.config.gripper_open_value
        )

    def test_gripper_endpoints_are_not_smoothed_by_the_human_filter(self):
        """A binary transition must reach its endpoint on the same tick."""
        config = replace(
            self.config,
            human_filter_min_cutoff_hz=1.0,
            human_filter_beta=0.15,
            human_filter_d_cutoff_hz=1.0,
        )
        core = HumanDaggerCore(config, clock_ns=self.clock)
        self.core = core
        self.enter_policy()
        self.clock.now_ns += MS
        stamp = self.clock.now_ns
        closed = config.gripper_closed_value
        core.update_feedback(
            arm_feedback(stamp, offset=0.0, gripper=closed),
            arm_feedback(stamp, offset=1.0, gripper=closed),
        )
        core.update_vr(vr_pose(stamp, gripper=5.0), vr_pose(stamp, gripper=5.0))
        core.handle_key(" ", stamp)
        held = core.tick(stamp)
        self.clock.now_ns += MS
        self.assertTrue(
            core.acknowledge_handoff_hold_published(
                held.snapshot.control_epoch, self.clock.now_ns
            )
        )
        stamp = self.clock.now_ns
        core.update_feedback(
            arm_feedback(stamp, offset=0.0, gripper=closed),
            arm_feedback(stamp, offset=1.0, gripper=closed),
        )
        core.update_vr(vr_pose(stamp, gripper=5.0), vr_pose(stamp, gripper=5.0))
        self.assertEqual(core.tick(stamp).snapshot.state, ControlState.HUMAN)

        # Release both triggers: the jaws must land exactly on the open endpoint
        # on this tick, not ramp toward it.
        self.clock.now_ns += 16 * MS
        stamp = self.clock.now_ns
        core.update_feedback(
            arm_feedback(stamp, offset=0.0, gripper=closed),
            arm_feedback(stamp, offset=1.0, gripper=closed),
        )
        core.update_vr(vr_pose(stamp, gripper=0.0), vr_pose(stamp, gripper=0.0))
        result = core.tick(stamp)
        self.assertEqual(result.command.left.gripper, config.gripper_open_value)
        self.assertEqual(result.command.right.gripper, config.gripper_open_value)


class CommandSafetyAndFaultTests(CoreFixture):
    def test_hold_policy_and_human_fill_both_joint_and_eef_fields(self):
        self.install_inputs()
        hold = self.core.tick(self.clock.now_ns).command
        self.assertEqual(hold.source, CommandSource.HOLD)
        self.assertEqual(len(hold.left.joint_pos), 6)
        self.assertEqual(len(hold.left.end_pos), 6)
        self.assertTrue(all(math.isfinite(value) for value in hold.left.joint_pos))
        self.assertTrue(all(math.isfinite(value) for value in hold.left.end_pos))

        self.enter_policy()
        policy = self.core.tick(self.clock.now_ns).command
        self.assertEqual(policy.source, CommandSource.POLICY)
        self.assertEqual(policy.left.end_pos, arm_feedback(self.clock.now_ns).eef_pose)
        self.assertEqual(policy.left.joint_pos, policy_action()[:6])

    def test_one_invalid_arm_feedback_faults_atomic_pair(self):
        self.enter_manual_reset()
        previous = self.core.tick(self.clock.now_ns).command
        bad_right = arm_feedback(
            self.clock.now_ns,
            1.0,
            eef_pose=(math.nan, 0, 0, 0, 0, 0),
        )
        self.assertFalse(
            self.core.update_feedback(arm_feedback(self.clock.now_ns), bad_right)
        )
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertEqual(result.command.source, CommandSource.HOLD)
        self.assertEqual(result.command.left.joint_pos, previous.left.joint_pos)
        self.assertEqual(result.command.right.joint_pos, previous.right.joint_pos)
        self.assertEqual(result.command.left.mode, int(CommandMode.POSITION_CONTROL))
        self.assertEqual(result.command.right.mode, int(CommandMode.POSITION_CONTROL))

    def test_tick_has_no_command_before_first_complete_feedback_pair(self):
        result = self.core.tick(self.clock.now_ns)
        self.assertIsNone(result.command)
        self.assertEqual(result.snapshot.state, ControlState.PRECHECK_HOLD)

    def test_handoff_timeout_faults_atomically(self):
        self.enter_policy()
        self.clock.now_ns += MS
        self.core.handle_key(" ", self.clock.now_ns)
        handoff = self.core.tick(self.clock.now_ns)
        self.assertEqual(handoff.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.clock.now_ns += 2 * SECOND + 1
        self.core.update_feedback(
            arm_feedback(self.clock.now_ns), arm_feedback(self.clock.now_ns, 1.0)
        )
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)
        self.assertEqual(result.command.source, CommandSource.HOLD)


class TriggerToggleTests(CoreFixture):
    """The physical trigger key "1" toggles POLICY <-> HUMAN.

    Mirrors the translation block in human_dagger.py's control process: only
    the settled POLICY/HUMAN states translate to Space/P; every other state
    leaves "1" unmapped so the core ignores it.
    """

    @staticmethod
    def translate_trigger(state: ControlState) -> str:
        key = "1"
        if state is ControlState.POLICY:
            key = " "
        elif state is ControlState.HUMAN:
            key = "p"
        return key

    def press_trigger(self) -> tuple[str, bool]:
        key = self.translate_trigger(self.core.state)
        return key, self.core.handle_key(key, self.clock.now_ns)

    def complete_handoff_to_human(self) -> None:
        held = self.core.tick()
        self.assertEqual(held.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.core.acknowledge_handoff_hold_published(
            held.snapshot.control_epoch, self.clock.now_ns
        )
        self.clock.now_ns += MS
        self.install_inputs()
        self.assertEqual(self.core.tick().snapshot.state, ControlState.HUMAN)

    def complete_handoff_to_policy(self) -> None:
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)
        epoch = result.snapshot.control_epoch
        self.clock.now_ns += MS
        self.install_inputs()
        self.assertTrue(self.core.acknowledge_policy_reset(epoch, self.clock.now_ns))
        target = policy_action()
        for frame in range(120):
            self.assertTrue(
                self.core.submit_policy_action(
                    PolicyActionPacket(epoch, frame, self.clock.now_ns, target),
                    self.clock.now_ns,
                )
            )
            if self.core.tick(self.clock.now_ns).snapshot.state is ControlState.POLICY:
                return
            self.clock.now_ns += 2 * SECOND // 120
            self.install_inputs()
        self.fail("policy handoff did not converge")

    def test_trigger_key_toggles_between_policy_and_human(self):
        self.enter_policy()

        # POLICY: press takes over.
        self.assertEqual(self.press_trigger(), (" ", True))
        self.core.tick()
        self.assertEqual(self.core.state, ControlState.HANDOFF_TO_HUMAN)
        # Mid-handoff press stays unmapped and ignored.
        self.assertEqual(self.press_trigger(), ("1", False))
        self.complete_handoff_to_human()

        # HUMAN: press resumes policy.
        self.assertEqual(self.press_trigger(), ("p", True))
        self.core.tick(self.clock.now_ns)
        # Mid-resume press must not cancel the resume (double-press guard).
        self.assertEqual(self.core.state, ControlState.HANDOFF_TO_POLICY)
        self.assertEqual(self.press_trigger(), ("1", False))
        self.complete_handoff_to_policy()

        # POLICY again: the cycle closes.
        self.assertEqual(self.press_trigger(), (" ", True))
        self.complete_handoff_to_human()

    def test_trigger_key_ignored_in_review_hold(self):
        self.enter_policy()
        self.core.handle_key("e", self.clock.now_ns)
        self.core.tick()
        self.assertEqual(self.core.state, ControlState.REVIEW_HOLD)
        self.assertEqual(self.press_trigger(), ("1", False))


if __name__ == "__main__":
    unittest.main()


READY_LEFT = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55)
READY_RIGHT = (1.05, 1.15, 1.25, 1.35, 1.45, 1.55)
READY_GRIPPERS = (-1.25, -2.25)


class ReadyMoveFixture(CoreFixture):
    """A core configured to park the arms at a ready pose before every episode."""

    def setUp(self) -> None:
        super().setUp()
        self.config = replace(
            self.config,
            ready_pose_left=READY_LEFT,
            ready_pose_right=READY_RIGHT,
            ready_gripper=READY_GRIPPERS,
            ready_move_timeout_ns=SECOND,
        )
        self.core = HumanDaggerCore(self.config, clock_ns=self.clock)


class ReadyMoveEntryTests(ReadyMoveFixture):
    def test_record_key_enters_ready_move_instead_of_policy_handoff(self):
        self.enter_manual_reset()
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.READY_MOVE)

    def test_record_key_marks_the_episode_active_before_the_arms_move(self):
        self.enter_manual_reset()
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertTrue(result.snapshot.episode_active)

    def test_ready_move_does_not_request_a_policy_reset_yet(self):
        # The policy handoff budget is two seconds; asking the policy to reset
        # before the arms have parked would spend it on the parking move.
        self.enter_manual_reset()
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertIsNone(result.snapshot.pending_policy_reset_epoch)


class ReadyMoveDisabledTests(CoreFixture):
    def test_record_key_goes_straight_to_policy_handoff_without_a_ready_pose(self):
        # Regression guard: a core with no configured ready pose must behave
        # exactly as it did before the ready move existed.
        self.enter_manual_reset()
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)
        self.assertEqual(
            result.snapshot.pending_policy_reset_epoch,
            result.snapshot.control_epoch,
        )


class ReadyMoveCommandTests(ReadyMoveFixture):
    def start_ready_move(self):
        self.enter_manual_reset()
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.READY_MOVE)
        return result

    def feed_from_command(self, command) -> None:
        """Feed back exactly what was commanded: a perfectly tracking arm."""
        self.assertTrue(
            self.core.update_feedback(
                ArmFeedback(
                    joint_pos=command.left.joint_pos,
                    eef_pose=(0.4, -0.1, 0.3, 0.2, -0.3, 0.4),
                    gripper=command.left.gripper,
                    timestamp_ns=self.clock.now_ns,
                ),
                ArmFeedback(
                    joint_pos=command.right.joint_pos,
                    eef_pose=(-0.4, 0.1, 0.35, -0.2, 0.1, -0.4),
                    gripper=command.right.gripper,
                    timestamp_ns=self.clock.now_ns,
                ),
            )
        )

    @staticmethod
    def emitted_action(command):
        return (
            *command.left.joint_pos,
            command.left.gripper,
            *command.right.joint_pos,
            command.right.gripper,
        )

    def test_first_ready_move_command_holds_the_measured_position(self):
        # MANUAL_RESET commands are END_CONTROL; one measured POSITION_CONTROL
        # frame lets the mode change land before any joint is asked to move.
        result = self.start_ready_move()
        self.assertEqual(result.command.source, CommandSource.HOLD)
        self.assertEqual(self.emitted_action(result.command), measured_action())

    def test_ready_move_commands_are_position_control(self):
        result = self.start_ready_move()
        self.clock.now_ns += MS
        self.feed_from_command(result.command)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.command.source, CommandSource.READY_MOVE)
        self.assertEqual(result.command.left.mode, int(CommandMode.POSITION_CONTROL))
        self.assertEqual(result.command.right.mode, int(CommandMode.POSITION_CONTROL))

    def test_ready_move_never_steps_further_than_the_configured_limit(self):
        result = self.start_ready_move()
        previous = measured_action()
        for _ in range(60):
            self.clock.now_ns += MS
            self.feed_from_command(result.command)
            result = self.core.tick(self.clock.now_ns)
            if result.snapshot.state is not ControlState.READY_MOVE:
                break
            emitted = self.emitted_action(result.command)
            for index, (old, new) in enumerate(zip(previous, emitted)):
                self.assertLessEqual(
                    abs(new - old),
                    self.config.ready_move_step_per_arm[index % 7] + 1e-9,
                    f"channel {index} stepped too far",
                )
            previous = emitted

    def test_ready_move_walks_the_commands_onto_the_ready_pose(self):
        result = self.start_ready_move()
        for _ in range(60):
            self.clock.now_ns += MS
            self.feed_from_command(result.command)
            result = self.core.tick(self.clock.now_ns)
            if self.emitted_action(result.command) == self.config.ready_action():
                return
        self.fail("ready move never commanded the configured ready pose")


class ReadyMoveArrivalTests(ReadyMoveCommandTests):
    def feed_frozen(self) -> None:
        """A blocked arm: feedback stays fresh but the joints never move."""
        self.install_inputs()

    def run_move(self, feeder, limit: int = 90):
        result = self.start_ready_move()
        for _ in range(limit):
            self.clock.now_ns += MS
            feeder(result.command)
            result = self.core.tick(self.clock.now_ns)
            if result.snapshot.state is not ControlState.READY_MOVE:
                return result
        return result

    def test_ready_move_completes_into_the_policy_handoff(self):
        result = self.run_move(lambda command: self.feed_from_command(command))
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)
        self.assertEqual(
            result.snapshot.pending_policy_reset_epoch,
            result.snapshot.control_epoch,
        )

    def test_blocked_arm_never_completes_the_ready_move(self):
        # The commands walk all the way onto the target, but the arm does not
        # follow.  Treating "the command arrived" as arrival would start the
        # episode from the wrong pose and record it as if it were right.
        result = self.run_move(lambda command: self.feed_frozen())
        self.assertEqual(result.snapshot.state, ControlState.READY_MOVE)

    def test_episode_start_request_lands_on_the_policy_handoff_tick(self):
        # The recorder opens on the tick that enters HANDOFF_TO_POLICY and
        # drops events emitted before that, so the request that opens the
        # recorded handoff row has to arrive on this exact tick.
        result = self.run_move(lambda command: self.feed_from_command(command))
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)
        names = {getattr(event.name, "value", str(event.name)) for event in result.events}
        self.assertIn("EPISODE_START_REQUEST", names)

    def test_arrival_tolerates_the_configured_tracking_error(self):
        # Real arms settle near the target, not on it.  Anything inside the
        # tolerance counts as arrived.
        offset = self.config.ready_arrival_tolerance * 0.9

        def near_miss(command):
            self.assertTrue(
                self.core.update_feedback(
                    ArmFeedback(
                        joint_pos=tuple(a + offset for a in command.left.joint_pos),
                        eef_pose=(0.4, -0.1, 0.3, 0.2, -0.3, 0.4),
                        gripper=command.left.gripper,
                        timestamp_ns=self.clock.now_ns,
                    ),
                    ArmFeedback(
                        joint_pos=tuple(a + offset for a in command.right.joint_pos),
                        eef_pose=(-0.4, 0.1, 0.35, -0.2, 0.1, -0.4),
                        gripper=command.right.gripper,
                        timestamp_ns=self.clock.now_ns,
                    ),
                )
            )

        result = self.run_move(near_miss)
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)


class ReadyMoveFailureTests(ReadyMoveCommandTests):
    def test_ready_move_that_never_arrives_latches_a_fault(self):
        result = self.start_ready_move()
        deadline = self.clock.now_ns + self.config.ready_move_timeout_ns + 10 * MS
        while self.clock.now_ns < deadline:
            self.clock.now_ns += 10 * MS
            # A blocked arm: feedback stays fresh, the joints never move.
            self.install_inputs()
            result = self.core.tick(self.clock.now_ns)
            if result.snapshot.state is ControlState.FAULT_HOLD:
                self.assertEqual(result.command.source, CommandSource.HOLD)
                return
        self.fail("a ready move that never arrives must latch a fault")

    def test_ready_move_faults_when_arm_feedback_goes_stale(self):
        self.start_ready_move()
        self.clock.now_ns += self.config.feedback_timeout_ns + MS
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.FAULT_HOLD)

    def test_ready_move_does_not_need_fresh_vr(self):
        # Nothing in the move reads the VR trackers, so a headset that drops
        # out while the arms park must not abort the episode.
        result = self.start_ready_move()
        for _ in range(30):
            self.clock.now_ns += self.config.vr_timeout_ns // 4
            self.feed_from_command(result.command)
            result = self.core.tick(self.clock.now_ns)
            self.assertNotEqual(result.snapshot.state, ControlState.FAULT_HOLD)
            if result.snapshot.state is ControlState.HANDOFF_TO_POLICY:
                return
        self.fail("ready move did not finish without VR")

    def test_end_key_aborts_the_ready_move_back_to_manual_reset(self):
        self.start_ready_move()
        self.clock.now_ns += MS
        self.install_inputs()
        self.core.handle_key("e", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.MANUAL_RESET)
        self.assertFalse(result.snapshot.episode_active)

    def test_aborting_the_ready_move_re_anchors_the_vr_teleop(self):
        # The arms have moved since MANUAL_RESET was last entered.  Reusing the
        # old anchor would make the first VR frame jump them back.
        result = self.start_ready_move()
        for _ in range(3):
            self.clock.now_ns += MS
            self.feed_from_command(result.command)
            self.core.update_vr(
                vr_pose(self.clock.now_ns, gripper=2.0),
                vr_pose(
                    self.clock.now_ns,
                    eef_pose=(-1.0, -2.0, 2.5, 0.1, -0.15, -0.2),
                    gripper=3.0,
                ),
            )
            result = self.core.tick(self.clock.now_ns)
        self.core.handle_key("e", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.MANUAL_RESET)
        # Anchored on the pose the arms are actually in: the VR pose has not
        # changed since the anchor, so the command repeats the measured EEF.
        self.assertEqual(result.command.source, CommandSource.HUMAN)
        self.assertEqual(result.command.left.mode, int(CommandMode.END_CONTROL))
        self.assertEqual(result.command.left.end_pos, (0.4, -0.1, 0.3, 0.2, -0.3, 0.4))


class ReadyMoveEveryEpisodeTests(ReadyMoveCommandTests):
    def test_the_second_episode_parks_the_arms_again(self):
        # The point of the ready move is that every recorded episode starts
        # from the same pose, not just the first one after startup.
        result = self.start_ready_move()
        for _ in range(90):
            self.clock.now_ns += MS
            self.feed_from_command(result.command)
            result = self.core.tick(self.clock.now_ns)
            if result.snapshot.state is ControlState.HANDOFF_TO_POLICY:
                break
        self.assertEqual(result.snapshot.state, ControlState.HANDOFF_TO_POLICY)

        self.clock.now_ns += MS
        self.install_inputs()
        self.core.handle_key("e", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.REVIEW_HOLD)

        self.clock.now_ns += MS
        self.install_inputs()
        self.assertTrue(self.core.reset_after_review(self.clock.now_ns))
        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        self.assertEqual(result.snapshot.state, ControlState.READY_MOVE)
