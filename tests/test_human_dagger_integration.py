from __future__ import annotations

import base64
import inspect
import multiprocessing as mp
import os
import pty
import queue
import signal
import sys
import tempfile
import termios
import threading
import time
import types
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from collection_ui import TerminalKeyReader  # noqa: E402
from human_dagger_core import (  # noqa: E402
    ArmFeedback,
    CommandMode,
    CommandSource,
    ControlState,
    HumanDaggerConfig,
    HumanDaggerCore,
    PolicyActionPacket,
    TimelineEventName,
    VrPose,
)
from human_dagger_recorder import (  # noqa: E402
    ControlMode,
    EventType,
    HumanDaggerRecorder,
    SOURCE_TIMESTAMP_NAMES,
)
# The mock worker and signature checks do not decode images. Keep this test
# runnable in lightweight development environments where production's OpenCV
# dependency is intentionally absent; deployment preflight validates real cv2.
try:  # pragma: no cover - deployment and robot environments take this branch
    import cv2 as _cv2  # noqa: F401
except ImportError:  # pragma: no cover - exercised by lightweight CI
    sys.modules["cv2"] = types.ModuleType("cv2")
    _remove_cv2_stub = True
else:
    _remove_cv2_stub = False

from human_dagger_policy import (  # noqa: E402
    mock_policy_worker_main,
    policy_worker_main,
)
from human_dagger import (  # noqa: E402
    _feedback_pair_acknowledges_hold,
    _retry_review_reset_after_close,
    _single_valid_feedback_for_hold,
    _timeline_event_records,
    recorder_worker_main,
)
if _remove_cv2_stub:
    del sys.modules["cv2"]
from validate_dagger_episode import validate_episode  # noqa: E402


MS = 1_000_000
SECOND = 1_000_000_000
CAMERAS = ("head", "left_wrist", "right_wrist")

# A real 2x2 JPEG keeps finalize-time image decoding in this integration test.
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQ"
    "FxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMK"
    "ChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
    "KCj/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL"
    "/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS"
    "0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlq"
    "c3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJ"
    "ytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAA"
    "AAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMi"
    "MoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
    "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0"
    "tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIR"
    "AxEAPwD58ooor0TgP//Z"
)


class FakeClock:
    def __init__(self, now_ns: int = 10 * SECOND) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, amount_ns: int = MS) -> int:
        self.now_ns += amount_ns
        return self.now_ns


class ForwardStartTimestamps:
    """Pickleable mapping that signals the exact pre-delay worker boundary."""

    def __init__(self, observation_ns: int, marker_path: str) -> None:
        self.observation_ns = observation_ns
        self.marker_path = marker_path

    def __getitem__(self, key: str) -> int:
        if key != "observation_ns":
            raise KeyError(key)
        # mock_policy_worker_main reads this immediately before its configured
        # delay, so the marker proves pause/reset are sent during the forward.
        Path(self.marker_path).touch()
        return self.observation_ns


def arm_pair(timestamp_ns: int) -> tuple[ArmFeedback, ArmFeedback]:
    left = ArmFeedback(
        joint_pos=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        eef_pose=(0.4, -0.1, 0.3, 0.2, -0.3, 0.4),
        gripper=-1.2,
        timestamp_ns=timestamp_ns,
    )
    right = ArmFeedback(
        joint_pos=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
        eef_pose=(-0.4, 0.1, 0.35, -0.2, 0.1, -0.4),
        gripper=-2.2,
        timestamp_ns=timestamp_ns,
    )
    return left, right


def vr_pair(timestamp_ns: int, movement: float = 0.0) -> tuple[VrPose, VrPose]:
    left = VrPose(
        eef_pose=(1.0 + movement, 2.0, 3.0, -0.1, 0.15, 0.25),
        gripper=2.0 + movement,
        timestamp_ns=timestamp_ns,
    )
    right = VrPose(
        eef_pose=(-1.0, -2.0 + movement, 2.5, 0.1, -0.15, -0.2),
        gripper=3.0 - movement,
        timestamp_ns=timestamp_ns,
    )
    return left, right


def measured_joint_action(timestamp_ns: int) -> tuple[float, ...]:
    left, right = arm_pair(timestamp_ns)
    return (
        *left.joint_pos,
        left.gripper,
        *right.joint_pos,
        right.gripper,
    )


class CoreHarness:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.core = HumanDaggerCore(
            HumanDaggerConfig(
                feedback_timeout_ns=5 * SECOND,
                vr_timeout_ns=5 * SECOND,
                policy_timeout_ns=5 * SECOND,
                handoff_timeout_ns=2 * SECOND,
                policy_slew_duration_ns=2 * SECOND,
            ),
            clock_ns=self.clock,
        )
        self.left_feedback, self.right_feedback = arm_pair(self.clock.now_ns)
        self.left_vr, self.right_vr = vr_pair(self.clock.now_ns)

    def install_inputs(self, *, movement: float = 0.0) -> None:
        self.left_feedback, self.right_feedback = arm_pair(self.clock.now_ns)
        self.left_vr, self.right_vr = vr_pair(self.clock.now_ns, movement)
        if not self.core.update_feedback(self.left_feedback, self.right_feedback):
            raise AssertionError("test feedback unexpectedly rejected")
        if not self.core.update_vr(self.left_vr, self.right_vr):
            raise AssertionError("test VR unexpectedly rejected")

    def enter_policy(self):
        self.install_inputs()
        self.core.mark_precheck_complete(self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        if result.snapshot.state is not ControlState.MANUAL_RESET:
            raise AssertionError(result.snapshot)

        self.core.handle_key("r", self.clock.now_ns)
        result = self.core.tick(self.clock.now_ns)
        epoch = result.snapshot.control_epoch
        if result.snapshot.state is not ControlState.HANDOFF_TO_POLICY:
            raise AssertionError(result.snapshot)

        self.clock.advance()
        self.install_inputs()
        if not self.core.acknowledge_policy_reset(epoch, self.clock.now_ns):
            raise AssertionError("policy reset acknowledgement unexpectedly rejected")
        if not self.core.submit_policy_action(
            PolicyActionPacket(
                epoch,
                0,
                self.clock.now_ns,
                measured_joint_action(self.clock.now_ns),
            ),
            self.clock.now_ns,
        ):
            raise AssertionError("first policy action unexpectedly rejected")
        result = self.core.tick(self.clock.now_ns)
        # The target equals feedback, so it is safe to activate immediately;
        # the deadline is an upper bound rather than a forced wait.
        if result.command is None or result.command.source is not CommandSource.POLICY:
            raise AssertionError(result.command)
        if result.snapshot.state is not ControlState.POLICY:
            raise AssertionError(result.snapshot)
        return result


def event_time(result, name: TimelineEventName) -> int:
    matches = [event.timestamp_ns for event in result.events if event.name is name]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name.value} event, got {result.events!r}")
    return matches[0]


class BlockingPolicyTakeoverIntegrationTests(unittest.TestCase):
    def test_space_gates_a_blocked_policy_forward_within_100ms(self) -> None:
        harness = CoreHarness()
        policy_result = harness.enter_policy()
        old_epoch = policy_result.snapshot.control_epoch
        invocation_timestamp = harness.clock.now_ns
        forward_started = threading.Event()
        late_packets: list[PolicyActionPacket] = []

        def blocking_mock_forward() -> None:
            forward_started.set()
            time.sleep(0.5)
            late_packets.append(
                PolicyActionPacket(
                    old_epoch,
                    99,
                    invocation_timestamp,
                    measured_joint_action(invocation_timestamp),
                )
            )

        worker = threading.Thread(target=blocking_mock_forward, daemon=True)
        worker.start()
        self.assertTrue(forward_started.wait(timeout=0.2))

        harness.clock.advance()
        harness.install_inputs()
        before_gate = time.perf_counter()
        self.assertTrue(harness.core.handle_key(" ", harness.clock.now_ns))
        takeover = harness.core.tick(harness.clock.now_ns)
        gate_latency = time.perf_counter() - before_gate

        self.assertLess(gate_latency, 0.100)
        self.assertEqual(takeover.snapshot.state, ControlState.HANDOFF_TO_HUMAN)
        self.assertEqual(takeover.snapshot.control_epoch, old_epoch + 1)
        self.assertIsNotNone(takeover.command)
        self.assertEqual(takeover.command.source, CommandSource.HOLD)
        self.assertEqual(takeover.command.left.mode, int(CommandMode.POSITION_CONTROL))
        self.assertEqual(takeover.command.right.mode, int(CommandMode.POSITION_CONTROL))
        self.assertEqual(takeover.command.left.joint_pos, harness.left_feedback.joint_pos)
        self.assertEqual(takeover.command.right.joint_pos, harness.right_feedback.joint_pos)
        # This models the ROS layer's acknowledgement only after the atomic
        # pair above has actually been published. It must not be needed to gate
        # the old policy epoch or produce the first HOLD.
        self.assertTrue(
            harness.core.acknowledge_handoff_hold_published(
                takeover.snapshot.control_epoch, harness.clock.now_ns
            )
        )

        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(late_packets), 1)
        acceptance = harness.core.submit_policy_action(
            late_packets[0], harness.clock.now_ns
        )
        self.assertFalse(acceptance)
        self.assertEqual(acceptance.reason, "control_epoch mismatch")


class CoordinatorSafetyHelperIntegrationTests(unittest.TestCase):
    def test_partial_arm_startup_selects_only_one_fresh_finite_feedback(self) -> None:
        now_ns = 20 * SECOND
        left, right = arm_pair(now_ns)
        left_only = {"feedback": {"left": (left, (), ())}}
        candidate = _single_valid_feedback_for_hold(
            left_only,
            now_ns,
            100 * MS,
            5 * MS,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate[0], "left")
        self.assertEqual(candidate[1], left)

        both = {
            "feedback": {
                "left": (left, (), ()),
                "right": (right, (), ()),
            }
        }
        self.assertIsNone(
            _single_valid_feedback_for_hold(both, now_ns, 100 * MS, 5 * MS)
        )

        stale_left = ArmFeedback(
            joint_pos=left.joint_pos,
            eef_pose=left.eef_pose,
            gripper=left.gripper,
            timestamp_ns=now_ns - 101 * MS,
        )
        self.assertIsNone(
            _single_valid_feedback_for_hold(
                {"feedback": {"left": (stale_left, (), ())}},
                now_ns,
                100 * MS,
                5 * MS,
            )
        )

    def test_review_close_latch_retries_after_stale_inputs_recover(self) -> None:
        harness = CoreHarness()
        harness.enter_policy()
        harness.core.handle_key("e", harness.clock.now_ns)
        review = harness.core.tick(harness.clock.now_ns)
        self.assertEqual(review.snapshot.state, ControlState.REVIEW_HOLD)

        # Model the recorder close acknowledgement arriving during a temporary
        # sensor gap. The close latch must remain set rather than being consumed
        # by this failed one-shot transition.
        harness.clock.advance(6 * SECOND)
        pending, completed = _retry_review_reset_after_close(
            harness.core,
            True,
            harness.clock.now_ns,
        )
        self.assertTrue(pending)
        self.assertFalse(completed)
        self.assertEqual(harness.core.state, ControlState.REVIEW_HOLD)

        harness.install_inputs()
        pending, completed = _retry_review_reset_after_close(
            harness.core,
            pending,
            harness.clock.now_ns,
        )
        self.assertFalse(pending)
        self.assertTrue(completed)
        self.assertEqual(harness.core.state, ControlState.MANUAL_RESET)

    def test_invalid_post_hold_feedback_pair_cannot_acknowledge(self) -> None:
        now_ns = 20 * SECOND
        left, right = arm_pair(now_ns)
        invalid_left = ArmFeedback(
            joint_pos=(float("nan"), *left.joint_pos[1:]),
            eef_pose=left.eef_pose,
            gripper=left.gripper,
            timestamp_ns=now_ns,
        )
        samples = {
            "feedback": {
                "left": (invalid_left, (), ()),
                "right": (right, (), ()),
            }
        }
        core = HumanDaggerCore(clock_ns=lambda: now_ns)
        feedback_pair_valid = core.update_feedback(invalid_left, right)
        self.assertFalse(feedback_pair_valid)
        self.assertFalse(
            _feedback_pair_acknowledges_hold(
                samples,
                now_ns - 1,
                feedback_pair_valid,
            )
        )

        valid_samples = {
            "feedback": {
                "left": (left, (), ()),
                "right": (right, (), ()),
            }
        }
        self.assertTrue(
            _feedback_pair_acknowledges_hold(
                valid_samples,
                now_ns - 1,
                True,
            )
        )

    def test_ending_during_policy_handoff_preserves_raw_request_and_gate(self) -> None:
        harness = CoreHarness()
        harness.install_inputs()
        harness.core.mark_precheck_complete(harness.clock.now_ns)
        harness.core.tick(harness.clock.now_ns)
        harness.core.handle_key("r", harness.clock.now_ns)
        handoff = harness.core.tick(harness.clock.now_ns)
        self.assertEqual(handoff.snapshot.state, ControlState.HANDOFF_TO_POLICY)

        pending, records = _timeline_event_records(handoff.events, None, 0)
        harness.clock.advance()
        harness.install_inputs()
        harness.core.handle_key("e", harness.clock.now_ns)
        ended = harness.core.tick(harness.clock.now_ns)
        pending, end_records = _timeline_event_records(ended.events, pending, 1)
        records.extend(end_records)

        by_name = {record["event"]: record for record in records}
        self.assertGreater(by_name["EPISODE_START_REQUEST"]["request_ns"], 0)
        self.assertEqual(by_name["EPISODE_START_REQUEST"]["epoch"], -1)
        self.assertGreater(by_name["CONTROL_GATE"]["gate_ns"], 0)
        self.assertEqual(
            by_name["POLICY_RESET_REQUEST"]["epoch"],
            handoff.snapshot.control_epoch,
        )
        self.assertGreater(by_name["EPISODE_END_REQUEST"]["request_ns"], 0)
        self.assertEqual(by_name["EPISODE_END_REQUEST"]["epoch"], -1)
        self.assertNotIn("HANDOFF_TO_POLICY", by_name)
        self.assertIsNotNone(pending)

    def test_ending_during_human_handoff_preserves_raw_request_and_gate(self) -> None:
        harness = CoreHarness()
        harness.enter_policy()
        harness.clock.advance()
        harness.install_inputs()
        harness.core.handle_key(" ", harness.clock.now_ns)
        handoff = harness.core.tick(harness.clock.now_ns)
        self.assertEqual(handoff.snapshot.state, ControlState.HANDOFF_TO_HUMAN)

        pending, records = _timeline_event_records(handoff.events, None, 0)
        harness.clock.advance()
        harness.install_inputs()
        harness.core.handle_key("e", harness.clock.now_ns)
        ended = harness.core.tick(harness.clock.now_ns)
        pending, end_records = _timeline_event_records(ended.events, pending, 1)
        records.extend(end_records)

        by_name = {record["event"]: record for record in records}
        self.assertGreater(by_name["TAKEOVER_REQUEST"]["request_ns"], 0)
        self.assertEqual(by_name["TAKEOVER_REQUEST"]["epoch"], -1)
        self.assertGreater(by_name["CONTROL_GATE"]["gate_ns"], 0)
        self.assertGreater(by_name["EPISODE_END_REQUEST"]["request_ns"], 0)
        self.assertEqual(by_name["EPISODE_END_REQUEST"]["epoch"], -1)
        self.assertNotIn("HANDOFF_TO_HUMAN", by_name)
        self.assertIsNotNone(pending)


class MultiprocessingPolicyWorkerIntegrationTests(unittest.TestCase):
    def test_production_and_mock_worker_queue_signatures_are_explicit(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(policy_worker_main).parameters),
            (
                "worker_config",
                "control_queue",
                "observation_queue",
                "result_queue",
                "status_queue",
            ),
        )
        mock_parameters = inspect.signature(mock_policy_worker_main).parameters
        self.assertEqual(
            tuple(mock_parameters),
            (
                "control_queue",
                "observation_queue",
                "result_queue",
                "status_queue",
                "delay_seconds",
            ),
        )
        self.assertEqual(mock_parameters["delay_seconds"].default, 0.0)

    def test_pause_and_new_reset_during_forward_do_not_pollute_new_epoch(self) -> None:
        context = mp.get_context("spawn")
        control_queue = context.Queue()
        observation_queue = context.Queue(maxsize=1)
        result_queue = context.Queue()
        status_queue = context.Queue()
        worker = context.Process(
            name="human-dagger-policy-mock-test",
            target=mock_policy_worker_main,
            args=(
                control_queue,
                observation_queue,
                result_queue,
                status_queue,
                0.5,
            ),
        )
        worker.start()
        stopped_cleanly = False
        try:
            # A spawn child imports NumPy/SciPy/HDF5 before entering the mock;
            # allow for a cold dynamic-loader cache on the robot computer.
            ready = status_queue.get(timeout=15.0)
            self.assertEqual(ready["kind"], "policy_ready")
            self.assertTrue(ready["mock"])

            old_epoch = 11
            new_epoch = 12
            control_queue.put({"kind": "reset", "control_epoch": old_epoch})
            first_ack = status_queue.get(timeout=2.0)
            self.assertEqual(
                (first_ack["kind"], first_ack["control_epoch"]),
                ("policy_reset_ack", old_epoch),
            )

            with tempfile.TemporaryDirectory() as marker_dir:
                marker_path = Path(marker_dir, "forward_started")
                observation_queue.put(
                    {
                        "kind": "observation",
                        "episode_id": 4,
                        "control_epoch": old_epoch,
                        "observation_seq": 101,
                        "observation": {
                            "timestamps": ForwardStartTimestamps(
                                time.monotonic_ns(), str(marker_path)
                            ),
                            "qpos": np.full(14, 1.0, dtype=np.float64),
                        },
                    }
                )

                deadline = time.monotonic() + 5.0
                while not marker_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue(marker_path.exists(), "mock forward did not start")

                # The marker is written immediately before the mock's 500 ms
                # sleep. Both messages therefore arrive while inference is in
                # flight, exactly as Space followed later by P would behave.
                control_queue.put({"kind": "pause"})
                control_queue.put({"kind": "reset", "control_epoch": new_epoch})

                old_result = result_queue.get(timeout=2.0)
                second_ack = status_queue.get(timeout=2.0)
                self.assertEqual(
                    (second_ack["kind"], second_ack["control_epoch"]),
                    ("policy_reset_ack", new_epoch),
                )

                # An unavoidable in-flight result remains tagged with its old
                # epoch, so the control process can discard it. It must never
                # be relabelled as the newly reset epoch.
                self.assertEqual(old_result["kind"], "policy_action")
                self.assertEqual(old_result["control_epoch"], old_epoch)
                self.assertEqual(old_result["observation_seq"], 101)
                np.testing.assert_array_equal(old_result["action"], np.ones(14))
                with self.assertRaises(queue.Empty):
                    result_queue.get(timeout=0.05)

                observation_queue.put(
                    {
                        "kind": "observation",
                        "episode_id": 4,
                        "control_epoch": new_epoch,
                        "observation_seq": 202,
                        "observation": {
                            "timestamps": {"observation_ns": time.monotonic_ns()},
                            "qpos": np.full(14, 2.0, dtype=np.float64),
                        },
                    }
                )
                new_result = result_queue.get(timeout=2.0)
                self.assertEqual(new_result["control_epoch"], new_epoch)
                self.assertEqual(new_result["observation_seq"], 202)
                np.testing.assert_array_equal(new_result["action"], np.full(14, 2.0))

                accepted_for_new_epoch = [
                    result
                    for result in (old_result, new_result)
                    if result["control_epoch"] == new_epoch
                ]
                self.assertEqual(
                    [result["observation_seq"] for result in accepted_for_new_epoch],
                    [202],
                )

            control_queue.put({"kind": "stop"})
            worker.join(timeout=2.0)
            stopped_cleanly = not worker.is_alive() and worker.exitcode == 0
        finally:
            if worker.is_alive():
                try:
                    control_queue.put_nowait({"kind": "stop"})
                except queue.Full:
                    pass
                worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
            for managed_queue in (
                control_queue,
                observation_queue,
                result_queue,
                status_queue,
            ):
                managed_queue.close()
                managed_queue.join_thread()

        self.assertTrue(stopped_cleanly, f"mock worker exit code: {worker.exitcode}")


@unittest.skipUnless(os.name == "posix", "signal test requires POSIX")
class RecorderWorkerSignalIntegrationTests(unittest.TestCase):
    def test_failed_frame_poisons_worker_before_queued_finalize(self) -> None:
        context = mp.get_context("spawn")
        commands = context.Queue()
        statuses = context.Queue()
        with tempfile.TemporaryDirectory() as output_dir:
            worker = context.Process(
                name="human-dagger-recorder-poison-test",
                target=recorder_worker_main,
                args=(commands, statuses),
            )
            worker.start()
            try:
                commands.put(
                    {
                        "kind": "start",
                        "output_dir": output_dir,
                        "episode_name": "episode_0",
                        "camera_names": CAMERAS,
                        "metadata": {"task": "poison_test"},
                        "flush_every": 1,
                    }
                )
                started = statuses.get(timeout=10.0)
                self.assertEqual(started["kind"], "recorder_started")

                # Queue finalize before the worker reports the bad frame.  The
                # worker itself, not the coordinator's later fault response,
                # must prevent this episode from entering the formal dataset.
                commands.put({"kind": "frame", "frame": {}})
                commands.put({"kind": "finalize"})

                received = [statuses.get(timeout=10.0) for _ in range(3)]
                self.assertEqual(
                    [message["kind"] for message in received],
                    ["recorder_quarantined", "recorder_error", "recorder_error"],
                )
                commands.put({"kind": "stop"})
                worker.join(timeout=5.0)
                self.assertFalse(worker.is_alive())
                self.assertEqual(worker.exitcode, 0)

                output_path = Path(output_dir)
                self.assertFalse((output_path / "episode_0.hdf5").exists())
                self.assertFalse((output_path / "episode_0.partial.hdf5").exists())
                quarantined = list(
                    (output_path / "quarantine").glob("episode_0.*.partial.hdf5")
                )
                self.assertEqual(len(quarantined), 1)
                with h5py.File(quarantined[0], "r") as root:
                    self.assertFalse(bool(root.attrs["finalized"]))
                    self.assertIn("frame mutation failed", root.attrs["quarantine_reason"])
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2.0)
                commands.close()
                commands.join_thread()
                statuses.close()
                statuses.join_thread()

    def test_sigint_quarantines_an_open_partial(self) -> None:
        context = mp.get_context("spawn")
        commands = context.Queue()
        statuses = context.Queue()
        with tempfile.TemporaryDirectory() as output_dir:
            worker = context.Process(
                name="human-dagger-recorder-signal-test",
                target=recorder_worker_main,
                args=(commands, statuses),
            )
            worker.start()
            try:
                commands.put(
                    {
                        "kind": "start",
                        "output_dir": output_dir,
                        "episode_name": "episode_0",
                        "camera_names": CAMERAS,
                        "metadata": {"task": "signal_test"},
                        "flush_every": 1,
                    }
                )
                started = statuses.get(timeout=10.0)
                self.assertEqual(started["kind"], "recorder_started")
                os.kill(worker.pid, signal.SIGINT)
                worker.join(timeout=5.0)
                self.assertFalse(worker.is_alive())
                self.assertEqual(worker.exitcode, 0)
                self.assertFalse(Path(output_dir, "episode_0.partial.hdf5").exists())
                quarantined = list(
                    Path(output_dir, "quarantine").glob("*.partial.hdf5")
                )
                self.assertEqual(len(quarantined), 1)
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2.0)
                commands.close()
                commands.join_thread()
                statuses.close()
                statuses.join_thread()


@unittest.skipUnless(os.name == "posix", "TerminalKeyReader requires a POSIX TTY")
class TerminalKeyReaderIntegrationTests(unittest.TestCase):
    def test_termios_is_restored_after_exception_on_a_pty(self) -> None:
        master_fd, slave_fd = pty.openpty()
        try:
            with os.fdopen(os.dup(slave_fd), "rb", buffering=0) as stream:
                canonical = termios.tcgetattr(stream.fileno())
                canonical[3] |= termios.ICANON | termios.ECHO
                termios.tcsetattr(stream.fileno(), termios.TCSANOW, canonical)
                original = termios.tcgetattr(stream.fileno())
                reader = TerminalKeyReader(stream)

                with self.assertRaisesRegex(RuntimeError, "simulated UI failure"):
                    with reader:
                        in_cbreak = termios.tcgetattr(stream.fileno())
                        self.assertFalse(in_cbreak[3] & termios.ICANON)
                        raise RuntimeError("simulated UI failure")

                restored = termios.tcgetattr(stream.fileno())
                # macOS may report the kernel-maintained PENDIN status bit
                # after tcsetattr even though no configured setting changed.
                # It is not a mode controlled by TerminalKeyReader.
                pendin = getattr(termios, "PENDIN", 0)
                restored[3] &= ~pendin
                original[3] &= ~pendin
                self.assertEqual(restored, original)
                self.assertIsNone(reader.fd)
                self.assertIsNone(reader.original_settings)
        finally:
            os.close(master_fd)
            os.close(slave_fd)


class RecorderTimelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def observation(harness: CoreHarness) -> dict[str, np.ndarray]:
        left = harness.left_feedback
        right = harness.right_feedback
        return {
            "qpos": np.asarray(
                (*left.joint_pos, left.gripper, *right.joint_pos, right.gripper),
                dtype=np.float32,
            ),
            "qvel": np.zeros(14, dtype=np.float32),
            "effort": np.zeros(14, dtype=np.float32),
            "eef": np.asarray(
                (*left.eef_pose, left.gripper, *right.eef_pose, right.gripper),
                dtype=np.float32,
            ),
            "robot_base": np.zeros(6, dtype=np.float32),
            "base_velocity": np.zeros(4, dtype=np.float32),
        }

    def append_result(
        self,
        recorder: HumanDaggerRecorder,
        harness: CoreHarness,
        result,
        mode: ControlMode,
        frame: int,
    ) -> None:
        kwargs = {}
        if mode is ControlMode.POLICY:
            command = result.command
            self.assertIsNotNone(command)
            kwargs["policy_action_joint"] = np.asarray(
                (
                    *command.left.joint_pos,
                    command.left.gripper,
                    *command.right.joint_pos,
                    command.right.gripper,
                ),
                dtype=np.float32,
            )
            kwargs["action_seq"] = result.snapshot.latest_policy_sequence
        elif mode is ControlMode.HUMAN:
            kwargs["expert_action_eef_raw"] = np.asarray(
                (
                    *harness.left_vr.eef_pose,
                    harness.left_vr.gripper,
                    *harness.right_vr.eef_pose,
                    harness.right_vr.gripper,
                ),
                dtype=np.float32,
            )
            self.assertIsNotNone(result.snapshot.latest_rebased_expert)
            kwargs["expert_action_eef_rebased"] = np.asarray(
                result.snapshot.latest_rebased_expert, dtype=np.float32
            )

        frame_time = 100 * SECOND + frame * MS
        recorder.append_frame(
            observation=self.observation(harness),
            images_jpeg={camera: JPEG for camera in CAMERAS},
            control_mode=mode,
            timestamps={
                "observation_ns": frame_time,
                "control_ns": frame_time + 100,
                **{
                    name: frame_time - 1000 + index
                    for index, name in enumerate(SOURCE_TIMESTAMP_NAMES)
                },
            },
            control_epoch=result.snapshot.control_epoch,
            **kwargs,
        )

    def test_policy_human_policy_timeline_finalizes_and_validates(self) -> None:
        harness = CoreHarness()
        recorder = HumanDaggerRecorder(
            self.output_dir,
            "episode_0",
            camera_names=CAMERAS,
            image_capacity=64,
            flush_every=2,
            metadata={
                "task": "integration_test",
                "height_command": 0.2,
                "dagger_round": 1,
                "policy_checkpoint": "/tmp/mock.ckpt",
                "policy_checkpoint_sha256": "mock",
                "git_commit": "integration-test",
                "nominal_fps": 60.0,
            },
        )

        policy = harness.enter_policy()
        self.append_result(recorder, harness, policy, ControlMode.POLICY, 0)

        harness.clock.advance()
        harness.install_inputs()
        harness.core.handle_key(" ", harness.clock.now_ns)
        to_human = harness.core.tick(harness.clock.now_ns)
        human_request = event_time(to_human, TimelineEventName.TAKEOVER_REQUEST)
        human_gate = event_time(to_human, TimelineEventName.CONTROL_GATE)
        handoff_pending, event_records = _timeline_event_records(
            to_human.events,
            None,
            1,
        )
        for event_record in event_records:
            recorder.record_event(**event_record)
        self.append_result(
            recorder, harness, to_human, ControlMode.HANDOFF_TO_HUMAN, 1
        )
        self.assertTrue(
            harness.core.acknowledge_handoff_hold_published(
                to_human.snapshot.control_epoch, harness.clock.now_ns
            )
        )

        harness.clock.advance()
        harness.install_inputs()
        human = harness.core.tick(harness.clock.now_ns)
        human_active = event_time(human, TimelineEventName.HUMAN_ACTIVE)
        handoff_pending, event_records = _timeline_event_records(
            human.events,
            handoff_pending,
            2,
        )
        for event_record in event_records:
            recorder.record_event(**event_record)
        self.assertIsNone(handoff_pending)
        self.append_result(recorder, harness, human, ControlMode.HUMAN, 2)

        harness.clock.advance()
        harness.install_inputs(movement=0.02)
        human = harness.core.tick(harness.clock.now_ns)
        self.append_result(recorder, harness, human, ControlMode.HUMAN, 3)

        harness.clock.advance()
        harness.install_inputs(movement=0.02)
        harness.core.handle_key("p", harness.clock.now_ns)
        to_policy = harness.core.tick(harness.clock.now_ns)
        policy_request = event_time(
            to_policy, TimelineEventName.POLICY_RESUME_REQUEST
        )
        policy_gate = event_time(to_policy, TimelineEventName.CONTROL_GATE)
        handoff_pending, event_records = _timeline_event_records(
            to_policy.events,
            None,
            4,
        )
        for event_record in event_records:
            recorder.record_event(**event_record)
        self.append_result(
            recorder, harness, to_policy, ControlMode.HANDOFF_TO_POLICY, 4
        )

        epoch = to_policy.snapshot.control_epoch
        harness.clock.advance()
        harness.install_inputs(movement=0.02)
        self.assertTrue(
            harness.core.acknowledge_policy_reset(epoch, harness.clock.now_ns)
        )
        self.assertTrue(
            harness.core.submit_policy_action(
                PolicyActionPacket(
                    epoch,
                    0,
                    harness.clock.now_ns,
                    measured_joint_action(harness.clock.now_ns),
                ),
                harness.clock.now_ns,
            )
        )
        policy = harness.core.tick(harness.clock.now_ns)
        self.assertEqual(policy.command.source, CommandSource.POLICY)
        self.assertEqual(policy.snapshot.state, ControlState.POLICY)
        policy_active = event_time(policy, TimelineEventName.POLICY_ACTIVE)
        handoff_pending, event_records = _timeline_event_records(
            policy.events,
            handoff_pending,
            5,
        )
        for event_record in event_records:
            recorder.record_event(**event_record)
        self.assertIsNone(handoff_pending)
        self.append_result(recorder, harness, policy, ControlMode.POLICY, 5)

        final_path = recorder.finalize()
        validation = validate_episode(final_path)
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(validation.num_frames, 6)

        with h5py.File(final_path, "r") as root:
            np.testing.assert_array_equal(
                root["/dagger/control_mode"][:],
                [
                    ControlMode.POLICY,
                    ControlMode.HANDOFF_TO_HUMAN,
                    ControlMode.HUMAN,
                    ControlMode.HUMAN,
                    ControlMode.HANDOFF_TO_POLICY,
                    ControlMode.POLICY,
                ],
            )
            np.testing.assert_array_equal(
                root["/dagger/intervention_mask"][:],
                [False, False, True, True, False, False],
            )
            np.testing.assert_array_equal(
                root["/dagger/supervision_valid"][:],
                [False, False, True, True, False, False],
            )
            events = root["/dagger/events"][:]
            handoff_events = [
                row
                for row in events
                if row["event"].decode().rstrip("\x00")
                in {EventType.HANDOFF_TO_HUMAN, EventType.HANDOFF_TO_POLICY}
            ]
            self.assertEqual(
                [row["event"].decode().rstrip("\x00") for row in handoff_events],
                [EventType.HANDOFF_TO_HUMAN, EventType.HANDOFF_TO_POLICY],
            )
            np.testing.assert_array_equal(
                [row["frame"] for row in handoff_events], [2, 5]
            )
            np.testing.assert_array_equal(
                [row["epoch"] for row in handoff_events], [2, 3]
            )

    def test_ended_incomplete_handoff_raw_events_finalize_and_validate(self) -> None:
        harness = CoreHarness()
        harness.install_inputs()
        harness.core.mark_precheck_complete(harness.clock.now_ns)
        harness.core.tick(harness.clock.now_ns)
        harness.core.handle_key("r", harness.clock.now_ns)
        handoff = harness.core.tick(harness.clock.now_ns)
        self.assertEqual(handoff.snapshot.state, ControlState.HANDOFF_TO_POLICY)

        recorder = HumanDaggerRecorder(
            self.output_dir,
            "episode_incomplete_handoff",
            camera_names=CAMERAS,
            image_capacity=64,
            flush_every=1,
            metadata={
                "task": "incomplete_handoff_test",
                "height_command": 0.2,
                "policy_checkpoint_sha256": "mock",
            },
        )
        pending, records = _timeline_event_records(handoff.events, None, 0)
        for event in records:
            recorder.record_event(**event)
        self.append_result(
            recorder,
            harness,
            handoff,
            ControlMode.HANDOFF_TO_POLICY,
            0,
        )

        harness.clock.advance()
        harness.install_inputs()
        harness.core.handle_key("e", harness.clock.now_ns)
        ended = harness.core.tick(harness.clock.now_ns)
        self.assertEqual(ended.snapshot.state, ControlState.REVIEW_HOLD)
        pending, records = _timeline_event_records(ended.events, pending, 1)
        for event in records:
            recorder.record_event(**event)
        self.assertIsNotNone(pending)

        final_path = recorder.finalize()
        validation = validate_episode(final_path)
        self.assertTrue(validation.valid, validation.errors)
        with h5py.File(final_path, "r") as root:
            events = root["/dagger/events"][:]
            decoded = [row["event"].decode().rstrip("\x00") for row in events]
            self.assertIn("EPISODE_START_REQUEST", decoded)
            self.assertIn("CONTROL_GATE", decoded)
            self.assertIn("EPISODE_END_REQUEST", decoded)
            self.assertNotIn(EventType.HANDOFF_TO_POLICY, decoded)


if __name__ == "__main__":
    unittest.main()
