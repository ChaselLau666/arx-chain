"""ROS-independent safety and arbitration core for Human DAgger collection.

The ROS-facing process is deliberately kept out of this module.  It should stamp
all incoming samples with ``time.monotonic_ns()``, feed them to
``HumanDaggerCore``, and translate the returned :class:`ArmCommand` objects to
``arx5_arm_msg/RobotCmd``.  A single call to :meth:`HumanDaggerCore.tick`
arbitrates both arms, so a fault can never leave one arm in a different mode.
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable, Optional, Sequence, Tuple

try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:  # pragma: no cover - exercised only on misconfigured robots
    Rotation = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _SCIPY_IMPORT_ERROR = None


Vector6 = Tuple[float, float, float, float, float, float]
Vector14 = Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


class ControlState(str, Enum):
    PRECHECK_HOLD = "PRECHECK_HOLD"
    MANUAL_RESET = "MANUAL_RESET"
    HANDOFF_TO_POLICY = "HANDOFF_TO_POLICY"
    POLICY = "POLICY"
    HANDOFF_TO_HUMAN = "HANDOFF_TO_HUMAN"
    HUMAN = "HUMAN"
    REVIEW_HOLD = "REVIEW_HOLD"
    FAULT_HOLD = "FAULT_HOLD"


class CommandMode(IntEnum):
    """X5 SDK modes used by the external ``normal`` controller."""

    END_CONTROL = 4
    POSITION_CONTROL = 5


class CommandSource(str, Enum):
    HOLD = "HOLD"
    POLICY = "POLICY"
    HUMAN = "HUMAN"
    POLICY_SLEW = "POLICY_SLEW"


class ControlEvent(str, Enum):
    PRECHECK_COMPLETE = "PRECHECK_COMPLETE"
    START_POLICY = "START_POLICY"
    TAKEOVER = "TAKEOVER"
    RESUME_POLICY = "RESUME_POLICY"
    END_EPISODE = "END_EPISODE"
    FAULT = "FAULT"


class EventPriority(IntEnum):
    NORMAL = 0
    TRANSITION = 10
    END = 20
    FAULT = 30


class TimelineEventName(str, Enum):
    PRECHECK_COMPLETE = "PRECHECK_COMPLETE"
    EPISODE_START_REQUEST = "EPISODE_START_REQUEST"
    TAKEOVER_REQUEST = "TAKEOVER_REQUEST"
    POLICY_RESUME_REQUEST = "POLICY_RESUME_REQUEST"
    EPISODE_END_REQUEST = "EPISODE_END_REQUEST"
    CONTROL_GATE = "CONTROL_GATE"
    HOLD_ACK = "HOLD_ACK"
    POLICY_RESET_REQUEST = "POLICY_RESET_REQUEST"
    POLICY_RESET_ACK = "POLICY_RESET_ACK"
    POLICY_ACTION_ACCEPTED = "POLICY_ACTION_ACCEPTED"
    POLICY_SLEW_STARTED = "POLICY_SLEW_STARTED"
    POLICY_ACTIVE = "POLICY_ACTIVE"
    HUMAN_ACTIVE = "HUMAN_ACTIVE"
    STATE_TRANSITION = "STATE_TRANSITION"
    FAULT = "FAULT"


_EVENT_PRIORITY = {
    ControlEvent.PRECHECK_COMPLETE: EventPriority.NORMAL,
    ControlEvent.START_POLICY: EventPriority.TRANSITION,
    ControlEvent.TAKEOVER: EventPriority.TRANSITION,
    ControlEvent.RESUME_POLICY: EventPriority.TRANSITION,
    ControlEvent.END_EPISODE: EventPriority.END,
    ControlEvent.FAULT: EventPriority.FAULT,
}


def _float_tuple(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(value) for value in values)


def _validate_vector(values: Sequence[float], size: int, name: str) -> Optional[str]:
    if len(values) != size:
        return f"{name} must have {size} elements, got {len(values)}"
    if not all(math.isfinite(float(value)) for value in values):
        return f"{name} contains a non-finite value"
    return None


def _validate_timestamp(timestamp_ns: int, name: str) -> Optional[str]:
    if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
        return f"{name} timestamp must be an integer monotonic nanosecond value"
    if timestamp_ns < 0:
        return f"{name} timestamp must be non-negative"
    return None


@dataclass(frozen=True)
class ArmFeedback:
    """Measured state for one arm.

    ``joint_pos`` excludes the gripper.  ``eef_pose`` is ``xyz + roll/pitch/yaw``
    in radians, matching ``RobotStatus.end_pos``.
    """

    joint_pos: Vector6
    eef_pose: Vector6
    gripper: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_pos", _float_tuple(self.joint_pos))
        object.__setattr__(self, "eef_pose", _float_tuple(self.eef_pose))
        object.__setattr__(self, "gripper", float(self.gripper))


@dataclass(frozen=True)
class VrPose:
    """Raw VR pose for one hand (xyz, RPY in radians, and raw gripper)."""

    eef_pose: Vector6
    gripper: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "eef_pose", _float_tuple(self.eef_pose))
        object.__setattr__(self, "gripper", float(self.gripper))


@dataclass(frozen=True)
class PolicyActionPacket:
    """One post-processed 14-D policy target.

    Layout is ``left(j0..j5, gripper), right(j0..j5, gripper)``.  Epoch,
    sequence, generation-time freshness, and source-observation freshness are
    checked by the core before use. Tests and simple callers may omit
    ``observation_timestamp_ns``; it then defaults to ``timestamp_ns``.
    """

    control_epoch: int
    sequence: int
    timestamp_ns: int
    action: Vector14
    observation_timestamp_ns: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _float_tuple(self.action))
        if self.observation_timestamp_ns is None:
            object.__setattr__(self, "observation_timestamp_ns", self.timestamp_ns)


@dataclass(frozen=True)
class ArmCommand:
    """Pure-data equivalent of the fields required by ``RobotCmd``."""

    joint_pos: Vector6
    end_pos: Vector6
    gripper: float
    mode: int


@dataclass(frozen=True)
class BimanualCommand:
    left: ArmCommand
    right: ArmCommand
    source: CommandSource
    state: ControlState
    control_epoch: int
    timestamp_ns: int


@dataclass(frozen=True)
class TimelineEvent:
    name: TimelineEventName
    timestamp_ns: int
    state: ControlState
    control_epoch: int
    detail: str = ""


@dataclass(frozen=True)
class ActionAcceptance:
    accepted: bool
    reason: str

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class CoreSnapshot:
    state: ControlState
    control_epoch: int
    episode_active: bool
    intervention_occurred: bool
    pending_policy_reset_epoch: Optional[int]
    policy_reset_acknowledged: bool
    latest_policy_action: Optional[Vector14]
    latest_policy_sequence: int
    latest_rebased_expert: Optional[Vector14]
    fault_reason: Optional[str]
    transition_revision: int
    timeline_size: int


@dataclass(frozen=True)
class TickResult:
    command: Optional[BimanualCommand]
    snapshot: CoreSnapshot
    state_changed: bool
    events: Tuple[TimelineEvent, ...]


class _OneEuroFilter:
    """One Euro filter over the 7 channels of one arm's rebased target.

    Channels 0-2 are metres, 3-5 are radians (unwrapped against the previous
    filtered value so a pi -> -pi step is not smoothed as a full turn), 6 is
    the gripper. First sample passes through unchanged, which keeps the
    exact-first-frame handoff equality intact.

    The gripper channel is passed through unfiltered: it carries a binary
    endpoint, and smoothing it would drag each transition through the very
    intermediate openings the binary mapping exists to avoid.
    """

    _ANGLE_CHANNELS = (3, 4, 5)
    _PASSTHROUGH_CHANNELS = (6,)

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        self._min_cutoff = float(min_cutoff_hz)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff_hz)
        self._prev_time_ns: Optional[int] = None
        self._prev_value: Optional[list] = None
        self._prev_derivative: Optional[list] = None

    def reset(self) -> None:
        self._prev_time_ns = None
        self._prev_value = None
        self._prev_derivative = None

    @staticmethod
    def _alpha(cutoff_hz: float, dt_s: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)

    def filter(self, values: Tuple[float, ...], now_ns: int) -> Tuple[float, ...]:
        sample = list(values)
        if self._prev_value is None or self._prev_time_ns is None:
            self._prev_time_ns = now_ns
            self._prev_value = sample
            self._prev_derivative = [0.0] * len(sample)
            return tuple(sample)
        dt_s = (now_ns - self._prev_time_ns) / 1e9
        if dt_s <= 0.0:
            return tuple(self._prev_value)
        assert self._prev_derivative is not None
        d_alpha = self._alpha(self._d_cutoff, dt_s)
        result = []
        for index, raw in enumerate(sample):
            prev = self._prev_value[index]
            if index in self._PASSTHROUGH_CHANNELS:
                self._prev_derivative[index] = 0.0
                result.append(raw)
                continue
            if index in self._ANGLE_CHANNELS:
                # Shortest-path unwrap so wrap-around is not seen as motion.
                raw = prev + math.atan2(math.sin(raw - prev), math.cos(raw - prev))
            derivative = (raw - prev) / dt_s
            smoothed_derivative = (
                d_alpha * derivative + (1.0 - d_alpha) * self._prev_derivative[index]
            )
            cutoff = self._min_cutoff + self._beta * abs(smoothed_derivative)
            alpha = self._alpha(cutoff, dt_s)
            filtered = alpha * raw + (1.0 - alpha) * prev
            if index in self._ANGLE_CHANNELS:
                filtered = math.atan2(math.sin(filtered), math.cos(filtered))
            self._prev_derivative[index] = smoothed_derivative
            result.append(filtered)
        self._prev_time_ns = now_ns
        self._prev_value = result
        return tuple(result)


@dataclass(frozen=True)
class HumanDaggerConfig:
    feedback_timeout_ns: int = 100_000_000
    vr_timeout_ns: int = 100_000_000
    policy_timeout_ns: int = 250_000_000
    handoff_timeout_ns: int = 2_000_000_000
    policy_slew_duration_ns: int = 2_000_000_000
    policy_slew_step_per_arm: Tuple[float, ...] = (
        0.05,
        0.05,
        0.03,
        0.05,
        0.05,
        0.05,
        0.2,
    )
    future_timestamp_tolerance_ns: int = 5_000_000
    # The gripper is a single-sided bounded actuator, so HUMAN drives it as
    # an absolute binary target rather than an anchor-relative delta.  A
    # relative delta saturates whenever the anchor lands on an endpoint:
    # taking over from a policy that had closed the gripper leaves the
    # trigger already at its own lower endpoint, and no amount of trigger
    # travel can reopen the jaw.  The thresholds are hysteretic so a hand
    # resting mid-travel does not chatter between the two endpoints.
    gripper_trigger_open_below: float = 2.0
    gripper_trigger_close_above: float = 3.0
    gripper_open_value: float = 0.0
    gripper_closed_value: float = -3.384
    # One Euro smoothing of the HUMAN rebased target; min_cutoff <= 0 disables.
    # Disabled by default: the exact SE(3) rebase output is a tested contract.
    # The Human DAgger app opts in via human_dagger.yaml.
    human_filter_min_cutoff_hz: float = 0.0
    human_filter_beta: float = 0.15
    human_filter_d_cutoff_hz: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_slew_step_per_arm",
            _float_tuple(self.policy_slew_step_per_arm),
        )
        positive = (
            "feedback_timeout_ns",
            "vr_timeout_ns",
            "policy_timeout_ns",
            "handoff_timeout_ns",
            "policy_slew_duration_ns",
        )
        for field_name in positive:
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.future_timestamp_tolerance_ns < 0:
            raise ValueError("future_timestamp_tolerance_ns must be non-negative")
        if len(self.policy_slew_step_per_arm) != 7:
            raise ValueError("policy_slew_step_per_arm must have 7 elements")
        if not all(
            math.isfinite(step) and step > 0
            for step in self.policy_slew_step_per_arm
        ):
            raise ValueError("policy_slew_step_per_arm values must be finite and positive")
        for field_name in (
            "gripper_trigger_open_below",
            "gripper_trigger_close_above",
            "gripper_open_value",
            "gripper_closed_value",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.gripper_trigger_open_below > self.gripper_trigger_close_above:
            raise ValueError(
                "gripper_trigger_open_below must not exceed "
                "gripper_trigger_close_above"
            )
        for field_name in (
            "human_filter_min_cutoff_hz",
            "human_filter_beta",
            "human_filter_d_cutoff_hz",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.human_filter_min_cutoff_hz > 0 and (
            self.human_filter_beta < 0 or self.human_filter_d_cutoff_hz <= 0
        ):
            raise ValueError(
                "human_filter_beta must be >= 0 and human_filter_d_cutoff_hz > 0"
            )


@dataclass(frozen=True)
class _QueuedEvent:
    event: ControlEvent
    timestamp_ns: int
    detail: str


@dataclass(frozen=True)
class _HumanAnchor:
    left_feedback: ArmFeedback
    right_feedback: ArmFeedback
    left_vr: VrPose
    right_vr: VrPose


class HumanDaggerCore:
    """Thread-safe bimanual state machine and command arbiter."""

    def __init__(
        self,
        config: HumanDaggerConfig = HumanDaggerConfig(),
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._state = ControlState.PRECHECK_HOLD
        self._control_epoch = 0
        self._episode_active = False
        self._intervention_occurred = False
        self._fault_reason: Optional[str] = None

        self._left_feedback: Optional[ArmFeedback] = None
        self._right_feedback: Optional[ArmFeedback] = None
        self._left_vr: Optional[VrPose] = None
        self._right_vr: Optional[VrPose] = None
        self._human_anchor: Optional[_HumanAnchor] = None
        self._human_filters: Optional[dict] = None
        if config.human_filter_min_cutoff_hz > 0:
            self._human_filters = {
                side: _OneEuroFilter(
                    config.human_filter_min_cutoff_hz,
                    config.human_filter_beta,
                    config.human_filter_d_cutoff_hz,
                )
                for side in ("left", "right")
            }
        self._latest_rebased_expert: Optional[Vector14] = None

        self._latest_policy_packet: Optional[PolicyActionPacket] = None
        self._last_policy_sequence = -1
        self._pending_policy_reset_epoch: Optional[int] = None
        self._policy_reset_acknowledged = False
        self._policy_reset_ack_ns: Optional[int] = None
        self._slew_start_ns: Optional[int] = None
        self._slew_start_action: Optional[Vector14] = None
        self._last_slew_action: Optional[Vector14] = None

        self._handoff_started_ns: Optional[int] = None
        self._gate_timestamp_ns: Optional[int] = None
        self._handoff_hold_published_ns: Optional[int] = None
        self._event_sequence = 0
        self._pending_events: list[tuple[int, int, _QueuedEvent]] = []
        self._timeline: list[TimelineEvent] = []
        self._tick_timeline_cursor = 0
        self._transition_revision = 0

    @property
    def state(self) -> ControlState:
        with self._lock:
            return self._state

    @property
    def control_epoch(self) -> int:
        with self._lock:
            return self._control_epoch

    def snapshot(self) -> CoreSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def timeline(self) -> Tuple[TimelineEvent, ...]:
        with self._lock:
            return tuple(self._timeline)

    def precheck_ready(self, now_ns: Optional[int] = None) -> bool:
        now = self._resolve_now(now_ns)
        with self._lock:
            return self._feedback_fresh_locked(now) and self._vr_fresh_locked(now)

    def update_feedback(
        self,
        left: ArmFeedback,
        right: ArmFeedback,
    ) -> bool:
        """Atomically install a pair of measured arm states.

        Invalid data faults the shared bimanual controller and neither invalid
        sample is installed, preserving the last safe values for HOLD.
        """

        with self._lock:
            error = self._validate_feedback(left, "left feedback")
            error = error or self._validate_feedback(right, "right feedback")
            if error:
                self._enter_fault_locked(error, self._clock_ns())
                return False
            self._left_feedback = left
            self._right_feedback = right
            return True

    def update_vr(self, left: VrPose, right: VrPose) -> bool:
        """Atomically install a pair of raw VR targets."""

        with self._lock:
            error = self._validate_vr(left, "left VR")
            error = error or self._validate_vr(right, "right VR")
            if error:
                self._enter_fault_locked(error, self._clock_ns())
                return False
            self._left_vr = left
            self._right_vr = right
            return True

    def submit_event(
        self,
        event: ControlEvent,
        timestamp_ns: Optional[int] = None,
        detail: str = "",
    ) -> None:
        """Queue an event from any thread.

        The next tick handles only the highest-priority event and discards all
        lower-priority events from the same control cycle.  FIFO order is used
        for events at equal priority.
        """

        if not isinstance(event, ControlEvent):
            event = ControlEvent(event)
        now = self._resolve_now(timestamp_ns)
        with self._lock:
            self._event_sequence += 1
            queued = _QueuedEvent(event=event, timestamp_ns=now, detail=str(detail))
            heapq.heappush(
                self._pending_events,
                (-int(_EVENT_PRIORITY[event]), self._event_sequence, queued),
            )

    def handle_key(self, key: str, timestamp_ns: Optional[int] = None) -> bool:
        """Map the fixed operator keys R/Space/P/E to control events."""

        normalized = key.lower()
        mapping = {
            "r": ControlEvent.START_POLICY,
            " ": ControlEvent.TAKEOVER,
            "space": ControlEvent.TAKEOVER,
            "p": ControlEvent.RESUME_POLICY,
            "e": ControlEvent.END_EPISODE,
        }
        event = mapping.get(normalized)
        if event is None:
            return False
        self.submit_event(event, timestamp_ns)
        return True

    def mark_precheck_complete(self, timestamp_ns: Optional[int] = None) -> None:
        self.submit_event(ControlEvent.PRECHECK_COMPLETE, timestamp_ns)

    def request_fault(self, reason: str, timestamp_ns: Optional[int] = None) -> None:
        self.submit_event(ControlEvent.FAULT, timestamp_ns, detail=reason)

    def acknowledge_policy_reset(
        self,
        control_epoch: int,
        timestamp_ns: Optional[int] = None,
    ) -> bool:
        """Acknowledge that the worker cleared temporal aggregation/caches."""

        now = self._resolve_now(timestamp_ns)
        with self._lock:
            if (
                self._state is not ControlState.HANDOFF_TO_POLICY
                or self._pending_policy_reset_epoch != control_epoch
                or control_epoch != self._control_epoch
            ):
                return False
            self._policy_reset_acknowledged = True
            self._policy_reset_ack_ns = now
            self._append_timeline_locked(
                TimelineEventName.POLICY_RESET_ACK,
                now,
                detail=f"epoch={control_epoch}",
            )
            return True

    def acknowledge_handoff_hold_published(
        self,
        control_epoch: int,
        timestamp_ns: Optional[int] = None,
    ) -> bool:
        """Latch the time at which both HANDOFF HOLD messages were published.

        A human handoff cannot become active based on samples that happened
        between the key/gate event and the actual ROS publish.  The integration
        layer calls this only after it has published both mode-5 commands.
        """

        now = self._resolve_now(timestamp_ns)
        with self._lock:
            if (
                self._state is not ControlState.HANDOFF_TO_HUMAN
                or control_epoch != self._control_epoch
                or self._handoff_hold_published_ns is not None
            ):
                return False
            if self._gate_timestamp_ns is not None and now < self._gate_timestamp_ns:
                return False
            self._handoff_hold_published_ns = now
            return True

    def submit_policy_action(
        self,
        packet: PolicyActionPacket,
        now_ns: Optional[int] = None,
    ) -> ActionAcceptance:
        """Validate and atomically accept a policy action packet."""

        now = self._resolve_now(now_ns)
        with self._lock:
            # A late result from a gated epoch is inert regardless of its
            # payload. Validate current-epoch data strictly, but never let an
            # expired CUDA forward inject a fault after takeover.
            if (
                isinstance(packet.control_epoch, int)
                and not isinstance(packet.control_epoch, bool)
                and packet.control_epoch >= 0
                and packet.control_epoch != self._control_epoch
            ):
                return ActionAcceptance(False, "control_epoch mismatch")
            error = self._validate_policy_packet(packet)
            if error:
                self._enter_fault_locked(f"invalid policy action: {error}", now)
                return ActionAcceptance(False, error)
            if packet.sequence <= self._last_policy_sequence:
                return ActionAcceptance(False, "sequence is not newer")
            if not self._timestamp_fresh(
                packet.timestamp_ns,
                now,
                self.config.policy_timeout_ns,
            ):
                return ActionAcceptance(False, "action is not fresh")
            assert packet.observation_timestamp_ns is not None
            if not self._timestamp_fresh(
                packet.observation_timestamp_ns,
                now,
                self.config.policy_timeout_ns,
            ):
                return ActionAcceptance(False, "source observation is not fresh")
            if self._state not in (
                ControlState.HANDOFF_TO_POLICY,
                ControlState.POLICY,
            ):
                return ActionAcceptance(False, "policy is not an active control source")
            if self._state is ControlState.HANDOFF_TO_POLICY:
                if not self._policy_reset_acknowledged:
                    return ActionAcceptance(False, "policy reset has not been acknowledged")
                assert self._policy_reset_ack_ns is not None
                if packet.observation_timestamp_ns < self._policy_reset_ack_ns:
                    return ActionAcceptance(
                        False,
                        "source observation predates policy reset acknowledgement",
                    )
                if not self._feedback_fresh_locked(now):
                    return ActionAcceptance(False, "arm feedback is not fresh")

            self._latest_policy_packet = packet
            self._last_policy_sequence = packet.sequence
            self._append_timeline_locked(
                TimelineEventName.POLICY_ACTION_ACCEPTED,
                now,
                detail=f"sequence={packet.sequence}",
            )

            if self._state is ControlState.HANDOFF_TO_POLICY and self._slew_start_ns is None:
                assert self._left_feedback is not None
                assert self._right_feedback is not None
                self._slew_start_ns = now
                self._slew_start_action = self._feedback_as_action_locked()
                self._last_slew_action = self._slew_start_action
                self._append_timeline_locked(TimelineEventName.POLICY_SLEW_STARTED, now)

            return ActionAcceptance(True, "accepted")

    def tick(self, now_ns: Optional[int] = None) -> TickResult:
        """Advance the machine and build one atomic bimanual command.

        ``command`` is ``None`` only before the first complete feedback pair has
        ever been received.  The ROS layer must never synthesize zero commands
        in that case.
        """

        now = self._resolve_now(now_ns)
        with self._lock:
            revision_before = self._transition_revision
            now = self._process_pending_event_locked(now)
            self._advance_automatic_transitions_locked(now)
            command = self._build_command_locked(now)
            events = tuple(self._timeline[self._tick_timeline_cursor :])
            self._tick_timeline_cursor = len(self._timeline)
            snapshot = self._snapshot_locked()
            return TickResult(
                command=command,
                snapshot=snapshot,
                state_changed=self._transition_revision != revision_before,
                events=events,
            )

    def reset_after_review(self, now_ns: Optional[int] = None) -> bool:
        """Explicitly prepare another episode after REVIEW (never after fault)."""

        now = self._resolve_now(now_ns)
        with self._lock:
            if self._state is not ControlState.REVIEW_HOLD:
                return False
            if not (self._feedback_fresh_locked(now) and self._vr_fresh_locked(now)):
                return False
            self._invalidate_control_locked()
            self._episode_active = False
            self._intervention_occurred = False
            self._fault_reason = None
            self._capture_human_anchor_locked()
            self._transition_locked(ControlState.MANUAL_RESET, now, "review complete")
            return True

    def _resolve_now(self, now_ns: Optional[int]) -> int:
        now = self._clock_ns() if now_ns is None else now_ns
        if not isinstance(now, int) or isinstance(now, bool) or now < 0:
            raise ValueError("timestamp must be a non-negative integer nanosecond value")
        return now

    def _snapshot_locked(self) -> CoreSnapshot:
        return CoreSnapshot(
            state=self._state,
            control_epoch=self._control_epoch,
            episode_active=self._episode_active,
            intervention_occurred=self._intervention_occurred,
            pending_policy_reset_epoch=(
                self._pending_policy_reset_epoch
                if not self._policy_reset_acknowledged
                else None
            ),
            policy_reset_acknowledged=self._policy_reset_acknowledged,
            latest_policy_action=(
                self._latest_policy_packet.action
                if self._latest_policy_packet is not None
                else None
            ),
            latest_policy_sequence=(
                self._latest_policy_packet.sequence
                if self._latest_policy_packet is not None
                else -1
            ),
            latest_rebased_expert=self._latest_rebased_expert,
            fault_reason=self._fault_reason,
            transition_revision=self._transition_revision,
            timeline_size=len(self._timeline),
        )

    def _process_pending_event_locked(self, now: int) -> int:
        if not self._pending_events:
            return now
        _, _, queued = heapq.heappop(self._pending_events)
        self._pending_events.clear()
        effective_now = max(now, queued.timestamp_ns)
        self._handle_event_locked(queued, effective_now)
        return effective_now

    def _handle_event_locked(self, queued: _QueuedEvent, now: int) -> None:
        # Cross-process UI events can be enqueued just after the ROS loop sampled
        # its tick clock. Never record a control gate/state transition before the
        # operator request that caused it.
        now = max(now, queued.timestamp_ns)
        event = queued.event
        if event is ControlEvent.FAULT:
            self._enter_fault_locked(queued.detail or "external fault", now)
            return

        if event is ControlEvent.END_EPISODE:
            if self._state in (
                ControlState.HANDOFF_TO_POLICY,
                ControlState.POLICY,
                ControlState.HANDOFF_TO_HUMAN,
                ControlState.HUMAN,
            ):
                self._append_timeline_locked(
                    TimelineEventName.EPISODE_END_REQUEST,
                    queued.timestamp_ns,
                )
                self._invalidate_control_locked()
                self._episode_active = False
                self._transition_locked(ControlState.REVIEW_HOLD, now, "operator end")
            return

        if event is ControlEvent.PRECHECK_COMPLETE:
            if self._state is ControlState.PRECHECK_HOLD and (
                self._feedback_fresh_locked(now) and self._vr_fresh_locked(now)
            ):
                self._capture_human_anchor_locked()
                self._append_timeline_locked(
                    TimelineEventName.PRECHECK_COMPLETE,
                    queued.timestamp_ns,
                )
                self._transition_locked(ControlState.MANUAL_RESET, now, "precheck complete")
            return

        if event is ControlEvent.START_POLICY:
            if self._state is ControlState.MANUAL_RESET:
                self._append_timeline_locked(
                    TimelineEventName.EPISODE_START_REQUEST,
                    queued.timestamp_ns,
                )
                self._episode_active = True
                self._intervention_occurred = False
                self._begin_policy_handoff_locked(now, "episode start")
            return

        if event is ControlEvent.TAKEOVER:
            if self._state in (ControlState.POLICY, ControlState.HANDOFF_TO_POLICY):
                self._append_timeline_locked(
                    TimelineEventName.TAKEOVER_REQUEST,
                    queued.timestamp_ns,
                )
                self._invalidate_control_locked()
                self._handoff_started_ns = now
                self._gate_timestamp_ns = now
                self._handoff_hold_published_ns = None
                self._append_timeline_locked(TimelineEventName.CONTROL_GATE, now, "policy")
                self._transition_locked(
                    ControlState.HANDOFF_TO_HUMAN,
                    now,
                    "operator takeover",
                )
            return

        if event is ControlEvent.RESUME_POLICY:
            if self._state is ControlState.HUMAN:
                self._append_timeline_locked(
                    TimelineEventName.POLICY_RESUME_REQUEST,
                    queued.timestamp_ns,
                )
                self._begin_policy_handoff_locked(now, "operator resume")

    def _begin_policy_handoff_locked(self, now: int, reason: str) -> None:
        self._invalidate_control_locked()
        self._handoff_started_ns = now
        self._gate_timestamp_ns = now
        self._pending_policy_reset_epoch = self._control_epoch
        self._append_timeline_locked(TimelineEventName.CONTROL_GATE, now, reason)
        self._append_timeline_locked(
            TimelineEventName.POLICY_RESET_REQUEST,
            now,
            detail=f"epoch={self._control_epoch}",
        )
        self._transition_locked(ControlState.HANDOFF_TO_POLICY, now, reason)

    def _invalidate_control_locked(self) -> None:
        self._control_epoch += 1
        self._latest_policy_packet = None
        self._last_policy_sequence = -1
        self._pending_policy_reset_epoch = None
        self._policy_reset_acknowledged = False
        self._policy_reset_ack_ns = None
        self._slew_start_ns = None
        self._slew_start_action = None
        self._last_slew_action = None
        self._human_anchor = None
        self._latest_rebased_expert = None
        self._handoff_hold_published_ns = None

    def _advance_automatic_transitions_locked(self, now: int) -> None:
        if self._state is ControlState.PRECHECK_HOLD:
            return

        if self._state in (
            ControlState.MANUAL_RESET,
            ControlState.HANDOFF_TO_POLICY,
            ControlState.POLICY,
            ControlState.HANDOFF_TO_HUMAN,
            ControlState.HUMAN,
        ) and not self._feedback_fresh_locked(now):
            self._enter_fault_locked("arm feedback timeout", now)
            return

        if self._state in (ControlState.MANUAL_RESET, ControlState.HUMAN):
            if not self._vr_fresh_locked(now):
                self._enter_fault_locked("VR timeout", now)
            return

        if self._state is ControlState.HANDOFF_TO_HUMAN:
            assert self._handoff_started_ns is not None
            if now - self._handoff_started_ns > self.config.handoff_timeout_ns:
                self._enter_fault_locked("human handoff timeout", now)
                return
            # The takeover request's own control cycle must publish a complete
            # POSITION_CONTROL HOLD pair before END_CONTROL can be enabled.
            # Timestamp equality remains valid for post-gate samples, but only
            # on a later tick.
            if self._handoff_hold_published_ns is None:
                return
            if not self._vr_fresh_locked(now):
                return
            assert self._left_feedback is not None
            assert self._right_feedback is not None
            post_hold_feedback = (
                self._left_feedback.timestamp_ns >= self._handoff_hold_published_ns
                and self._right_feedback.timestamp_ns >= self._handoff_hold_published_ns
            )
            assert self._left_vr is not None
            assert self._right_vr is not None
            post_hold_vr = (
                self._left_vr.timestamp_ns >= self._handoff_hold_published_ns
                and self._right_vr.timestamp_ns >= self._handoff_hold_published_ns
            )
            if not (post_hold_feedback and post_hold_vr):
                return
            self._capture_human_anchor_locked()
            self._intervention_occurred = True
            self._append_timeline_locked(
                TimelineEventName.HOLD_ACK,
                now,
                "feedback and VR received after HOLD publish",
            )
            self._transition_locked(ControlState.HUMAN, now, "human inputs ready")
            self._append_timeline_locked(TimelineEventName.HUMAN_ACTIVE, now)
            return

        if self._state is ControlState.HANDOFF_TO_POLICY:
            assert self._handoff_started_ns is not None
            policy_handoff_budget_ns = min(
                self.config.handoff_timeout_ns,
                self.config.policy_slew_duration_ns,
            )
            elapsed_ns = now - self._handoff_started_ns
            if self._slew_start_ns is None:
                if elapsed_ns >= policy_handoff_budget_ns:
                    seconds = policy_handoff_budget_ns / 1_000_000_000
                    self._enter_fault_locked(
                        f"policy cold-start handoff timed out after {seconds:g}s",
                        now,
                    )
                return
            if not self._policy_packet_fresh_locked(now):
                self._enter_fault_locked("policy action timeout during handoff", now)
                return
            assert self._latest_policy_packet is not None
            assert self._last_slew_action is not None
            # The handoff budget is a hard end-to-end deadline.  A target that
            # only becomes reachable on or after the deadline must not be
            # allowed to enter POLICY merely because it is now one slew step
            # away; the approved failure behavior is an atomic HOLD.
            if elapsed_ns >= policy_handoff_budget_ns:
                seconds = policy_handoff_budget_ns / 1_000_000_000
                self._enter_fault_locked(
                    f"policy handoff failed to converge within {seconds:g}s",
                    now,
                )
            elif self._within_one_slew_step_locked(
                self._last_slew_action,
                self._latest_policy_packet.action,
            ):
                # The POLICY command built later in this same tick is at most one
                # configured step away from the last command actually emitted.
                self._pending_policy_reset_epoch = None
                self._transition_locked(ControlState.POLICY, now, "policy slew complete")
                self._append_timeline_locked(TimelineEventName.POLICY_ACTIVE, now)
            return

        if self._state is ControlState.POLICY and not self._policy_packet_fresh_locked(now):
            self._enter_fault_locked("policy action timeout", now)

    def _build_command_locked(self, now: int) -> Optional[BimanualCommand]:
        if self._left_feedback is None or self._right_feedback is None:
            return None

        if self._state in (ControlState.MANUAL_RESET, ControlState.HUMAN):
            if self._human_anchor is None or not self._vr_fresh_locked(now):
                return self._hold_command_locked(now)
            left_target = self._rebase_one(
                self._human_anchor.left_feedback,
                self._human_anchor.left_vr,
                self._left_vr,
            )
            right_target = self._rebase_one(
                self._human_anchor.right_feedback,
                self._human_anchor.right_vr,
                self._right_vr,
            )
            if self._human_filters is not None:
                # Smooth the commanded target, not the raw VR: this also
                # absorbs the 0-17ms sampling-age jitter of the fixed-rate
                # tick. The same filtered value feeds the recorded expert
                # action below, preserving recorded == emitted.
                left_filtered = self._human_filters["left"].filter(
                    (*left_target[0], left_target[1]), now
                )
                right_filtered = self._human_filters["right"].filter(
                    (*right_target[0], right_target[1]), now
                )
                left_target = (_float_tuple(left_filtered[:6]), float(left_filtered[6]))
                right_target = (_float_tuple(right_filtered[:6]), float(right_filtered[6]))
            assert self._left_vr is not None
            assert self._right_vr is not None
            self._latest_rebased_expert = (
                *left_target[0],
                left_target[1],
                *right_target[0],
                right_target[1],
            )
            return BimanualCommand(
                left=self._human_arm_command(self._left_feedback, left_target),
                right=self._human_arm_command(self._right_feedback, right_target),
                source=CommandSource.HUMAN,
                state=self._state,
                control_epoch=self._control_epoch,
                timestamp_ns=now,
            )

        if self._state is ControlState.POLICY:
            if self._latest_policy_packet is None:
                return self._hold_command_locked(now)
            return self._policy_command_locked(
                self._latest_policy_packet.action,
                CommandSource.POLICY,
                now,
            )

        if (
            self._state is ControlState.HANDOFF_TO_POLICY
            and self._slew_start_ns is not None
            and self._slew_start_action is not None
            and self._last_slew_action is not None
            and self._latest_policy_packet is not None
        ):
            slewed = self._slew_one_tick_locked(
                self._last_slew_action,
                self._latest_policy_packet.action,
            )
            self._last_slew_action = slewed
            return self._policy_command_locked(
                slewed,
                CommandSource.POLICY_SLEW,
                now,
            )

        return self._hold_command_locked(now)

    def _slew_one_tick_locked(
        self,
        previous: Sequence[float],
        target: Sequence[float],
    ) -> Vector14:
        steps = self.config.policy_slew_step_per_arm * 2
        values = []
        for old, wanted, step in zip(previous, target, steps):
            delta = wanted - old
            values.append(old + max(-step, min(step, delta)))
        return tuple(values)  # type: ignore[return-value]

    def _within_one_slew_step_locked(
        self,
        previous: Sequence[float],
        target: Sequence[float],
    ) -> bool:
        steps = self.config.policy_slew_step_per_arm * 2
        return all(
            abs(wanted - old) <= step + 1e-12
            for old, wanted, step in zip(previous, target, steps)
        )

    def _hold_command_locked(self, now: int) -> BimanualCommand:
        assert self._left_feedback is not None
        assert self._right_feedback is not None
        return BimanualCommand(
            left=self._hold_arm_command(self._left_feedback),
            right=self._hold_arm_command(self._right_feedback),
            source=CommandSource.HOLD,
            state=self._state,
            control_epoch=self._control_epoch,
            timestamp_ns=now,
        )

    @staticmethod
    def _hold_arm_command(feedback: ArmFeedback) -> ArmCommand:
        return ArmCommand(
            joint_pos=feedback.joint_pos,
            end_pos=feedback.eef_pose,
            gripper=feedback.gripper,
            mode=int(CommandMode.POSITION_CONTROL),
        )

    @staticmethod
    def _human_arm_command(
        feedback: ArmFeedback,
        target: tuple[Vector6, float],
    ) -> ArmCommand:
        return ArmCommand(
            joint_pos=feedback.joint_pos,
            end_pos=target[0],
            gripper=target[1],
            mode=int(CommandMode.END_CONTROL),
        )

    def _policy_command_locked(
        self,
        action: Sequence[float],
        source: CommandSource,
        now: int,
    ) -> BimanualCommand:
        assert self._left_feedback is not None
        assert self._right_feedback is not None
        left = action[:7]
        right = action[7:14]
        return BimanualCommand(
            left=ArmCommand(
                joint_pos=_float_tuple(left[:6]),
                end_pos=self._left_feedback.eef_pose,
                gripper=float(left[6]),
                mode=int(CommandMode.POSITION_CONTROL),
            ),
            right=ArmCommand(
                joint_pos=_float_tuple(right[:6]),
                end_pos=self._right_feedback.eef_pose,
                gripper=float(right[6]),
                mode=int(CommandMode.POSITION_CONTROL),
            ),
            source=source,
            state=self._state,
            control_epoch=self._control_epoch,
            timestamp_ns=now,
        )

    def _feedback_as_action_locked(self) -> Vector14:
        assert self._left_feedback is not None
        assert self._right_feedback is not None
        return (
            *self._left_feedback.joint_pos,
            self._left_feedback.gripper,
            *self._right_feedback.joint_pos,
            self._right_feedback.gripper,
        )

    def _capture_human_anchor_locked(self) -> None:
        assert self._left_feedback is not None
        assert self._right_feedback is not None
        assert self._left_vr is not None
        assert self._right_vr is not None
        self._human_anchor = _HumanAnchor(
            left_feedback=self._left_feedback,
            right_feedback=self._right_feedback,
            left_vr=self._left_vr,
            right_vr=self._right_vr,
        )
        self._latest_rebased_expert = None
        if self._human_filters is not None:
            for side_filter in self._human_filters.values():
                side_filter.reset()

    def _resolve_gripper(self, current_command: float, trigger: float) -> float:
        """Absolute binary gripper target from the raw VR trigger value.

        ``current_command`` is only consulted inside the hysteresis band, so
        a hand resting mid-travel holds the jaw where it is instead of
        chattering.  Outside the band the trigger alone decides, which is
        what makes "release to open" hold regardless of what the policy had
        commanded before the takeover.
        """
        open_value = float(self.config.gripper_open_value)
        closed_value = float(self.config.gripper_closed_value)
        if trigger <= self.config.gripper_trigger_open_below:
            return open_value
        if trigger >= self.config.gripper_trigger_close_above:
            return closed_value
        # Inside the band: keep whichever endpoint the jaw is nearer to.
        if abs(current_command - closed_value) <= abs(current_command - open_value):
            return closed_value
        return open_value

    def _rebase_one(
        self,
        feedback_anchor: ArmFeedback,
        vr_anchor: VrPose,
        vr_current: Optional[VrPose],
    ) -> tuple[Vector6, float]:
        assert vr_current is not None
        if Rotation is None:  # pragma: no cover - depends on deployment packaging
            raise RuntimeError(
                "Human DAgger SE(3) rebase requires scipy"
            ) from _SCIPY_IMPORT_ERROR

        # Avoid Euler canonicalisation changing an equivalent representation on
        # the very first HUMAN command.  Exact first-frame equality is useful to
        # both the hardware handoff and its acceptance test.
        if vr_current.eef_pose == vr_anchor.eef_pose:
            # The pose stays bit-exact, but the gripper is absolute: it must
            # follow the trigger from the very first frame, otherwise a
            # takeover that begins with the hand still would inherit the
            # policy's closed jaw and never reopen.
            return feedback_anchor.eef_pose, self._resolve_gripper(
                feedback_anchor.gripper, vr_current.gripper
            )

        robot_position = feedback_anchor.eef_pose[:3]
        robot_rpy = feedback_anchor.eef_pose[3:]
        vr_anchor_position = vr_anchor.eef_pose[:3]
        vr_anchor_rpy = vr_anchor.eef_pose[3:]
        vr_position = vr_current.eef_pose[:3]
        vr_rpy = vr_current.eef_pose[3:]

        position = tuple(
            robot + current - anchor
            for robot, current, anchor in zip(
                robot_position,
                vr_position,
                vr_anchor_position,
            )
        )
        robot_rotation = Rotation.from_euler("xyz", robot_rpy)
        anchor_rotation = Rotation.from_euler("xyz", vr_anchor_rpy)
        current_rotation = Rotation.from_euler("xyz", vr_rpy)
        command_rotation = robot_rotation * (anchor_rotation.inv() * current_rotation)
        command_rpy = command_rotation.as_euler("xyz")
        end_pos = _float_tuple((*position, *command_rpy))
        gripper = self._resolve_gripper(
            feedback_anchor.gripper, vr_current.gripper
        )
        return end_pos, float(gripper)

    def _feedback_fresh_locked(self, now: int) -> bool:
        return bool(
            self._left_feedback is not None
            and self._right_feedback is not None
            and self._timestamp_fresh(
                self._left_feedback.timestamp_ns,
                now,
                self.config.feedback_timeout_ns,
            )
            and self._timestamp_fresh(
                self._right_feedback.timestamp_ns,
                now,
                self.config.feedback_timeout_ns,
            )
        )

    def _vr_fresh_locked(self, now: int) -> bool:
        return bool(
            self._left_vr is not None
            and self._right_vr is not None
            and self._timestamp_fresh(
                self._left_vr.timestamp_ns,
                now,
                self.config.vr_timeout_ns,
            )
            and self._timestamp_fresh(
                self._right_vr.timestamp_ns,
                now,
                self.config.vr_timeout_ns,
            )
        )

    def _policy_packet_fresh_locked(self, now: int) -> bool:
        return bool(
            self._latest_policy_packet is not None
            and self._latest_policy_packet.control_epoch == self._control_epoch
            and self._timestamp_fresh(
                self._latest_policy_packet.timestamp_ns,
                now,
                self.config.policy_timeout_ns,
            )
        )

    def _timestamp_fresh(self, timestamp_ns: int, now: int, timeout_ns: int) -> bool:
        age = now - timestamp_ns
        return (
            -self.config.future_timestamp_tolerance_ns <= age <= timeout_ns
        )

    @staticmethod
    def _validate_feedback(sample: ArmFeedback, name: str) -> Optional[str]:
        return (
            _validate_vector(sample.joint_pos, 6, f"{name}.joint_pos")
            or _validate_vector(sample.eef_pose, 6, f"{name}.eef_pose")
            or (
                f"{name}.gripper must be finite"
                if not math.isfinite(sample.gripper)
                else None
            )
            or _validate_timestamp(sample.timestamp_ns, name)
        )

    @staticmethod
    def _validate_vr(sample: VrPose, name: str) -> Optional[str]:
        return (
            _validate_vector(sample.eef_pose, 6, f"{name}.eef_pose")
            or (
                f"{name}.gripper must be finite"
                if not math.isfinite(sample.gripper)
                else None
            )
            or _validate_timestamp(sample.timestamp_ns, name)
        )

    @staticmethod
    def _validate_policy_packet(packet: PolicyActionPacket) -> Optional[str]:
        return (
            _validate_vector(packet.action, 14, "action")
            or (
                "control_epoch must be a non-negative integer"
                if not isinstance(packet.control_epoch, int)
                or isinstance(packet.control_epoch, bool)
                or packet.control_epoch < 0
                else None
            )
            or (
                "sequence must be a non-negative integer"
                if not isinstance(packet.sequence, int)
                or isinstance(packet.sequence, bool)
                or packet.sequence < 0
                else None
            )
            or _validate_timestamp(packet.timestamp_ns, "policy action")
            or _validate_timestamp(
                packet.observation_timestamp_ns,
                "policy source observation",
            )
        )

    def _enter_fault_locked(self, reason: str, now: int) -> None:
        if self._state is ControlState.FAULT_HOLD:
            return
        self._fault_reason = str(reason)
        self._episode_active = False
        self._invalidate_control_locked()
        self._append_timeline_locked(TimelineEventName.FAULT, now, self._fault_reason)
        self._transition_locked(ControlState.FAULT_HOLD, now, self._fault_reason)

    def _transition_locked(
        self,
        new_state: ControlState,
        now: int,
        detail: str,
    ) -> None:
        if new_state is self._state:
            return
        previous = self._state
        self._state = new_state
        self._transition_revision += 1
        self._append_timeline_locked(
            TimelineEventName.STATE_TRANSITION,
            now,
            detail=f"{previous.value}->{new_state.value}: {detail}",
        )

    def _append_timeline_locked(
        self,
        name: TimelineEventName,
        timestamp_ns: int,
        detail: str = "",
    ) -> None:
        self._timeline.append(
            TimelineEvent(
                name=name,
                timestamp_ns=timestamp_ns,
                state=self._state,
                control_epoch=self._control_epoch,
                detail=detail,
            )
        )


__all__ = [
    "ActionAcceptance",
    "ArmCommand",
    "ArmFeedback",
    "BimanualCommand",
    "CommandMode",
    "CommandSource",
    "ControlEvent",
    "ControlState",
    "CoreSnapshot",
    "EventPriority",
    "HumanDaggerConfig",
    "HumanDaggerCore",
    "PolicyActionPacket",
    "TickResult",
    "TimelineEvent",
    "TimelineEventName",
    "VrPose",
]
