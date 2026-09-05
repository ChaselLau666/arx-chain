"""Remote Tau0VLA policy worker for the Human DAgger controller.

Drop-in alternative to the local ACT worker in human_dagger_policy.py: same
multiprocessing entry-point signature, same message contract on all four
queues, so human_dagger_core's epoch gating, freshness checks and handoff
logic apply unchanged.

The difference is inference topology. ACT runs a CUDA forward per observation;
Tau0VLA lives on a dedicated-link HTTP server and answers with a 30-step,
30 Hz action chunk. The worker therefore keeps a ChunkScheduler buffer
(tau0vla_protocol) filled by a single in-flight background request, and each
incoming observation may pull ONE buffered step, at most once per 30 Hz slot.
Early observations return their credit without advancing the trajectory; the
60 Hz controller continues publishing its current target and servicing keys,
and the core's policy_timeout_ns freshness check remains the safety backstop:
if the buffer starves for longer than that window while POLICY is active, the
core faults to HOLD, which is the intended failure behavior.

Latency calibration happens during worker startup (inside the PRECHECK budget),
against a throwaway session and synthetic observations, so the first
HANDOFF_TO_POLICY only ever pays one real round trip inside its 2 s cold-start
budget.

Never raises across the process boundary; failures surface as policy_error
status messages, exactly like the ACT worker.
"""

from __future__ import annotations

import argparse
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from tau0vla_protocol import (
    ACTION_DIM,
    ActionEMA,
    BinaryGripperStabilizer,
    CAMERA_NAMES,
    FPS,
    ChunkScheduler,
    Observation,
    ProtocolError,
    Tau0VLAHttpClient,
    resolve_replan_steps,
)


@dataclass(frozen=True)
class Tau0VLAWorkerConfig:
    server_url: str
    task_instruction: str
    request_timeout_s: float = 5.0
    max_response_age_ms: float = 2000.0
    replan_steps: str = "auto"
    chunk_blend_steps: int = 6
    gripper_blend_steps: int | None = None
    gripper_debounce_frames: int = 0
    gripper_low_threshold: float = -2.1
    gripper_high_threshold: float = -1.05
    gripper_low_value: float = -3.384
    gripper_high_value: float = 0.0
    arm_ema_alpha: float = 1.0
    gripper_ema_alpha: float = 1.0
    benchmark_warmup: int = 3
    benchmark_requests: int = 30
    latency_margin_ms: float = 100.0
    max_observation_age_ns: int = 250_000_000


def _worker_observation(message_observation: Mapping[str, Any]) -> Observation:
    """Convert a control-process observation dict into a protocol Observation.

    The dagger observation carries images keyed head/left_wrist/right_wrist as
    raw JPEG bytes and qpos as a 14-vector -- exactly the protocol layout, so
    this is a re-wrap, not a transformation.
    """

    qpos = np.asarray(message_observation["qpos"], dtype=np.float32).reshape(-1)
    images = {
        name: bytes(payload)
        for name, payload in message_observation["images_jpeg"].items()
    }
    return Observation(
        qpos=qpos,
        images=images,
        sample_monotonic_ns=int(message_observation["policy_basis_ns"]),
    )


def _calibration_observation() -> Observation:
    """Synthetic observation for latency benchmarking.

    Noise images compress WORSE than real scenes, so the measured RTT is a
    conservative overestimate -- the safe direction for replan selection. cv2
    is imported lazily; it is already a dependency of the dagger frontend.
    """

    import cv2

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ProtocolError("failed to encode the calibration image")
    payload = encoded.tobytes()
    return Observation(
        qpos=np.zeros(ACTION_DIM, dtype=np.float32),
        images={name: payload for name in CAMERA_NAMES},
        sample_monotonic_ns=time.monotonic_ns(),
    )


class _Tau0VLARuntime:
    """Session, latency calibration and chunk buffer for one worker process."""

    def __init__(self, config: Tau0VLAWorkerConfig) -> None:
        self.config = config
        self.client = Tau0VLAHttpClient(
            config.server_url,
            request_timeout=config.request_timeout_s,
            max_response_age_ms=config.max_response_age_ms,
        )
        self.client.health()
        self.client.policy_contract()
        self._request_id = 0

        # Calibrate against a throwaway session so synthetic frames never
        # enter the real task session's context.
        self.client.create_session("latency calibration (discard)")
        self.replan_steps = self._calibrate()

        # The real session; garbage actions from calibration are gone with
        # the calibration session.
        self.client.create_session(config.task_instruction)
        # Request IDs are session-local and consecutive; calibration used a
        # different session. reset() for R/P keeps this session and its counter.
        self._request_id = 0

        self.scheduler: Optional[ChunkScheduler] = None
        self.ema: Optional[ActionEMA] = None
        self.gripper: Optional[BinaryGripperStabilizer] = None
        self.filters_seeded = False

    def next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _calibrate(self) -> int:
        observation = _calibration_observation()
        rtts = []
        total = self.config.benchmark_warmup + self.config.benchmark_requests
        for index in range(total):
            chunk = self.client.infer(observation, self.next_request_id())
            if index >= self.config.benchmark_warmup:
                rtts.append(chunk.round_trip_ms)
        steps, self.p99_ms = resolve_replan_steps(
            str(self.config.replan_steps), rtts, self.config.latency_margin_ms,
        )
        return steps

    def reset(self) -> None:
        self.scheduler = ChunkScheduler(
            self.replan_steps,
            blend_steps=self.config.chunk_blend_steps,
            gripper_blend_steps=self.config.gripper_blend_steps,
        )
        self.ema = ActionEMA(
            arm_alpha=self.config.arm_ema_alpha,
            gripper_alpha=self.config.gripper_ema_alpha,
        )
        self.gripper = BinaryGripperStabilizer(
            self.config.gripper_debounce_frames,
            low_threshold=self.config.gripper_low_threshold,
            high_threshold=self.config.gripper_high_threshold,
            low_value=self.config.gripper_low_value,
            high_value=self.config.gripper_high_value,
        )
        self.filters_seeded = False


def tau0vla_policy_worker_main(
    worker_config: Tau0VLAWorkerConfig,
    control_queue: Any,
    observation_queue: Any,
    result_queue: Any,
    status_queue: Any,
) -> None:
    """Multiprocessing entry point. Messages are intentionally plain dicts."""

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        runtime = _Tau0VLARuntime(worker_config)
        status_queue.put({"kind": "policy_ready", "time_ns": time.monotonic_ns()})
    except BaseException as exc:  # propagate initialization failures to the safety process
        status_queue.put({"kind": "policy_error", "error": repr(exc), "time_ns": time.monotonic_ns()})
        executor.shutdown(wait=False)
        return

    active_epoch: int | None = None
    action_seq = 0  # global, never reset: the core requires monotone sequences
    pending: Optional[Future] = None
    pending_epoch: Optional[int] = None
    # Integer wall-clock slots, not observation count: the ROS loop is 60 Hz.
    action_origin_ns: int | None = None
    last_action_slot = -1

    def drop(epoch: int) -> None:
        status_queue.put(
            {
                "kind": "policy_observation_dropped",
                "control_epoch": epoch,
                "time_ns": time.monotonic_ns(),
            }
        )

    while True:
        try:
            message = control_queue.get_nowait()
        except queue.Empty:
            try:
                message = observation_queue.get(timeout=0.02)
            except queue.Empty:
                continue
        kind = message.get("kind")
        if kind == "stop":
            executor.shutdown(wait=False, cancel_futures=True)
            return
        if kind == "pause":
            # Gate the epoch; a response landing later is screened out by
            # pending_epoch before adoption.
            active_epoch = None
            continue
        if kind == "reset":
            runtime.reset()
            action_origin_ns = None
            last_action_slot = -1
            active_epoch = int(message["control_epoch"])
            status_queue.put(
                {
                    "kind": "policy_reset_ack",
                    "control_epoch": active_epoch,
                    "time_ns": time.monotonic_ns(),
                }
            )
            continue
        if kind != "observation" or active_epoch is None:
            continue
        if int(message["control_epoch"]) != active_epoch:
            continue

        observation_ns = int(message["observation"]["timestamps"]["observation_ns"])
        try:
            policy_basis_ns = int(message["observation"]["policy_basis_ns"])
        except KeyError:
            policy_basis_ns = observation_ns
        if time.monotonic_ns() - policy_basis_ns > worker_config.max_observation_age_ns:
            drop(active_epoch)
            continue

        now_ns = time.monotonic_ns()
        if action_origin_ns is not None:
            slot = (now_ns - action_origin_ns) * FPS // 1_000_000_000
            if slot <= last_action_slot:
                drop(active_epoch)
                continue

        try:
            protocol_observation = _worker_observation(message["observation"])
            scheduler = runtime.scheduler
            ema = runtime.ema
            gripper = runtime.gripper
            if scheduler is None or ema is None or gripper is None:
                raise ProtocolError("observation before any reset")
            if not runtime.filters_seeded:
                gripper.reset(protocol_observation.qpos)
                ema.reset(gripper.apply(protocol_observation.qpos))
                runtime.filters_seeded = True

            if pending is not None and pending.done():
                future, pending = pending, None
                adopted_epoch, pending_epoch = pending_epoch, None
                if adopted_epoch == active_epoch:
                    chunk = future.result()  # only current-epoch errors may fault
                    scheduler.adopt(
                        chunk,
                        initial=False,
                        arrival_monotonic_ns=time.monotonic_ns(),
                    )
                # Do not even unwrap a gated future: its exception is stale too.

            if scheduler.remaining == 0 and pending is None:
                # Cold start after reset, or recovery after a stall: fetch
                # synchronously; there is nothing to execute meanwhile anyway.
                chunk = runtime.client.infer(
                    protocol_observation, runtime.next_request_id()
                )
                scheduler.adopt(
                    chunk,
                    initial=True,
                    arrival_monotonic_ns=time.monotonic_ns(),
                )
            elif pending is None and scheduler.should_request(False):
                pending = executor.submit(
                    runtime.client.infer,
                    protocol_observation,
                    runtime.next_request_id(),
                )
                pending_epoch = active_epoch

            try:
                scheduled = scheduler.next_action()
            except BufferError:
                # Transient starvation: return the credit and let the next
                # observation retry; the core's policy timeout is the backstop.
                drop(active_epoch)
                continue

            action = np.asarray(ema.apply(gripper.apply(scheduled.action)), dtype=np.float64)
            if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                raise ProtocolError("scheduled action is not a finite 14-vector")

            action_seq += 1
            generated_ns = time.monotonic_ns()
            if action_origin_ns is None:
                action_origin_ns = generated_ns
            last_action_slot = (generated_ns - action_origin_ns) * FPS // 1_000_000_000
            result_queue.put(
                {
                    "kind": "policy_action",
                    "episode_id": int(message["episode_id"]),
                    "control_epoch": active_epoch,
                    "observation_seq": int(message["observation_seq"]),
                    "action_seq": action_seq,
                    "generated_ns": generated_ns,
                    "observation_ns": observation_ns,
                    "policy_basis_ns": policy_basis_ns,
                    "action": action,
                }
            )
        except Exception as exc:
            status_queue.put(
                {
                    "kind": "policy_error",
                    "control_epoch": active_epoch,
                    "error": repr(exc),
                    "time_ns": time.monotonic_ns(),
                }
            )
            active_epoch = None
            pending = None
            pending_epoch = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the Tau0VLA server contract for Human DAgger"
    )
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--max-response-age-ms", type=float, default=2000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = Tau0VLAHttpClient(
        args.server_url,
        request_timeout=args.request_timeout,
        max_response_age_ms=args.max_response_age_ms,
    )
    client.health()
    client.policy_contract()
    client.create_session(args.task_instruction)
    print(f"Tau0VLA policy preflight passed (model_id={client.model_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
