"""Human DAgger policy worker for the calibrated arx-feedback-v4 backend.

Why this exists next to human_dagger_tau0vla_policy.py:

  human_dagger_tau0vla_policy.py speaks the original tau0vla_protocol, whose
  HTTP surface is /api/v1/arx-lift2s/... That path is gone from the deployed
  server (404); only /arx/v4 remains. The v4 wire format also differs in ways
  no amount of configuration can bridge:

    * the request carries raw_joint_feedback + open_baselines + calibration_version
      instead of observation_state;
    * the response field is calibrated_action_chunk with
      wire_action_is_robot_command=false, so the returned values are NOT robot
      commands -- they must pass through CalibratedGripperMapper before being
      published;
    * the gripper is handled by a per-run measured calibration artifact rather
      than the old fixed BinaryGripperStabilizer thresholds.

So this worker reuses the same protocol stack the calibrated rollout uses
(CalibratedHttpClient + CalibratedChunkScheduler + tau0vla_calibration), while
keeping the worker contract the dagger core expects: a frozen config dataclass
plus a *_worker_main(config, control_q, observation_q, result_q, status_q).

Calibration lifecycle differs from the rollout on purpose. The rollout calibrates,
runs once, and marks the artifact consumed. A dagger session is long and is
interrupted by human takeover, so re-calibrating mid-session is not possible
without driving the grippers. This worker therefore validates and loads one
artifact at startup and keeps it for the whole session; the open-baseline recheck
that the rollout performs against live samples (assert_current_open) is not
available here because the worker has no ROS node of its own.
"""
from __future__ import annotations

import argparse
import queue
import socket
import time
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from tau0vla_calibrated_protocol import (
    ACTION_DIM,
    CAMERA_NAMES,
    FEEDBACK_PROTOCOL_VERSION,
    FPS,
    ActionEMA,
    CalibratedChunkScheduler,
    CalibratedHttpClient,
    Observation,
    ProtocolError,
    resolve_replan_steps,
)
from tau0vla_calibration import (
    current_robot_identity,
    load_artifact,
    mark_consumed,
    require_unconsumed,
    validate_artifact,
)


@dataclass(frozen=True)
class CalibratedWorkerConfig:
    server_url: str
    task_instruction: str
    calibration_file: str
    experiment: str = "joint-feedback"
    protocol_version: str = FEEDBACK_PROTOCOL_VERSION
    request_timeout_s: float = 5.0
    max_response_age_ms: float = 500.0
    # A fixed replan matches the calibrated rollout (REPLAN_STEPS=15). "auto"
    # benchmarks first, which costs a throwaway session of synthetic frames on
    # a server that records every request, so it is opt-in.
    replan_steps: str = "15"
    chunk_blend_steps: int = 6
    gripper_blend_steps: int = 6
    arm_ema_alpha: float = 0.6
    gripper_ema_alpha: float = 1.0
    calibration_max_age_s: float = 900.0
    consume_calibration: bool = True
    # 05_human_dagger.sh starts the frontend (and therefore this worker) at line
    # 521, but the arms it calibrates against only come up at 540/550 -- the
    # frontend must already be the sole command publisher before an arm spins.
    # So the artifact cannot exist yet at construction time; wait for the
    # launcher to drop it in.
    calibration_wait_s: float = 180.0
    arm_stack: str = "dagger"
    benchmark_warmup: int = 3
    benchmark_requests: int = 30
    latency_margin_ms: float = 100.0
    max_observation_age_ns: int = 250_000_000


def _worker_observation(message_observation: Mapping[str, Any]) -> Observation:
    """Re-wrap a dagger observation dict as a calibrated protocol Observation.

    qpos is the raw 14-vector joint feedback the v4 contract wants as
    raw_joint_feedback. eef stays None: requires_eef is false for v4 and the
    dagger observation carries no EEF channel.
    """

    qpos = np.asarray(message_observation["qpos"], dtype=np.float32).reshape(-1)
    images = {
        name: bytes(payload)
        for name, payload in message_observation["images_jpeg"].items()
    }
    return Observation(
        qpos=qpos,
        eef=None,
        images=images,
        sample_monotonic_ns=int(message_observation["policy_basis_ns"]),
    )


def _benchmark_observation() -> Observation:
    """Synthetic observation for latency benchmarking.

    Noise images compress WORSE than real scenes, so the measured RTT is a
    conservative overestimate -- the safe direction for replan selection.
    """

    import cv2

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ProtocolError("failed to encode the benchmark image")
    payload = encoded.tobytes()
    return Observation(
        qpos=np.zeros(ACTION_DIM, dtype=np.float32),
        eef=None,
        images={name: payload for name in CAMERA_NAMES},
        sample_monotonic_ns=time.monotonic_ns(),
    )


def _wait_for_calibration(config: CalibratedWorkerConfig) -> None:
    """Block until the launcher has written a complete calibration artifact.

    save_artifact writes in place, so a partially written file is observable.
    Requiring two identical sizes a beat apart is enough to avoid parsing a
    half-flushed artifact without reaching for filesystem locks.
    """
    path = Path(config.calibration_file)
    deadline = time.monotonic() + max(0.0, config.calibration_wait_s)
    previous = -1
    while True:
        if path.is_file():
            size = path.stat().st_size
            if size > 0 and size == previous:
                return
            previous = size
        elif previous != -1:
            raise ProtocolError(f"calibration artifact disappeared: {path}")
        if time.monotonic() >= deadline:
            raise ProtocolError(
                f"calibration artifact did not appear within "
                f"{config.calibration_wait_s:.0f}s: {path}"
            )
        time.sleep(0.25)


def build_client(config: CalibratedWorkerConfig) -> tuple[CalibratedHttpClient, Any]:
    """Validate the artifact against this robot and build a checked client.

    Shared by the worker and by --preflight so both refuse for the same reasons.
    """

    _wait_for_calibration(config)
    calibration = load_artifact(config.calibration_file)
    hostname, domain, boot_id, controllers = current_robot_identity(config.arm_stack)
    validate_artifact(
        calibration,
        hostname=hostname,
        ros_domain_id=domain,
        boot_id=boot_id,
        controller_identity=controllers,
        max_age_s=config.calibration_max_age_s,
    )
    client = CalibratedHttpClient(
        config.server_url,
        experiment=config.experiment,
        calibration=calibration,
        robot_id=hostname,
        protocol_version=config.protocol_version,
        request_timeout=config.request_timeout_s,
        max_response_age_ms=config.max_response_age_ms,
    )
    client.health()
    client.policy_contract()
    return client, calibration


class _CalibratedRuntime:
    """Session, replan selection and chunk buffer for one worker process."""

    def __init__(self, config: CalibratedWorkerConfig) -> None:
        self.config = config
        self.client, self.calibration = build_client(config)
        if config.consume_calibration:
            require_unconsumed(config.calibration_file)
        self._request_id = 0
        self.p99_ms = float("nan")

        if str(config.replan_steps).strip().lower() == "auto":
            # Throwaway session so synthetic frames never enter the real task
            # session's context or its server-side recording.
            self.client.create_session("latency calibration (discard)")
            self.replan_steps = self._benchmark()
        else:
            self.replan_steps = int(config.replan_steps)

        session = self.client.create_session(config.task_instruction)
        self.session_id = str(session["session_id"])
        # Request IDs are session-local and consecutive; a benchmark used a
        # different session. reset() for R/P keeps this session and its counter.
        self._request_id = 0
        if config.consume_calibration:
            mark_consumed(config.calibration_file, session_id=self.session_id)

        self.scheduler: Optional[CalibratedChunkScheduler] = None
        self.ema: Optional[ActionEMA] = None
        self.filters_seeded = False

    def next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _benchmark(self) -> int:
        observation = _benchmark_observation()
        rtts = []
        total = self.config.benchmark_warmup + self.config.benchmark_requests
        for index in range(total):
            chunk = self.client.infer(observation, self.next_request_id())
            if index >= self.config.benchmark_warmup:
                rtts.append(chunk.round_trip_ms)
        steps, self.p99_ms = resolve_replan_steps(
            "auto", rtts, self.config.latency_margin_ms,
        )
        return steps

    def reset(self) -> None:
        self.scheduler = CalibratedChunkScheduler(
            self.replan_steps,
            blend_steps=self.config.chunk_blend_steps,
            gripper_blend_steps=self.config.gripper_blend_steps,
            max_response_age_ms=self.config.max_response_age_ms,
        )
        self.ema = ActionEMA(
            arm_alpha=self.config.arm_ema_alpha,
            gripper_alpha=self.config.gripper_ema_alpha,
        )
        self.filters_seeded = False


def calibrated_policy_worker_main(
    worker_config: CalibratedWorkerConfig,
    control_queue: Any,
    observation_queue: Any,
    result_queue: Any,
    status_queue: Any,
) -> None:
    """Multiprocessing entry point. Messages are intentionally plain dicts."""

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        runtime = _CalibratedRuntime(worker_config)
        status_queue.put(
            {
                "kind": "policy_ready",
                "time_ns": time.monotonic_ns(),
                "session_id": runtime.session_id,
                "model_id": runtime.client.model_id,
                "replan_steps": runtime.replan_steps,
            }
        )
    except BaseException as exc:  # propagate init failures to the safety process
        status_queue.put(
            {"kind": "policy_error", "error": repr(exc), "time_ns": time.monotonic_ns()}
        )
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
            if scheduler is None or ema is None:
                raise ProtocolError("observation before any reset")
            if not runtime.filters_seeded:
                # Seed on raw feedback: the mapper already returns native
                # command coordinates, so no gripper stabilizer is involved.
                ema.reset(protocol_observation.qpos)
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

            action = np.asarray(ema.apply(scheduled.action), dtype=np.float64)
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
        description="Validate the calibrated arx-feedback-v4 contract for Human DAgger"
    )
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--experiment", default="joint-feedback")
    parser.add_argument("--protocol-version", default=FEEDBACK_PROTOCOL_VERSION)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--max-response-age-ms", type=float, default=500.0)
    parser.add_argument("--calibration-max-age-s", type=float, default=900.0)
    parser.add_argument("--calibration-wait-s", type=float, default=180.0)
    parser.add_argument("--arm-stack", choices=("rollout", "dagger"), default="dagger")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Contract-only preflight.

    This runs before any hardware is started, so there is no calibration
    artifact yet and no arm process to identify -- the launcher calibrates only
    once its own arms are up. Therefore the preflight deliberately checks just
    the server side: health, the v4 policy contract, and the offsets derived
    from it. create_session is not called; it would need a real artifact and
    would leave an unused session behind.
    """
    args = _build_parser().parse_args(argv)
    client = CalibratedHttpClient(
        args.server_url,
        experiment=args.experiment,
        calibration=None,
        robot_id=socket.gethostname(),
        protocol_version=args.protocol_version,
        request_timeout=args.request_timeout,
        max_response_age_ms=args.max_response_age_ms,
    )
    client.health()
    client.policy_contract()
    if not args.task_instruction.strip():
        raise ProtocolError("task instruction must not be empty")
    print(
        "Calibrated policy preflight passed "
        f"(model_id={client.model_id}, experiment={client.experiment}, "
        f"protocol={client.protocol_version}, "
        f"arm_offset={client.arm_offset_steps}, gripper_offset={client.gripper_offset_steps})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
