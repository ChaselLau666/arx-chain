"""Standalone ARX client for sessioned Tau0VLA calibrated-v3 inference."""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import yaml

from safe_height import is_safe_and_stable
from tau0vla_calibrated_protocol import (
    ACTION_DIM,
    FPS,
    ActionEMA,
    CalibratedActionChunk,
    CalibratedChunkScheduler,
    CalibratedHttpClient,
    Observation,
    ProtocolError,
    resolve_replan_steps,
)
from tau0vla_calibrated_trace import TraceWriter, analyze_trace, plot_trace, write_summary
from tau0vla_calibration import (
    assert_current_open,
    current_robot_identity,
    load_artifact,
    mark_consumed,
    require_unconsumed,
    return_trajectory,
    validate_artifact,
)
from utils.setup_loader import setup_loader


ROOT = Path(__file__).resolve().parent
ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])


class TeeStream:
    """Line-buffered stdout/stderr duplication without a shell pipeline."""
    def __init__(self, terminal, log_stream, lock: threading.Lock):
        self.terminal = terminal
        self.log_stream = log_stream
        self.lock = lock
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, value):
        with self.lock:
            terminal_result = self.terminal.write(value)
            self.log_stream.write(value)
            self.log_stream.flush()
        return terminal_result

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log_stream.flush()

    def isatty(self):
        return self.terminal.isatty()


def create_observation_node(config: dict, *, max_observation_age_ms: float, max_camera_skew_ms: float):
    from rclpy.node import Node

    class ObservationNode(Node):
        def __init__(self):
            super().__init__("tau0vla_calibrated_client")
            from arm_control.msg import PosCmd
            from arx5_arm_msg.msg import RobotStatus
            from sensor_msgs.msg import CompressedImage

            self._lock = threading.Lock()
            self._latest = {}
            self._height_samples = collections.deque(maxlen=2000)
            self._gripper_samples = collections.deque(maxlen=4000)
            self._message_type = RobotStatus
            self._left_publisher = None
            self._right_publisher = None
            self._max_age_ns = int(max_observation_age_ms * 1e6)
            self._max_camera_skew_ns = int(max_camera_skew_ms * 1e6)
            cameras = {
                "head": config["camera_config"]["img_head_topic"],
                "left_wrist": config["camera_config"]["img_left_topic"],
                "right_wrist": config["camera_config"]["img_right_topic"],
            }
            for name, topic in cameras.items():
                self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda message, key=name: self._receive(f"camera:{key}", message),
                    10,
                )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["follow_arm_left_feedback_topic"],
                lambda message: self._receive_arm("left", message),
                20,
            )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["follow_arm_right_feedback_topic"],
                lambda message: self._receive_arm("right", message),
                20,
            )
            self.create_subscription(
                PosCmd,
                config["robot_base_config"]["robot_base_topic"],
                self._receive_height,
                10,
            )

        def _receive(self, key: str, message) -> None:
            with self._lock:
                self._latest[key] = (message, time.monotonic_ns())

        def _receive_arm(self, side: str, message) -> None:
            now = time.monotonic_ns()
            with self._lock:
                self._latest[f"arm:{side}"] = (message, now)
                left = self._latest.get("arm:left")
                right = self._latest.get("arm:right")
                if left is not None and right is not None:
                    self._gripper_samples.append(
                        (
                            time.monotonic(),
                            float(left[0].joint_pos[6]),
                            float(right[0].joint_pos[6]),
                        )
                    )

        def _receive_height(self, message) -> None:
            with self._lock:
                self._height_samples.append((time.monotonic(), float(message.height)))

        @staticmethod
        def _stamp_ns(message) -> int:
            stamp = message.header.stamp
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def snapshot(self) -> Observation:
            keys = [
                "camera:head",
                "camera:left_wrist",
                "camera:right_wrist",
                "arm:left",
                "arm:right",
            ]
            now = time.monotonic_ns()
            with self._lock:
                missing = [key for key in keys if key not in self._latest]
                if missing:
                    raise ProtocolError(f"missing ROS observations: {missing}")
                latest = {key: self._latest[key] for key in keys}
            stale = [key for key, (_, received) in latest.items() if now - received > self._max_age_ns]
            if stale:
                raise ProtocolError(f"stale ROS observations: {stale}")
            stamps = [self._stamp_ns(latest[f"camera:{name}"][0]) for name in ("head", "left_wrist", "right_wrist")]
            if max(stamps) - min(stamps) > self._max_camera_skew_ns:
                raise ProtocolError("camera timestamp skew exceeds configured limit")
            left_message = latest["arm:left"][0]
            right_message = latest["arm:right"][0]
            left_joint = np.asarray(left_message.joint_pos, dtype=np.float32)
            right_joint = np.asarray(right_message.joint_pos, dtype=np.float32)
            left_eef = np.asarray(left_message.end_pos, dtype=np.float32)
            right_eef = np.asarray(right_message.end_pos, dtype=np.float32)
            qpos = np.concatenate((left_joint[:7], right_joint[:7]))
            eef = np.concatenate((left_eef[:6], left_joint[6:7], right_eef[:6], right_joint[6:7]))
            if qpos.shape != (ACTION_DIM,) or eef.shape != (ACTION_DIM,):
                raise ProtocolError("ROS joint/EEF feedback has an invalid shape")
            if not np.isfinite(qpos).all() or not np.isfinite(eef).all():
                raise ProtocolError("ROS joint/EEF feedback contains NaN or Inf")
            images = {
                name: bytes(latest[f"camera:{name}"][0].data)
                for name in ("head", "left_wrist", "right_wrist")
            }
            if any(not value for value in images.values()):
                raise ProtocolError("ROS camera message contains an empty JPEG")
            return Observation(qpos=qpos, eef=eef, images=images, sample_monotonic_ns=now)

        def height_samples(self):
            with self._lock:
                return tuple(self._height_samples)

        def open_samples(self, seconds: float) -> np.ndarray:
            deadline = time.monotonic() + seconds
            start = time.monotonic()
            while time.monotonic() < deadline:
                time.sleep(0.02)
            with self._lock:
                rows = [row[1:] for row in self._gripper_samples if row[0] >= start]
            return np.asarray(rows, dtype=np.float64)

        def current_qpos(self) -> np.ndarray:
            with self._lock:
                left = self._latest.get("arm:left")
                right = self._latest.get("arm:right")
                if left is None or right is None:
                    raise ProtocolError("joint feedback is unavailable")
                values = np.concatenate((
                    np.asarray(left[0].joint_pos, dtype=np.float32)[:7],
                    np.asarray(right[0].joint_pos, dtype=np.float32)[:7],
                ))
            if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
                raise ProtocolError("joint feedback is not a finite 14-vector")
            return values

        def enable_publishers(self) -> None:
            if self._left_publisher is None:
                self._left_publisher = self.create_publisher(
                    self._message_type, config["arm_config"]["follow_arm_left_cmd_topic"], 10
                )
                self._right_publisher = self.create_publisher(
                    self._message_type, config["arm_config"]["follow_arm_right_cmd_topic"], 10
                )

        def publish(self, action: np.ndarray) -> None:
            if self._left_publisher is None or self._right_publisher is None:
                raise RuntimeError("action publishers are disabled")
            values = np.asarray(action, dtype=np.float32)
            if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
                raise ProtocolError("refusing malformed calibrated action at ROS boundary")
            left = self._message_type()
            right = self._message_type()
            left.joint_pos[:7] = values[:7].astype(float).tolist()
            right.joint_pos[:7] = values[7:].astype(float).tolist()
            self._left_publisher.publish(left)
            self._right_publisher.publish(right)

    return ObservationNode()


def return_to_initial_pose(node, target, trace, abort: threading.Event, args) -> dict[str, float]:
    start = node.current_qpos()
    trajectory = return_trajectory(
        start,
        target,
        rate_hz=FPS,
        minimum_duration_s=args.return_duration_s,
        max_arm_step=args.return_arm_step,
        max_gripper_step=args.return_gripper_step,
    )
    period = 1.0 / FPS
    deadline = time.monotonic()
    for index, command in enumerate(trajectory):
        if abort.is_set():
            raise RuntimeError("return interrupted; publication stopped")
        node.publish(command)
        feedback = node.current_qpos()
        trace.return_tick(
            monotonic_ns=time.monotonic_ns(),
            return_step=index,
            command=command,
            feedback=feedback,
        )
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))
    hold_deadline = time.monotonic() + 1.0
    while time.monotonic() < hold_deadline:
        if abort.is_set():
            raise RuntimeError("return interrupted while holding target")
        node.publish(target)
        time.sleep(period)
    samples = []
    verify_deadline = time.monotonic() + 2.0
    while time.monotonic() < verify_deadline:
        samples.append(node.current_qpos())
        time.sleep(period)
    values = np.asarray(samples, dtype=np.float32)
    arm_error = float(np.max(np.abs(values[-1, ARM_INDICES] - target[ARM_INDICES])))
    gripper_error = float(np.max(np.abs(values[-1, GRIPPER_INDICES] - target[GRIPPER_INDICES])))
    spread = float(np.max(np.ptp(values, axis=0)))
    detail = {
        "arm_error_max": arm_error,
        "gripper_error_max": gripper_error,
        "feedback_spread_max": spread,
        "trajectory_steps": int(len(trajectory)),
    }
    if arm_error > .05 or gripper_error > .1 or spread > .01:
        trace.return_result(status="failed", target=target, detail=detail)
        raise RuntimeError(f"return-to-initial verification failed: {detail}")
    trace.return_result(status="complete", target=target, detail=detail)
    return detail


def spin_node(node, stop: threading.Event) -> None:
    import rclpy

    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=.01)


def wait_observation(node, timeout: float) -> Observation:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            return node.snapshot()
        except ProtocolError as error:
            last = error
            time.sleep(.05)
    raise RuntimeError(f"sensor preflight timed out: {last}")


def verify_height(node, expected: float, tolerance: float, window: float, timeout: float) -> None:
    from rclpy.parameter_client import AsyncParameterClient

    client = AsyncParameterClient(node, "/lift")
    if not client.wait_for_services(timeout_sec=5.0):
        raise RuntimeError("/lift parameter service is unavailable")
    future = client.get_parameters(["fixed_height"])
    deadline = time.monotonic() + 5.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(.02)
    if not future.done() or future.result() is None:
        raise RuntimeError("timed out reading /lift fixed_height")
    fixed = float(future.result().values[0].double_value)
    if not np.isclose(fixed, expected, atol=1e-6):
        raise RuntimeError(f"/lift fixed_height={fixed}, expected {expected}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples = node.height_samples()
        if is_safe_and_stable(samples, float("inf"), tolerance, window):
            print(f"Height verified: fixed_height={fixed:.6f}, stable_feedback={samples[-1][1]:.6f}")
            return
        time.sleep(.05)
    raise RuntimeError("height feedback did not become stable")


def benchmark(client, node, request_id: int, warmup: int, samples: int):
    latencies = []
    for index in range(warmup + samples):
        request_id += 1
        result = client.infer(wait_observation(node, 5.0), request_id)
        if index >= warmup:
            latencies.append(result.round_trip_ms)
        print(
            f"Benchmark {index+1}/{warmup+samples}: request={request_id}, "
            f"RTT={result.round_trip_ms:.1f} ms, inference={result.inference_ms:.1f} ms"
        )
    return request_id, latencies


def run(args) -> None:
    import rclpy

    calibration = load_artifact(args.calibration_file)
    hostname, domain, boot_id, controllers = current_robot_identity()
    validate_artifact(
        calibration,
        hostname=hostname,
        ros_domain_id=domain,
        boot_id=boot_id,
        controller_identity=controllers,
        max_age_s=args.calibration_max_age_s,
    )
    if args.execute:
        require_unconsumed(args.calibration_file)

    setup_loader(ROOT)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rclpy.init()
    node = create_observation_node(
        config,
        max_observation_age_ms=args.max_observation_age_ms,
        max_camera_skew_ms=args.max_camera_skew_ms,
    )
    policy_stop = threading.Event()
    spin_stop = threading.Event()
    return_abort = threading.Event()
    spin = threading.Thread(target=spin_node, args=(node, spin_stop), daemon=True)
    spin.start()

    def handle_interrupt(*_):
        if policy_stop.is_set():
            return_abort.set()
            print("Second Ctrl-C: return motion aborted; no more commands will be published.")
        else:
            policy_stop.set()
            print("Ctrl-C: policy publication stopping; return confirmation will follow.")

    signal.signal(signal.SIGINT, handle_interrupt)
    executor = ThreadPoolExecutor(max_workers=1)
    pending: Future[CalibratedActionChunk] | None = None
    trace = TraceWriter(args.trace_path)
    session_id = ""
    try:
        verify_height(
            node,
            args.expected_height,
            args.height_stability_tolerance,
            args.height_stability_window,
            args.height_timeout,
        )
        wait_observation(node, args.sensor_timeout)
        assert_current_open(calibration, node.open_samples(2.0))
        client = CalibratedHttpClient(
            args.server_url,
            experiment=args.experiment,
            calibration=calibration,
            robot_id=hostname,
            request_timeout=args.request_timeout,
            max_response_age_ms=args.max_response_age_ms,
        )
        health = client.health()
        contract = client.policy_contract()
        session = client.create_session(args.task_instruction)
        session_id = session["session_id"]
        print(f"Server health: {json.dumps(health, sort_keys=True)}")
        print(f"Policy contract: {json.dumps(contract, sort_keys=True)}")
        print(f"Session: {session_id}")
        trace.metadata(
            session_id=session_id,
            server_url=args.server_url,
            task_instruction=args.task_instruction,
            model_id=health.get("model_id"),
            route=health.get("route"),
            checkpoint_sha256=health.get("checkpoint_sha256"),
            experiment=args.experiment,
            calibration_id=calibration.calibration_id,
            calibration=calibration.to_dict(),
            execute=args.execute,
            replan_steps=args.replan_steps,
            blend_steps=args.chunk_blend_steps,
            gripper_blend_steps=args.gripper_blend_steps,
            arm_ema_alpha=args.arm_ema_alpha,
            gripper_ema_alpha=args.gripper_ema_alpha,
        )

        request_id, latencies = benchmark(
            client, node, 0, args.benchmark_warmup, args.benchmark_requests
        )
        replan_steps, p99 = resolve_replan_steps(
            args.replan_steps, latencies, args.latency_margin_ms
        )
        print(f"Selected replan_steps={replan_steps}; measured p99 RTT={p99:.1f} ms")
        request_id += 1
        first = client.infer(wait_observation(node, 5.0), request_id)
        scheduler = CalibratedChunkScheduler(
            replan_steps,
            blend_steps=args.chunk_blend_steps,
            gripper_blend_steps=args.gripper_blend_steps,
        )
        ema = ActionEMA(args.arm_ema_alpha, args.gripper_ema_alpha)
        initial = node.snapshot().qpos
        trace.return_result(status="initial_pose", target=initial, detail={})
        ema.reset(initial)
        adoption = scheduler.adopt(first, initial=True, arrival_monotonic_ns=time.monotonic_ns())
        trace.adoption(first.request_id, adoption)

        if args.execute:
            confirmation = input(
                "Workspace clear, grippers calibrated, emergency stop reachable. "
                "Type EXECUTE CALIBRATED TAU0VLA to publish: "
            )
            if confirmation != "EXECUTE CALIBRATED TAU0VLA":
                raise RuntimeError("execution cancelled; no action publisher was created")
            mark_consumed(args.calibration_file, session_id=session_id)
            node.enable_publishers()
            print("EXECUTE mode: calibration consumed; publishing mapped finite 14D commands.")
        else:
            print("DRY-RUN mode: no action publisher was created; calibration remains reusable.")

        period = 1.0 / FPS
        deadline = time.monotonic()
        step = 0
        starved = False
        published = 0
        publish_started = time.monotonic()
        while rclpy.ok() and not policy_stop.is_set() and step < args.max_steps:
            now = time.monotonic()
            if now < deadline:
                time.sleep(min(deadline-now, .005))
                continue
            deadline += period
            if now - deadline > period:
                deadline = now + period
            if pending is not None and pending.done():
                result = pending.result()
                adoption = scheduler.adopt(result, arrival_monotonic_ns=time.monotonic_ns())
                trace.adoption(result.request_id, adoption)
                print(
                    f"Response request={result.request_id}, RTT={result.round_trip_ms:.1f} ms, "
                    f"inference={result.inference_ms:.1f} ms, "
                    f"skip arm/gripper={adoption.arm_skipped}/{adoption.gripper_skipped}, "
                    f"buffer={scheduler.remaining}"
                )
                pending = None
                starved = False
            if scheduler.should_request(pending is not None):
                request_id += 1
                pending = executor.submit(client.infer, node.snapshot(), request_id)
            try:
                scheduled = scheduler.next_action()
            except BufferError:
                if not starved:
                    print("BUFFER STARVED: publication paused until a fresh action chunk arrives.")
                    trace.starvation(time.monotonic_ns(), step)
                    starved = True
                continue
            command = ema.apply(scheduled.action)
            feedback = node.snapshot().qpos
            trace.tick(
                monotonic_ns=time.monotonic_ns(),
                control_step=step,
                scheduled=scheduled,
                command=command,
                feedback=feedback,
                execute=args.execute,
            )
            if args.execute:
                node.publish(command)
                published += 1
                elapsed = now - publish_started
                if elapsed >= 1.0:
                    print(f"Publish rate={published/elapsed:.2f} Hz, step={step}, buffer={scheduler.remaining}")
                    published = 0
                    publish_started = now
            elif step % FPS == 0:
                print(f"DRY-RUN step={step}, command={np.array2string(command, precision=4)}")
            step += 1
        if args.execute and not args.no_return_to_initial:
            policy_stop.set()
            if pending is not None:
                pending.cancel()
                pending = None
            print(f"Policy stopped at step={step}; model publication is paused.")
            confirmation = input(
                "Clear the return path and keep the emergency stop reachable. "
                "Type RETURN TO INITIAL POSE to move back: "
            )
            if confirmation != "RETURN TO INITIAL POSE":
                raise RuntimeError("return-to-initial cancelled; robot remains at current pose")
            detail = return_to_initial_pose(node, initial, trace, return_abort, args)
            print(f"RETURN TO INITIAL COMPLETE: {json.dumps(detail, sort_keys=True)}")
        print("CALIBRATED_ROLLOUT_COMPLETE")
    finally:
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        trace.close()
        spin_stop.set()
        node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2.0)
        if args.trace_path is not None and args.trace_path.exists():
            try:
                summary = analyze_trace(args.trace_path)
                summary_path = write_summary(args.trace_path, summary)
                plots = plot_trace(args.trace_path)
                print(f"Trace: {args.trace_path}")
                print(f"Trace summary: {json.dumps(summary, sort_keys=True)}")
                print(f"Trace summary file: {summary_path}")
                print("Trace plots: " + ", ".join(str(path) for path in plots))
            except Exception as error:
                print(f"Trace report generation failed: {error}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://192.168.50.2:8000")
    parser.add_argument("--experiment", choices=("joint-feedback", "joint-vr"), required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config.yaml")
    parser.add_argument("--expected-height", type=float, default=15.5)
    parser.add_argument("--height-stability-tolerance", type=float, default=.05)
    parser.add_argument("--height-stability-window", type=float, default=2.0)
    parser.add_argument("--height-timeout", type=float, default=15.0)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument("--max-observation-age-ms", type=float, default=100.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=50.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--max-response-age-ms", type=float, default=500.0)
    parser.add_argument("--latency-margin-ms", type=float, default=100.0)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-requests", type=int, default=30)
    parser.add_argument("--replan-steps", default="15")
    parser.add_argument("--chunk-blend-steps", type=int, default=6)
    parser.add_argument("--gripper-blend-steps", type=int, default=6)
    parser.add_argument("--arm-ema-alpha", type=float, default=.6)
    parser.add_argument("--gripper-ema-alpha", type=float, default=1.0)
    parser.add_argument("--calibration-max-age-s", type=float, default=900.0)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-return-to-initial", action="store_true")
    parser.add_argument("--return-duration-s", type=float, default=5.0)
    parser.add_argument("--return-arm-step", type=float, default=.02)
    parser.add_argument("--return-gripper-step", type=float, default=.05)
    args = parser.parse_args()
    if args.benchmark_warmup < 0 or args.benchmark_requests < 1:
        parser.error("benchmark counts must be non-negative with at least one request")
    if not 0 <= args.chunk_blend_steps < 30 or not 0 <= args.gripper_blend_steps < 30:
        parser.error("blend steps must be in [0,29]")
    if not 0 < args.arm_ema_alpha <= 1 or not 0 < args.gripper_ema_alpha <= 1:
        parser.error("EMA alpha must be in (0,1]")
    if args.return_duration_s < 5.0:
        parser.error("--return-duration-s may not be shorter than 5.0")
    if not 0 < args.return_arm_step <= .02 or not 0 < args.return_gripper_step <= .05:
        parser.error("return step limits exceed the reviewed envelope")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.log_path is None:
        run(parsed)
    else:
        parsed.log_path.parent.mkdir(parents=True, exist_ok=True)
        with parsed.log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
            lock = threading.Lock()
            stdout, stderr = sys.stdout, sys.stderr
            sys.stdout = TeeStream(stdout, log_stream, lock)
            sys.stderr = TeeStream(stderr, log_stream, lock)
            try:
                run(parsed)
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
