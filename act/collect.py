"""ARX LIFT2s HDF5 v2 collector with explicit keyboard review."""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset_v2 import EpisodeValidationError, EpisodeWriter, next_episode_path
from pipeline_contract import FPS
from utils.setup_loader import setup_loader


class TerminalKeys:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("collection requires an interactive terminal")
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, traceback):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> str | None:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1).lower() if readable else None

    def wait_for(self, allowed: set[str]) -> str:
        while True:
            key = self.poll()
            if key in allowed:
                return key
            time.sleep(0.02)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def spin_node(node):
    import rclpy

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)


def preflight_height(node, expected_height: float | None, tolerance: float) -> dict:
    status = node.height_status()
    print(
        f"Body height: current={status['current_height']:.6f}, "
        f"commanded={status['commanded_height']:.6f}, locked={status['locked']}"
    )
    if not status["locked"]:
        raise RuntimeError("height is not locked; use lift_height.py lock before collection")
    if expected_height is not None and abs(status["current_height"] - expected_height) > tolerance:
        raise RuntimeError(
            f"current feedback differs from expected {expected_height:.6f} by more than {tolerance:.3f}"
        )
    return status


def record_episode(args, node, keys: TerminalKeys, height_status: dict):
    writer = EpisodeWriter(
        dataset_dir=args.datasets,
        task_name=args.task,
        task_instruction=args.task_instruction,
        expected_height=height_status["current_height"],
        commanded_height=height_status["commanded_height"],
        repo_root=REPO_ROOT,
    )
    print(f"Recording to pending file {writer.path.name}; press E to end")
    period_ns = int(1_000_000_000 / FPS)
    deadline_ns = time.monotonic_ns()
    attempted = 0
    dropped = 0
    previous = None
    transitions = 0
    try:
        # Keep one slot for the terminal state that supplies action(t)=state(t+1).
        while transitions < args.max_timesteps - 1:
            if keys.poll() == "e":
                break
            now_ns = time.monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep(min((deadline_ns - now_ns) / 1e9, 0.005))
                continue
            if now_ns - deadline_ns >= period_ns:
                skipped = (now_ns - deadline_ns) // period_ns
                attempted += int(skipped)
                dropped += int(skipped)
                deadline_ns += int(skipped) * period_ns
            attempted += 1
            try:
                current = node.snapshot()
            except RuntimeError as error:
                dropped += 1
                print(f"\nDropped tick: {error}")
            else:
                if previous is not None:
                    writer.append_transition(previous, current)
                    transitions += 1
                    print(f"\rTransitions: {transitions}", end="", flush=True)
                previous = current
            deadline_ns += period_ns

        if previous is None:
            raise EpisodeValidationError("no valid samples captured")
        terminal_deadline = time.monotonic() + 0.5
        terminal = None
        while terminal is None and time.monotonic() < terminal_deadline:
            try:
                terminal = node.snapshot()
            except RuntimeError:
                time.sleep(0.01)
        if terminal is None:
            raise EpisodeValidationError("could not capture terminal qpos for state(t+1)")
        writer.append_transition(previous, terminal)
        attempted += 1
        writer.set_sampling_stats(attempted, dropped)
        print()
        summary = writer.finalize()
        return writer, summary, None
    except Exception as error:
        print()
        try:
            writer.set_sampling_stats(attempted, dropped)
            if transitions:
                writer.finalize()
        except Exception:
            pass
        return writer, None, error


def run(args) -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", "62")
    setup_loader(ROOT)
    import rclpy
    from collection_node import create_collection_node

    rclpy.init()
    node = create_collection_node(load_yaml(args.config))
    spin_thread = threading.Thread(target=spin_node, args=(node,), daemon=True)
    spin_thread.start()
    try:
        height_status = preflight_height(node, args.expected_height, args.height_tolerance)
        Path(args.datasets).mkdir(parents=True, exist_ok=True)
        print("Ready: R record | Q quit")
        with TerminalKeys() as keys:
            while rclpy.ok():
                key = keys.wait_for({"r", "q"})
                if key == "q":
                    return
                writer, summary, error = record_episode(args, node, keys, height_status)
                print(f"Review PASS: {summary}" if error is None else f"Review FAILED: {error}")
                print("S save | D discard | Q quit (failed episodes cannot be saved)")
                while True:
                    decision = keys.wait_for({"s", "d", "q"})
                    if decision == "s" and error is not None:
                        print("Save refused: episode validation failed; press D")
                        continue
                    if decision == "s":
                        final_path = next_episode_path(args.datasets, args.episode_idx)
                        writer.save_as(final_path)
                        print(f"Saved {final_path}")
                    else:
                        writer.discard()
                        print("Pending episode discarded")
                    if decision == "q":
                        return
                    break
                print("Ready: R record | Q quit")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    parser.add_argument("--episode_idx", type=int, default=-1)
    parser.add_argument("--max_timesteps", type=int, default=900)
    parser.add_argument("--frame_rate", type=int, choices=[FPS], default=FPS)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config.yaml")
    parser.add_argument("--task", required=True, help="stable task slug")
    parser.add_argument("--task_instruction", required=True, help="natural-language task instruction")
    parser.add_argument("--expected_height", type=float, default=None)
    parser.add_argument("--height_tolerance", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
