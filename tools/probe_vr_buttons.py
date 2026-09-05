#!/usr/bin/env python3
"""Print VR controller button/trigger changes, one line per change.

The VR serial node publishes at roughly 500 Hz, so ``ros2 topic echo`` scrolls
far too fast to read.  This watches only the fields that could carry a button
state and prints a line when one of them actually changes, which makes both the
button-to-field mapping and its level-versus-pulse behaviour obvious by eye.

Read-only: it subscribes and prints, and never publishes a command.

    ros2 run serial_port serial_port_node --ros-args \
      -r /ARX_VR_L:=/vr_probe/left -r /ARX_VR_R:=/vr_probe/right

    python3 tools/probe_vr_buttons.py
"""

from __future__ import annotations

import argparse
import time
import json
import select
import sys
from pathlib import Path
from datetime import datetime

import rclpy
from arm_control.msg import PosCmd
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# mode1/mode2 are the only discrete channels in PosCmd; gripper is the side
# trigger and is watched here to confirm its travel range at the same time.
INT_FIELDS = ("mode1", "mode2")
FLOAT_FIELDS = ("gripper", "height", "chx", "chy", "chz", "head_pit", "head_yaw")


class SessionLog:
    def __init__(self, stream):
        self.stream = stream
        self.label = 'baseline'
        self.write('label', label=self.label)

    def write(self, kind, **fields):
        self.stream.write(json.dumps(dict(kind=kind, wall_ns=time.time_ns(),
                                        monotonic_ns=time.monotonic_ns(), **fields),
                                     ensure_ascii=False,
                                     default=lambda value: value.tolist() if hasattr(value, 'tolist')
                                     else list(value)) + '\n')

    def set_label(self, text):
        self.label = text.strip() or 'baseline'
        self.write('label', label=self.label)
        self.stream.flush()

    def sample(self, hand, message):
        fields = {name: getattr(message, name) for name in message.get_fields_and_field_types()}
        self.write('sample', label=self.label, hand=hand, fields=fields)


class Probe(Node):
    def __init__(self, epsilon: float, log: SessionLog) -> None:
        super().__init__("vr_button_probe")
        self.epsilon = epsilon
        self.log = log
        self.started = time.monotonic()
        self.last: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        self.counts: dict[str, int] = {"left": 0, "right": 0}
        for side, topic in (("left", "/vr_probe/left"), ("right", "/vr_probe/right")):
            self.create_subscription(
                PosCmd,
                topic,
                lambda message, hand=side: self.on_message(hand, message),
                qos_profile_sensor_data,
            )
        print("Watching. Press one button at a time, ~2s apart.")
        print("Integer fields change exactly; floats need to move by "
              f"more than {epsilon} to count.\n")
        print(f"{'time':>9}  {'hand':<5}  change")
        print("-" * 60)

    def on_message(self, hand: str, message: PosCmd) -> None:
        self.counts[hand] += 1
        self.log.sample(hand, message)
        previous = self.last[hand]
        changes = []
        for name in INT_FIELDS:
            value = float(int(getattr(message, name, 0)))
            if name not in previous:
                previous[name] = value
                continue
            if value != previous[name]:
                changes.append(f"{name}: {previous[name]:.0f} -> {value:.0f}")
                previous[name] = value
        for name in FLOAT_FIELDS:
            value = float(getattr(message, name, 0.0))
            if name not in previous:
                previous[name] = value
                continue
            if abs(value - previous[name]) > self.epsilon:
                changes.append(f"{name}: {previous[name]:+.3f} -> {value:+.3f}")
                previous[name] = value
        if changes:
            stamp = time.monotonic() - self.started
            for change in changes:
                print(f"{stamp:9.3f}  [{self.log.label}] {hand:<5}  {change}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None, help='new JSONL file; never overwrite')
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="minimum float movement to report; raises to filter tracker noise",
    )
    args = parser.parse_args()
    if not 0 < args.epsilon < float('inf'):
        parser.error('--epsilon must be finite and positive')
    output = args.output or (Path(__file__).resolve().parents[1] / 'logs' /
                            ('vr_buttons_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jsonl'))
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open('x', encoding='utf-8')
    log = SessionLog(stream)

    rclpy.init()
    probe = Probe(args.epsilon, log)
    print(f'LOG: {output}', flush=True)
    print('输入按键名称并回车（如 right_A），再按住/松开该键，重复3次。', flush=True)
    print('空行回到 baseline；输入 /quit 或 Ctrl+C 结束。全部消息都会记录，不自动猜测映射。', flush=True)
    last_report = time.monotonic()
    last_counts = dict(probe.counts)
    try:
        while rclpy.ok():
            rclpy.spin_once(probe, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line or line.strip() == '/quit':
                    break
                log.set_label(line)
                print(f'LABEL = {log.label}; 现在操作这个按键。', flush=True)
            now = time.monotonic()
            if now - last_report >= 5:
                rates = {s: round((probe.counts[s] - last_counts[s]) / (now - last_report), 1)
                         for s in probe.counts}
                print(f'[接收频率 Hz] {rates}; label={log.label}', flush=True)
                stream.flush()
                last_counts = dict(probe.counts)
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        log.write('end', label=log.label, counts=probe.counts)
        stream.close()
        print("\n" + "-" * 60)
        print(f"messages seen: left={probe.counts['left']} right={probe.counts['right']}")
        for hand in ("left", "right"):
            if probe.last[hand]:
                summary = "  ".join(
                    f"{name}={probe.last[hand][name]:.2f}"
                    for name in INT_FIELDS + FLOAT_FIELDS
                    if name in probe.last[hand]
                )
                print(f"final {hand:<5} {summary}")
        if probe.counts["left"] == 0 and probe.counts["right"] == 0:
            print("\nNo messages arrived. Check that serial_port_node is running,")
            print("that it was remapped to /vr_probe/*, and that ROS_DOMAIN_ID matches.")
        if rclpy.ok():
            probe.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
