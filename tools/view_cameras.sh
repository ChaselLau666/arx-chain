#!/usr/bin/env bash
# Three-way live camera viewer for the Human DAgger rig.
#
# Subscribes to the three compressed color streams and shows them side by side
# with a per-camera fps/staleness overlay, so a dead or stalled camera (e.g.
# camera_h's "Frames didn't arrived within 5 seconds") is visible at a glance.
#
# Run it in a terminal that can open windows (robot desktop or ToDesk), with
# the cameras already running (05_/10_ stack, or any realsense launch).
# View-only: subscribes, never publishes; safe next to a live dagger session.
# Quit with q or Esc in the window, or Ctrl-C in the terminal.
set -Eeuo pipefail

# Domain must come from the machine (/etc/environment), same rule as 05_/10_.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set (ark-1=62, ark-2=63); refusing to guess which robot to watch}"

ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}

set +u
source /opt/ros/jazzy/setup.bash
set -u

exec "$ACT_PYTHON" - "$@" << 'EOF'
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

CAMERAS = ("camera_h", "camera_l", "camera_r")
TOPIC_TEMPLATE = "/camera/{name}/color/image_rect_raw/compressed"
TILE_W, TILE_H = 640, 480
STALE_AFTER_S = 1.0


class Viewer(Node):
    def __init__(self) -> None:
        super().__init__("camera_viewer")
        self.frames = {}   # name -> (bgr image, wall time)
        self.counts = {name: 0 for name in CAMERAS}
        self.fps = {name: 0.0 for name in CAMERAS}
        self._fps_window_start = time.monotonic()
        for name in CAMERAS:
            self.create_subscription(
                CompressedImage,
                TOPIC_TEMPLATE.format(name=name),
                lambda msg, name=name: self._on_image(name, msg),
                qos_profile_sensor_data,
            )

    def _on_image(self, name: str, msg: CompressedImage) -> None:
        image = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            self.frames[name] = (image, time.monotonic())
            self.counts[name] += 1

    def refresh_fps(self) -> None:
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            for name in CAMERAS:
                self.fps[name] = self.counts[name] / elapsed
                self.counts[name] = 0
            self._fps_window_start = now


def tile_for(viewer: Viewer, name: str) -> np.ndarray:
    entry = viewer.frames.get(name)
    now = time.monotonic()
    if entry is None:
        tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        label, color = f"{name}: NO DATA", (0, 0, 255)
    else:
        image, stamp = entry
        tile = cv2.resize(image, (TILE_W, TILE_H))
        age = now - stamp
        if age > STALE_AFTER_S:
            tile = (tile // 2).astype(np.uint8)  # dim a frozen stream
            label, color = f"{name}: STALE {age:.1f}s", (0, 0, 255)
        else:
            label, color = f"{name}: {viewer.fps[name]:.1f}fps", (0, 255, 0)
    cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return tile


def main() -> int:
    rclpy.init()
    viewer = Viewer()
    signal.signal(signal.SIGINT, lambda *_: rclpy.try_shutdown())
    window = "Human DAgger cameras (q/Esc to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, TILE_W * 3 // 2, TILE_H // 2)
    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.02)
            viewer.refresh_fps()
            mosaic = cv2.hconcat([tile_for(viewer, name) for name in CAMERAS])
            cv2.imshow(window, mosaic)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.try_shutdown()
    return 0


sys.exit(main())
EOF
