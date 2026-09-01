"""Wait until body feedback is low and stable; this node never publishes commands."""

from __future__ import annotations

import argparse
import collections
import os
import time

import rclpy
from rclpy.node import Node
from arm_control.msg import PosCmd
from safe_height import is_safe_and_stable


class HeightMonitor(Node):
    def __init__(self):
        super().__init__('safe_shutdown_height_monitor')
        self.samples = collections.deque(maxlen=2000)
        self.create_subscription(PosCmd, '/body_information', self.callback, 10)

    def callback(self, message):
        value = float(message.height)
        self.samples.append((time.monotonic(), value))
        print(f'height={value:.6f}', flush=True)


def main(args):
    os.environ.setdefault('ROS_DOMAIN_ID', '62')
    rclpy.init()
    node = HeightMonitor()
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if is_safe_and_stable(node.samples, args.safe_max, args.tolerance, args.window):
                print(f'SAFE_LOW_STABLE height={node.samples[-1][1]:.6f}')
                return 0
        print('REFUSED: height did not become low and stable before timeout')
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--safe-max', type=float, default=1.0)
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=90.0)
    raise SystemExit(main(parser.parse_args()))
