"""ROS-level regression tests for the VR filter startup safety gate."""
from __future__ import annotations

import os
import sys
import time
import unittest
import uuid
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'act'))

import rclpy
from arm_control.msg import PosCmd
from arx5_arm_msg.msg import RobotStatus
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from vr_pose_filter import build_node


class VrPoseFilterStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        suffix = f'{os.getpid()}_{uuid.uuid4().hex[:8]}'
        self.raw_topic = f'/codex_test_vr_raw_{suffix}'
        self.filtered_topic = f'/codex_test_vr_filtered_{suffix}'
        self.status_topic = f'/codex_test_arm_status_{suffix}'
        args = Namespace(
            node_name=f'codex_test_vr_filter_{suffix}',
            in_topic=self.raw_topic,
            out_topic=self.filtered_topic,
            arm_status_topic=self.status_topic,
            tau=0.01,
            dt=1 / 60.0,
            home_mute=0.5,
            rebase=True,
            report_period=3600.0,
            arm_status_timeout=0.5,
            max_position_jump=0.08,
            max_angle_jump_deg=45.0,
        )
        _, self.filter_node = build_node(args)
        self.driver = Node(f'codex_test_vr_driver_{suffix}')
        self.raw_pub = self.driver.create_publisher(PosCmd, self.raw_topic, 10)
        self.status_pub = self.driver.create_publisher(RobotStatus, self.status_topic, 10)
        self.outputs = []
        self.driver.create_subscription(
            PosCmd, self.filtered_topic, self.outputs.append, 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.filter_node)
        self.executor.add_node(self.driver)
        self.assertTrue(self.spin_until(
            lambda: self.raw_pub.get_subscription_count() == 1 and
                    self.status_pub.get_subscription_count() == 1,
            timeout=2.0,
        ), 'ROS publishers did not discover the filter subscriptions')

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.executor.remove_node(self.filter_node)
        self.driver.destroy_node()
        self.filter_node.destroy_node()
        self.executor.shutdown()

    def spin_until(self, condition, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            self.executor.spin_once(timeout_sec=0.02)
        return condition()

    def publish_arm_status(self, end_pos):
        arm = RobotStatus()
        arm.end_pos = end_pos
        self.status_pub.publish(arm)
        self.executor.spin_once(timeout_sec=0.1)

    def publish_raw_and_wait(self, raw):
        before = len(self.outputs)
        self.raw_pub.publish(raw)
        self.assertTrue(
            self.spin_until(lambda: len(self.outputs) > before, timeout=0.5),
            'the filter did not publish a safe output',
        )
        return self.outputs[-1]

    def test_zero_vr_pose_is_held_until_fresh_arm_feedback_then_rebased(self):
        raw = PosCmd()
        self.raw_pub.publish(raw)
        self.spin_until(lambda: bool(self.outputs), timeout=0.3)
        self.assertEqual(
            self.outputs, [],
            'the filter forwarded an absolute zero target before it knew the arm pose',
        )

        arm_end_pos = [0.31, -0.12, 0.44, 0.10, -0.20, 0.30]
        self.publish_arm_status(arm_end_pos)
        out = self.publish_raw_and_wait(raw)
        for actual, expected in zip(
                [out.x, out.y, out.z, out.roll, out.pitch, out.yaw], arm_end_pos):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_non_finite_pose_is_never_forwarded(self):
        self.publish_arm_status([0.31, -0.12, 0.44, 0.0, 0.0, 0.0])
        self.publish_raw_and_wait(PosCmd())
        before = len(self.outputs)
        invalid = PosCmd()
        invalid.x = float('nan')
        self.raw_pub.publish(invalid)
        self.spin_until(lambda: len(self.outputs) > before, timeout=0.2)
        self.assertEqual(len(self.outputs), before)

    def test_large_raw_jump_rebases_to_current_arm_pose(self):
        self.publish_arm_status([0.31, -0.12, 0.44, 0.0, 0.0, 0.0])
        self.publish_raw_and_wait(PosCmd())

        current_arm_pose = [0.32, -0.11, 0.43, 0.0, 0.0, 0.0]
        self.publish_arm_status(current_arm_pose)
        jumped = PosCmd()
        jumped.x = 0.20
        out = self.publish_raw_and_wait(jumped)
        for actual, expected in zip(
                [out.x, out.y, out.z, out.roll, out.pitch, out.yaw], current_arm_pose):
            self.assertAlmostEqual(actual, expected, places=6)


if __name__ == '__main__':
    unittest.main()
