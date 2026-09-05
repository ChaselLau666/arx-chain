"""Opt-in ROS regression: synthetic feedback only, never robot commands.

Run with HUMAN_DAGGER_ROS_TEST=1 ROS_DOMAIN_ID=197 ROS_LOCALHOST_ONLY=1.
"""
import ast
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any
import unittest
import uuid


@unittest.skipUnless(os.environ.get("HUMAN_DAGGER_ROS_TEST") == "1", "requires isolated ROS test environment")
class HoldExecutorTests(unittest.TestCase):
    def test_hold_wait_does_not_block_feedback_and_stale_feedback_still_fails(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
        from rclpy.experimental.events_executor import EventsExecutor
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        # Exercise the actual production executor wiring and HOLD callback,
        # replacing the hardware node with a synthetic feedback subscriber.
        source = Path(__file__).resolve().parents[1] / "act/human_dagger.py"
        tree = ast.parse(source.read_text())
        runtime = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_ros_control")
        node_class = next(n for n in runtime.body if isinstance(n, ast.ClassDef) and n.name == "HumanDaggerRosNode")
        callback = next(n for n in node_class.body if isinstance(n, ast.FunctionDef) and n.name == "_request_hold")
        ack_helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_feedback_pair_acknowledges_hold")
        scope = {"Any": Any, "Mapping": dict, "monotonic_ns": time.monotonic_ns}
        exec(compile(ast.Module(body=[callback, ack_helper], type_ignores=[]), str(source), "exec"), scope)
        suffix = uuid.uuid4().hex
        service_name = "/hold_regression_" + suffix
        feedback_name = "/feedback_regression_" + suffix
        gaps = []

        class FeedbackNode(Node):
            _request_hold = scope["_request_hold"]

            def __init__(self):
                super().__init__("feedback_test_" + suffix)
                self.external_hold_ack = threading.Event()
                self.external_hold_requested = threading.Event()
                self.external_hold_request_ns = 0
                self.external_hold_published_ns = 0
                self.service_callback_group = MutuallyExclusiveCallbackGroup()
                self.last_feedback_ns = 0
                self.create_subscription(String, feedback_name, self.receive_feedback, 10)
                self.publisher = self.create_publisher(String, feedback_name, 10)
                self.timer = self.create_timer(0.02, lambda: self.publisher.publish(String()))

            def receive_feedback(self, _message):
                now = time.monotonic_ns()
                if self.last_feedback_ns:
                    gaps.append((now - self.last_feedback_ns) / 1e9)
                self.last_feedback_ns = now
                sample = (SimpleNamespace(timestamp_ns=now),)
                if scope["_feedback_pair_acknowledges_hold"](
                    {"feedback": {"left": sample, "right": sample}},
                    self.external_hold_published_ns, True,
                ):
                    self.external_hold_ack.set()

        statements = runtime.body
        start = next(i for i, n in enumerate(statements) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "HumanDaggerRosNode")
        end = next(i for i, n in enumerate(statements) if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "frame_period" for t in n.targets))
        rclpy.init()
        scope.update(Node=Node, Trigger=Trigger, HumanDaggerRosNode=FeedbackNode,
                     MultiThreadedExecutor=MultiThreadedExecutor, SingleThreadedExecutor=SingleThreadedExecutor,
                     EventsExecutor=EventsExecutor,
                     threading=threading, os=os, ros_config={}, topic=lambda *args: service_name)
        exec(compile(ast.Module(body=statements[start:end], type_ignores=[]), str(source), "exec"), scope)
        node = scope["node"]
        stop = threading.Event()

        def fake_control_loop():
            while not stop.wait(0.005):
                if node.external_hold_requested.is_set() and node.external_hold_published_ns == 0:
                    # Simulate a HOLD publication; do not publish RobotCmd.
                    node.external_hold_published_ns = time.monotonic_ns()

        control_thread = threading.Thread(target=fake_control_loop)
        control_thread.start()
        client_node = Node("hold_client_" + suffix)
        client = client_node.create_client(Trigger, service_name)
        try:
            self.assertTrue(client.wait_for_service(timeout_sec=5))
            time.sleep(0.2)
            for _ in range(2):
                future = client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(client_node, future, timeout_sec=3)
                self.assertTrue(future.done())
                self.assertTrue(future.result().success)
            self.assertTrue(gaps)
            self.assertLess(max(gaps), 0.5, "HOLD stalled feedback callbacks")
            node.destroy_timer(node.timer)
            time.sleep(0.2)
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(client_node, future, timeout_sec=3)
            self.assertTrue(future.done())
            self.assertFalse(future.result().success, "stale feedback must not acknowledge HOLD")
        finally:
            stop.set()
            control_thread.join(timeout=1)
            for name in ("hold_executor", "executor"):
                scope[name].shutdown(timeout_sec=2)
            for name in ("hold_thread", "spin_thread"):
                scope[name].join(timeout=2)
            scope["hold_node"].destroy_node()
            node.destroy_node()
            client_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
