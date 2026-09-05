"""Request X5's native HOME, then gate lowering on fresh, stable joint feedback.

No CAN owner is started here. Command sources must already be stopped by shutdown.
The gripper is not commanded. On failure drivers remain alive; use physical E-stop
if necessary. This is not a collision-free planner or an emergency stop.
"""
import math
import time


HOME_TOPICS = (
    '/arx_joy',
    '/human_dagger/isolated/arm/left/joy_disabled',
    '/human_dagger/isolated/arm/right/joy_disabled',
)


def home_routes(subscribers):
    """Accept a complete legacy pair OR the exact isolated DAgger pair."""
    legacy, left, right = (subscribers.get(topic, []) for topic in HOME_TOPICS)
    if legacy and not left and not right:
        if len(legacy) == 2 and len(set(legacy)) == 2 and all(
                name.startswith('/') and '_UNKNOWN_' not in name for name in legacy):
            return {HOME_TOPICS[0]: tuple(sorted(legacy))}
    if not legacy and left == ['/human_dagger_arm_left'] and right == ['/human_dagger_arm_right']:
        return {HOME_TOPICS[1]: tuple(left), HOME_TOPICS[2]: tuple(right)}
    raise ValueError('Expected a complete legacy OR isolated DAgger HOME pair; '
                     f'found {subscribers}')


def at_home(position, velocity, target):
    return (
        len(position) == 7 and len(velocity) == 7 and len(target) == 6
        and all(math.isfinite(x) for x in (*position, *velocity, *target))
        and all(abs(position[i] - target[i]) <= 0.05 for i in range(6))
        and all(abs(velocity[i]) <= 0.1 for i in range(6))
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Check graph, parameters and feedback without creating command publishers')
    args = parser.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.qos import qos_profile_sensor_data
    from arx5_arm_msg.msg import RobotStatus
    from std_msgs.msg import Int32MultiArray

    rclpy.init()
    node = Node('shutdown_arm_home')
    samples = {}

    def spin_for(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)

    def fail(reason):
        raise RuntimeError(reason + '; platform NOT lowered; drivers remain alive. '
                           'Use physical emergency stop if motion is unsafe.')

    def endpoint_name(info):
        return info.node_namespace.rstrip('/') + '/' + info.node_name

    def read_routes():
        return home_routes({topic: [endpoint_name(info) for info in
                           node.get_subscriptions_info_by_topic(topic)]
                           for topic in HOME_TOPICS})

    try:
        # Allow DDS discovery to settle before checking command exclusivity.
        deadline = time.monotonic() + 10.0
        routes = None
        route_error = ''
        while time.monotonic() < deadline:
            spin_for(0.2)
            try:
                routes = read_routes()
                break
            except ValueError as exc:
                route_error = str(exc)
        if routes is None:
            fail(route_error)
        names = [name for group in routes.values() for name in group]
        arms = []
        for name in names:
            client = AsyncParameterClient(node, name)
            if not client.wait_for_services(timeout_sec=3.0):
                fail('Missing parameter service: ' + name)
            future = client.get_parameters(['arm_can_id', 'go_home_position'])
            rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
            if not future.done() or future.result() is None:
                fail('Could not read HOME parameters: ' + name)
            values = future.result().values
            can = values[0].string_value
            if HOME_TOPICS[0] not in routes:
                expected_can = {'/human_dagger_arm_left': 'can1',
                                '/human_dagger_arm_right': 'can3'}[name]
                if can != expected_can:
                    fail('Isolated arm CAN identity mismatch: ' + name)
            target = list(values[1].double_array_value)
            if len(target) != 6 or not all(math.isfinite(x) for x in target):
                fail('Invalid go_home_position: ' + name)
            namespace, _, short = name.rpartition('/')
            namespace = namespace or '/'
            publications = node.get_publisher_names_and_types_by_node(short, namespace)
            statuses = [topic for topic, types in publications
                        if 'arx5_arm_msg/msg/RobotStatus' in types]
            if len(statuses) != 1 or node.count_publishers(statuses[0]) != 1:
                fail('Ambiguous RobotStatus publisher: ' + name)
            subscriptions = node.get_subscriber_names_and_types_by_node(short, namespace)
            commands = [topic for topic, types in subscriptions
                        if any(t in types for t in ('arx5_arm_msg/msg/RobotCmd',
                               'arx5_arm_msg/msg/RobotStatus', 'arm_control/msg/PosCmd',
                               'arm_control/msg/JointControl'))]
            if len(commands) != 1:
                fail('Unsupported X5 command interface: ' + name)
            arms.append((name, can, target, statuses[0], commands[0]))
        if {arm[1] for arm in arms} != {'can1', 'can3'}:
            fail('HOME subscribers are not the can1/can3 pair')

        def receive(msg, name):
            samples[name] = (time.monotonic(), list(msg.joint_pos), list(msg.joint_vel))

        for name, _, _, topic, _ in arms:
            node.create_subscription(RobotStatus, topic,
                                     lambda msg, name=name: receive(msg, name),
                                     qos_profile_sensor_data)
        spin_for(1.0)

        def check_feedback():
            now = time.monotonic()
            for name, _, _, _, command in arms:
                if node.count_publishers(command):
                    fail('Command source still active on ' + command)
                if name not in samples or now - samples[name][0] > 0.25:
                    fail('Stale or missing arm feedback: ' + name)
                _, pos, vel = samples[name]
                if len(pos) != 7 or len(vel) != 7 or not all(
                        math.isfinite(x) for x in (*pos, *vel)):
                    fail('Invalid arm feedback: ' + name)

        def check_home_graph(own_publishers):
            try:
                current = read_routes()
            except ValueError as exc:
                fail(str(exc))
            if current != routes:
                fail('HOME subscribers changed')
            for topic in HOME_TOPICS:
                expected = 1 if own_publishers and topic in routes else 0
                if node.count_publishers(topic) != expected:
                    fail('Unexpected HOME publisher count on ' + topic)

        check_feedback()
        check_home_graph(False)
        if args.check_only:
            print(f'HOME_PRECHECK_OK: routes={routes}; no command publishers created.', flush=True)
            return
        pubs = {topic: node.create_publisher(Int32MultiArray, topic, 1) for topic in routes}
        spin_for(0.5)
        check_feedback()
        check_home_graph(True)
        if any(pub.get_subscription_count() != len(routes[topic]) for topic, pub in pubs.items()):
            fail('HOME subscriptions not matched during precheck')
        for name, can, target, _, _ in arms:
            print(f'HOME {name} ({can}): {target}; no extra gripper command', flush=True)
        # Native callback reads data[0] and data[1]; do not publish an empty array.
        # Check BOTH routes before sending either. Separate DDS sends are not atomic.
        for pub in pubs.values():
            pub.publish(Int32MultiArray(data=[0, 1]))
        started = time.monotonic()
        stable_since = None
        while time.monotonic() - started < 30.0:
            rclpy.spin_once(node, timeout_sec=0.02)
            check_feedback()
            check_home_graph(True)
            now = time.monotonic()
            # Require post-command feedback from BOTH arms, not cached precheck.
            reached = all(samples[name][0] > started and at_home(
                samples[name][1], samples[name][2], target)
                for name, _, target, _, _ in arms)
            stable_since = (stable_since if stable_since is not None else now) if reached else None
            if stable_since is not None and now - stable_since >= 2.0:
                print('ARMS_HOME_STABLE: both arms reached HOME; lowering may proceed.', flush=True)
                return
        fail('HOME timed out after 30 seconds')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
