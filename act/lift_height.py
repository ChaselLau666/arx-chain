"""Fixed lift height configuration shared by collection and replay."""

from __future__ import annotations

import time


def feedback_is_stable(samples, tolerance=0.01, window_seconds=2.0):
    if len(samples) < 2:
        return False
    all_samples = list(samples)
    newest_time = all_samples[-1][0]
    cutoff = newest_time - window_seconds
    start = 0
    for index, sample in enumerate(all_samples):
        if sample[0] <= cutoff:
            start = index
        else:
            break
    recent = all_samples[start:]
    if recent[-1][0] - recent[0][0] < window_seconds:
        return False
    values = [value for _, value in recent]
    return max(values) - min(values) <= tolerance


def configure_fixed_height(node, height, timeout=60.0, require_vr=True, should_stop=None,
                           tolerance=0.01):
    """Pin /lift to a fixed height and wait for its feedback to settle.

    Collection runs alongside VR and refuses to start until VR is publishing,
    so the body never briefly follows a raw VR height. Replay must not start
    VR at all - the patched body enforces fixed_height from its own control
    loop - so it passes require_vr=False.
    """
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.parameter_client import AsyncParameterClient

    target = -1.0 if height is None else float(height)
    if target != -1.0 and not 0.0 <= target <= 20.0:
        raise ValueError('height must be within [0, 20], or omitted to follow VR')

    client = AsyncParameterClient(node, '/lift')
    if not client.wait_for_services(timeout_sec=5.0):
        raise RuntimeError('/lift parameter service unavailable; body must already be running')
    future = client.set_parameters([Parameter('fixed_height', Parameter.Type.DOUBLE, target)])
    deadline = time.monotonic() + 5.0
    while not future.done() and rclpy.ok() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not future.done() or future.result() is None:
        raise RuntimeError('timed out setting /lift fixed_height')
    response = future.result()
    results = getattr(response, 'results', response)
    if not results or not all(result.successful for result in results):
        reason = '; '.join(result.reason for result in results if not result.successful)
        raise RuntimeError(f'failed to set /lift fixed_height: {reason}')
    print(f'/lift fixed_height set to {target:.6f}')

    def interrupted():
        return should_stop is not None and should_stop()

    def freeze_here():
        """Pin the lift where it is now, so an abort actually stops the platform.

        Simply raising would leave fixed_height at the requested value and the
        body control loop would keep driving towards it at 400 Hz.
        """
        if not node.height_feedback_deque:
            print('Interrupted before any lift feedback arrived; fixed_height left unchanged.')

            return
        current = float(node.height_feedback_deque[-1][1])
        frozen = client.set_parameters(
            [Parameter('fixed_height', Parameter.Type.DOUBLE, current)])
        freeze_deadline = time.monotonic() + 2.0
        while not frozen.done() and rclpy.ok() and time.monotonic() < freeze_deadline:
            time.sleep(0.02)
        print(f'Interrupted: /lift fixed_height frozen at {current:.6f}')

    if target < 0:
        return None
    if require_vr:
        vr_deadline = time.monotonic() + 10.0
        while rclpy.ok() and not node.controller_left_deque and time.monotonic() < vr_deadline:
            if interrupted():
                freeze_here()
                raise RuntimeError('interrupted while waiting for VR')
            time.sleep(0.05)
        if not node.controller_left_deque:
            raise RuntimeError('/ARX_VR_L unavailable; fixed height was not applied, refused')
    node.height_feedback_deque.clear()

    # Fail fast and loudly when no feedback arrives at all: that means the node
    # never subscribed to /body_information, not that the lift is still moving.
    first_deadline = time.monotonic() + 5.0
    while rclpy.ok() and not node.height_feedback_deque:
        if interrupted():
            freeze_here()
            raise RuntimeError('interrupted while waiting for the first lift sample')
        if time.monotonic() > first_deadline:
            raise RuntimeError(
                'no /body_information sample in 5s; the node is not subscribed to lift '
                'feedback (RosOperator needs args.height set before construction)')
        time.sleep(0.05)

    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        if interrupted():
            freeze_here()
            raise RuntimeError('interrupted while waiting for the lift to settle')
        if feedback_is_stable(node.height_feedback_deque, tolerance=tolerance):
            settled = node.height_feedback_deque[-1][1]
            print(f'Lift feedback settled at {settled:.6f} for command {target:.6f}')
            return settled
        time.sleep(0.1)

    # Report what the feedback actually did: a range wider than the tolerance
    # usually means this machine's encoder quantisation is coarser than the
    # default, not that the platform is still moving.
    values = [value for _, value in node.height_feedback_deque]
    observed = (max(values) - min(values)) if values else float('nan')
    raise RuntimeError(
        f'lift feedback did not settle within {timeout:.0f}s; refused. '
        f'observed range {observed:.6f} over {len(values)} samples '
        f'(min {min(values):.6f}, max {max(values):.6f}) vs tolerance {tolerance:.6f}; '
        f'raise --height-tolerance above the observed range if the platform is visibly still')
