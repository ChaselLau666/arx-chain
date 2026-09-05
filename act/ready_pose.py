"""Where the arms park between episodes, and how arriving there is judged.

No ROS imports: the decisions below are the part worth testing without hardware,
and collect.py supplies the messages and the feedback. The pose itself is
deliberately absent - it is passed to X5Controller as go_home_position by
tools/08_collect_ready_pose.sh and read back off the arms, so there is one copy
of the numbers rather than two to keep in step.
"""
from __future__ import annotations

import numpy as np

# The two node names 08_collect_ready_pose.sh remaps its X5Controllers to.
READY_ARM_NODES = ('/vr_arm_l', '/vr_arm_r')

# data[1] == 1 selects GO_HOME; data[0] is left at 0 because a 1 there selects
# gravity compensation instead. arxJoyCB indexes both without checking the
# length, so the array has to carry two elements.
GO_HOME_JOY = (0, 1)

# go_home_position covers six joints and not the gripper, so this is the one
# part of the ready pose that has to be named here rather than read back.
READY_GRIPPER = (-2.9717, -2.9675)

# X5Controller multiplies a VR gripper by this before use (X5Controller.cpp:165),
# so a command has to carry the inverse.
VR_GRIPPER_SCALE = -3.4 / 5


def vr_gripper_command(gripper: float) -> float:
    """The VR-units gripper value that lands on `gripper` after the controller."""
    return gripper / VR_GRIPPER_SCALE


def joints_are_still(samples, tolerance: float = 0.004,
                     settle_window: float = 0.4) -> bool:
    """Whether every joint held within `tolerance` across the whole window.

    `samples` is (timestamp, joints) oldest first. A window shorter than the one
    asked for is not still, only unmeasured: without that check the first two
    samples after a move starts are trivially identical and read as settled.
    """
    if len(samples) < 2:
        return False
    if samples[-1][0] - samples[0][0] < settle_window * 0.8:
        return False
    spread = np.ptp(np.array([joints for _, joints in samples]), axis=0).max()
    return bool(spread < tolerance)


def arms_have_arrived(current, target, arrival_tolerance: float = 0.05) -> bool:
    """Whether the arms stopped where they were asked to, not merely stopped.

    A target of None means go_home_position could not be read, so there is
    nothing to compare against and coming to rest is all that can be claimed.
    Callers say so rather than reporting a verified arrival.
    """
    if target is None:
        return True
    return bool(np.abs(np.asarray(current, dtype=float)
                       - np.asarray(target, dtype=float)).max() <= arrival_tolerance)
