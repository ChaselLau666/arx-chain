"""What host-side kinematics needs to know about the ARX X5, in one place.

Every value here was established against recorded data by
tools/validate_ik_offline.py rather than read off a datasheet; see that script
and its commit message before changing any of them.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

VENDOR_URDF = Path('/home/arx/LIFT/ARX_X5/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/x5.urdf')
JOINTS = tuple(f'joint{i}' for i in range(1, 7))
EE_FRAME = 'link6'

# The arm reports and accepts end-effector positions relative to where link6
# sits at the home configuration, not relative to the base link. Placo works in
# the base frame, so this is added to every target and subtracted from every
# FK result. Identical for both arms; per-frame spread over an episode is 0.
HOME_EE = np.array([0.0952, 0.0010, 0.1565])

# The home configuration the vendor app's "reset" returns to: every joint at
# zero. Recorded episodes start within a few milliradians of this.
HOME_Q = np.zeros(6)

# Extrinsic xyz, i.e. scipy Rotation.from_euler('xyz', [roll, pitch, yaw]).
# The "# eef:ZXY" comment in ros_operator.py is wrong; xyz matched to 0.000 deg.
EULER = 'xyz'

# vr_slave mode scales the VR gripper (0..5) by this before setCatch, and
# remote_slave passes joint_pos[6] to setCatch unscaled, so a host-side node
# has to apply it itself to behave the same way.
GRIPPER_SCALE = -3.4 / 5.0

# The vendor URDF carries +/-10 rad on every joint, which is no limit at all.
# These start from the arx5-sdk X5.urdf values for joints 2-6; joint 1 is not
# given there and +/-3 rad is wider than anything an episode has used. The
# arm's own controller still enforces its real limits, so a solver limit that
# is looser than reality is harmless while one that is tighter makes the QP
# refuse poses the arm can reach. That is why joints 2 and 3 get -0.1 rather
# than the sdk's 0.0: the home pose sits exactly there, and recordings dip to
# -0.01, so 0.0 would pin the solver against the bound at rest.
JOINT_LIMITS = {
    'joint1': (-3.0, 3.0),
    'joint2': (-0.1, 3.14),
    'joint3': (-0.1, 3.14),
    'joint4': (-1.5708, 1.5708),
    'joint5': (-1.67, 1.67),
    'joint6': (-1.57, 1.57),
}


def kinematic_urdf(dst: Path, src: Path = VENDOR_URDF, limits: dict | None = JOINT_LIMITS) -> Path:
    """Write a copy of the vendor URDF that Placo can load.

    The original references meshes under a package that does not exist on
    the robot, and FK/IK need none of them, so visual and collision blocks are
    dropped. Joint limits are replaced when given, since the vendor's are
    placeholders.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    for link in root.iter('link'):
        for tag in ('visual', 'collision'):
            for element in list(link.findall(tag)):
                link.remove(element)
    if limits:
        for joint in root.iter('joint'):
            bounds = limits.get(joint.get('name'))
            limit = joint.find('limit')
            if bounds and limit is not None:
                limit.set('lower', repr(float(bounds[0])))
                limit.set('upper', repr(float(bounds[1])))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding='unicode')
    return dst


def target_transform(xyz, rpy) -> np.ndarray:
    """Base-frame 4x4 for a pose given the way the arm gives and takes them."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_euler(EULER, rpy).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=float) + HOME_EE
    return T
