"""Pure helpers shared by the replay entry point and offline tests."""

from __future__ import annotations

import numpy as np


def split_bimanual_frame(frame, name='frame'):
    """Split one [left 7, right 7] frame after validating its shape."""
    frame = np.asarray(frame, dtype=float)
    if frame.shape != (14,):
        raise ValueError(f'expected a 14-D {name}, got shape {frame.shape}')

    return frame[:7].tolist(), frame[7:14].tolist()


def episode_start_pose(trajectory):
    """Split the first recorded frame into per-arm 7-DoF start targets.

    The start pose comes from the episode itself rather than a hardcoded
    constant, so every commanded value - the gripper included - stays in the
    same units the arm controller reports back.
    """
    if len(trajectory) == 0:
        raise ValueError('episode contains no frames; nothing to replay')

    return split_bimanual_frame(trajectory[0], name='recorded frame')


def select_replay_trajectory(mode, qpos, action, action_poscmd, states_replay=False):
    """Select a trajectory without silently crossing action representations.

    Legacy episodes have no ``/action_poscmd`` and remain valid for joint
    replay. PosCmd replay is deliberately strict because measured EEF is not a
    substitute for the Cartesian command that originally entered ``vr_slave``.
    """
    if mode == 'joint':
        return np.asarray(action if states_replay else qpos)
    if mode == 'poscmd':
        if action_poscmd is None:
            raise ValueError(
                'episode has no /action_poscmd; recollect it with a filtered collection '
                'launcher before using --replay-mode poscmd')
        trajectory = np.asarray(action_poscmd)
        if trajectory.ndim != 2 or trajectory.shape[1] != 14:
            raise ValueError(
                f'expected /action_poscmd shape (T, 14), got {trajectory.shape}')
        return trajectory

    raise ValueError(f'unknown replay mode: {mode}')


def build_poscmd_start_ramp(current_eef, target_poscmd,
                            max_xyz_step=0.005, max_angle_step=0.03):
    """Interpolate measured EEF pose to the first recorded PosCmd.

    ``current_eef`` contains two measured xyz/rpy blocks (12 values), while the
    command contains two xyz/rpy/gripper blocks (14 values). Euler deltas take
    the equivalent shortest wrapped direction. The recorded gripper command is
    held throughout because measured gripper feedback uses motor units, not the
    VR command units stored in PosCmd.
    """
    current = np.asarray(current_eef, dtype=float)
    target = np.asarray(target_poscmd, dtype=float)
    if current.shape != (12,):
        raise ValueError(f'expected 12-D measured EEF, got shape {current.shape}')
    if target.shape != (14,):
        raise ValueError(f'expected 14-D PosCmd target, got shape {target.shape}')
    if max_xyz_step <= 0 or max_angle_step <= 0:
        raise ValueError('PosCmd ramp step limits must be positive')

    starts, deltas = [], []
    steps = 1
    for arm in range(2):
        measured = current[arm * 6:(arm + 1) * 6]
        command = target[arm * 7:arm * 7 + 6]
        delta = command - measured
        delta[3:6] = np.arctan2(np.sin(delta[3:6]), np.cos(delta[3:6]))
        starts.append(measured)
        deltas.append(delta)
        steps = max(
            steps,
            int(np.ceil(np.max(np.abs(delta[:3])) / max_xyz_step)),
            int(np.ceil(np.max(np.abs(delta[3:6])) / max_angle_step)),
        )

    ramp = np.repeat(target[None, :], steps, axis=0)
    fractions = np.arange(1, steps + 1, dtype=float) / steps
    for arm in range(2):
        pose_columns = slice(arm * 7, arm * 7 + 6)
        ramp[:, pose_columns] = starts[arm] + fractions[:, None] * deltas[arm]

    return ramp


def resolve_replay_height(recorded, requested):
    """Pick the lift command for a replay, refusing ambiguous combinations.

    The episode records the height it was collected at. A mismatch between
    that and an explicit --height is a physical hazard, not a preference, so
    it is refused rather than silently resolved in either direction.
    """
    if requested is None:
        if recorded is None:
            raise ValueError('episode has no height_command; pass --height explicitly')
        return float(recorded)

    requested = float(requested)
    if recorded is not None and abs(float(recorded) - requested) > 1e-6:
        raise ValueError(
            f'--height {requested} conflicts with recorded height_command {float(recorded)}')

    return requested


def best_lag(command, actual, max_lag=30):
    """Frame lag that best explains the difference between command and actual.

    The arm answers a position target a few cycles late, so a raw sample-wise
    error mixes real tracking error with that constant delay. Shifting the
    feedback back by the lag that minimises RMSE separates the two.
    """
    command = np.asarray(command, dtype=float)
    actual = np.asarray(actual, dtype=float)
    best, best_err = 0, float('inf')
    for lag in range(0, min(max_lag, len(command) - 1) + 1):
        error = actual[lag:] - command[:len(command) - lag]
        rmse = float(np.sqrt(np.mean(error ** 2)))
        if rmse < best_err:
            best, best_err = lag, rmse
    return best, best_err


GRIPPER_INDICES = (6, 13)
ARM_INDICES = tuple(i for i in range(14) if i not in GRIPPER_INDICES)


TWO_PI = 2.0 * np.pi


def tracking_report(command, actual, max_lag=30):
    """Compare what replay commanded against what the arms actually did.

    Arm joints and grippers are reported separately because they are not on the
    same scale. Arm feedback shares the commanded frame, so its error is
    meaningful directly. Gripper feedback does not: the motor's position count
    has no absolute reference, so each time the arm stack starts it can land a
    whole revolution away from the frame the recorded commands were written in.
    Measured on hardware that offset is 2*pi to within 0.1%, so it is snapped to
    the nearest whole turn rather than fitted - a residual that stays large
    after removing whole turns is what indicates the gripper failed to follow.
    """
    command = np.asarray(command, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if command.ndim != 2 or command.shape != actual.shape:
        raise ValueError(f'need two matching (T, N) arrays, got {command.shape} and {actual.shape}')
    if len(command) < 2:
        raise ValueError('need at least two frames to compare')
    if command.shape[1] < 14:
        raise ValueError(f'need at least 14 columns, got {command.shape[1]}')

    arm_c, arm_a = command[:, ARM_INDICES], actual[:, ARM_INDICES]
    lag, _ = best_lag(arm_c, arm_a, max_lag)
    keep = len(command) - lag
    arm_error = arm_a[lag:] - arm_c[:keep]

    grip_raw = actual[lag:, GRIPPER_INDICES] - command[:keep, GRIPPER_INDICES]
    turns = np.round(np.median(grip_raw, axis=0) / TWO_PI)
    grip_error = grip_raw - turns * TWO_PI

    return {
        'frames': int(len(command)),
        'lag_frames': int(lag),
        'arm_rmse': float(np.sqrt(np.mean(arm_error ** 2))),
        'arm_max': float(np.max(np.abs(arm_error))),
        'arm_per_joint_rmse': np.sqrt(np.mean(arm_error ** 2, axis=0)),
        'arm_per_joint_max': np.max(np.abs(arm_error), axis=0),
        'gripper_turns': turns.astype(int),
        'gripper_rmse': np.sqrt(np.mean(grip_error ** 2, axis=0)),
        'gripper_max': np.max(np.abs(grip_error), axis=0),
    }


def ema_alpha(tau, dt):
    """One-pole low-pass coefficient, as teleop-app derives it.

    Writing alpha in terms of a time constant rather than fixing it directly
    keeps the filter's behaviour the same when the frame rate changes.
    """
    tau = max(float(tau), 1e-6)

    return 1.0 - np.exp(-float(dt) / tau)


def smooth_causal(trajectory, tau, dt, columns=ARM_INDICES):
    """Low-pass the listed columns with the same one-pole filter teleop-app uses.

    This is the causal form: each output depends only on samples up to that
    point, so it also delays the signal by roughly tau. That is the price of a
    filter which could equally run online, and replay pays it on purpose - the
    aim is to behave the way inference will have to.

    Grippers are excluded by default. Their trajectory is two steps, and
    rounding those off would move the moment the grasp closes.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError(f'need a (T, N) array, got shape {trajectory.shape}')
    if tau <= 0:
        return trajectory.copy()

    alpha = ema_alpha(tau, dt)
    out = trajectory.copy()
    columns = [c for c in columns if c < trajectory.shape[1]]
    for i in range(1, len(out)):
        out[i, columns] = out[i - 1, columns] + alpha * (trajectory[i, columns] - out[i - 1, columns])

    return out
