#!/usr/bin/env python3
"""Offline check that host-side IK can stand in for the arm's built-in solver.

Runs Placo against a recorded episode, no hardware involved, and answers the
questions that decide whether moving IK to the host is viable:

  1. Does FK on the vendor URDF reproduce the eef the arm itself reported?
     This pins down the frame conventions: the arm reports position relative
     to its home end-effector pose (a constant offset), and orientation as
     extrinsic xyz Euler angles. The values found are compared against the
     constants act/x5_model.py carries, so drift between the two is caught.
  2. Does IK on that eef recover the recorded joint angles - converged, as a
     real-time node would (one solve per frame), and with the joint limits
     from x5_model enabled, which is how act/vr_ik_node.py runs?
  3. Do consecutive solutions stay continuous, or does the solver flip branch?
  4. How close does the trajectory pass to singularities?

Reproducing the vendor's own FK exactly is expected - the recorded eef is that
FK. What the run proves is that the URDF, frames, limits and solver are right;
it does not exercise unreachable or lagged targets, which only live teleop will.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'act'))
from x5_model import EE_FRAME, EULER, HOME_EE, JOINTS, VENDOR_URDF, kinematic_urdf, target_transform  # noqa: E402

ARM_SLICE = {'left': slice(0, 6), 'right': slice(7, 13)}
EULER_CONVENTIONS = ('xyz', 'XYZ', 'zyx', 'ZYX', 'zxy', 'ZXY')


def load_episode(path: Path, arm: str):
    with h5py.File(path, 'r') as root:
        qpos = root['/observations/qpos'][...]
        eef = root['/observations/eef'][...]
    s = ARM_SLICE[arm]
    return qpos[:, s], eef[:, s]


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


class Arm:
    def __init__(self, urdf: Path, placo):
        self.robot = placo.RobotWrapper(str(urdf), placo.Flags.ignore_collisions)

    def set_q(self, q):
        for name, value in zip(JOINTS, q):
            self.robot.set_joint(name, float(value))
        self.robot.update_kinematics()

    def q(self) -> np.ndarray:
        return np.array([self.robot.get_joint(n) for n in JOINTS])

    def T_ee(self) -> np.ndarray:
        return np.array(self.robot.get_T_world_frame(EE_FRAME))

    def jacobian(self) -> np.ndarray:
        J = np.array(self.robot.frame_jacobian(EE_FRAME, 'local_world_aligned'))
        return J[:, -6:] if J.shape[1] > 6 else J


def check_fk(arm: Arm, qpos, eef, stride=4) -> bool:
    """Rediscover the offset and Euler convention; flag if x5_model disagrees."""
    offsets, angles = [], {c: [] for c in EULER_CONVENTIONS}
    for k in range(0, len(qpos), stride):
        arm.set_q(qpos[k])
        T = arm.T_ee()
        offsets.append(T[:3, 3] - eef[k, :3])
        for conv in EULER_CONVENTIONS:
            angles[conv].append(rotation_angle_deg(T[:3, :3], R.from_euler(conv, eef[k, 3:6]).as_matrix()))
    offsets = np.array(offsets)
    found_offset, found_conv = offsets.mean(0), min(angles, key=lambda c: np.mean(angles[c]))
    print('  FK vs recorded eef')
    print(f'    position offset FK - eef = {found_offset.round(4)}  (std {offsets.std(0).max() * 1000:.3f} mm; ~0 means constant)')
    for conv in sorted(angles, key=lambda c: np.mean(angles[c]))[:3]:
        print(f'    orientation as {conv:4s}: mean {np.mean(angles[conv]):7.3f} deg  max {np.max(angles[conv]):7.3f} deg')
    ok = np.allclose(found_offset, HOME_EE, atol=1e-3) and found_conv == EULER
    print(f'    x5_model.py says HOME_EE={HOME_EE} EULER={EULER!r}: ' + ('matches' if ok else 'DOES NOT MATCH - update x5_model.py'))
    return ok


def moving_span(qpos):
    moving = np.flatnonzero(np.abs(np.diff(qpos, axis=0)).sum(1) > 1e-9)
    return int(moving.min()), int(moving.max()) + 2


def check_ik(urdf: Path, placo, qpos, eef, iters: int, dt: float, limits: bool, label: str):
    arm = Arm(urdf, placo)
    solver = placo.KinematicsSolver(arm.robot)
    solver.mask_fbase(True)
    solver.enable_joint_limits(limits)
    task = solver.add_frame_task(EE_FRAME, np.eye(4))
    task.configure(EE_FRAME, 'soft', 1.0, 1.0)
    solver.add_regularization_task(1e-5)
    solver.dt = dt

    s0, s1 = moving_span(qpos)
    arm.set_q(qpos[s0])
    prev, solutions, pos_err, jumps = arm.q(), [], [], []
    t0 = time.perf_counter()
    for k in range(s0, s1):
        T = target_transform(eef[k, :3], eef[k, 3:6])
        task.T_world_frame = T
        for _ in range(iters):
            solver.solve(True)
            arm.robot.update_kinematics()
        q = arm.q()
        solutions.append(q)
        pos_err.append(np.linalg.norm(arm.T_ee()[:3, 3] - T[:3, 3]))
        jumps.append(np.abs(q - prev).max())
        prev = q
    per_frame_ms = (time.perf_counter() - t0) / (s1 - s0) * 1000
    err = np.degrees(np.array(solutions) - qpos[s0:s1])
    recorded_jump = np.degrees(np.abs(np.diff(qpos[s0:s1], axis=0)).max())
    print(f'  IK, {label}  ({per_frame_ms:.2f} ms/frame over frames {s0}..{s1})')
    print(f'    end-effector position residual: mean {np.mean(pos_err) * 1000:.2f} mm  max {np.max(pos_err) * 1000:.2f} mm')
    print('    joint error vs recorded qpos, RMSE deg: ' + ' '.join(f'j{i + 1}:{np.sqrt((err[:, i] ** 2).mean()):.2f}' for i in range(6)))
    print(f'    largest joint error {np.abs(err).max():.2f} deg; largest frame-to-frame jump {np.degrees(max(jumps)):.2f} deg (recording itself: {recorded_jump:.2f} deg)')


def check_singularity(arm: Arm, qpos, stride=2):
    conds, manip = [], []
    for k in range(0, len(qpos), stride):
        arm.set_q(qpos[k])
        J = arm.jacobian()
        conds.append(np.linalg.cond(J))
        manip.append(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
    conds = np.array(conds)
    print('  Singularity margin along the trajectory')
    print(f'    Jacobian condition number: median {np.median(conds):.1f}  P95 {np.percentile(conds, 95):.1f}  max {conds.max():.1f}  (frames >100: {(conds > 100).mean() * 100:.1f}%)')
    print(f'    manipulability: median {np.median(manip):.4f}  min {np.min(manip):.4f} at frame {int(np.argmin(manip)) * stride}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--arm', choices=['left', 'right', 'both'], default='right')
    parser.add_argument('--urdf', type=Path, default=VENDOR_URDF, help='vendor URDF; a mesh-free copy with x5_model limits is cached beside the datasets')
    parser.add_argument('--frame-rate', type=float, default=60.0)
    args = parser.parse_args()

    try:
        import placo
    except ImportError:
        sys.exit('placo is not installed in this environment: pip install placo')

    urdf = kinematic_urdf(args.episode.resolve().parent.parent / 'x5_kin.urdf', src=args.urdf)
    arms = ['left', 'right'] if args.arm == 'both' else [args.arm]
    for arm_name in arms:
        qpos, eef = load_episode(args.episode, arm_name)
        span = np.degrees(qpos.max(0) - qpos.min(0))
        print(f'\n=== {args.episode.name}  {arm_name} arm  (joint travel deg: ' + ' '.join(f'{v:.0f}' for v in span) + ') ===')
        if span.max() < 5:
            print('  arm did not move in this episode; skipping')
            continue
        arm = Arm(urdf, placo)
        check_fk(arm, qpos, eef)
        rt = 1.0 / args.frame_rate
        check_ik(urdf, placo, qpos, eef, iters=20, dt=0.05, limits=False, label='converged (20 iterations/frame)')
        check_ik(urdf, placo, qpos, eef, iters=1, dt=rt, limits=False, label=f'real-time (1 solve/frame at {args.frame_rate:.0f} Hz)')
        check_ik(urdf, placo, qpos, eef, iters=1, dt=rt, limits=True, label='real-time with x5_model joint limits enabled, as vr_ik_node runs')
        check_singularity(arm, qpos)


if __name__ == '__main__':
    main()
