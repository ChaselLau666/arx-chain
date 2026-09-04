# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import yaml
import h5py
import argparse
import signal
import time

import rclpy

import threading

import numpy as np

np.set_printoptions(linewidth=200)

from functools import partial

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader
from lift_height import configure_fixed_height
from replay_support import (episode_start_pose, resolve_replay_height, tracking_report,
                            smooth_causal)


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None


def load_hdf5(dataset_path):
    dataset_path = Path.joinpath(ROOT, dataset_path)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at: {dataset_path}")

    try:
        with h5py.File(dataset_path, 'r') as root:
            qposes = root.get('/observations/qpos')
            eefs = root.get('/observations/eef')
            actions = root.get('/action')
            actions_eefs = root.get('/action_eef')
            action_base = root.get('/action_base')
            action_velocity = root.get('/action_velocity')

            # 确保所有所需的数据集都存在
            if any(item is None for item in [qposes, eefs, actions, actions_eefs, action_base, action_velocity]):
                missing_datasets = [name for name, item in zip(
                    ['/observations/qpos', '/observations/eef', '/action', '/action_eef',
                     '/action_base', '/action_velocity'],
                    [qposes, eefs, actions, actions_eefs, action_base, action_velocity]
                ) if item is None]

                raise ValueError(f"Missing datasets in HDF5 file: {', '.join(missing_datasets)}")

            recorded_height = root.attrs.get('height_command')

            return (qposes[()], eefs[()], actions[()], actions_eefs[()],
                    action_base[()], action_velocity[()],
                    None if recorded_height is None else float(recorded_height))
    except Exception as e:
        raise RuntimeError(f"Error occurred while loading the HDF5 file: {e}")


def robot_action(ros_operator, args, action, action_base, actions_velocity):
    gripper_idx = [6, 13]

    left_action = action[:gripper_idx[0] + 1]  # 取8维度
    right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]  # action[7:14]

    print(f'{left_action=}')

    ros_operator.follow_arm_publish(left_action, right_action)  # follow_arm_publish_continuous_thread

    if args.use_base:
        ros_operator.set_robot_base_target(np.concatenate([action_base, actions_velocity]))


def current_qpos(ros_operator):
    """Latest 14-D arm feedback, or None until both arms have reported."""
    left = ros_operator.follow_left_arm_deque
    right = ros_operator.follow_right_arm_deque
    if not left or not right:
        return None

    return np.concatenate([list(left[-1].joint_pos)[:7], list(right[-1].joint_pos)[:7]])


def init_robot(ros_operator, use_base, start_pose):
    left_start, right_start = start_pose

    ros_operator.follow_arm_publish_continuous(left_start, right_start)

    if use_base:
        input("Enter any key to continue :")

        ros_operator.start_base_control_thread()
        ros_operator.follow_arm_publish_continuous(left_start, right_start)


stop_requested = threading.Event()


def signal_handler(signal, frame, ros_operator):
    """Ask the replay loop to stop; never command the lift or the base here.

    Publishing /body_control from an interrupt would drive the platform to the
    height carried in that message. Stopping simply stops sending arm targets,
    which leaves both arms holding their last commanded pose.
    """
    if stop_requested.is_set():
        print('\nSecond interrupt; exiting immediately.')
        sys.exit(1)

    print('\nCaught Ctrl+C / SIGINT; stopping after the current frame.')
    stop_requested.set()
    ros_operator.request_arm_publish_stop()


def summarise_tracking(args, times, command, actual):
    """Report how closely the arms followed the replayed trajectory."""
    report = tracking_report(command, actual)
    elapsed = times[-1] - times[0]
    achieved = (len(times) - 1) / elapsed if elapsed > 0 else float('nan')

    print('\n--- tracking report ---')
    print(f'frames compared : {report["frames"]}')
    print(f'replay rate     : {achieved:.2f} Hz measured vs {args.frame_rate} Hz requested')
    print(f'feedback lag    : {report["lag_frames"]} frames '
          f'({report["lag_frames"] / args.frame_rate * 1000:.1f} ms)')
    print(f'arm joints      : rmse {report["arm_rmse"]:.4f} rad, max {report["arm_max"]:.4f} rad')
    print(f'  left  rmse {np.round(report["arm_per_joint_rmse"][:6], 4)}')
    print(f'  right rmse {np.round(report["arm_per_joint_rmse"][6:], 4)}')
    print(f'grippers        : feedback sits {report["gripper_turns"]} whole turn(s) '
          f'from the commanded frame (removed before comparing)')
    print(f'  residual      rmse {np.round(report["gripper_rmse"], 4)} rad, '
          f'max {np.round(report["gripper_max"], 4)}')

    os.makedirs(os.path.dirname(os.path.abspath(args.record_actual)) or '.', exist_ok=True)
    np.savez(args.record_actual, t=times, command=command, actual=actual,
             frame_rate=args.frame_rate, episode=str(args.episode_path))
    print(f'raw log saved to {args.record_actual}')


def main(args):
    armed = bool(args.execute)
    if armed:
        confirmation = input('Type EXECUTE REPLAY to publish arm targets: ')
        if confirmation != 'EXECUTE REPLAY':
            raise RuntimeError('replay cancelled; nothing was published')
    else:
        print('DRY-RUN: no arm publisher is created; pass --execute to move the arms.')

    setup_loader(ROOT)

    (qpoes, eefs, actions, actions_eefs, action_base, actions_velocity,
     recorded_height) = load_hdf5(args.episode_path)

    # Resolve the height before the node is built. RosOperator only subscribes
    # to /body_information when args.height is set, and configure_fixed_height
    # waits on exactly that feedback.
    args.height = resolve_replay_height(recorded_height, args.height)

    rclpy.init()

    config = load_yaml(args.data)
    ros_operator = RosOperator(args, config, in_collect=False)

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()

    signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))

    if armed:
        try:
            settled = configure_fixed_height(ros_operator, args.height, require_vr=False,
                                             should_stop=stop_requested.is_set,
                                             tolerance=args.height_tolerance)
        except Exception:
            ros_operator.destroy_node()
            rclpy.shutdown()
            spin_thread.join(timeout=2.0)
            raise
        print(f'Fixed lift ready: command={args.height:.6f}, feedback={settled:.6f}')
    else:
        print(f'DRY-RUN: would set /lift fixed_height to {args.height:.6f}')

    if args.states_replay:
        replay_actions = actions
    else:
        replay_actions = qpoes

    if args.smooth_tau > 0:
        raw = replay_actions
        replay_actions = smooth_causal(replay_actions, args.smooth_tau, 1.0 / args.frame_rate)
        moved = np.abs(replay_actions - raw).max()
        print(f'Smoothed arm joints with a causal one-pole filter: tau={args.smooth_tau:.3f}s, '
              f'alpha={1 - np.exp(-1 / (args.frame_rate * args.smooth_tau)):.4f}; '
              f'largest change {moved:.4f} rad. Grippers were left untouched.')

    start_pose = episode_start_pose(replay_actions)
    if armed:
        init_robot(ros_operator, args.use_base, start_pose)
    else:
        print('DRY-RUN start pose:')
        print(f'  left : {np.round(start_pose[0], 4)}')
        print(f'  right: {np.round(start_pose[1], 4)}')

    rate = Rate(args.frame_rate)
    total = len(replay_actions)
    log_t, log_cmd, log_actual = [], [], []
    for idx in range(total):
        if stop_requested.is_set():
            print(f'Replay stopped by operator at frame {idx}/{total}.')
            break

        if armed:
            robot_action(ros_operator, args, replay_actions[idx],
                         action_base[idx], actions_velocity[idx])
            if args.record_actual:
                measured = current_qpos(ros_operator)
                if measured is not None:
                    log_t.append(time.monotonic())
                    log_cmd.append(np.asarray(replay_actions[idx], dtype=float))
                    log_actual.append(measured)
        elif idx % args.frame_rate == 0:
            print(f'DRY-RUN frame {idx}/{total}: {np.round(replay_actions[idx], 4)}')

        rate.sleep()
    else:
        print(f'Replay finished: {total} frames.')

    if armed and args.record_actual and len(log_cmd) >= 2:
        summarise_tracking(args, np.array(log_t), np.array(log_cmd), np.array(log_actual))

    ros_operator.base_enable = False

    ros_operator.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--episode_path', type=str, help='episode_path', required=True)
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')
    parser.add_argument('--data', type=str, default=Path.joinpath(ROOT, 'data/config.yaml'), help='config file')

    parser.add_argument('--use_base', action='store_true', help='use base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')

    parser.add_argument('--states_replay', action='store_true', help='use qpos replay')
    parser.add_argument('--height', type=float, default=None,
                        help='fixed lift command in [0, 20]; defaults to the recorded height_command')
    parser.add_argument('--execute', action='store_true',
                        help='publish arm targets and set the lift; default is dry-run')
    parser.add_argument('--smooth-tau', type=float, default=0.0,
                        help='time constant of a causal one-pole filter applied to the arm '
                             'joints before replay, in seconds; 0 disables it. Matches the '
                             'teleop-app filter, and delays the trajectory by roughly tau')
    parser.add_argument('--record-actual', type=str,
                        default=str(Path.home() / 'replay_logs' / 'replay_actual.npz'),
                        help='write per-frame commanded/actual qpos here and print a tracking '
                             'report; pass an empty string to disable')
    parser.add_argument('--height-tolerance', type=float, default=0.06,
                        help='lift feedback is settled when its 2s range stays within this; '
                             'must exceed the encoder quantisation of the machine in use')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
