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

import rclpy

import threading

import numpy as np

np.set_printoptions(linewidth=200)

from functools import partial

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader
from lift_height import configure_fixed_height
from replay_support import episode_start_pose, resolve_replay_height


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

    start_pose = episode_start_pose(replay_actions)
    if armed:
        init_robot(ros_operator, args.use_base, start_pose)
    else:
        print('DRY-RUN start pose:')
        print(f'  left : {np.round(start_pose[0], 4)}')
        print(f'  right: {np.round(start_pose[1], 4)}')

    rate = Rate(args.frame_rate)
    total = len(replay_actions)
    for idx in range(total):
        if stop_requested.is_set():
            print(f'Replay stopped by operator at frame {idx}/{total}.')
            break

        if armed:
            robot_action(ros_operator, args, replay_actions[idx],
                         action_base[idx], actions_velocity[idx])
        elif idx % args.frame_rate == 0:
            print(f'DRY-RUN frame {idx}/{total}: {np.round(replay_actions[idx], 4)}')

        rate.sleep()
    else:
        print(f'Replay finished: {total} frames.')

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
    parser.add_argument('--height-tolerance', type=float, default=0.06,
                        help='lift feedback is settled when its 2s range stays within this; '
                             'must exceed the encoder quantisation of the machine in use')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
