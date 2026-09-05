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

import time
import h5py
import argparse
import rclpy
import cv2
import yaml
import threading
import pyttsx3

import numpy as np

from copy import deepcopy

from utils.ros_operator import Rate, RosOperator
from utils.setup_loader import setup_loader
from collection_paths import normalize_task_name, task_dataset_dir
from collection_ui import TerminalKeyReader, prompt_episode_decision, prompt_start_decision
from lift_height import configure_fixed_height

np.set_printoptions(linewidth=200)

voice_engine = pyttsx3.init()
voice_engine.setProperty('voice', 'en')
voice_engine.setProperty('rate', 120)  # 设置语速

voice_lock = threading.Lock()




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


def voice_process(voice_engine, line):
    with voice_lock:
        voice_engine.say(line)
        voice_engine.runAndWait()
        print(line)

    return


def collect_information(args, ros_operator, voice_engine, key_reader):
    timesteps = []
    actions = []
    actions_eef = []
    action_bases = []
    action_velocities = []
    count = 0
    rate = Rate(args.frame_rate)

    # 初始化机器人基础位置
    # ros_operator.init_robot_base_pose()

    gripper_idx = [6, 13]
    # gripper_close = 3
    gripper_close = -2.1

    print('RECORDING: press [e] to end and review this episode.')
    while rclpy.ok():
        key = key_reader.poll_key()
        if key == 'e':
            print('\n[e] received; recording stopped for review.')
            break
        if key is not None:
            print(f"\nIgnored key '{key}' while recording; press [e] to end.")

        obs_dict = ros_operator.get_observation(ts=count)
        action_dict = ros_operator.get_action()

        # 同步帧检测
        if obs_dict is None or action_dict is None:
            print("Synchronization frame")
            rate.sleep()

            continue

        # 获取动作和观察值
        action = deepcopy(obs_dict['qpos'])
        action_eef = deepcopy(obs_dict['eef'])
        action_base = obs_dict['robot_base']
        action_velocity = obs_dict['base_velocity']

        # 夹爪动作处理
        for idx in gripper_idx:
            action[idx] = 0 if action[idx] > gripper_close else action[idx]
        action_eef[6] = 0 if action_eef[6] > gripper_close else action_eef[6]
        action_eef[13] = 0 if action_eef[13] > gripper_close else action_eef[13]

        # 收集数据
        timesteps.append(obs_dict)
        actions.append(action)
        actions_eef.append(action_eef)
        action_bases.append(action_base)
        action_velocities.append(action_velocity)

        count += 1
        print(f"Frame data: {count}")

        if not rclpy.ok():
            exit(-1)

        rate.sleep()

    print(f"\nlen(timesteps): {len(timesteps)}")
    print(f"len(actions)  : {len(actions)}")

    return timesteps, actions, actions_eef, action_bases, action_velocities


def compress_and_pad_images(data_dict, camera_names, use_depth, quality=50):
    def compress_and_pad(key_prefix):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        all_encoded = []

        for cam in camera_names:
            key = f'/observations/{key_prefix}/{cam}'
            encoded_list = []
            for img in data_dict[key]:
                _, enc = cv2.imencode('.jpg', img, encode_param)
                encoded_list.append(enc)
                all_encoded.append(len(enc))
            data_dict[key] = encoded_list

            # Empty whenever the run has no cameras at all, where max() raises.
        padded_size = max(all_encoded) if all_encoded else 0

        for cam in camera_names:
            key = f'/observations/{key_prefix}/{cam}'
            padded = [np.pad(enc, (0, padded_size - len(enc)), constant_values=0) for enc in data_dict[key]]
            data_dict[key] = padded

        return padded_size

    # RGB
    padded_size = compress_and_pad('images')

    # Depth
    padded_size_depth = compress_and_pad('images_depth') if use_depth else 0

    return padded_size, padded_size_depth


def create_and_write_hdf5(args, data_dict, dataset_path, data_size, padded_size, padded_size_depth):
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['task'] = str(args.task)
        if not args.camera_names:
            # No images in this episode, so it can verify the recording path
            # but cannot train a policy. Marked so a loader refuses it rather
            # than silently reading an empty images group.
            root.attrs['no_images'] = True
        if args.height is not None:
            root.attrs['height_command'] = float(args.height)

        obs_dict = root.create_group('observations')
        image = obs_dict.create_group('images')
        if args.use_depth_image:
            depth = obs_dict.create_group('images_depth')

        for cam_name in args.camera_names:
            img_shape = (data_size, padded_size)
            img_chunk = (1, padded_size)
            if args.use_depth_image:
                depth_shape = (data_size, padded_size_depth)
                depth_chunk = (1, padded_size_depth)

            image.create_dataset(cam_name, img_shape, 'uint8', chunks=img_chunk)
            if args.use_depth_image:
                depth.create_dataset(cam_name, depth_shape, 'uint8', chunks=depth_chunk)

        # 创建观测和动作数据集
        state_dim = 14
        eef_dim = 14
        obs_specs = {'qpos': state_dim, 'eef': eef_dim, 'qvel': state_dim, 'effort': state_dim,
                     'robot_base': 6, 'base_velocity': 4}
        act_specs = {'action': state_dim, 'action_eef': eef_dim, 'action_base': 6, 'action_velocity': 4}

        for name, dim in obs_specs.items():
            obs_dict.create_dataset(name, (data_size, dim))
        for name, dim in act_specs.items():
            root.create_dataset(name, (data_size, dim))

        for name, arr in data_dict.items():
            root[name][...] = arr


# 保存数据函数
def save_data(args, timesteps, actions, actions_eef, action_bases, action_velocities, ros_operator, dataset_path):
    data_size = len(actions)

    # 数据字典
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/eef': [],
        '/observations/robot_base': [],
        '/action': [],
        '/action_eef': [],
        '/action_base': [],
        '/action_velocity': [],
    }

    # 初始化相机字典
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # 遍历并收集数据
    while actions and rclpy.ok():
        action = actions.pop(0)  # 动作  当前动作
        action_eef = actions_eef.pop(0)
        action_base = action_bases.pop(0)
        action_velocity = action_velocities.pop(0)
        ts = timesteps.pop(0)  # 奖励  前一帧

        # 填充数据
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        data_dict['/observations/eef'].append(ts['eef'])
        data_dict['/observations/effort'].append(ts['effort'])
        data_dict['/observations/robot_base'].append(ts['robot_base'])
        data_dict['/action'].append(action)
        data_dict['/action_eef'].append(action_eef)
        data_dict['/action_base'].append(action_base)
        data_dict['/action_velocity'].append(action_velocity)

        # 相机数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts['images_depth'][cam_name])

    # 压缩图像数据
    padded_size, padded_size_depth = compress_and_pad_images(data_dict, args.camera_names, args.use_depth_image)

    # 文本的属性：
    # 1 是否仿真
    # 2 图像是否压缩
    t0 = time.time()
    create_and_write_hdf5(args, data_dict, dataset_path, data_size, padded_size, padded_size_depth)

    voice_process(voice_engine, "Save")
    print(f"\033[32m\nSaved in {time.time() - t0:.1f}s: {dataset_path}\033[0m\n")

    return


def main(args):
    setup_loader(ROOT)

    rclpy.init()

    config = load_yaml(args.config)

    ros_operator = RosOperator(args, config, in_collect=True)

    def _spin_loop(node):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    try:
        settled_height = configure_fixed_height(ros_operator, args.height)
    except Exception:
        ros_operator.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        raise
    if settled_height is not None:
        print(f'Fixed lift ready: command={args.height:.6f}, feedback={settled_height:.6f}')

    datasets_root = Path(args.datasets)
    if not sys.stdin.isatty() and not datasets_root.is_absolute():
        datasets_root = ROOT / datasets_root
    datasets_dir = task_dataset_dir(datasets_root, args.task)
    print(f'Dataset directory: {datasets_dir}')

    num_episodes = 1000 if args.episode_idx == -1 else 1
    current_episode = 0 if args.episode_idx == -1 else args.episode_idx

    # 查找最大episode序号
    max_episode = -1
    if os.path.exists(datasets_dir):
        for filename in os.listdir(datasets_dir):
            if filename.startswith('episode_') and filename.endswith('.hdf5'):
                try:
                    episode_num = int(filename.split('_')[1].split('.')[0])
                    max_episode = max(max_episode, episode_num)
                except ValueError:
                    continue

    # 如果找到了已存在的episode，从最大序号的下一个开始
    if max_episode >= 0:
        current_episode = max_episode + 1

    episode_num = 0
    with TerminalKeyReader() as key_reader:
        while episode_num < num_episodes and rclpy.ok():
            start_decision = prompt_start_decision(current_episode, key_reader.read_key)
            if start_decision == 'q':
                print('Collection stopped while idle; no episode was recorded.')
                break

            print(f"Start recording episode {current_episode}")
            timesteps, actions, actions_eef, action_bases, action_velocities = collect_information(
                args, ros_operator, voice_engine, key_reader
            )

            decision = prompt_episode_decision(
                key_reader.read_key,
                allow_save=bool(timesteps),
            )
            if decision == 'd':
                voice_process(voice_engine, 'Discard')
                print(f'Episode {current_episode} discarded; the number will be reused.')
                continue
            if decision == 'q':
                voice_process(voice_engine, 'Discard and quit')
                print(f'Episode {current_episode} discarded; collection stopped.')
                break

            if not os.path.exists(datasets_dir):
                os.makedirs(datasets_dir)

            dataset_path = os.path.join(datasets_dir, "episode_" + str(current_episode))
            save_data(args, timesteps, actions, actions_eef, action_bases, action_velocities,
                      ros_operator, dataset_path)

            episode_num = episode_num + 1
            current_episode = current_episode + 1

    ros_operator.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


def parse_arguments(known=False):
    parser = argparse.ArgumentParser()

    # 数据集配置
    parser.add_argument('--datasets', type=str, default=Path.joinpath(ROOT, 'datasets'),
                        help='dataset dir')
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help=argparse.SUPPRESS)  # Deprecated and intentionally ignored.
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')

    # 配置文件
    parser.add_argument('--config', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 图像处理选项
    parser.add_argument('--camera_names', nargs='*', type=str,
                        choices=['head', 'left_wrist', 'right_wrist', ],
                        default=['head', 'left_wrist', 'right_wrist'], help='camera names')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')

    # 机器人选项
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')

    # 数据采集选项
    parser.add_argument('--key_collect', action='store_true',
                        help='deprecated compatibility flag; single-key collection is always enabled')
    parser.add_argument('--height', type=float, default=None,
                        help='fixed lift command in [0, 20]; omitted means follow VR height')

    parser.add_argument('--task', type=normalize_task_name, required=True,
                        help='task name and dataset subdirectory')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
