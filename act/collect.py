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

from collections import deque
from copy import deepcopy

from utils.ros_operator import Rate, RosOperator
from utils.setup_loader import setup_loader
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
    while (count < args.max_timesteps) and rclpy.ok():
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

    if count >= args.max_timesteps:
        print(f'Hard limit reached at {args.max_timesteps} frames; entering review.')

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
def read_ready_pose(node, arm_nodes=('/vr_arm_l', '/vr_arm_r'), timeout=4.0):
    """Read the go_home_position each arm controller was launched with.

    This is the single source of truth for where the arms park:
    tools/06_collect_filtered.sh passes it, X5Controller hands it to the SDK,
    and reading it back is what lets return_to_ready tell arriving from merely
    stopping. Returns None if either arm cannot be asked, in which case the
    caller falls back to checking that the arms came to rest.
    """
    from rclpy.parameter_client import AsyncParameterClient

    poses = []
    for name in arm_nodes:
        client = AsyncParameterClient(node, name)
        if not client.wait_for_services(timeout_sec=timeout):
            return None
        future = client.get_parameters(['go_home_position'])
        deadline = time.monotonic() + timeout
        while not future.done() and rclpy.ok() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done() or future.result() is None:
            return None
        values = future.result().values
        if not values or not values[0].double_array_value:
            return None
        poses.append(np.array(values[0].double_array_value, dtype=float))
    return np.concatenate(poses)


# The ready pose's gripper, in the arms' own units. go_home_position covers six
# joints and not the seventh, so this is the one part of the pose that has to be
# named here rather than read back from the arms.
READY_GRIPPER = (-2.9717, -2.9675)
# X5Controller multiplies a VR gripper by this before use, so command the inverse.
VR_GRIPPER_SCALE = -3.4 / 5


def hold_ready_pose(ros_operator, seconds=0.4, period=0.02):
    """Hold the pose the arms just reached, commanded the way teleop commands it.

    GO_HOME gets them there but leaves them in a state the next VR frame
    cancels. Re-sending the pose they are already at, as an end-effector
    command, leaves them in END_CONTROL holding it, so an episode starts in the
    state teleop runs in rather than in a fight between the two.

    The pose is read back from the arms rather than written down here, so it
    follows go_home_position without a second copy to keep in step.

    Runs inside the /arx_joy mute the caller has been holding, so nothing else
    is publishing to the arms while it does.
    """
    poses = []
    for arm, gripper in zip((ros_operator.follow_left_arm_deque,
                             ros_operator.follow_right_arm_deque), READY_GRIPPER):
        if not arm:
            print('WARNING: no arm state to read the ready pose back from; not holding it.')
            return False
        poses.append((np.array(arm[-1].end_pos, dtype=float), gripper / VR_GRIPPER_SCALE))

    end = time.time() + seconds
    while time.time() < end:
        ros_operator.publish_ready_pose(poses)
        time.sleep(period)
    print(f'  held as an end-effector command: '
          f'left {np.round(poses[0][0][:3], 4)} right {np.round(poses[1][0][:3], 4)}')
    return True


def return_to_ready(ros_operator, timeout=12.0, min_wait=1.0, settle_window=0.4,
                    tolerance=0.004, arrival_tolerance=0.05):
    """Walk both arms to the go_home_position their controllers were launched with.

    Publishing /arx_joy repeatedly does two jobs at once: it holds the arms in
    GO_HOME against the END_CONTROL that every VR frame would otherwise restore,
    and it keeps vr_pose_filter muted so the VR stream cannot fight the move.
    Stopping lets the filter resume, so the mute lasts exactly as long as this
    takes. The target lives in each arm's go_home_position parameter, set by
    tools/06_collect_filtered.sh, and is deliberately not repeated here.

    Coming to rest is not the same as arriving. When the mute does not take -
    no filter to mute under SMOOTH_TAU=0, or one left running from an older
    build - GO_HOME and END_CONTROL fight to a standstill partway, and treating
    that as success would put a wrong first frame in the episode without saying
    so. The pose read back from the arms decides it whenever it is available.

    Returns True once both arms are at the ready pose and still.
    """
    def joints():
        both = []
        for arm in (ros_operator.follow_left_arm_deque, ros_operator.follow_right_arm_deque):
            if not arm:
                return None
            both.append(np.array(arm[-1].joint_pos[:6], dtype=float))
        return np.concatenate(both)

    target = read_ready_pose(ros_operator)
    if target is None:
        print('Could not read go_home_position from the arms; '
              'will only check that they come to rest.')

    print('Returning the arms to the ready pose; stand clear.')
    start = time.time()
    recent = deque()
    while time.time() - start < timeout:
        ros_operator.request_go_home()
        time.sleep(0.05)
        current = joints()
        if current is None:
            continue
        now = time.time()
        recent.append((now, current))
        while recent and now - recent[0][0] > settle_window:
            recent.popleft()
        # min_wait covers the case where the arms have not started moving yet:
        # without it the first samples are trivially identical and settled.
        if now - start < min_wait or now - recent[0][0] < settle_window * 0.8:
            continue
        spread = np.ptp(np.array([q for _, q in recent]), axis=0).max()
        if spread >= tolerance:
            continue
        if target is not None and np.abs(current - target).max() > arrival_tolerance:
            continue          # stopped, but not where it was asked to go
        left, right = np.degrees(current[:6]), np.degrees(current[6:])
        print(f'  arms parked after {now - start:.1f}s: '
              f'left {np.round(left, 1)} right {np.round(right, 1)}')
        hold_ready_pose(ros_operator)
        return True

    current = joints()
    where = np.round(np.degrees(current), 1) if current is not None else 'unknown'
    if target is not None and current is not None:
        print(f'WARNING: the arms are {np.degrees(np.abs(current - target).max()):.1f} deg from the '
              f'ready pose after {timeout:.0f}s, at {where}. Recording anyway, so this episode does '
              f'not start where the others do. Is a pose filter running to be muted by /arx_joy?')
    else:
        print(f'WARNING: the arms had not settled after {timeout:.0f}s, at {where}. Recording anyway. '
              f'Is a pose filter running to be muted, or is something blocking the arms?')
    return False


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

    datasets_dir = args.datasets if sys.stdin.isatty() else Path.joinpath(ROOT, args.datasets)

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

            # Here rather than after save or discard: the arms follow the VR pose
            # again the moment the filter unmutes, so parking them only holds until
            # the operator next moves a hand. Doing it as recording is about to
            # start is what actually puts the same first frame in every episode.
            return_to_ready(ros_operator)

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
    parser.add_argument('--max_timesteps', type=int, default=800, help='max timesteps')
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

    parser.add_argument('--task', type=str, default='', help='task name')
    parser.add_argument('--ready_pose_topics', nargs=2,
                        default=['/ARX_VR_L_filtered', '/ARX_VR_R_filtered'],
                        metavar=('LEFT', 'RIGHT'),
                        help='the pose topics the arms subscribe to, left then right; '
                             'these are the raw VR topics when the filters are off')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
