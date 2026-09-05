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
from collection_paths import normalize_task_name, task_dataset_dir
from collection_ui import TerminalKeyReader, prompt_episode_decision, prompt_start_decision
from lift_height import configure_fixed_height
from ready_pose import (READY_ARM_NODES, READY_GRIPPER, arms_have_arrived,
                        joints_are_still, vr_gripper_command)

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


# 归位到 ready pose ----------------------------------------------------------
# 归位的目标只写在启动脚本里，由它传给 X5Controller 的 go_home_position，这里读回来
# 用作校验，避免同一组数值在脚本和 Python 两处各存一份、改一处忘另一处。
def read_ready_pose(node, arm_nodes=READY_ARM_NODES, timeout=4.0):
    """把每条手臂启动时拿到的 go_home_position 读回来。

    读不到就返回 None，此时调用方退化成只检查手臂停稳，并在日志里说清楚。
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


def hold_ready_pose(ros_operator, seconds=0.4, period=0.02):
    """把手臂刚到达的位姿按住，用遥操本来的命令方式。

    GO_HOME 能把手臂送到位，但留下的状态会被下一帧 VR 消息取消。把它已经在的
    位姿作为末端命令重发一遍，手臂就留在 END_CONTROL 按住不动，episode 从遥操
    本来运行的状态开始，而不是从两种状态互相打架开始。

    位姿是从手臂反馈读回来的，不在这里另写一份，所以它跟着 go_home_position 走。
    """
    poses = []
    for arm, gripper in zip((ros_operator.follow_left_arm_deque,
                             ros_operator.follow_right_arm_deque), READY_GRIPPER):
        if not arm:
            print('WARNING: 没有手臂反馈可以读回位姿，不按住。')
            return False
        poses.append((np.array(arm[-1].end_pos, dtype=float), vr_gripper_command(gripper)))

    end = time.time() + seconds
    while time.time() < end:
        ros_operator.publish_ready_pose(poses)
        time.sleep(period)
    print(f'  已作为末端命令按住：左 {np.round(poses[0][0][:3], 4)} 右 {np.round(poses[1][0][:3], 4)}')
    return True


def return_to_ready(ros_operator, timeout=12.0, min_wait=1.0, settle_window=0.4,
                    tolerance=0.004, arrival_tolerance=0.05):
    """把两条手臂走到它们控制器启动时拿到的 go_home_position。

    /arx_joy 要反复发，这一条消息干两件事：把手臂按在 GO_HOME 状态里，对抗每一帧
    VR 消息都会恢复的 END_CONTROL；同时让 vr_pose_filter 静音，这样 VR 流不会和
    归位打架。停止发布滤波器就恢复，所以静音时长正好等于归位耗时。目标写在每条
    手臂的 go_home_position 参数里，由 tools/08_collect_ready_pose.sh 传入，这里
    刻意不重复一份。

    停下来不等于到位。静音没生效时——SKIP_FILTER=1 下没有滤波器可静音，或者跑的是
    旧版滤波器——GO_HOME 和 END_CONTROL 会在半路打成僵持，把那当成成功就会让这个
    episode 的第一帧和别的不一样却不说。所以只要能读到 go_home_position，就用它
    判定是否真的到位。

    返回 True 表示两条手臂都到了 ready pose 且已静止。
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
        print('读不到手臂的 go_home_position，只能检查它们是否停稳。')

    print('正在把手臂归位到 ready pose；请与手臂保持距离。')
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
        # min_wait 覆盖手臂还没开始动的情况：没有它，最初几个采样天然相同、
        # 会被当成已经停稳。
        if now - start < min_wait:
            continue
        if not joints_are_still(recent, tolerance, settle_window):
            continue
        if not arms_have_arrived(current, target, arrival_tolerance):
            continue          # 停了，但不在它被要求去的地方
        left, right = np.degrees(current[:6]), np.degrees(current[6:])
        print(f'  手臂归位耗时 {now - start:.1f}s：左 {np.round(left, 1)} 右 {np.round(right, 1)}')
        hold_ready_pose(ros_operator)
        return True

    current = joints()
    where = np.round(np.degrees(current), 1) if current is not None else 'unknown'
    if target is not None and current is not None:
        gap = np.degrees(np.abs(current - target).max())
        print(f'WARNING: {timeout:.0f}s 后手臂距 ready pose 还有 {gap:.1f} 度，停在 {where}。'
              f'仍会录制，所以这个 episode 的起点和别的不一样。滤波节点是否在运行？')
    else:
        print(f'WARNING: {timeout:.0f}s 后手臂仍未停稳，在 {where}。仍会录制。'
              f'滤波节点是否在运行可被 /arx_joy 静音，或有东西挡住手臂？')
    return False


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

            # 放在这里而不是保存/丢弃之后：手臂一旦回到 ready pose，操作者下次
            # 动手就会把它带走，所以只有在即将开始录制时归位，才真的让每个
            # episode 的第一帧一致。
            if args.ready_pose:
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

    # 每个 episode 录制前把手臂归位到启动时设定的 go_home_position。默认关闭，
    # 因为只有用 ros2 run 传了 go_home_position 的启动脚本（tools/08_）才有意义；
    # 走厂商 v2_pos_control.launch.py 的 01_collect.sh 没有那个参数。
    parser.add_argument('--ready_pose', action='store_true',
                        help='park both arms at go_home_position before each episode records')
    # An arm in vr_slave mode reads poses from exactly one topic, so commanding
    # the ready pose means publishing where that arm is listening. Which topic
    # that is depends on whether the pose filters were started, so the launcher
    # passes the pair it actually wired.
    parser.add_argument('--ready_pose_topics', nargs=2,
                        default=['/ARX_VR_L_filtered', '/ARX_VR_R_filtered'],
                        metavar=('LEFT', 'RIGHT'),
                        help='topics the arms subscribe to, left first')

    parser.add_argument('--task', type=normalize_task_name, required=True,
                        help='task name and dataset subdirectory')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
