# ARX LIFT2s 数据采集流程

适用分支：`zjy_dev`
当前采集格式：官方 HDF5
默认采样配置：60 FPS、三相机、双臂 14 维、`use_base=False`

## 1. 启动前安全检查

每次开机后先确认：

- 平台处于安全低位，工作区清空，急停可触达。
- VR、三台 D405、三个 USB2CAN 均已连接。
- `ROS_DOMAIN_ID=62`。
- can1、can3、can5 均为 `UP`。

CAN 必须在启动任何 body/双臂节点前逐个配置。不要在接口缺失时直接依次运行官方 CAN watchdog，因为其中的全局 `pkill slcand` 可能终止其他接口。

```bash
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up

sudo slcand -o -f -s8 /dev/arxcan3 can3
sudo ip link set can3 up

sudo slcand -o -f -s8 /dev/arxcan5 can5
sudo ip link set can5 up

ip -br link show can1
ip -br link show can3
ip -br link show can5
```

## 2. 采集参数

标准启动需要两个环境变量：

```bash
export ROS_DOMAIN_ID=62
export LIFT_HEIGHT=15.04
export TASK_NAME=pickplace_right_to_bowl
```

- `LIFT_HEIGHT`：传给 body 的固定高度命令，范围 `[0,20]`。命令值和实际反馈可能存在校准偏差。
- `TASK_NAME`：写入 HDF5 的 `task` 属性。
- 不传 `--use_base`：底盘不作为模型输入/action；HDF5 中对应 base 数组保持 0。

当前脚本实际启动 collector 的等价命令为：

```bash
python collect.py \
  --episode_idx -1 \
  --height 15.04 \
  --task pickplace_right_to_bowl
```

重要参数：

- `--episode_idx -1`：连续采集，从数据目录现有最大编号加一开始。
- `--max_timesteps 800`：单条最多 800 帧。
- `--frame_rate 60`：采集器名义频率 60 FPS。
- `--height`：固定高度命令。
- `--task`：任务名称。
- `--key_collect`：仅为旧命令兼容保留；当前默认始终使用单键状态机。

## 3. 启动完整链路

必须从 `tools` 目录执行：

```bash
cd /home/arx/ROS2_LIFT_Play/tools
./01_collect.sh
```

脚本依次启动 CAN watchdog、body、双臂、三相机、VR 和 collector。body 在 VR 启动前收到 `fixed_height`，collector 会再次确认参数，并等待：

1. `/ARX_VR_L` 已有数据；
2. `/body_information.height` 连续 2 秒稳定；
3. 然后才进入 episode 交互。

正常输出包括：

```text
/lift fixed_height set to 15.040000
Lift feedback settled at ... for command 15.040000
Fixed lift ready: command=15.040000, feedback=...
```

## 4. 开始一条 episode

假设数据目录当前最大文件为 `episode_8.hdf5`，下一条编号就是 `episode_9`。

collector 就绪后显示：

```text
Ready for episode 9: [r]ecord, [q]uit:
```

- 单按 `R`：立即开始记录当前编号。
- 单按 `Q`：在未录制状态正常退出 collector。
- 不按 `R` 就不会采集帧，也不会写入 HDF5。
- 所有按键都不需要回车；旧的复位/夹爪准备手势不再触发开始。

## 5. 结束当前 episode

录制期间显示：

```text
RECORDING: press [e] to end and review this episode.
```

- 单按 `E`：结束记录并进入审核；机械臂复位不再自动结束。
- 即使相机或双臂同步失败、当前有效帧数为 0，`E` 也必须立即生效；0 帧 episode 禁止保存，只能丢弃或丢弃并退出。
- 达到 `--max_timesteps`（默认 800 帧）时仍会强制结束并进入审核，防止无限录制。
- 录制期间其他按键会被忽略，不能误保存或误开始下一条。

结束后程序不会立即保存，而是显示：

```text
Episode ended: [s]ave, [d]iscard, [q]discard and quit:
```

## 6. 保存、丢弃和下一条

### 保存当前条

单按：

```text
s
```

不需要回车。程序同步压缩并写完 HDF5，语音提示 `Save`。确认终端出现 `Saved in ...` 后：

- 当前编号被占用；
- 编号加一；
- 程序停在下一条确认提示，不会自动开始。

### 丢弃当前条并重采

单按：

```text
d
```

不需要回车。当前内存数据直接丢弃：

- 不生成 HDF5；
- 不占用 episode 编号；
- 程序停在重试确认提示，不会自动重采。

## 7. 明确开始下一次采集

保存或丢弃处理结束后，程序显示：

```text
Ready for episode N: [r]ecord, [q]uit:
```

- 单按 `R`：开始下一条。上一条已保存时使用新编号；上一条已丢弃时复用原编号。
- 单按 `Q`：不开始下一条，collector 正常退出。

因此，任何下一条 episode 都必须在新的等待提示出现后由操作者显式按 `R`，不会因为上一条保存或丢弃完成而自动开始。保存期间提前按下的键会在进入等待状态时清除，避免误触发下一条。

### 丢弃并退出

审核阶段单按：

```text
q
```

不需要回车。当前条不保存，collector 正常退出。其他由 `01_collect.sh` 启动的硬件节点仍在运行，需要按安全顺序单独关闭。

## 8. 每条保存后的检查

查看文件：

```bash
ls -lh /home/arx/ROS2_LIFT_Play/act/datasets/episode_*.hdf5
```

可视化一条：

```bash
cd /home/arx/ROS2_LIFT_Play/act
conda activate act
python visualize.py --datasets ./datasets --episode_idx 9
```

输出包括：

```text
datasets/episode_9_video.mp4
datasets/episode_9_qpos.png
datasets/episode_9_qvel.png
datasets/episode_9_eef.png
datasets/episode_9_action_base.png
datasets/episode_9_action_velocity.png
```

## 9. 当前数据语义说明

- 三相机图像为 640×480。
- state/action 均为双臂 14 维。
- 不使用 `--use_base`，base 数据不参与模型。
- 当前仍是官方 action 逻辑：12 个手臂关节 action 等于当前 qpos，夹爪经过官方阈值处理；不是真实 joint command。
- HDF5 没有真实时间戳，60 FPS 是采集器配置值。
- 转换到 30 FPS LeRobot 时按索引 `0,2,4,...` 同步抽取图像、state 和 action。

## 10. 一批采集结束后的安全处理

### 10.1 很快继续下一批：推荐保持硬件链路运行

如果平台仍在工作高位并且稍后还要继续采集，不要关闭或重启 body。先在 collector 的等待或审核提示中单按 `Q`，使 collector 正常退出；body、双臂、相机、VR 和 CAN 保持运行。

下一批只启动 collector，不重复执行 `01_collect.sh`：

```bash
cd /home/arx/ROS2_LIFT_Play/act
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
conda activate act

python collect.py \
  --episode_idx -1 \
  --height 15.04 \
  --task pickplace_right_to_bowl
```

collector 会从当前最大 HDF5 编号加一继续。

### 10.2 完整关闭所有控制程序

平台高位时禁止直接关闭 body。推荐使用一键安全关闭脚本：

```bash
cd /home/arx/ROS2_LIFT_Play/tools
./04_safe_shutdown.sh
```

脚本需要依次输入两个完整确认文本：

```text
LOWER AND SHUTDOWN
CONFIRM LOW
```

它会设置 `fixed_height=0.0`，同时发送一次不依赖 VR 的 `/body_control` 低位命令；只有反馈连续 2 秒稳定且不高于 1.0，并经过操作者肉眼确认后，才按 collector、VR、相机、双臂、body、CAN watchdog 顺序停止程序。底层 CAN 接口和 `slcand` 保持 UP，供下一次启动复用。

如果控制栈已经不完整，且 `/lift` 和 body 进程都不在，脚本无法读取或驱动高度。此时必须先肉眼确认平台已经处于安全低位，再输入：

```text
CONFIRM ALREADY LOW
```

脚本随后只清理仍在运行的采集、VR、相机、双臂和 CAN watchdog。如果仍存在 body 进程但 `/lift` 在当前 `ROS_DOMAIN_ID` 中不可见，脚本会拒绝关停；先检查 domain 和 body 日志，禁止盲目停止不可观测的 body。

对应的手动流程如下。首先保持 body 和 VR 运行，明确：

```text
当前高度：工作高位
目标命令：0.0
运动方向：DOWN
预期行为：平台下降并稳定在安全低位
```

在新终端执行：

```bash
export ROS_DOMAIN_ID=62
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash

ros2 param set /lift fixed_height 0.0
ros2 topic echo /body_information
```

观察 `height` 持续下降。反馈稳定后按 `Ctrl+C` 退出 echo，并肉眼确认平台已经到达安全低位。

然后在各启动终端按以下顺序执行 `Ctrl+C`，等待对应进程退出：

1. collector（如果尚未通过 `q` 退出）；
2. VR serial node 及 VR echo/hz 终端；
3. 三个 RealSense 相机终端；
4. 双臂 `v2_pos_control` 终端；
5. body 终端，必须最后关闭；
6. CAN watchdog 终端。

关闭后确认：

```bash
ps -eo pid,args | grep -E \
  '(lift_controller|X5Controller|serial_port_node|realsense2_camera_node|collect.py)' \
  | grep -v grep
```

没有输出表示 ROS 控制程序已退出。

默认建议保留 can1、can3、can5 的 `slcand` 和接口为 `UP`，这样同一次开机内再次执行 `01_collect.sh` 时，官方 CAN watchdog 不会进入全局 `pkill slcand` 分支。

### 10.3 完整关闭后再次开始

再次确认平台处于安全低位、工作区清空、急停可触达，然后：

```bash
export ROS_DOMAIN_ID=62
export LIFT_HEIGHT=15.04
export TASK_NAME=pickplace_right_to_bowl

cd /home/arx/ROS2_LIFT_Play/tools
./01_collect.sh
```
