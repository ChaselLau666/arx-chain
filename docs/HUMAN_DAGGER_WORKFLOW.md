# Human DAgger 双臂采集流程

本文适用于隔离副本 `/home/arx/ROS2_LIFT_Play_wy_dev` 的 `wy_dev` 分支。Human DAgger 只控制双臂，升降平台保持固定高度；底盘和头部不接收 VR 命令。

> 键盘接管不是急停。开始任何实机测试前，清空工作区并确保物理急停始终可触达。

中央控制进程会在可捕获异常和单臂启动不完整时尽力发布实测姿态 HOLD；但外部 X5 驱动没有可验证的命令 TTL。进程被 `SIGKILL`、整机掉电或驱动自身失效时，软件无法保证补发 HOLD，必须使用物理急停处置。

## 1. 启动前准备

必须在机器人桌面的本地 GNOME Terminal 中运行，不能通过 SSH 运行。先准备模型目录，目录内至少包含 checkpoint 和对应的训练统计文件：

```bash
export TASK_NAME=pickplace_right_to_bowl
export LIFT_HEIGHT=15.04
export CKPT_DIR=/home/arx/ROS2_LIFT_Play_wy_dev/act/weights/example

# 可选，以下为默认值
export CKPT_NAME=policy_best.ckpt
export STATS_NAME=dataset_stats.pkl
export DAGGER_ROUND=0
export MAX_TIMESTEPS=800
# 数据盘至少保留 5 GiB；可按现场容量提高阈值
export HUMAN_DAGGER_MIN_FREE_GIB=5
# 可选：把独立数据集放到其他数据盘
# export HUMAN_DAGGER_DATASET_DIR=/data/human_dagger
```

`LIFT_HEIGHT` 必须在 `[0, 20]`，整数写法（例如 `15`）也可以；脚本会先统一转换成 DOUBLE 字面量，再把同一个值传给 ROS、Python 和采集元数据。脚本强制使用 `ROS_DOMAIN_ID=62`，并从脚本自身位置解析仓库，因而不依赖当前工作目录。

启动前还要确认：

- 三台 RealSense、VR 串口和 can1/can3/can5 均已连接；三个 CAN 接口必须已由操作者安全配置为 `UP`。
- 没有运行 `v2_pos_control`、`v2_joint_control`、`open_double_arm`、其他 `X5Controller`、旧 collector/inference 或另一套 Human DAgger。
- `act/data/human_dagger.yaml` 中的话题和控制超时未被临时改坏。
- checkpoint 与 `dataset_stats.pkl` 来自同一次训练。
- 数据目录可写、剩余空间不低于阈值，且根目录没有待人工处理的 `*.partial.hdf5`。

## 2. 启动

```bash
/home/arx/ROS2_LIFT_Play_wy_dev/tools/05_human_dagger.sh
```

脚本依次完成以下工作：

1. 确认没有竞争的 CAN owner 或重复传感器栈。
2. 要求 can1、can3、can5 已经为 UP。脚本不会启动外部 `arx_can*.sh` watchdog；这些脚本会忙循环，并可能在单路掉线时全局杀死全部 `slcand`。
3. 使用规范化后的 DOUBLE 启动参数直接启动 body，并在节点可见后再次设置和确认 `/lift.fixed_height`，避免默认高度产生启动瞬态。
4. 先启动中央仲裁器；随后各启动一个 `X5Controller(normal)`：左臂 can1、右臂 can3、`arm_end_type=2`。X5 的 joy 输入被重映射到隔离话题，不能绕过仲裁器。
5. 启动三台 RealSense。
6. 直接启动 VR serial node，并将 `/ARX_VR_L/R` 重映射到 `/human_dagger/vr/left_raw` 和 `/human_dagger/vr/right_raw`。body 的 VR、joy 与普通 `/body_control` 输入也被隔离，因此底盘和头部不会被外部输入驱动。
7. 等待 body、双臂、三路 RGB 与双侧 VR 都收到新消息；中央仲裁器的 UI 始终位于当前终端。

后台组件日志和精确 PID 清单位于：

```text
${XDG_RUNTIME_DIR:-/tmp}/human_dagger-$UID/<session-id>/
```

PID 清单同时保存 supervisor、coordinator、policy、writer 以及各 ROS 组件的 Linux 进程启动时刻。安全关停只会对 PID 和启动时刻同时匹配的会话进程发信号，既能回收 supervisor 异常退出后的子进程，也避免 PID 被复用后误杀别的程序。

## 3. Episode 操作

就绪后机械臂处于 HOLD。先在 `MANUAL_RESET` 中将场景恢复到起始状态，然后使用单键操作（均不需要回车）：

- `R`：启动 policy 并开始录制。
- `Space`：从 policy 请求人工接管；`R` / `P` 后的冷启动和渐进衔接期间也可接管，重复按不会切回 policy。
- `P`：从人工控制请求重新启动 policy。
- `E`：结束当前 episode。
- 有人工介入时，审核阶段按 `S` 保存、`D` 丢弃、`Q` 丢弃并退出。
- 无人工介入且正常结束时自动保存。
- 活动 episode 中的 `Q` 被忽略。

屏幕显示的 `REQUESTED` 只是收到按键，只有出现 `HOLD`，随后出现 `HUMAN ACTIVE` 或 `POLICY ACTIVE`，控制权才真正切换。切换期间不要主动快速移动 VR。

程序退出或启动后预检失败时，启动脚本会尝试请求 HOLD，但不会直接停止高位的 body。必须继续执行安全关停流程。

### Tau0VLA 后端

在机器人本地终端运行当前仓库的 `tools/05_tau0vla_pickplace.sh`，它会选择 Tau0VLA 后端并调用同一采集入口。模型服务器由 `MODEL_SERVER_URL` 指定，任务文本由 `TASK_INSTRUCTION` 指定，无需本地 ACT checkpoint。

Tau0VLA 动作轨迹按独立的 **30 Hz** 时间槽推进；控制与录制默认仍为 **60 Hz**，中间周期沿用当前目标，不会把模型的动作块加倍速播放。人工遥操作不经过 policy 的频率限制或夹爪滤波。

启动脚本的 Tau0VLA 平滑参数与 `03_tau0vla_inference.sh` 对齐：`CHUNK_BLEND_STEPS=6`、`GRIPPER_BLEND_STEPS=0`、`GRIPPER_DEBOUNCE_FRAMES=12`、`ARM_EMA_ALPHA=0.6`、`GRIPPER_EMA_ALPHA=0.6`。夹爪阈值为 `-2.1/-1.05`，输出端点为 `-3.384/0.0`；可通过对应的 `GRIPPER_LOW/HIGH_THRESHOLD` 与 `GRIPPER_LOW/HIGH_VALUE` 环境变量调整。12 帧确认约为 0.4 秒，会有意过滤短暂的夹爪开合意图。每次 `R` / `P` 都清空动作缓冲与滤波状态，并从最新反馈重新初始化。

`REPLAN_STEPS=auto` 根据延迟选择重规划时机。手动设置时，数值表示消费多少步后请求下一块；必须让剩余动作时长覆盖实测 p99 RTT 与余量，过晚重规划会被拒绝。旧 epoch 的 HTTP 结果和异常都不会被新一轮 policy 采用。

## 4. 数据与校验

默认正式文件写入（设置 `HUMAN_DAGGER_DATASET_DIR` 时以该目录为准）：

```text
/home/arx/ROS2_LIFT_Play/dagger_datasets_YYYYMMDD_HHMMSS_NNNNNNNNN/episode_N.hdf5
```

每次通过 `05_human_dagger.sh` 或其任务启动脚本启动时，按机器人本地时间创建一个新目录（末尾为纳秒，避免快速重启重名），从 `episode_0.hdf5` 开始；同一次启动的后续 episode 在该目录递增编号。实际路径会打印并记入 session manifest。旧数据不移动；显式设置 `HUMAN_DAGGER_DATASET_DIR` 时仍使用指定目录。直接运行 Python 入口时以 `--datasets` 为准。

录制中先写为 `*.partial.hdf5`。控制、相机或写盘故障会进入 HOLD，并把未完成文件隔离到本次目录的 `quarantine/`，不会混入正式数据。每个文件包含完整 policy/HUMAN/交接时间线和事件日志；`/dagger/supervision_valid[t]` 只标记人工实际拥有 `[t,t+1)` 控制权的下一帧关节监督。

保存后运行离线校验：

```bash
cd /home/arx/ROS2_LIFT_Play
/home/arx/miniconda3/envs/act/bin/python \
  act/validate_dagger_episode.py dagger_datasets_YYYYMMDD_HHMMSS_NNNNNNNNN/episode_N.hdf5
```

需要机器可读结果时增加 `--json`。校验失败的文件不得加入后续训练数据。

## 5. 安全关停

不要用 `pkill`、关闭终端窗口或直接停止 body。运行：

```bash
/home/arx/ROS2_LIFT_Play_wy_dev/tools/04_safe_shutdown.sh
```

该脚本严格按以下顺序执行：

1. 调用 `/human_dagger/request_hold` 并要求成功确认。只要会话清单中的左臂或右臂控制进程仍存活，服务缺失、调用超时或返回失败都会在发送任何升降命令前中止；即使 coordinator 已退出也不会绕过 HOLD。
2. 将固定高度设为 0，等待平台反馈连续稳定在安全低位。
3. 要求操作者目视确认平台已降低。
4. 停止 Human DAgger 前台进程，并按本次会话 PID 清单的逆启动顺序停止 VR、相机、双臂和 body。历史会话若记录了 CAN watchdog，也会停止 watchdog shell。
5. 检查控制进程残留；存在残留时返回非零状态。

底层 CAN 接口保持 UP，供下次运行复用。

## 6. 常见故障

- **提示 competing process**：不要绕过检查。先判断它属于谁；仅对自己的会话执行安全关停。
- **某个 topic 超时**：查看启动信息输出的会话日志目录。禁止通过跳过双臂/VR freshness 检查来实机运行。
- **Space 后无法进入 HUMAN ACTIVE**：保持不动；双侧 VR 或机械臂反馈超过 100 ms 会触发 `FAULT_HOLD`。
- **P 后恢复失败**：policy 必须在从按键请求起的 2 秒总预算内清空时间聚合、用最新观测推理并通过逐关节渐进衔接；一旦收敛会立即进入 POLICY，不会强制等待满 2 秒。超时则 episode 故障隔离，不可自动恢复。
- **终端异常关闭**：不要重新启动另一套控制栈。打开新的本地终端，直接运行 `04_safe_shutdown.sh`；其 active manifest 用于识别原会话进程。
- **没有有效 active manifest**：为避免广泛 `pgrep` 误停另一位开发者的进程，关停脚本默认拒绝。仅在明确关停旧版、非 Human DAgger 栈时，先检查屏幕列出的 PID，再以 `HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN=1` 重跑并输入二次确认短语。
