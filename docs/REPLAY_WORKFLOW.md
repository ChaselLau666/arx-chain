# ARX LIFT2s 轨迹回放

把一条录好的 HDF5 episode 回放到真实双臂上。

适用分支：`hjs_dev`
入口脚本：`tools/05_replay.sh`
默认模式：**干跑**（不创建发布器、不设高度、不产生任何运动）

回放是**开环播放**，没有人在控制回路里 —— 与采集时人手握 VR 构成闭环不同。因此所有前置条件一律检查后拒绝，不做自动修复。

---

# 第一部分：为什么需要改动

`act/replay.py` 是 fork 进来的原始代码（`git log` 只有仓库第一个提交），此后所有安全改造都没同步到它。直接运行会出现下列问题。

| # | 问题 | 后果 | 处理 |
|---|---|---|---|
| 1 | 起始位姿硬编码 `init0 = [0,0,0,0,0,0,4]` | 夹爪那个 `4` 属于 `PosCmd` 的 0..5 约定，却发到 `RobotStatus` 通道（该通道原值直传，合法区间约 `[-3.4, 0]`）。越界且符号相反 | 改为取轨迹第一帧的实际值 |
| 2 | `init_robot()` 调用 `robot_base_shutdown()` | 向 `/body_control` 发 `height=0.0`，把平台命令到最低位，然后在错误高度上回放 | 删除。高度统一走 `/lift` 的 `fixed_height` 参数 |
| 3 | 完全不读也不设高度 | 轨迹在特定平台高度录制，回放高度不确定 | 从 HDF5 读 `height_command`，设参数并等反馈稳定 |
| 4 | `follow_arm_publish_continuous` 的插值被丢弃 | 计算了逐步逼近的位置，发布的却是终点 —— 手臂直冲目标 | 改为发布插值结果 |
| 5 | SIGINT handler 访问默认为 `None` 的属性 | Ctrl+C 抛 `AttributeError`，没有干净的中断路径 | 改为停止标志；三阶段全程可中断 |
| 6 | 没有干跑档 | 一启动就具备发指令能力 | 新增 `--execute`，默认干跑 |
| 7 | 没有启动脚本 | 手动拉起链路，容易漏步骤或误启 VR | 新增 `tools/05_replay.sh`，fail-closed |
| 8 | `robot_action()` 第五个实参传的是循环下标 | `--use_base` 路径必然抛异常 | 传正确的速度数组 |
| 9 | 每帧 `print` 整条轨迹数组 | 遗留调试代码，打印的是错误对象 | 删除 |

---

# 第二部分：改了什么

## 新增

| 文件 | 作用 |
|---|---|
| `act/lift_height.py` | 高度配置逻辑，采集与回放共用。模块级不 import rclpy，纯函数部分可脱离 ROS 测试 |
| `act/replay_support.py` | 回放的纯逻辑：起始位姿拆分、高度决策规则 |
| `tools/05_replay.sh` | 回放启动脚本，全部前置检查 fail-closed |
| `tests/test_replay_support.py` | 上述纯逻辑的测试 |
| `docs/REPLAY_WORKFLOW.md` | 本文档 |

## 修改

| 文件 | 改动 |
|---|---|
| `act/replay.py` | 上表 1–9 的主体改动；新增 `--execute`、`--height`、`--height-tolerance` |
| `act/collect.py` | 高度逻辑移出，改为 import `lift_height`（行为不变） |
| `act/utils/ros_operator.py` | 修复渐进移动；新增 `request_arm_publish_stop()` 中止钩子；手臂反馈等待加超时 |
| `tools/04_safe_shutdown.sh` | 补上 `replay.py` 进程与 `v2_joint_control` 手臂栈的清理 |
| `tests/test_fixed_height.py` | 改为 import `lift_height` |

## 测试

从 11 个增加到 **29 个**，且**整个套件不再需要先 source ROS** —— 此前 `test_fixed_height.py` 因为 import `collect.py` 而连带依赖 rclpy。

```bash
cd ~/ROS2_LIFT_Play
python -m unittest discover -s tests
```

---

# 第三部分：启动流程

## 0. 启动前确认

- 平台处于**安全低位**（见第五部分）
- 工作区清空，桌面上没有会被撞到的东西
- **急停可触达，人站在双臂工作半径之外**
- `/dev/arxcan1`、`/dev/arxcan3`、`/dev/arxcan5` 在位
- 确认 `ROS_DOMAIN_ID` 是本机的值（见第四部分）

回放**不需要** VR，也**不需要**相机。VR 在运行时脚本会拒绝执行。

## 1. CAN（整机重启后必须重做）

```bash
sudo slcand -o -f -s8 /dev/arxcan1 can1 && sudo ip link set can1 up
sudo slcand -o -f -s8 /dev/arxcan3 can3 && sudo ip link set can3 up
sudo slcand -o -f -s8 /dev/arxcan5 can5 && sudo ip link set can5 up

ip -br link show type can        # 三条都必须是 UP 才能继续
```

不要直接运行 `~/LIFT/ARX_CAN/arx_can/arx_can*.sh` —— 其中的全局 `sudo pkill -9 slcand` 会在接口缺失时连带杀掉另外两路。

`slcand` 与 ROS domain 无关，`04_safe_shutdown.sh` 会刻意保留接口 UP 供下次复用；只有整机重启才需要重做这一步。

## 2. body（平台必须在低位）

```bash
cd ~/LIFT/body/ROS2 && source install/setup.bash
ros2 launch arx_lift_controller lift.launch.py
```

启动后 `fixed_height` 是默认的 `-1.0`，且没有任何发布者推 `/ARX_VR_L`、`/body_control`、`/joy`，平台不受指令驱动。

body 由操作者手动启动，脚本不代劳 —— 它需要人先确认平台的物理状态。

## 3. 干跑

```bash
cd ~/ROS2_LIFT_Play/tools
EPISODE=datasets/episode_19.hdf5 ./05_replay.sh
```

脚本在手臂话题缺失时自动启动 `v2_joint_control.launch.py`。**手臂上电会回零复位，这一刻请站开。**

干跑不创建任何发布器（结构性的，不是调用点加判断），也不设 `/lift`，因此不产生运动。输出形如：

```text
Reusing the running joint-command arm stack.
Episode: /home/arx/ROS2_LIFT_Play/act/datasets/episode_19.hdf5
DRY-RUN: no arm publisher is created; pass --execute to move the arms.
DRY-RUN: would set /lift fixed_height to 15.500000
DRY-RUN start pose:
  left : [...]
  right: [...]
DRY-RUN frame 0/230: [...]
Replay finished: 230 frames.
```

**重点看两件事**：打印的起始位姿与手臂当前姿态差多远；高度值是否符合预期。

## 4. 真跑

```bash
EPISODE=datasets/episode_19.hdf5 ./05_replay.sh --execute 2>&1 | tee /tmp/replay_run.log
```

提示出现后一字不差地输入：

```text
EXECUTE REPLAY
```

输错即取消，不会创建任何发布器。

随后三段运动：

| 阶段 | 动作 | Ctrl+C 行为 |
|---|---|---|
| 1 | 设 `fixed_height`，等反馈稳定 | 把 `fixed_height` 冻结在当前反馈高度，平台原地停住 |
| 2 | 双臂匀速移到轨迹首帧 | 立即停止发布，手臂停在最后目标 |
| 3 | 回放全部帧 | 16.7 ms 内停止 |

第 2 段速度由 `arm_steps_length = [0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.2]` 决定，每 33 ms 一步。

## 5. 收工

```bash
./04_safe_shutdown.sh
```

依次输入两段确认文本：

```text
LOWER AND SHUTDOWN
CONFIRM LOW
```

脚本设 `fixed_height=0.0`，等反馈连续 2 秒稳定且不高于 1.0，人眼确认后按 collector / inference / replay、VR、相机、双臂、body、CAN watchdog 的顺序停止。CAN 接口保持 UP。

---

# 第四部分：ROS_DOMAIN_ID 与多机隔离

**这是实际调试中唯一真正阻塞回放的问题，且与代码无关。**

## 现象

两台机器原本都在 `.bashrc` 里设 `ROS_DOMAIN_ID=62`，且处于同一网段。DDS 的默认多播发现会把它们合并成**同一个 ROS 图**。因为节点名和话题名完全相同，后果是：

| 话题/节点 | 后果 |
|---|---|
| `/lift` | 两个同名节点，`ros2 param set /lift ...` 打到哪台不确定 |
| `/body_information` | 两台的高度反馈交错，读到的数据来源无法分辨 |
| `/arm_master_l_status` | **在一台上跑回放，另一台的手臂会同步执行** |

实测确认：ark-2 在**本机零个 ROS 进程**的情况下，仍能看到并读取 ark-1 的节点与话题内容。

这直接导致连续三次回放失败 —— `configure_fixed_height` 读到的是两台机器交错的高度，2 秒窗口的峰峰值恒等于两台平台的高度差，永远判不出"稳定"，只能走到超时。

## 当前配置

| 机器 | 地址 | ROS_DOMAIN_ID |
|---|---|---|
| ark-1 | 192.168.31.57 | 62 |
| ark-2 | 192.168.31.218 | **63** |

设在各自的 `~/.bashrc`。仓库脚本用 `${ROS_DOMAIN_ID:-62}`，会自动跟随环境变量，无需改动。

## 注意事项

- **文档里不要写死 domain 数字**，以本机 `.bashrc` 为准。
- 改 `.bashrc` **不影响已经在运行的进程** —— 它们的 domain 在启动时就固定了。切换 domain 后要操作旧进程，必须显式指定旧值，例如 `ROS_DOMAIN_ID=62 ./04_safe_shutdown.sh`。
- 开跑前用 `ros2 node list` 确认**只看到本机应有的节点**。看到多余的手臂或相机节点，说明串到另一台去了。
- `ros2 daemon` 会缓存陈旧注册。节点列表出现重复或幽灵节点时，先 `ros2 daemon stop` 再查。

---

# 第五部分：中断与安全

## Ctrl+C 不是安全装置

三个阶段现在全程可中断，但**停止发指令不等于手臂立刻静止** —— 手臂控制器收到的是位置目标，会走完最后一个目标再保持。中断后手臂仍会移动最多一个步长（关节 ≤0.05 rad、夹爪 ≤0.2 rad）。

**能保护人的只有物理急停。**

第二次 Ctrl+C 立即强制退出。中断不会降台：平台停在当前高度，降台是收工流程的事。

## 为什么要求平台在低位启动 body

`lift_head_control_loop.h` 里有这些成员：

```cpp
double gravity_compensation_torque = 0;
double lift_calibrate_vel = 3;
bool lift_is_calibrated_ = false;
std::chrono::system_clock::time_point calibrated_start_time_;
int calibrated_count_ = 0;
```

**推断**（实现在预编译的 `libarx_lift_src.so` 里，标定逻辑内联在 `update()` 中，无法直接查证）：升降柱电机没有绝对编码器，上电后需要以固定速度开到机械限位找参考点。这段行程不受指令控制，高位启动意味着长距离的不可控移动。腰部和头部有一组同构的标志位。

**已观测到的事实**：一次整机重启时平台停在约 15 的高度，重启后平台**没有下落**。所以"控制环停止后平台会因重力坠落"这一点**未被验证**，实际可能存在机械自锁或阻尼。

结论：低位启动的规则保留（这也是 `03_inference.sh` 拒绝信息里的说法），但理由以标定行程为准，不要依赖"会坠落"的假设。

---

# 第六部分：脚本的拒绝条件

`05_replay.sh` 在下列任一情况下拒绝执行，不做修复：

| 检查 | 拒绝条件 |
|---|---|
| `EPISODE` | 未设置 |
| episode 文件 | 不存在（相对路径按 `act/` 解析） |
| `/lift` | 不在 `ros2 node list` 中 |
| CAN | can1/can3/can5 任一不为 UP |
| VR serial node | 进程存在 |
| `/ARX_VR_L` | 有发布者 |
| `/vr_arm_l`、`/vr_arm_r` | 在运行（VR 遥操臂栈，订阅 `ARX_VR_*` 而非 `arm_master_*_status`） |
| 手臂栈 | 反馈话题只有 1/2（0 个则自动启动并轮询等待，2 个则复用） |

手臂栈以无头后台进程启动，日志在 `/tmp/replay_arms.log`，不依赖图形终端。

---

# 第七部分：高度语义

## 决策规则

| 情况 | 行为 |
|---|---|
| HDF5 有 `height_command`，未传 `--height` | 用文件里的值 |
| 两者都有且相等 | 放行 |
| 两者都有但不等 | **拒绝执行** |
| 文件没有，也没传 `--height` | **拒绝执行** |

不一致时不会默默采用命令行的值 —— 高度错了是物理事故。

## 命令值与实测值存在稳态差

实测两台机器在同一命令下的表现：

| 机器 | `fixed_height` | 实测 height | 差 |
|---|---|---|---|
| ark-1 | 15.5 | ≈15.03 | ≈0.47 |
| ark-2 | 15.5 | ≈15.15 | ≈0.35 |

控制器补不上最后这一段（配置中只有 `lift_kp`，没有积分项），且移动呈走走停停的静摩擦卡滞特征，而非连续爬升。

**因此 HDF5 中的 `height_command` 只是命令值，不代表录制时的实际高度。** 跨机器回放时两台的实际高度可能相差约 0.1，而 HDF5 中没有记录高度反馈（`observations/robot_base` 因未使用 `--use_base` 而全为 0），无法据此校验。

`--height-tolerance` 默认 0.06，用于判断反馈是否稳定。在 domain 隔离正确的前提下，单机高度反馈的方差实测为 0，该判据轻松通过。

---

# 第八部分：已知限制

- **不产生可对比的验证数据。** 当前版本只发指令，不记录手臂实际到达的位置。要量化回放忠实度（逐关节误差、跟踪延迟），需要另外实现"回放时记录实际 qpos 并与源轨迹比对"。
- **跨机器回放的标定差异无法校验。** 见第七部分。
- **默认回放 `qpos` 而非 `action`。** `action` 的夹爪列被 `collect.py` 的 `-2.1` 阈值二值化过（实测约 29.6% 的帧被压成 0），丢失了夹爪开合的过程信息。`--states_replay` 会切换到 `action`，其 help 文字与实际逻辑相反，不建议使用。
- **`--use_base` 路径未验证。** 当前数据集未使用底盘。另外 `ros_operator.py` 中 `use_base` 与 `height` 的订阅是 `if/elif` 互斥的，两者同时启用会导致高度反馈永远为空。
- **`replay.py` 与 `collect.py` 顶部重开 stdout/stderr 文件描述符**，多个此类模块被同时 import 时会互相关闭 fd。当前测试已避开该路径，但隐患仍在。
- **`04_safe_shutdown.sh` 的 `--tolerance 0.02`** 低于某些情况下观测到的反馈波动。若收工时卡在"等待高度稳定"，参照 `COLLECTION_WORKFLOW.md` 第 10.2 节的手动流程。
