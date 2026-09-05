# ark-1 DAgger 交接记录（2026-09-05）

新会话开场请说：**"读取 `/home/arx/ROS2_LIFT_Play/docs/DAGGER_HANDOFF_20260905.md`，继续之前的工作"**

---

## 0. 立刻要处理的两件事 ⚠️

### (1) 串口被占，现在起 DAgger 栈会失败

我为了测按钮起了两个后台进程，**还在跑**：

```bash
pgrep -af "[s]erial_port_node"      # 占着 /dev/ttyACM0
pgrep -af "[p]robe_vr_buttons"      # 探针
```

`05_human_dagger.sh:538` 会起它自己的 `vr_serial`，两者抢同一个串口。**起栈前必须先停**：

```bash
pkill -f probe_vr_buttons
pkill -f serial_port_node
```

日志在 `/home/arx/logs/vr_probe/`。

### (2) `vr_engage_enabled: true` 的配置已被实测证伪，有误触风险

`act/data/human_dagger.yaml` 里现在是：

```yaml
vr_engage_enabled: true
vr_engage_field: mode1
vr_engage_active_value: 1
```

但实测 `mode1` **当前就是 1**（且见过 2）。也就是说 `mode1` 从 2 变 1 的那一刻会**自动触发接管**。建议立刻改回 `vr_engage_enabled: false`。

（改完要同步 `tests/test_human_dagger_scripts.py` 里那条 `'vr_engage_enabled: true'` 断言，否则测试挂。）

---

## 1. SSH 连接方式（重要）

本会话 `$HOME` 被指到 `/Users/xiangchengliu/.claude-modes/jd`，那里的 `.ssh/config` 只有一条 `aliyun`。**必须显式指定真实 config**：

```bash
ssh -F /Users/xiangchengliu/.ssh/config arx1
```

- `arx1` → `192.168.31.57`，主机名 `ark-1`，用户 `arx`，`sudo` 免密
- `arx2` → `192.168.31.218`（今天关机中）
- 裸写 `ssh arx` 会走 DNS 到已失效的 `192.168.31.43`，超时
- conda 环境：`source /home/arx/miniconda3/etc/profile.d/conda.sh && conda activate act`
- 无 `pytest`，用 `python -m unittest tests.<module>`
- 仓库 `/home/arx/ROS2_LIFT_Play`，分支 `main`，`ROS_DOMAIN_ID=62`

远程编辑技巧：复杂 Python 补丁脚本先写本地 `/tmp`，再 `base64 -i xx.py | ssh ... 'base64 -d > /tmp/xx.py && python3 /tmp/xx.py'`，避免多层引号地狱。补丁里用 `assert s.count(old) == 1` 锚定。

---

## 2. 已完成的代码改动

### 改动 A：夹爪改绝对二值映射 ✅ 已生效

**原问题**：按空格接管时，如果 policy 已把夹爪合死，操作员**无法用 VR 张开**。

**根因**（`act/human_dagger_core.py:1192` 原代码）：

```python
gripper = feedback_anchor.gripper + (vr_current.gripper - vr_anchor.gripper) * (-0.68)
```

夹爪是**锚点增量**控制。接管瞬间 policy 合死（反馈 ≈ -3.384，行程最低端）+ 操作员手松（扳机 ≈ 0，也在行程最低端），**两个端点同向对齐**。扳机只能往正方向走（扣紧），经 -0.68 缩放后只会产生更负（更关）的命令。张开方向可用行程恰好为 0 —— 不是量不够，是方向上完全没有。

单侧有界执行器用相对锚点，锚点错位必然单侧饱和。

**改法**：新增 `_resolve_gripper()`，绝对二值映射 + 滞回，锚点彻底不参与夹爪：

| 扳机值 | 结果 |
|---|---|
| ≤ 2.0 | `0.0` 张开 |
| ≥ 3.0 | `-3.384` 闭合 |
| 2.0~3.0 | 保持当前那一端（防抖） |

**涉及文件**：
- `act/human_dagger_core.py` — `_resolve_gripper()` 新函数；`_rebase_one` 换公式；首帧特判改为「姿态仍 bit-exact，夹爪走绝对映射」；One Euro 滤波把通道 6 设为 `_PASSTHROUGH_CHANNELS`（否则二值跳变被磨成百毫秒斜坡，正好穿过要避开的中间开度）；配置字段 `gripper_delta_scale` → `gripper_trigger_open_below/close_above` + `gripper_open_value/closed_value` + 校验
- `act/human_dagger.py` — 读新的 4 个 yaml key
- `act/data/human_dagger.yaml` — 同上
- `tests/test_human_dagger_core.py` — 3 处旧断言更新 + 2 个新回归测试
- `tests/test_human_dagger_scripts.py` — yaml key 断言

**⚠️ 行为变化（现场必须知道）**：首帧不再保证夹爪连续。姿态仍 bit-exact，但夹爪立刻跟扳机走 —— 这是「松开就张开」的必要代价。**如果 policy 正夹着工具、你接管时手是松的，进入 HUMAN 第一帧夹爪就张开，工具会掉。** 想接手时保持夹住，接管前先把扳机扣过 3.0。

**备份**：`.codex-backups/gripper_binary_20260905/`

### 改动 B：VR 按钮 hold-to-engage ⚠️ 已实现但信号源不可用

`_vr_callback` 多抓一个 `mode1`/`mode2`，控制循环在 `core.update_vr()` 之后做**上升沿检测**，检到注入 `ControlEvent.TAKEOVER`。状态机一行没改，走的就是空格那条完全相同的路径。

安全性质（都有测试钉住，`tests/test_vr_engage_edge.py` 6 个）：只在上升沿触发 / 首个样本不触发（启动时按钮已按住不会自己接管）/ 松开永不触发任何事件 / 缺样本(`None`)被忽略且不重置状态。

**但按钮实测后证明 `mode1`/`mode2` 都不能用**（见第 3 节）。所以这个功能**该关掉**，代码留着等有可用信号再启用。

**涉及文件**：`act/human_dagger.py`、`act/data/human_dagger.yaml`、`tests/test_vr_engage_edge.py`（新）、`tests/test_human_dagger_scripts.py`

**备份**：`.codex-backups/vr_auto_takeover_20260905/`

### 改动 C：`tools/00_hw_up.sh` 新脚本 ✅

幂等硬件 bringup，三段：CAN×3 → body/lift → `fixed_height`。每段满足就跳过。

```bash
./00_hw_up.sh --check          # 纯只读看状态
LIFT_HEIGHT=14.0 ./00_hw_up.sh # 实际起（DAgger 用 14.0，脚本默认 15.5）
./00_hw_up.sh --yes            # 跳过确认短语
./00_hw_up.sh --no-height      # 只起 CAN + body
```

**不用厂商 `arx_can1/3/5.sh`** —— 那三个是 `while true` 看门狗，每个都跑全局 `sudo pkill -9 slcand`，三个一起跑会互杀。新脚本只对绑在该设备的 daemon 发信号（`pkill -f "slcand.*${device}"`）。

**顺手补了一个必踩的坑**：`03_tau0vla_inference.sh` 从不设 `fixed_height`，但 `tau0vla_client.py:204` 严格校验它必须等于 `--expected-height`，而 `lift_controller.cpp:28` 默认 `-1.0`（回落跟随 VR 高度）。全仓库只有 `01_collect.sh` 会设。**所以光起 CAN+body，推理会在高度校验处被拒。** 第 3 段补这个。

**不起两个臂控制器**（故意）：起它就可能让手臂动（SDK 初始化调 `arx_x(...)`），且 ark-1 **没有** `act/hold_guard.py`（ark-2 才有），起臂到模型接管之间那段窗口没有任何东西托住手臂。必须人在现场清场 + 急停在手边手动起。

**已验证**：`--check`、`--help`、参数守卫、CAN 三段实跑成功（`can1/can3/can5` 都 UP，每设备恰好一个 slcand，无互杀）。
**未验证**：body 段的 `setsid` 起法、第 3 段升高度、`read_fixed_height` 的 awk 解析。

### 改动 D：`tools/probe_vr_buttons.py` 新脚本（只读探针）

只在字段变化时打一行，用于测按钮映射。**有已知缺陷见第 4 节。**

### 测试状态

**21 个测试模块全部通过**（含新增的 `test_vr_engage_edge`）。
注意：`python -m unittest discover -s tests -t .` 会因 `tests/` 无 `__init__.py` 报 `ImportError`，要逐模块跑。另外 `test_act_range_tools` / `test_collection_review` 的 robomimic 警告会污染 `tail -1`，判断 OK/FAIL 要 `grep -E "^(OK|FAILED|ERROR)"`。

**所有改动都没有上过真机。**

---

## 3. VR 按钮实测结果（决定性，不用再试）

用 `tools/probe_vr_buttons.py` 实测。**结论：DAgger 能用的可靠信号只有 `gripper`（侧扳机）一个。**

| VR 按钮 | 信号 | 在哪 |
|---|---|---|
| **侧扳机（中指）** | ✅ 有 | `gripper`，**0..5 连续模拟量** |
| 食指扳机 | ❌ 无独立字段 | app 内部 gate 位姿（未验证） |
| X / Y / A / B | ❌ 无 | app 内部消化 |
| 摇杆按下 | ❌ 无 | app 内部消化 |
| 摇杆推动 | ⚠️ 间接 | 改 `height` / `chx/chy/chz` |
| 头部转动 | ✅ 有 | `head_pit` / `head_yaw` |

**根本原因：协议层没有按钮位域。** `serial_parser.hpp` 那个 94 字节结构体里离散字段只有 `mode1`/`mode2` 两个 `uint8`。不是 app 没发，是线上没地方放。

**关键推论**：探针和 dagger 订阅同一个 topic、同一份数据 —— **探针看不到的，dagger 永远也看不到**。任何基于食指扳机/ABXY/摇杆的方案，在当前 VR SDK 下都不可能，除非改头显侧 `X5_MR_Control` app（闭源）或串口协议。

### `mode1` / `mode2` 为什么不能用

- `mode2`：全程恒 0，一次都没变过
- `mode1`：是**模式档位**（见过 2 和 1，现在稳定在 1），不是按钮按下/松开；只在左手消息填（`serial_port.cpp:121-122`，右手恒 0）；且被 `lift_controller.cpp:69` 的 `setChassisCmd(0,0,0,msg.mode1)` 当底盘指令消费 —— 拿来当接管键会耦合底盘

### Topic 清单

DAgger 运行时只有两个（`05_human_dagger.sh:540-541` remap）：

```
/ARX_VR_L → /human_dagger/vr/left_raw
/ARX_VR_R → /human_dagger/vr/right_raw
```

**没有 `/joy`，没有独立按钮 topic**。（`lift_controller.cpp:72` 订阅的 `/joy` 是给实体手柄的，VR 链路不发。）

### 其它实测机器事实

- **`gripper` 范围确认 0..5**，厂商 `X5Controller.cpp:163-166` 用 `×(-3.4/5)` 映射 —— 二值夹爪阈值 2.0/3.0 **不用改**
- **`gripper` 是真正的连续量**，能停在 2.05 / 2.26 / 3.29 等中间值。所以 2.0~3.0 滞回带**会被真实扫过**，滞回逻辑是必需的，不是装饰
- **VR 实际发布 ~105Hz**，不是我早先按 `serial_port.cpp` 2ms 定时器推的 500Hz。`vr_timeout_ms: 100` 仍有约 10 帧余量
- VR 接收器是 `/dev/serial/by-id/usb-1a86_USB_Single_Serial_*` → `/dev/ttyACM0`，`serial_port.cpp:13` 按 `"USB_Single_Serial"` 字符串匹配挑设备
- 启动瞬间那条 `Frame HEAD check failed: 00 00` WARN 是首帧未对齐，之后正常

---

## 4. 我犯过的错误（避免重复踩）

1. **探针漏了位姿字段**：`probe_vr_buttons.py:29-30` 的监视列表**没有 `x/y/z/roll/pitch/yaw`**。所以「食指扳机没留下任何痕迹」这个结论**不成立**，我根本没在看那六个字段。**要继续查食指扳机/A 键，必须先补上这些字段。**

2. **误判侧扳机是二值的**：曾说它"9ms 切换、几乎二值"，后续数据显示是连续量，之前看着像二值是因为按得快。

3. **方案 2「锚点行程对齐」在代数上等于绝对映射**：把 `vr_anchor.gripper = feedback/scale` 代入公式，锚点整个消掉，`command = g × scale`。它**能**张开（不是打不开），但"没有首帧跳变"是错的，跳变照样有。

4. **误报 VR 500Hz**（实际 ~105Hz）。

---

## 5. 未解决 / 待办

### (1) 「按 A 键自动复位」的机制还没搞清

文档说「长按右手 A → 右侧机械臂归 0」。两种可能，现有数据分不出来：
- **可能一（证据更强）**：头显 app 生成归零轨迹，通过 `x/y/z/rpy` 逐帧发出，下游只是跟随
- **可能二**：`mode1` 携带指令。但 A 键测试期间 `mode1` 只在 39.6s 跳了一次，之后 250 秒没变 —— 跟"长按 A 归零"对不上

**查法**：补全探针的位姿字段，然后手放着不动长按 A，看 `x/y/z` 是否自己变化。

### (2) 食指扳机是否 gate 位姿 —— 未验证

补全探针后做对照测试：
1. 松开食指扳机 + 移动手 → `x/y/z` 是否静止
2. 按住食指扳机 + 移动手 → `x/y/z` 是否跟随

这决定「松开自动退出」到底有没有可用信号。文档说「松开食指扳机 → 机械臂保持空间位姿」，暗示是可能一。

### (3) 接管方式最终选哪个 —— 待决策

| 方案 | 评价 |
|---|---|
| **A. 退回空格** | 确定可靠，代价是要碰键盘（frontend 必须真实终端跑，`</dev/tty` 读按键，SSH 起不了） |
| **B. 双手同时扣满侧扳机 → 接管** | 唯一可靠通道。单手扣是正常夹爪操作不会误触，双手同时扣满不会偶然发生；接管后松手就是张开夹爪，跟改动 A 天然衔接。**我推荐这个** |
| C. 再测一轮 `mode1` | 底盘耦合问题仍在，不推荐 |

### (4) 「松开自动退出」—— 刻意没做

用户原始诉求是「激活 VR 自动接管、松开自动结束」。**接管方向已实现（等信号源），退出方向刻意未做。**

原因：`HUMAN → POLICY` 是最危险的方向。`human_dagger_core.py:949-1000` —— policy 先 reset 清空动作缓存，然后从当前反馈位姿向 policy **重新预测**的目标做限步 slew（每 tick 每关节 0.03~0.05 rad，夹爪 0.2），2 秒收敛不了直接 `FAULT_HOLD`。policy reset 后从新观测重新预测，想去的是它自己轨迹的延续，**不是你刚拖到的地方** → 手臂被匀速拉回（这就是用户说的「P 之后会退回」，是设计不是 bug）。

把这个方向绑到最容易误触的信号上：手抖、按钮接触不良、扳机滑出阈值 → 立刻触发 policy 恢复 + 拉回；反复误触 → 反复 invalidate epoch + policy reset + 2 秒 slew 窗口 → `FAULT_HOLD`。

真要做必须配：**停留计时**（连续松开 1.5s 才触发）+ **滞回**，且最好先解决「拉回」问题（需要给 policy 目标做 rebase，另一个工程）。

中间方案：**同一颗信号「再按一次退出」**（toggle）比「松开退出」安全得多 —— 需要明确的按下动作，抖动不会触发。

### (5) 「拉回」问题本身 —— 未解决

不管用什么方式触发恢复都存在。要消掉得给 policy 目标做 rebase。

---

## 6. 当前接管逻辑（改动后的实际状态）

### 进 HUMAN（接管）

两个触发源，注入**完全相同**的 `ControlEvent.TAKEOVER`：

1. **空格** — `human_dagger_core.py:567-568`，一直可用
2. **VR `mode1` 上升沿** — 已启用但**信号源不可用，应关掉**

**只在 `POLICY` / `HANDOFF_TO_POLICY` 状态下生效**（`human_dagger_core.py:839-840`）。其它状态（`MANUAL_RESET`/`HUMAN`/`REVIEW_HOLD`/`FAULT_HOLD`）按了**静默丢弃** —— 所以 MANUAL_RESET 阶段是测按钮的安全窗口。

过程（零跳变）：作废在飞的 policy 包（epoch+1）→ `HANDOFF_TO_HUMAN` 两臂发 `POSITION_CONTROL` HOLD → 等 HOLD 之后的新反馈+新 VR 都到齐 → 捕获锚点 → `HUMAN`。2 秒走不完 → `FAULT_HOLD`。

### 退出 HUMAN

| 触发 | 去哪 |
|---|---|
| **P** | → `HANDOFF_TO_POLICY` → `POLICY`（会被拉回） |
| **E** | → `REVIEW_HOLD`，再 S 存 / D 弃 |
| VR 流断 >100ms | → `FAULT_HOLD`（掉线即停，不是交回 policy） |

**再按 VR 按钮无效**（HUMAN 态 TAKEOVER 被丢弃）。**松开无效**（只认上升沿）。

### 状态机

```
PRECHECK_HOLD
   ↓ 前置检查通过
MANUAL_RESET  ← 用 VR 摆初始姿态（此处按空格无效）
   │ R
   ↓
HANDOFF_TO_POLICY ──┐
   │ 收敛            │空格
   ↓                ↓
POLICY ──空格──→ HANDOFF_TO_HUMAN
   │                ↓ HOLD+锚点就绪
   │             HUMAN
   │                │ P → 回 HANDOFF_TO_POLICY（会被拉回）
   │                │ E → REVIEW_HOLD ──S/D──→ 下一 episode
   │                │ VR断100ms → FAULT_HOLD
   └── E ────────────────────→ REVIEW_HOLD
```

---

## 7. 起栈完整流程

机器 17:07 重启过，栈是空的。

```bash
# 0. 先停掉测试进程（见第 0 节）
pkill -f probe_vr_buttons; pkill -f serial_port_node

# 1. CAN + body + 高度（注意 14.0）
cd /home/arx/ROS2_LIFT_Play/tools
LIFT_HEIGHT=14.0 ./00_hw_up.sh

# 2. 【危险，人在现场，清场 + 急停在手】两个臂控制器
cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws && source install/setup.bash
ros2 launch arx_x5_controller v2_joint_control.launch.py
# 必须恰好 2 个 X5Controller 且都带 v2_joint_control.yaml

# 3. DAgger（用户给的完整命令）
cd /home/arx/ROS2_LIFT_Play/tools
export ROS_DOMAIN_ID=62
DAGGER_EXECUTOR=events HUMAN_DAGGER_ALLOW_SSH=1 \
MODEL_SERVER_URL=http://192.168.50.2:8000 \
TASK_NAME=pickplace_tau0vla_dagger \
TASK_INSTRUCTION='Pick up the tool and place it into the tray.' \
LIFT_HEIGHT=14.0 REPLAN_STEPS=15 CHUNK_BLEND_STEPS=6 GRIPPER_BLEND_STEPS=6 \
ARM_EMA_ALPHA=0.6 GRIPPER_EMA_ALPHA=1.0 GRIPPER_DEBOUNCE_FRAMES=0 \
MAX_TIMESTEPS=3600 COLOR_PROFILE=640x480x30 DEPTH_PROFILE=640x480x30 \
ENABLE_DEPTH=false \
bash 05_tau0vla_pickplace.sh
```

操作键：等 `MANUAL_RESET` → VR 摆初始姿态 → `R` 开始 → 空格接管 → `P` 恢复 → `E` 结束（`S` 存 / `D` 弃）。

**关栈**：`04_safe_shutdown.sh`。**绝不要直接 SIGINT lift 进程** —— 厂商 `arx_lift_controller` 收 SIGINT 会 `pure virtual method called` 崩在析构里，平台升起时崩它很危险。

### 环境事实

- 模型服务器 `192.168.50.2:8000` 健康，直连 `enp130s0 src 192.168.50.1` 已验证，模型 `tau0vla-arx-pickplace-tool-yipan-h200-step10000`
- `GRIPPER_BLEND_STEPS`/`GRIPPER_EMA_ALPHA`/`GRIPPER_DEBOUNCE_FRAMES` 链路确实存在且被使用（`05_human_dagger.sh:413-421` → `human_dagger.py:1732-1739` → `Tau0VLAWorkerConfig` → `human_dagger_tau0vla_policy.py:169-181`），但**只作用于 POLICY 态**输出。HUMAN 态夹爪不受它们影响 —— 调它们治不了改动 A 那个病
- ark-1 **没有** `hold_guard.py` / `lib_human_dagger.sh` / 10_/20_/30_ 三段式（那些是 ark-2 的）
- 每次重启 CAN 必掉；`/dev/arxcan1,3,5` udev 软链正常（→ ttyACM1/3/2）
