# ARX LIFT2s 官方 ACT 范围训练与开环回放

适用分支：`zjy_dev`

数据契约：三相机、双臂物理 state/action 14 维、官方 ACT 内部 action 28 维、60 FPS、`use_base=False`、`state(t+1)` fallback。

## 1. 当前数据

正式训练使用：

```text
/home/arx/ROS2_LIFT_Play/act/datasets/episode_0.hdf5
...
/home/arx/ROS2_LIFT_Play/act/datasets/episode_49.hdf5
```

`episode_50.hdf5` 只用于共同开环回放。51 条数据均标记 `height_command=15.5`；模型不读取高度，但后续在线 dry-run 和真机推理必须复现同一高度命令。

源 HDF5 `/action(t)` 是当前 `qpos(t)` 加官方夹爪阈值。loader 只后移一次，形成训练目标 `action(t)=source_action(t+1)`；末帧没有伪造 action。

## 2. 代表 episode 可视化

在 ARX 上执行：

```bash
cd /home/arx/ROS2_LIFT_Play/act
conda activate act

for episode in 0 19 25 34 50; do
  python visualize.py \
    --datasets ./datasets \
    --episode_idx "${episode}" \
    --output_dir ./data_review
done
```

重点人工检查：抓取和放置是否完整、左臂是否静止、三相机是否连续、右腕遮挡是否仅发生在正常近距离抓取阶段。`episode_25` 中段右腕视角应重点复核。

## 3. 4090 独立环境

训练服务器：`xiangchengliu@192.168.31.83`，代码与现有 `tau-0-vla` 分开保存。

```bash
conda create -n arx-act python=3.11 -y
conda activate arx-act

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r act/requirements-training.txt
```

首次构建模型可能需要 torchvision 的 ResNet-18 权重。smoke test 前必须确认权重已缓存且环境能够执行 CUDA 前后向。

## 4. 通用范围接口

`--start` 和 `--end` 都包含端点；评估 episode 不允许出现在训练范围中。

```bash
python tools/run_act_experiment.py \
  --start 0 \
  --end 24 \
  --eval-episode 50 \
  --source-dir /home/xiangchengliu/data/arx_act_pickplace/hdf5 \
  --view-root /home/xiangchengliu/data/arx_act_pickplace/views \
  --run-root /home/xiangchengliu/data/arx_act_pickplace/runs \
  --epochs 2
```

脚本会校验 HDF5、计算 SHA-256、建立连续编号链接、写入 split manifest，并拒绝覆盖已有 run 目录。

三组 smoke 与正式训练可顺序执行：

```bash
python tools/run_act_three_ranges.py \
  --phase smoke \
  --source-dir /home/xiangchengliu/data/arx_act_pickplace/hdf5 \
  --view-root /home/xiangchengliu/data/arx_act_pickplace/views \
  --run-root /home/xiangchengliu/data/arx_act_pickplace/runs

# smoke 全部通过后才执行
python tools/run_act_three_ranges.py \
  --phase full \
  --source-dir /home/xiangchengliu/data/arx_act_pickplace/hdf5 \
  --view-root /home/xiangchengliu/data/arx_act_pickplace/views \
  --run-root /home/xiangchengliu/data/arx_act_pickplace/runs
```

每个 run 必须包含：

```text
policy_best.ckpt
dataset_stats.pkl
args.yaml
data_contract.yaml
best_checkpoint.yaml
policy_epoch*_pretrained_all_info.ckpt
TensorBoard 日志和训练曲线
```

## 5. episode_50 开环回放

三组正式训练结束后：

```bash
python tools/eval_act_openloop.py \
  --episode /home/xiangchengliu/data/arx_act_pickplace/hdf5/episode_50.hdf5 \
  --run first25=/home/xiangchengliu/data/arx_act_pickplace/runs/act_ep000_024_seed0_full \
  --run second25=/home/xiangchengliu/data/arx_act_pickplace/runs/act_ep025_049_seed0_full \
  --run all50=/home/xiangchengliu/data/arx_act_pickplace/runs/act_ep000_049_seed0_full \
  --output-dir /home/xiangchengliu/data/arx_act_pickplace/openloop_episode50
```

输出包含三组关节与夹爪叠加图、误差 JSON、三相机开环视频和模型对比图。该过程不导入 ROS，也不会发布机器人动作。

## 6. ARX 在线 dry-run

当前推理默认 dry-run。body 必须已在安全低位启动并提前设为固定高度 15.5；脚本不会启动或重启 body，也不会修改高度。

```bash
export ROS_DOMAIN_ID=62
export LIFT_HEIGHT=15.5
export CKPT_DIR=/home/arx/ROS2_LIFT_Play/act/weights/act_ep000_049_seed0_full
export CKPT_NAME=policy_best.ckpt

cd /home/arx/ROS2_LIFT_Play/tools
./03_inference.sh
```

dry-run 不创建双臂或 `/body_control` publisher，只打印模型 action。checkpoint 契约、相机顺序、14/28 维关系、action semantics 或高度不一致时直接拒绝。

高度稳定判定使用连续 2 秒峰峰值不超过 `0.05`。LIFT2s 的高度反馈会在相邻编码档位间产生约 `0.047` 的量化跳变，因此不得使用小于量化步长的容差；超过 `0.05` 仍拒绝推理。

LIFT2s 推理双臂必须使用 `v2_joint_control.launch.py`：它发布 `/arm_slave_l_status`、`/arm_slave_r_status`，并订阅模型端的 `/arm_master_l_status`、`/arm_master_r_status`。旧的 `open_double_arm.launch.py` 使用 `/joint_information*` 与 `/joint_control*`，和 `inference.py` 不兼容，不得用于本链路。

`X5Controller` 在建立模型命令订阅器之前会调用底层 `arx_x(500, 2000, 10)` 初始化，现场已观察到该阶段可能产生一段 replay-like 机械臂运动。因此 `03_inference.sh` 禁止自动启动双臂控制器，只能复用由操作者在工作区清空、急停可触达时手动启动并确认初始化完成的两个 v2 controller。此初始化动作不是 checkpoint 输出；“dry-run 不发布动作”只描述模型进程，不代表启动硬件驱动本身绝对无运动。

如果出现 `there is no head queue`（或 left/right wrist queue），表示推理订阅存在但对应图像没有实时 publisher。`03_inference.sh` 必须按 `Publisher count > 0` 判断双臂和三相机是否在线，并在启动后等待全部 publisher 就绪；不能仅凭 ROS daemon 中残留的 topic 名复用旧栈。检查命令：

```bash
ros2 topic info /camera/camera_h/color/image_rect_raw/compressed
ros2 topic info /camera/camera_l/color/image_rect_raw/compressed
ros2 topic info /camera/camera_r/color/image_rect_raw/compressed
```

每路必须显示 `Publisher count: 1`。启动脚本不能用 ROS graph 中可能残留的 topic/publisher 判断是否复用硬件栈；双臂必须同时存在两个加载 `v2_joint_control.yaml` 的 X5Controller 实际进程，相机必须存在零个或三个实际进程。默认 dry-run 下机器人不运动是正确行为；只有日志持续打印 `DRY-RUN action[0:14]` 才表示模型链路已经运行。

真机执行必须额外提供经过审核的 14 维关节限位 YAML，并显式使用 `--execute --joint-limits ...`。当前没有审核过的限位文件，因此禁止真机执行。

在正式限位尚未提供时，只允许使用 `--single-step-test` 验证真实发布链：左臂固定为启动时反馈，右臂六关节每轴相对当前值默认不超过 `0.02 rad`，右夹爪不超过 `0.2`；超限时零发布，合格时只发布一条并永久 disarm。该模式不等价于连续真机推理或任务成功验收。
