# 0909 模型测试

`MODEL_VARIANT=0909` 选择期望的模型 route：
`arx-lift2s-0909-all-joint-feedback-64g50k-ft`。
它适用于下表七个任务 profile。服务器必须已经部署该 route；
客户端会在启动硬件前检查，不匹配时退出。此参数不会切换服务器模型。

在方舟桌面终端先执行只读检查：

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4
export ROS_DOMAIN_ID=62
unset TASK_INSTRUCTION

MODEL_VARIANT=0909 MODEL_PROFILE=all-blue-feedback \
MODEL_SERVER_URL=http://192.168.50.2:8000 LIFT_HEIGHT=12.5 \
./tools/05_tau0vla_calibrated_rollout.sh --check
```

清空运动路径后，将 `--check` 换成 `--execute` 开始测试。
该命令会启动硬件、标定夹爪、归到固定初始姿态并开始推理。

七个任务文本已核对训练数据 `data/0909_mixed_v1/All/meta/tasks.parquet`：

| `MODEL_PROFILE` | 精确任务文本 |
|---|---|
| `all-l-feedback` | Pick up the L-shaped part and place it in its designated position on the board. |
| `all-t-feedback` | Pick up the T-shaped part and place it in its designated position on the board. |
| `all-banana-feedback` | Pick up the banana and place it in its designated position on the board. |
| `all-red-feedback` | Pick up the red object and place it in its designated position on the board. |
| `all-blue-feedback` | Pick up the blue box and place it in its designated position on the board. |
| `all-cylinder-upper-feedback` | Pick up the cylindrical part and place it in the upper hole on the board. |
| `all-cylinder-lower-feedback` | Pick up the cylindrical part and place it in the lower hole on the board. |

`all-circle-feedback` 只属于 0908；在 0909 下使用会被拒绝，并提示选择圆柱上孔或下孔任务。
两个圆柱任务只支持 `MODEL_VARIANT=0909`。
如显式设置 `TASK_INSTRUCTION`，它会覆盖 profile 文本。

服务器切回 0908 模型后，使用 `MODEL_VARIANT=0908`；省略时仍默认 0908。
旧的 0907 v3 profile 保持原行为，不能与 `MODEL_VARIANT=0909` 混用。
升降高度仍为 `12.5`。
