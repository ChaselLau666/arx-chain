# 0909 模型测试

`MODEL_VARIANT=0909` 选择期望的模型 route：
`arx-lift2s-0909-all-joint-feedback-64g50k-ft`。
它只适用于六个 `all-*-feedback` 任务 profile。服务器必须已经部署该 route；
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

任务可替换为 `all-l-feedback`、`all-t-feedback`、`all-banana-feedback`、
`all-red-feedback`、`all-blue-feedback` 或 `all-circle-feedback`。
各 profile 沿用现有任务文本；此文档不宣称新 checkpoint 的训练任务表已核实。
如显式设置 `TASK_INSTRUCTION`，它会覆盖 profile 文本。

服务器切回 0908 模型后，使用 `MODEL_VARIANT=0908`；省略时仍默认 0908。
旧的 0907 v3 profile 保持原行为，不能与 `MODEL_VARIANT=0909` 混用。
升降高度仍为 `12.5`。
