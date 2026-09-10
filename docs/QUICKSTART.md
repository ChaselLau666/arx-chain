# Quick start

Copy-paste commands for `ark-1`. Details in `DAGGER_CALIBRATED_WORKFLOW.md`
(DAgger) and `TAU0VLA_CALIBRATED_V3.md` (rollout).

The rollout stack and a DAgger stack cannot coexist: DAgger starts and tracks its
own lift, arms and cameras, and refuses if anything else owns them. Switching
between the two always means a shutdown first.

## Human DAgger, calibrated feedback-v4

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && ./tools/04_safe_shutdown.sh
```

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && MODEL_PROFILE=all-t-feedback TASK_NAME=pickplace_t_dagger LIFT_HEIGHT=12.5 ./tools/05_dagger_calibrated.sh
```

Change the object by changing `MODEL_PROFILE`; move `TASK_NAME` with it so the
recorded episodes stay separable.

| `MODEL_PROFILE` | prompt object | suggested `TASK_NAME` |
| --- | --- | --- |
| `all-t-feedback` | T-shaped part | `pickplace_t_dagger` |
| `all-l-feedback` | L-shaped part | `pickplace_l_dagger` |
| `all-blue-feedback` | blue box | `pickplace_blue_dagger` |
| `all-red-feedback` | red object | `pickplace_red_dagger` |
| `all-banana-feedback` | banana | `pickplace_banana_dagger` |
| `all-circle-feedback` | circular part | `pickplace_circle_dagger` |

Keys once the UI is up: `R` park at the ready pose and start, `Space` human,
`P` policy, `E` end, `S`/`D`/`Q` review. The foot trigger is enabled by default and
toggles policy/human with one press.

Startup does a **gripper calibration** partway through — the grippers move. It has
not been verified on hardware yet, so keep a hand on the emergency stop for the
first run.

## Calibrated rollout

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && ./tools/04_safe_shutdown.sh
```

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && export ROS_DOMAIN_ID=62 && unset TASK_INSTRUCTION && MAX_STEPS=10800 MODEL_PROFILE=all-t-feedback MODEL_SERVER_URL=http://192.168.50.2:8000 LIFT_HEIGHT=12.5 ./tools/05_tau0vla_calibrated_rollout.sh --execute
```

`MAX_STEPS` is in 30 Hz ticks: `3600` = 2 min (the script's `--execute` default),
`10800` = 6 min. The client itself allows up to 10000 unless overridden.

Server and config checks only, no hardware and no motion:

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && export ROS_DOMAIN_ID=62 && MODEL_PROFILE=all-t-feedback LIFT_HEIGHT=12.5 ./tools/05_tau0vla_calibrated_rollout.sh --check
```

Return to the fixed training pose, standalone:

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && ROS_DOMAIN_ID=62 LIFT_HEIGHT=12.5 ./tools/06_tau0vla_return_fixed.sh --execute
```

## If CAN went down

DAgger validates `can1/can3/can5` but never brings them up, so bring them up
before launching:

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4 && ./tools/00_can_up.sh
```

```bash
for i in can1 can3 can5; do ip -br link show $i; done
```

## Checking state

What owns the hardware right now:

```bash
pgrep -af 'lift_controller|X5Controller|realsense2_camera_node|tau0vla|human_dagger'
```

Model server identity and readiness:

```bash
curl -s http://192.168.50.2:8000/health
```

Most recent rollout artifacts:

```bash
ls -lt /home/arx/logs/tau0vla-calibrated/ | head
```

## Notes

`ROS_DOMAIN_ID` is deliberately absent from the DAgger command: `05_human_dagger.sh`
requires the robot identity to come from `/etc/environment` (ark-1=62, ark-2=63) and
refuses to guess. The rollout does want it exported.

`LIFT_HEIGHT=12.5` matches the v4 checkpoint. The older `05_dagger_pickplace.sh`
pins `14.0`, which belongs to the ACT era.

`unset TASK_INSTRUCTION` in the rollout command lets the profile supply the prompt.
Setting it overrides the profile for both paths, which will not match training.
