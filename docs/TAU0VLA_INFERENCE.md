# Tau0VLA remote inference on ARX LIFT2s

Tau0VLA runs on `192.168.31.83:8000`; ARX1 keeps ROS subscriptions and arm
publication local. This is an independent inference path and does not modify
`act/inference.py` or the ACT checkpoint path.

## Contract

- cameras: `head`, `left_wrist`, `right_wrist`, transported as their original
  ROS compressed JPEG bytes;
- state/action: 14D
  `left_j0..j5, left_gripper, right_j0..j5, right_gripper`;
- control rate: 30 Hz;
- server output: 30-step `state_t_plus_1` action chunk;
- base motion: disabled.

The client rejects malformed, non-finite, stale, timed-out, or out-of-session
responses. It deliberately does not apply joint limits, per-step clipping, or
left-arm replacement. In execute mode the model's finite 14D values are sent
directly to the two arm command topics.

## Start

The lift, exactly two operator-started `v2_joint_control` processes, and three
RealSense processes must be live. Starting the arm controllers can itself move
the robot before a model publisher exists, so clear the workspace and make the
emergency stop reachable first.

Dry-run is the default:

```bash
cd /home/arx/ROS2_LIFT_Play/tools
MODEL_SERVER_URL=http://192.168.31.83:8000 \
  ./03_tau0vla_inference.sh
```

The client performs three warmups and thirty measured requests, then selects a
replanning interval from the measured p99 latency. Override it when comparing
chunk schedules:

```bash
REPLAN_STEPS=10 ./03_tau0vla_inference.sh
```

For physical execution:

```bash
./03_tau0vla_inference.sh --execute
```

No command publisher is created until the operator types the complete phrase
shown by the client. On protocol failure or buffer starvation, publication is
paused rather than repeating an old trajectory. Stop with Ctrl+C; use
`04_safe_shutdown.sh` for the normal lower-and-shutdown workflow.
