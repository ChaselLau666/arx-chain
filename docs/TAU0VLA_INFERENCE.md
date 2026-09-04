# Tau0VLA remote inference on ARX LIFT2s

Tau0VLA runs on the dedicated Ethernet link at `192.168.77.1:8000`; ARX1 keeps ROS subscriptions and arm
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
MODEL_SERVER_URL=http://192.168.77.1:8000 \
  ./03_tau0vla_inference.sh
```

The client performs three warmups and thirty measured requests, then selects a
replanning interval from the measured p99 latency. Override it when comparing
chunk schedules:

```bash
REPLAN_STEPS=10 ./03_tau0vla_inference.sh
```

The default `CHUNK_BLEND_STEPS=6` performs a smoothstep transition between
time-aligned old and new chunk tails instead of replacing the active plan in
one control tick. `ARM_EMA_ALPHA=1.0` and `GRIPPER_EMA_ALPHA=1.0` leave EMA
disabled. For a second-stage arm-only smoothing trial, set
`ARM_EMA_ALPHA=0.4`; keep the gripper at `1.0` unless its timing is separately
validated. Every run writes a JSONL command/feedback trace beside the client
log; summarize it with `python act/tau0vla_trace.py TRACE.jsonl`.

The launcher refuses a direct URL unless `ip route get 192.168.77.1` resolves
through `enp130s0` with source `192.168.77.2`. Wi-Fi is an explicit diagnostic
fallback only: it requires both a Wi-Fi URL and
`ALLOW_NON_DIRECT_MODEL_SERVER=1`.

For physical execution:

```bash
./03_tau0vla_inference.sh --execute
```

No command publisher is created until the operator types the complete phrase
shown by the client. On protocol failure or buffer starvation, publication is
paused rather than repeating an old trajectory. Stop with Ctrl+C; use
`04_safe_shutdown.sh` for the normal lower-and-shutdown workflow.
