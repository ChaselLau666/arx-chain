# Tau0VLA remote inference on ARX LIFT2s

Tau0VLA runs on the dedicated Ethernet link at `192.168.50.2:8000`; ARX1 keeps ROS subscriptions and arm
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
responses. It deliberately does not apply arm joint limits, per-step clipping,
or left-arm replacement. The tool-yipan launcher treats each gripper as a
binary actuator and debounces its state intent as described below.

## Start

The lift, exactly two operator-started `v2_joint_control` processes, and three
RealSense processes must be live. Starting the arm controllers can itself move
the robot before a model publisher exists, so clear the workspace and make the
emergency stop reachable first.

Dry-run is the default:

```bash
cd /home/arx/ROS2_LIFT_Play/tools
MODEL_SERVER_URL=http://192.168.50.2:8000 \
  ./03_tau0vla_inference.sh
```

The client performs three warmups and thirty measured requests, then selects a
replanning interval from the measured p99 latency. Override it when comparing
chunk schedules:

```bash
REPLAN_STEPS=10 ./03_tau0vla_inference.sh
```

The default `CHUNK_BLEND_STEPS=6` performs a smoothstep transition between
time-aligned old and new arm tails. Binary grippers do not use that linear
cross-fade (`GRIPPER_BLEND_STEPS=0`): interpolating two disagreeing open/close
plans can create extra threshold crossings.

The tool-yipan defaults are `ARM_EMA_ALPHA=0.6`,
`GRIPPER_EMA_ALPHA=0.6`, and `GRIPPER_DEBOUNCE_FRAMES=12`. A gripper target at
or below `-2.1` votes for the low endpoint (`-3.384`); a target at or above
`-1.05` votes for the high endpoint (`0.0`); the middle gap retains the current
state. Twelve consecutive opposite votes are required before switching. EMA
then smooths that one accepted transition. Set `GRIPPER_DEBOUNCE_FRAMES=0` to
restore the exact model-valued path for an explicit A/B test.

The default task text matches the single-task training data exactly:
`Pick up the tool and place it into the tray.` Every run writes a JSONL
command/feedback trace beside the client log; summarize it with
`python act/tau0vla_trace.py TRACE.jsonl`.

The launcher refuses a direct URL unless `ip route get 192.168.50.2` resolves
through `enp130s0` with source `192.168.50.1`. Wi-Fi is an explicit diagnostic
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
