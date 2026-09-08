# Tau0VLA calibrated-v3 on ARX LIFT2s

This path is only for Blue/T `joint-feedback` and `joint-vr` checkpoints. The existing `03_tau0vla_inference.sh` remains the v1 client.

## Preconditions

- Set `ROS_DOMAIN_ID=63` on ark-2.
- Bring up `can1/can3/can5`, `/lift` at `fixed_height=15.5`, and exactly two `v2_joint_control` arm processes.
- Start the three RealSense cameras sequentially and verify one compressed-image publisher per camera.
- Verify the direct route is `192.168.50.2 dev enp130s0 src 192.168.50.1`.

## Mandatory calibration

Run once before every execute rollout:

```bash
export ROS_DOMAIN_ID=63
cd /home/arx/ROS2_LIFT_Play/tools
./02_tau0vla_calibrated_gripper.sh --execute
```

Type `CALIBRATE BOTH GRIPPERS` only with hands clear. The tool holds the six arm joints, moves one gripper at a time through `-3.39,-2.55,-1.70,-0.85,0.0`, returns both to `-3.39`, validates the feedback fit, and writes an immutable JSON artifact. A dry-run may inspect the artifact without consuming it; the first execute session marks it consumed.

## Policy dry-run and execute

Set the checkpoint experiment and its exact task text:

```bash
export ROS_DOMAIN_ID=63
export CALIBRATED_EXPERIMENT=joint-feedback  # or joint-vr
export TASK_INSTRUCTION='Pick up the blue box and place it in its designated position on the board.'
cd /home/arx/ROS2_LIFT_Play/tools
./03_tau0vla_calibrated_inference.sh --max-steps 900
```

For T use `Pick up the T-shaped part and place it in its designated position on the board.`. After dry-run succeeds, reuse the same still-valid calibration once:

```bash
./03_tau0vla_calibrated_inference.sh --execute --max-steps 3600
```

Type `EXECUTE CALIBRATED TAU0VLA` only after confirming the model ID, calibration ID, clear workspace, and reachable emergency stop. Arm actions are not clipped; invalid intent, calibration, mapping, response age, session ordering, or finite-value checks stop publication.

For `joint-feedback`, a flow sample may fall slightly beyond a demonstrated gripper endpoint. The client accepts at most `0.075` command units of endpoint error and saturates it to the measured command endpoint; the extrapolated command is never published. Larger errors still terminate the session. Every saturation records its count and maximum excess in the response log and trace summary. Arm actions are never clipped. `joint-vr` intent remains strictly constrained to `[0,1]`.

Logs, immutable calibration artifacts, JSONL traces, summaries and plots are written under `/home/arx/logs/tau0vla-calibrated/`.

## One-command workflow

After deployment, the same command is used for every rollout. It idempotently starts or reuses the direct network, CAN, lift, v2 arms and sequential cameras; performs a new full calibration; moves both arms to the fixed pose used at the beginning of the 0907 training demonstrations; runs the selected policy; and offers a guarded return to that same fixed pose:

```bash
export ROS_DOMAIN_ID=63
cd /home/arx/ROS2_LIFT_Play/tools
MODEL_PROFILE=blue-feedback \
MODEL_SERVER_URL=http://192.168.50.2:8000 \
./05_tau0vla_calibrated_rollout.sh --execute
```

During validation use candidate port `8001`. Profiles are `blue-feedback`, `t-feedback`, `blue-vr`, and `t-vr`. Before inference, `MOVE TO FIXED INITIAL POSE` is required and arrival is checked against `act/data/tau0vla_calibrated_ready.yaml`. Grippers remain at the full-open feedback measured by the current calibration, with their commands obtained through the fitted inverse rather than by treating feedback as command coordinates.

The one-command execute path is non-interactive: it starts/reuses the hardware stack, performs the full two-gripper calibration, moves to the fixed training pose, starts policy publication, and returns to the fixed pose when either the operator presses `Ctrl-C` once or `MAX_STEPS` is reached. It prints `AUTO-CONFIRM` at each movement transition instead of reading confirmation text. The workspace must therefore be clear and the emergency stop reachable before the command is launched. A second `Ctrl-C` during return aborts movement.

Protocol, calibration, sensor, mapping, response-age, and other unexpected errors remain fail-stop and never trigger automatic movement. After such an error, or if the policy process is externally killed, use the independent return command below. It locates the newest calibration from the current boot, rejects changed controller identities or an artifact older than 15 minutes, and does not require the rollout trace or model server:

```bash
./06_tau0vla_return_fixed.sh --execute
```

The standalone command is also non-interactive and begins the fixed-pose move immediately after all identity and age checks pass. After a verified return, rerun the rollout command; it creates a fresh one-use calibration.
