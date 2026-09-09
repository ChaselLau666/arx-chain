# Tau0VLA feedback-v4 and calibrated-v3 on ARX LIFT2s

The 0908 All checkpoint-30000 uses `arx-feedback-v4`, contract `arx-feedback-open-v1`, route `arx-lift2s-0908-all-joint-feedback-ft`. It uploads 14 joint feedback values and head/left-wrist/right-wrist JPEGs, without EEF feedback. The policy produces 30 steps at 30 Hz; both arm and gripper offsets are one uploaded frame. RTC is disabled. Legacy 0907 Blue/T feedback and VR profiles retain their v3 protocol.

## This deployment

The installed client directory is `/home/arx/ROS2_LIFT_Play_feedback-v4`. The connected robot is `ark-1`, `ROS_DOMAIN_ID=62`; its head/left/right camera serials are `260522272299`, `260422271992`, `260522274175`. The reviewed `ark-2`, domain `63` profile retains `260522275257`, `260422273222`, `260422272473`. Unknown hosts or mismatched ROS domains are rejected. Reused camera nodes must report the configured serials.

Model traffic uses `192.168.50.2 dev enp130s0 src 192.168.50.1`. Formal service port is `8000`; candidate validation uses `8001`. **The lift command defaults to `12.5`**. `LIFT_HEIGHT` is exported to hardware bring-up, target waiting, inference height checks and standalone return. The waiter/client verify `/lift fixed_height`, then require fresh stable feedback. Command and feedback coordinates have an offset and are not compared directly.

## Check without starting hardware

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4
export ROS_DOMAIN_ID=62
MODEL_PROFILE=all-blue-feedback MODEL_SERVER_URL=http://192.168.50.2:8000 \
LIFT_HEIGHT=12.5 ./tools/05_tau0vla_calibrated_rollout.sh --check
```

This only validates robot configuration and model health, route, protocol, contract, model identity and camera order in the contract. It creates no model session, ROS publisher, calibration or motion. It does not claim that physical cameras or the hardware stack are running. The same model preflight also runs before any hardware start in `--execute`.

For an observation-based dry-run, the hardware must already be running and an explicit valid calibration artifact must exist:

```bash
CALIBRATION_FILE=/home/arx/logs/tau0vla-calibrated/calibration_TIMESTAMP.json \
MODEL_PROFILE=all-blue-feedback LIFT_HEIGHT=12.5 \
./tools/05_tau0vla_calibrated_rollout.sh --dry-run
```

Dry-run does not start CAN/lift/arms/cameras, calibrate grippers, move to a pose, publish robot commands or return. Missing/invalid calibration, unavailable or stale observations, wrong height, camera skew or invalid protocol responses cause an explicit failure. Real observations are sent to the model and a client trace plus server NPZ records are created. Calibration identity, age and current full-open feedback are checked, but the artifact is not consumed.

## Execute on site

Clear the workspace and keep the emergency stop reachable before running:

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4
export ROS_DOMAIN_ID=62
MODEL_PROFILE=all-blue-feedback \
MODEL_SERVER_URL=http://192.168.50.2:8000 \
LIFT_HEIGHT=12.5 \
./tools/05_tau0vla_calibrated_rollout.sh --execute
```

This command starts/reuses CAN, lift, both v2 arms and three cameras in sequence; performs fresh full two-gripper calibration; moves to the fixed training initial pose and verifies arrival; benchmarks the model with three warmups and 30 requests; then publishes policy commands. These motion transitions are non-interactive. Each execute rollout requires a new one-use calibration. A per-robot lock prevents overlapping rollouts, and both execute/dry-run reject existing policy, calibration or return processes before starting hardware or creating a model session.

Available v4 profiles and exact training text:

| Profile | Task instruction |
|---|---|
| `all-l-feedback` | Pick up the L-shaped part and place it in its designated position on the board. |
| `all-t-feedback` | Pick up the T-shaped part and place it in its designated position on the board. |
| `all-banana-feedback` | Pick up the banana and place it in its designated position on the board. |
| `all-red-feedback` | Pick up the red object and place it in its designated position on the board. |
| `all-blue-feedback` | Pick up the blue box and place it in its designated position on the board. |
| `all-circle-feedback` | Pick up the circular part and place it in its designated position on the board. |

V3 profiles are `blue-feedback`, `t-feedback`, `blue-vr`, `t-vr`; choose the matching deployed route. The legacy low-level launcher accepts `PROTOCOL_VERSION=arx-feedback-v4` when supplying experiment and task text manually.

Defaults: replan every 15 steps, blend 6 arm/gripper steps, arm EMA `0.6`, gripper EMA `1.0`, maximum response age `500 ms`. Joint actions are never clipped. Existing measured gripper endpoint saturation limits remain enforced and recorded. Invalid mapping, stale observations/responses, session ordering errors or buffer exhaustion stop publication. Calibration artifacts, logs and JSONL traces live in `/home/arx/logs/tau0vla-calibrated/`; trace metadata identifies model, session, calibration and height for matching server NPZ records.

## Stop and return

Press `Ctrl-C` once during normal policy execution to stop policy commands and run the guarded return to the fixed initial pose; reaching `MAX_STEPS` does the same. Press `Ctrl-C` again during a return to abort further motion. Unexpected protocol, calibration or sensor failures stop commands and do not automatically move the robot.

For a separate return after the policy process has stopped:

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4
ROS_DOMAIN_ID=62 LIFT_HEIGHT=12.5 ./tools/06_tau0vla_return_fixed.sh --execute
```

This checks the robot/domain, calibration age and boot/controller identity, expected lift parameter and fresh stable height before moving. It uses the newest calibration without requiring the model server or trace. For an artifact/target inspection without movement, omit `--execute`. After return, start the one-command rollout again to create fresh calibration.

Hardware shutdown, when needed, remains `./tools/04_safe_shutdown.sh`; inspect its existing on-site procedure before use. This deployment does not certify closed-loop performance: real observation quality, gripper calibration and task execution must be checked at the first attended rollout.
