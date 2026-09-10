# Human DAgger with the calibrated feedback-v4 policy

The DAgger counterpart of `05_tau0vla_calibrated_rollout.sh`. The policy comes from
the same `arx-feedback-v4` server and the same profile table, so the task instruction,
experiment and protocol version have one definition rather than one per launcher.

Entry point: `tools/05_dagger_calibrated.sh`.

## Start a session

The rollout stack and a DAgger stack cannot coexist — DAgger starts and tracks
its own lift, arms and cameras, and refuses if anything else owns them. Stop the
current stack first:

```bash
cd /home/arx/ROS2_LIFT_Play_feedback-v4
./tools/04_safe_shutdown.sh
```

Then launch:

```bash
MODEL_PROFILE=all-t-feedback \
TASK_NAME=pickplace_t_dagger \
LIFT_HEIGHT=12.5 \
./tools/05_dagger_calibrated.sh
```

`ROS_DOMAIN_ID` is not passed on the command line: `05_human_dagger.sh` requires the
robot identity to come from `/etc/environment` (ark-1=62, ark-2=63) and refuses to
guess. CAN must already be UP; the launcher validates `can1/can3/can5` but never
brings them up. Run `./tools/00_can_up.sh` first if needed.

Controls once the UI is up: `R` park at the ready pose and start, `Space` human
takeover, `P` resume policy, `E` end the episode, `S`/`D`/`Q` review. The physical
foot trigger is enabled by default and toggles between policy and human.

## TASK_INSTRUCTION vs TASK_NAME

Two unrelated things that are easy to conflate.

**`TASK_INSTRUCTION`** is the natural-language prompt sent to the model. Do not set it
by hand — the profile provides it, which is what keeps DAgger and the rollout aligned.
`MODEL_PROFILE=all-t-feedback` yields:

```
Pick up the T-shaped part and place it in its designated position on the board.
```

Available profiles, all `arx-feedback-v4` on route `arx-lift2s-0908-all-joint-feedback-ft`:

| `MODEL_PROFILE` | object in the prompt |
| --- | --- |
| `all-t-feedback` | T-shaped part |
| `all-l-feedback` | L-shaped part |
| `all-blue-feedback` | blue box |
| `all-red-feedback` | red object |
| `all-banana-feedback` | banana |
| `all-circle-feedback` | circular part |

The legacy `t-feedback`, `blue-feedback`, `t-vr`, `blue-vr` profiles are `arx-calibrated-v3`.
The calibrated DAgger backend implements v4 only and refuses them explicitly.

Exporting `TASK_INSTRUCTION` does override the profile (`load_tau0vla_model_profile`
ends with `task=${TASK_INSTRUCTION:-${task}}`), but a prompt that does not match
training will change model behaviour. Prefer switching profile.

**`TASK_NAME`** is a recording label. It is written into each episode's metadata as
`task` and is never sent to the model. Episode filenames are `episode_N.hdf5`
regardless. Default is `pickplace_dagger`; set it per object, otherwise data for
different objects shares one label and cannot be separated later.

## Lift height

`LIFT_HEIGHT=12.5` matches the v4 checkpoint. The older `05_dagger_pickplace.sh`
pins `14.0`, which belongs to the ACT era — do not carry it over.

## Startup order and why calibration happens mid-launch

The gripper calibration artifact is bound to the `(pid, start_ticks)` of the arms it
was measured against, so it cannot be produced in advance or reused from a rollout.
But the frontend must be the sole command publisher before any arm spins up, so it
starts *before* the arms. The result:

```
505  lift
530  frontend            ← the calibrated policy worker starts here and blocks
549  arm_left
559  arm_right
598  wait for both arm status topics
607  gripper calibration ← THE GRIPPERS MOVE
     → the worker picks up the artifact, validates it, creates the session
614  cameras, VR serial
```

The worker waits up to `CALIBRATION_WAIT_S` (default 180 s) for the artifact and
fails with an explicit timeout if it never lands. `policy_ready` has no deadline on
the frontend side, so the delay is safe.

The artifact goes to the session directory as `gripper_calibration.json`, with its log
next to it. Calibration runs once per session, not once per episode. Human takeover
does not invalidate it: the fit measures the full-open reference position, which
takeover does not move.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PROFILE` | `all-t-feedback` | Selects prompt, experiment, route |
| `MODEL_SERVER_URL` | `http://192.168.50.2:8000` | Must be the direct-link server |
| `TASK_NAME` | `pickplace_dagger` | Recording label |
| `LIFT_HEIGHT` | `12.5` | Fixed lift command |
| `MAX_TIMESTEPS` | `7200` | Per-episode frame cap |
| `REPLAN_STEPS` | `15` | As in the rollout |
| `CHUNK_BLEND_STEPS` | `6` | As in the rollout |
| `GRIPPER_BLEND_STEPS` | `6` | As in the rollout |
| `ARM_EMA_ALPHA` | `0.6` | As in the rollout |
| `GRIPPER_EMA_ALPHA` | `1.0` | As in the rollout |
| `MAX_RESPONSE_AGE_MS` | `500` | As in the rollout |
| `CALIBRATION_WAIT_S` | `180` | Worker's wait for the artifact |
| `HUMAN_DAGGER_DATASET_DIR` | timestamped dir | Dataset destination |
| `TRIGGER_DEVICE` | `/dev/input/dagger_trigger` | Foot trigger |
| `COLOR_PROFILE` | `640x480x30` | 90 fps saturates the rclpy frontend |
| `PROFILE_DAGGER` | `1` | py-spy whole-session profile |

These defaults deliberately follow the calibrated rollout, **not** the older
`--policy-backend tau0vla` DAgger path, whose gripper handling is a different
mechanism (fixed thresholds plus a debounce, rather than a measured calibration).

## Foot trigger

Bound by `/etc/udev/rules.d/99-dagger-trigger.rules` on vendor `8088`, product `0015`,
serial `BF6EA7EC` → `/dev/input/dagger_trigger`, mode `0666`, so no root is needed.
The hardware is flashed to emit `KEY_1`; a reader thread turns that into operator key
`"1"`, which borrows the existing Space/P paths.

One button toggles, with `core.state` as the memory:

```
state POLICY → Space  (human takeover)
state HUMAN  → p      (resume policy)
```

Only those two settled states translate. A press during a handoff falls through
unmapped, so a double press cannot cancel an in-flight resume — aborting a resume
stays on the keyboard `Space`. If the device is absent the launcher prints a notice
and continues keyboard-only.

The device registers as an ordinary keyboard (`sysrq kbd` handlers), so pressing it
while another window has focus types `1` into that window.

## Not yet verified on hardware

Calibration under the DAgger stack has never been run. Two things can only be
confirmed on the robot:

* whether `RobotCmd` with `mode=5` is accepted for gripper commands by the DAgger
  X5Controller pair — the rollout drives `RobotStatus` on different topics;
* whether the frontend's concurrent HOLD publications conflict with the calibration
  commands.

Keep a hand on the emergency stop for the first run. Confirm handoff with the
keyboard `Space`/`P` before trying the trigger, so a failure is attributable.

The `quarantine` directory of the existing dataset contains
`policy_worker_HTTPError_409_Client_Error_Conflict` partials from the old tau0vla
backend. The calibrated worker reuses one session across `reset()` instead of
recreating it, so 409s should not recur — untested. Report one if it appears.

## Related

* `docs/TAU0VLA_CALIBRATED_V3.md` — the rollout path and the v4 protocol
* `docs/HUMAN_DAGGER_WORKFLOW.md` — DAgger mechanics, recording, review
* `tools/05_dagger_pickplace_v4.sh` — ACT-backend DAgger, for when no model server
  is available
