#!/usr/bin/env bash
# Human DAgger with the calibrated arx-feedback-v4 policy backend.
#
# This is the DAgger counterpart of 05_tau0vla_calibrated_rollout.sh. It reuses
# the rollout profile table so the task instruction, experiment and protocol
# version come from one place instead of being retyped per run.
#
# Note the differences from the rollout, which are structural, not cosmetic:
#   * the gripper calibration happens inside 05_human_dagger.sh, after it starts
#     its own arms -- the artifact is bound to those PIDs, so it cannot be
#     produced in advance or reused from a rollout;
#   * defaults follow the rollout (REPLAN_STEPS=15, ARM_EMA_ALPHA=0.6,
#     GRIPPER_EMA_ALPHA=1.0) rather than the older tau0vla DAgger backend.
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/tau0vla_robot_profile.sh"
load_tau0vla_robot_profile
: "${MODEL_PROFILE:=all-t-feedback}"
: "${MODEL_SERVER_URL:=http://192.168.50.2:8000}"
load_tau0vla_model_profile

export POLICY_BACKEND=calibrated
export MODEL_SERVER_URL
export TASK_INSTRUCTION="${task}"
export EXPERIMENT="${experiment}"
export PROTOCOL_VERSION="${protocol_version}"

export TASK_NAME=${TASK_NAME:-pickplace_dagger}
export LIFT_HEIGHT=${LIFT_HEIGHT:-12.5}
export MAX_TIMESTEPS=${MAX_TIMESTEPS:-7200}

# Match the calibrated rollout, not the old tau0vla DAgger defaults.
export REPLAN_STEPS=${REPLAN_STEPS:-15}
export CHUNK_BLEND_STEPS=${CHUNK_BLEND_STEPS:-6}
export GRIPPER_BLEND_STEPS=${GRIPPER_BLEND_STEPS:-6}
export ARM_EMA_ALPHA=${ARM_EMA_ALPHA:-0.6}
export GRIPPER_EMA_ALPHA=${GRIPPER_EMA_ALPHA:-1.0}
export MAX_RESPONSE_AGE_MS=${MAX_RESPONSE_AGE_MS:-500}

# 90fps floods the rclpy frontend (single-core GIL saturation: tick drops to
# ~53Hz, all-stream gaps spike together). The 60Hz control loop samples the
# latest frame, so 30fps is sufficient for teleop, recording and ACT.
export COLOR_PROFILE=${COLOR_PROFILE:-640x480x30}
export DEPTH_PROFILE=${DEPTH_PROFILE:-640x480x30}

if [[ "${protocol_version}" != arx-feedback-v4 ]]; then
  echo "Refused: MODEL_PROFILE=${MODEL_PROFILE} selects ${protocol_version}." >&2
  echo "The calibrated DAgger backend only implements arx-feedback-v4; use an all-*-feedback profile." >&2
  exit 1
fi

# 05_human_dagger.sh validates CAN but never brings it up.
for interface in can1 can3 can5; do
  ip -o link show dev "${interface}" 2>/dev/null \
    | awk -F"[<>]" "\$2 ~ /(^|,)UP(,|\$)/ { found=1 } END { exit !found }" \
    || { echo "Refused: ${interface} is not UP; run tools/00_can_up.sh first" >&2; exit 1; }
done

conflicts=$(pgrep -af "[/]arx_lift_controller/[l]ift_controller|[r]ealsense2_camera_node|[/]arx_x5_controller/[X]5Controller|[v]2_joint_control" || true)
if [[ -n "${conflicts}" ]]; then
  echo "Refused: another control or sensor stack owns the hardware:" >&2
  printf "%s\n" "${conflicts}" >&2
  echo "Human DAgger starts and tracks its own stack; stop the current one with" >&2
  echo "  ${repo_root}/tools/04_safe_shutdown.sh" >&2
  exit 1
fi

echo "Profile ${MODEL_PROFILE}: route=${expected_route}, experiment=${experiment}, protocol=${protocol_version}"
echo "Task instruction: ${task}"
echo "Task=${TASK_NAME}, lift=${LIFT_HEIGHT}, max_timesteps=${MAX_TIMESTEPS}"
exec "${script_dir}/05_human_dagger.sh" "$@"
