#!/usr/bin/env bash
# Human DAgger pickplace launcher for the feedback-v4 worktree.
#
# Why this exists instead of tools/05_dagger_pickplace.sh:
#   1. That script pins CKPT_DIR to "${repo_root}/act/weights/...", but weights
#      are gitignored (.gitignore:145 "**/weights"), so a linked worktree has no
#      act/weights at all. The checkpoints only live in the primary worktree.
#   2. It names act_ep000_024_seed0_epochs25000, which does not exist. The real
#      directory is act_ep000_024_seed0_full.
#
# POLICY_BACKEND is deliberately left at "act". The tau0vla DAgger backend
# cannot run against the current model server: human_dagger_tau0vla_policy.py
# imports the old tau0vla_protocol, which calls /api/v1/arx-lift2s/... and that
# path now returns 404. The server serves only /arx/v4 (arx-feedback-v4), whose
# request fields and calibrated_action_chunk mapping the old client lacks.
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# Checkpoints live outside any worktree. Override WEIGHTS_ROOT/CKPT_DIR to move.
WEIGHTS_ROOT=${WEIGHTS_ROOT:-/home/arx/ROS2_LIFT_Play/act/weights}

export TASK_NAME=${TASK_NAME:-pickplace_dagger}
export LIFT_HEIGHT=${LIFT_HEIGHT:-14.0}
export MAX_TIMESTEPS=${MAX_TIMESTEPS:-7200}
export CKPT_DIR=${CKPT_DIR:-"${WEIGHTS_ROOT}/act_ep000_024_seed0_full"}
export CKPT_NAME=${CKPT_NAME:-policy_best.ckpt}
export STATS_NAME=${STATS_NAME:-dataset_stats.pkl}

# 90fps floods the rclpy frontend (single-core GIL saturation: tick drops to
# ~53Hz, all-stream gaps spike together). The 60Hz control loop samples the
# latest frame, so 30fps is sufficient for teleop, recording and ACT.
export COLOR_PROFILE=${COLOR_PROFILE:-640x480x30}
export DEPTH_PROFILE=${DEPTH_PROFILE:-640x480x30}

# Fail before touching hardware if the checkpoint set is incomplete.
[[ -d "${CKPT_DIR}" ]] || { echo "Refused: CKPT_DIR is not a directory: ${CKPT_DIR}" >&2; exit 1; }
for f in "${CKPT_NAME}" "${STATS_NAME}"; do
  [[ -f "${CKPT_DIR}/${f}" ]] || { echo "Refused: missing ${CKPT_DIR}/${f}" >&2; exit 1; }
done

# 05_human_dagger.sh validates CAN but never brings it up, and it refuses to
# start when a competing stack owns the lift, arms or cameras. Report both here
# so the failure is legible before it dies deeper in the launcher.
for interface in can1 can3 can5; do
  ip -o link show dev "${interface}" 2>/dev/null \
    | awk -F'[<>]' '$2 ~ /(^|,)UP(,|$)/ { found=1 } END { exit !found }' \
    || { echo "Refused: ${interface} is not UP; run tools/00_can_up.sh first" >&2; exit 1; }
done

conflicts=$(pgrep -af '[/]arx_lift_controller/[l]ift_controller|[r]ealsense2_camera_node|[/]arx_x5_controller/[X]5Controller|[v]2_joint_control' || true)
if [[ -n "${conflicts}" ]]; then
  echo "Refused: another control or sensor stack owns the hardware:" >&2
  printf '%s\n' "${conflicts}" >&2
  echo "Human DAgger starts and tracks its own stack; stop the current one with" >&2
  echo "  ${repo_root}/tools/04_safe_shutdown.sh" >&2
  exit 1
fi

echo "Checkpoint: ${CKPT_DIR}/${CKPT_NAME}"
echo "Task=${TASK_NAME}, lift=${LIFT_HEIGHT}, max_timesteps=${MAX_TIMESTEPS}"
exec "${repo_root}/tools/05_human_dagger.sh" "$@"
