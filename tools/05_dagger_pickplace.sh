#!/usr/bin/env bash
# One-command Human DAgger launch for the pickplace task on this robot.
# All run configuration is pinned here; the robot identity (ROS_DOMAIN_ID)
# still comes from /etc/environment as required by 05_human_dagger.sh.
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export TASK_NAME=pickplace_dagger
export LIFT_HEIGHT=15.5
export CKPT_DIR="${repo_root}/act/weights/act_ep000_024_seed0_epochs25000"
export CKPT_NAME=policy_best.ckpt
export STATS_NAME=dataset_stats.pkl

exec "${script_dir}/05_human_dagger.sh" "$@"
