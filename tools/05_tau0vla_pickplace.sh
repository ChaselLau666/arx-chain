#!/usr/bin/env bash
# One-command Tau0VLA-backed Human DAgger launch for the pickplace task.
# Same frontend, state machine, takeover and recording as 05_dagger_pickplace;
# only the policy worker differs: actions come from the Tau0VLA server over
# the dedicated direct link instead of a local ACT checkpoint.
#
# The robot identity (ROS_DOMAIN_ID) still comes from /etc/environment.
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POLICY_BACKEND=tau0vla
export MODEL_SERVER_URL=${MODEL_SERVER_URL:-http://192.168.50.2:8000}
export TASK_INSTRUCTION=${TASK_INSTRUCTION:-"Pick up the handle and place it into the tray."}
export TASK_NAME=${TASK_NAME:-pickplace_tau0vla_dagger}
export LIFT_HEIGHT=${LIFT_HEIGHT:-14.0}
export MAX_TIMESTEPS=${MAX_TIMESTEPS:-7200}

# Same camera policy as the ACT pickplace launcher: 30fps color only.
export COLOR_PROFILE=${COLOR_PROFILE:-640x480x30}
export DEPTH_PROFILE=${DEPTH_PROFILE:-640x480x30}

exec "${script_dir}/05_human_dagger.sh" "$@"
