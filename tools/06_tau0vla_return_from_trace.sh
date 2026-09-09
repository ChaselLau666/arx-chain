#!/bin/bash
# Recover the fixed training initial pose using the newest rollout calibration.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/tau0vla_robot_profile.sh"
load_tau0vla_robot_profile
: "${LOG_DIR:=/home/arx/logs/tau0vla-calibrated}"

if pgrep -f '[t]au0vla_.*client.py|[t]au0vla_calibrate_gripper.py' >/dev/null; then
  echo "Refused: policy/calibration process is still active." >&2
  exit 1
fi
trace=${TRACE_FILE:-$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'trace_*.jsonl' \
  -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)}
if [[ -z "${trace}" || ! -f "${trace}" ]]; then
  echo "Refused: no calibrated trace found." >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate act
set -u

cd "${repo_root}/act"
exec python tau0vla_return_from_trace.py --expected-height "${LIFT_HEIGHT}" --trace "${trace}" "$@"
