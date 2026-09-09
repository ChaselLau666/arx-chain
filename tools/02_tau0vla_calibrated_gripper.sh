#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/tau0vla_robot_profile.sh"
load_tau0vla_robot_profile
: "${LOG_DIR:=/home/arx/logs/tau0vla-calibrated}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

if pgrep -f '[t]au0vla_.*client.py' >/dev/null; then
  echo "Refused: a Tau0VLA client is already running." >&2
  exit 1
fi
for interface in can1 can3 can5; do
  if ! ip link show "${interface}" 2>/dev/null | grep -q 'UP'; then
    echo "Refused: ${interface} is not UP." >&2
    exit 1
  fi
done
if ! ros2 node list 2>/dev/null | grep -qx '/lift'; then
  echo "Refused: /lift is not running." >&2
  exit 1
fi
arm_pids=$(pgrep -f '/arx_x5_controller/[X]5Controller.*v2_joint_control.yaml' || true)
if [[ $(wc -w <<<"${arm_pids}") -ne 2 ]]; then
  echo "Refused: expected exactly two v2_joint_control processes." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
stamp=$(date +%Y%m%d_%H%M%S)
output="${LOG_DIR}/calibration_${stamp}.json"
log="${LOG_DIR}/calibration_${stamp}.log"
output_q=$(printf '%q' "${output}")
log_q=$(printf '%q' "${log}")
extra_q=""
for argument in "$@"; do
  extra_q+=" $(printf '%q' "${argument}")"
done

gnome-terminal --title="tau0vla-calibrated-gripper" -- bash -ic \
  "set -o pipefail; cd $(printf '%q' "${repo_root}/act"); conda activate act; \
   python tau0vla_calibrate_gripper.py --output ${output_q}${extra_q} \
   2>&1 | tee -a ${log_q}; exec bash"

echo "Calibration terminal launched."
echo "Calibration artifact (created only on success): ${output}"
echo "Calibration log: ${log}"
