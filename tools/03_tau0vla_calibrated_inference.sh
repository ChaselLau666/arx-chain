#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/tau0vla_robot_profile.sh"
load_tau0vla_robot_profile
: "${MODEL_SERVER_URL:=http://192.168.50.2:8000}"
: "${DIRECT_SERVER_IP:=192.168.50.2}"
: "${DIRECT_INTERFACE:=enp130s0}"
: "${DIRECT_CLIENT_IP:=192.168.50.1}"
: "${PROTOCOL_VERSION:=arx-calibrated-v3}"
: "${CALIBRATED_EXPERIMENT:?Set CALIBRATED_EXPERIMENT=joint-feedback or joint-vr}"
: "${TASK_INSTRUCTION:?Set the exact checkpoint task instruction}"
: "${LIFT_HEIGHT:=12.5}"
: "${REPLAN_STEPS:=15}"
: "${CHUNK_BLEND_STEPS:=6}"
: "${GRIPPER_BLEND_STEPS:=6}"
: "${ARM_EMA_ALPHA:=0.6}"
: "${GRIPPER_EMA_ALPHA:=1.0}"
: "${MAX_RESPONSE_AGE_MS:=500}"
: "${LOG_DIR:=/home/arx/logs/tau0vla-calibrated}"

if [[ "${CALIBRATED_EXPERIMENT}" != joint-feedback && "${CALIBRATED_EXPERIMENT}" != joint-vr ]]; then
  echo "Refused: CALIBRATED_EXPERIMENT must be joint-feedback or joint-vr." >&2
  exit 1
fi
if [[ -z "${CALIBRATION_FILE:-}" ]]; then
  CALIBRATION_FILE=$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'calibration_*.json' \
    ! -name '*.consumed' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
fi
if [[ -z "${CALIBRATION_FILE:-}" || ! -f "${CALIBRATION_FILE}" ]]; then
  echo "Refused: no calibration artifact; run 02_tau0vla_calibrated_gripper.sh --execute." >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

server_host=${MODEL_SERVER_URL#*://}
server_host=${server_host%%[:/]*}
if [[ "${server_host}" != "${DIRECT_SERVER_IP}" ]]; then
  echo "Refused: calibrated execution only supports the reviewed direct server IP." >&2
  exit 1
fi
route_info=$(ip route get "${DIRECT_SERVER_IP}" 2>/dev/null || true)
if [[ "${route_info}" != *"dev ${DIRECT_INTERFACE}"* || "${route_info}" != *"src ${DIRECT_CLIENT_IP}"* ]]; then
  echo "Refused: direct model route is not active: ${route_info:-unavailable}" >&2
  exit 1
fi
if ! curl --fail --silent --show-error --max-time 3 "${MODEL_SERVER_URL}/health" >/dev/null; then
  echo "Refused: calibrated Tau0VLA server is not ready." >&2
  exit 1
fi
if ! ros2 node list 2>/dev/null | grep -qx '/lift'; then
  echo "Refused: /lift is not running." >&2
  exit 1
fi
for interface in can1 can3 can5; do
  if ! ip link show "${interface}" 2>/dev/null | grep -q 'UP'; then
    echo "Refused: ${interface} is not UP." >&2
    exit 1
  fi
done
if pgrep -f '[t]au0vla_.*client.py' >/dev/null; then
  echo "Refused: another Tau0VLA client is already running." >&2
  exit 1
fi
arm_pids=$(pgrep -f '/arx_x5_controller/[X]5Controller' || true)
v2_pids=$(pgrep -f '/arx_x5_controller/[X]5Controller.*v2_joint_control.yaml' || true)
if [[ $(wc -w <<<"${arm_pids}") -ne 2 || $(wc -w <<<"${v2_pids}") -ne 2 ]]; then
  echo "Refused: expected exactly two operator-started v2_joint_control processes." >&2
  exit 1
fi

publisher_count() {
  ros2 topic info "$1" 2>/dev/null | awk '/^Publisher count:/{print $3; found=1} END{if (!found) print 0}'
}
for topic in \
  /arm_slave_l_status /arm_slave_r_status \
  /camera/camera_h/color/image_rect_raw/compressed \
  /camera/camera_l/color/image_rect_raw/compressed \
  /camera/camera_r/color/image_rect_raw/compressed
do
  if [[ $(publisher_count "${topic}") -lt 1 ]]; then
    echo "Refused: no publisher for ${topic}. Start cameras sequentially before calibrated inference." >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"
stamp=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/client_${stamp}.log"
trace_file="${LOG_DIR}/trace_${stamp}.jsonl"
quote() { printf '%q' "$1"; }
extra_q=""
for argument in "$@"; do
  extra_q+=" $(quote "${argument}")"
done

gnome-terminal --title="tau0vla-calibrated-inference" -- bash -ic \
  "set -o pipefail; cd $(quote "${repo_root}/act"); conda activate act; \
   python tau0vla_calibrated_client.py \
     --server-url $(quote "${MODEL_SERVER_URL}") \
     --experiment $(quote "${CALIBRATED_EXPERIMENT}") \
     --protocol-version $(quote "${PROTOCOL_VERSION}") \
     --task-instruction $(quote "${TASK_INSTRUCTION}") \
     --calibration-file $(quote "${CALIBRATION_FILE}") \
     --expected-height $(quote "${LIFT_HEIGHT}") \
     --replan-steps $(quote "${REPLAN_STEPS}") \
     --chunk-blend-steps $(quote "${CHUNK_BLEND_STEPS}") \
     --gripper-blend-steps $(quote "${GRIPPER_BLEND_STEPS}") \
     --arm-ema-alpha $(quote "${ARM_EMA_ALPHA}") \
     --gripper-ema-alpha $(quote "${GRIPPER_EMA_ALPHA}") \
     --max-response-age-ms $(quote "${MAX_RESPONSE_AGE_MS}") \
     --trace-path $(quote "${trace_file}") \
     --log-path $(quote "${log_file}")${extra_q}; exec bash"

echo "Calibrated client launched."
echo "Calibration: ${CALIBRATION_FILE}"
echo "Client log: ${log_file}"
echo "Trace: ${trace_file}"
