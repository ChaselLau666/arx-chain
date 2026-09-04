#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}
: "${MODEL_SERVER_URL:=http://192.168.77.1:8000}"
: "${DIRECT_SERVER_IP:=192.168.77.1}"
: "${DIRECT_INTERFACE:=enp130s0}"
: "${DIRECT_CLIENT_IP:=192.168.77.2}"
: "${ALLOW_NON_DIRECT_MODEL_SERVER:=0}"
: "${TASK_INSTRUCTION:=Pick up the handle and place it into the tray.}"
: "${LIFT_HEIGHT:=15.5}"
: "${REPLAN_STEPS:=auto}"
: "${CHUNK_BLEND_STEPS:=6}"
: "${ARM_EMA_ALPHA:=1.0}"
: "${GRIPPER_EMA_ALPHA:=1.0}"
: "${LOG_DIR:=/home/arx/logs/tau0vla}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

server_host=${MODEL_SERVER_URL#*://}
server_host=${server_host%%[:/]*}
if [[ "${server_host}" != "${DIRECT_SERVER_IP}" ]]; then
  if [[ "${ALLOW_NON_DIRECT_MODEL_SERVER}" != "1" ]]; then
    echo "Refused: MODEL_SERVER_URL=${MODEL_SERVER_URL} is not the reviewed direct-link server." >&2
    echo "Set ALLOW_NON_DIRECT_MODEL_SERVER=1 only for an explicit Wi-Fi diagnostic." >&2
    exit 1
  fi
else
  route_info=$(ip route get "${DIRECT_SERVER_IP}" 2>/dev/null || true)
  if [[ "${route_info}" != *"dev ${DIRECT_INTERFACE}"* || "${route_info}" != *"src ${DIRECT_CLIENT_IP}"* ]]; then
    echo "Refused: direct-link route is not active." >&2
    echo "Expected: ${DIRECT_SERVER_IP} dev ${DIRECT_INTERFACE} src ${DIRECT_CLIENT_IP}" >&2
    echo "Actual: ${route_info:-unavailable}" >&2
    exit 1
  fi
  echo "Direct model route verified: ${route_info}"
fi

if ! curl --fail --silent --show-error --max-time 3 "${MODEL_SERVER_URL}/health" >/dev/null; then
  echo "Refused: Tau0VLA server is not ready at ${MODEL_SERVER_URL}." >&2
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

publisher_count() {
  local topic=$1
  local info
  info=$(ros2 topic info "${topic}" 2>/dev/null || true)
  awk '/^Publisher count:/{print $3; found=1} END{if (!found) print 0}' <<<"${info}"
}

wait_for_publishers() {
  local label=$1
  shift
  local topic count all_ready
  for _ in $(seq 1 30); do
    all_ready=true
    for topic in "$@"; do
      count=$(publisher_count "${topic}")
      if ! [[ "${count}" =~ ^[0-9]+$ ]] || (( count < 1 )); then
        all_ready=false
        break
      fi
    done
    if [[ "${all_ready}" == true ]]; then
      echo "${label}: all publishers are live."
      return 0
    fi
    sleep 1
  done
  echo "Refused: ${label} publishers did not become live within 30 seconds." >&2
  return 1
}

arm_feedback_topics=(/arm_slave_l_status /arm_slave_r_status)
arm_pids=$(pgrep -f '/arx_x5_controller/[X]5Controller' || true)
arm_processes=$(wc -w <<<"${arm_pids}")
v2_arm_pids=$(pgrep -f '/arx_x5_controller/[X]5Controller.*v2_joint_control.yaml' || true)
v2_arm_processes=$(wc -w <<<"${v2_arm_pids}")
if [[ ${arm_processes} -ne 2 || ${v2_arm_processes} -ne 2 ]]; then
  echo "Refused: expected exactly two operator-started v2_joint_control processes; " \
       "found ${arm_processes} arm processes (${v2_arm_processes} v2)." >&2
  echo "Start them manually only after clearing the workspace and making the emergency stop reachable." >&2
  exit 1
fi
wait_for_publishers "running inference arms" "${arm_feedback_topics[@]}"

camera_image_topics=(
  /camera/camera_h/color/image_rect_raw/compressed
  /camera/camera_l/color/image_rect_raw/compressed
  /camera/camera_r/color/image_rect_raw/compressed
)
camera_pids=$(pgrep -f '/realsense2_camera/[r]ealsense2_camera_node' || true)
camera_processes=$(wc -w <<<"${camera_pids}")
if [[ ${camera_processes} -eq 0 ]]; then
  gnome-terminal --title="inference-cameras" -- bash -ic \
    "cd ${repo_root}/realsense; ./realsense.sh; exec bash"
  wait_for_publishers "cameras" "${camera_image_topics[@]}"
elif [[ ${camera_processes} -ne 3 ]]; then
  echo "Refused: expected zero or three RealSense processes; found ${camera_processes}." >&2
  exit 1
else
  wait_for_publishers "running cameras" "${camera_image_topics[@]}"
fi

server_q=$(printf '%q' "${MODEL_SERVER_URL}")
task_q=$(printf '%q' "${TASK_INSTRUCTION}")
height_q=$(printf '%q' "${LIFT_HEIGHT}")
replan_q=$(printf '%q' "${REPLAN_STEPS}")
blend_q=$(printf '%q' "${CHUNK_BLEND_STEPS}")
arm_ema_q=$(printf '%q' "${ARM_EMA_ALPHA}")
gripper_ema_q=$(printf '%q' "${GRIPPER_EMA_ALPHA}")
mkdir -p "${LOG_DIR}"
log_file="${LOG_DIR}/client_$(date +%Y%m%d_%H%M%S).log"
trace_file="${LOG_DIR}/trace_$(date +%Y%m%d_%H%M%S).jsonl"
log_q=$(printf '%q' "${log_file}")
trace_q=$(printf '%q' "${trace_file}")
extra_args_q=""
for argument in "$@"; do
  extra_args_q+=" $(printf '%q' "${argument}")"
done

gnome-terminal --title="tau0vla-inference" -- bash -ic \
  "set -o pipefail; cd ${repo_root}/act; conda activate act; python tau0vla_client.py \
    --server-url ${server_q} --task-instruction ${task_q} \
    --expected-height ${height_q} --replan-steps ${replan_q} \
    --chunk-blend-steps ${blend_q} --arm-ema-alpha ${arm_ema_q} \
    --gripper-ema-alpha ${gripper_ema_q} --trace-path ${trace_q}${extra_args_q} \
    2>&1 | tee -a ${log_q}; exec bash"

echo "Tau0VLA inference terminal launched. Default mode is DRY-RUN; no action publisher is created."
echo "Client log: ${log_file}"
echo "Action trace: ${trace_file}"
