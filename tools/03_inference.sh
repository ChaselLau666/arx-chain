#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Domain must come from the machine (/etc/environment); a default would let
# this script silently join another robot's graph on a shared LAN.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. The robot identity lives in /etc/environment (ark-1=62, ark-2=63); refusing to guess which robot to talk to}"
: "${CKPT_DIR:?Set CKPT_DIR to the selected ACT run directory}"
: "${CKPT_NAME:=policy_best.ckpt}"
: "${LIFT_HEIGHT:=15.5}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

if ! ros2 node list 2>/dev/null | grep -qx '/lift'; then
  echo "Refused: /lift is not running. Start body only while the platform is at a safe low position." >&2
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

has_publisher() {
  local count
  count=$(publisher_count "$1")
  [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 ))
}

wait_for_publishers() {
  local label=$1
  shift
  local topic all_ready
  for _ in $(seq 1 30); do
    all_ready=true
    for topic in "$@"; do
      if ! has_publisher "${topic}"; then
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
if [[ ${arm_processes} -eq 0 ]]; then
  echo "Refused: v2 inference arm controllers are not running." >&2
  echo "Start v2_joint_control.launch.py manually with the workspace clear and emergency stop reachable;" >&2
  echo "the SDK calls arx_x(...) during initialization and may move the arms before any model publisher exists." >&2
  exit 1
elif [[ ${arm_processes} -ne 2 || ${v2_arm_processes} -ne 2 ]]; then
  echo "Refused: expected exactly two v2_joint_control X5Controller processes; " \
       "found ${arm_processes} arm processes (${v2_arm_processes} v2)." >&2
  exit 1
else
  wait_for_publishers "running inference arms" "${arm_feedback_topics[@]}"
  echo "Reusing two operator-started v2 inference arm controllers."
fi

camera_image_topics=()
for camera in camera_h camera_l camera_r; do
  camera_image_topics+=("/camera/${camera}/color/image_rect_raw/compressed")
done
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
  echo "Reusing three running cameras."
fi

ckpt_dir_q=$(printf '%q' "${CKPT_DIR}")
ckpt_name_q=$(printf '%q' "${CKPT_NAME}")
height_q=$(printf '%q' "${LIFT_HEIGHT}")
extra_args_q=""
for argument in "$@"; do
  extra_args_q+=" $(printf '%q' "${argument}")"
done

gnome-terminal --title="inference" -- bash -ic \
  "cd ${repo_root}/act; conda activate act; python inference.py --ckpt_dir ${ckpt_dir_q} --ckpt_name ${ckpt_name_q} --expected-height ${height_q}${extra_args_q}; exec bash"

echo "Inference terminal launched. Default mode is DRY-RUN; no arm/body publisher is created."
