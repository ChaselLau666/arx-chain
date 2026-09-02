#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}
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

arm_topics=0
for topic in /arm_slave_l_status /arm_slave_r_status; do
  ros2 topic list 2>/dev/null | grep -qx "${topic}" && arm_topics=$((arm_topics + 1))
done
if [[ ${arm_topics} -eq 0 ]]; then
  gnome-terminal --title="inference-arms" -- bash -ic \
    "cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller open_double_arm.launch.py; exec bash"
  sleep 2
elif [[ ${arm_topics} -ne 2 ]]; then
  echo "Refused: inference arm stack is partial (${arm_topics}/2 feedback topics)." >&2
  exit 1
else
  echo "Reusing running inference arm stack."
fi

camera_topics=0
for camera in camera_h camera_l camera_r; do
  topic="/camera/${camera}/color/image_rect_raw/compressed"
  ros2 topic list 2>/dev/null | grep -qx "${topic}" && camera_topics=$((camera_topics + 1))
done
if [[ ${camera_topics} -eq 0 ]]; then
  gnome-terminal --title="inference-cameras" -- bash -ic \
    "cd ${repo_root}/realsense; ./realsense.sh; exec bash"
  sleep 4
elif [[ ${camera_topics} -ne 3 ]]; then
  echo "Refused: camera stack is partial (${camera_topics}/3 image topics)." >&2
  exit 1
else
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
