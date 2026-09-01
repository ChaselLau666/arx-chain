#!/bin/bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${repo_root}/tools"
: "${TASK_NAME:?Set TASK_NAME, for example pickplace_right_to_bowl}"
: "${TASK_INSTRUCTION:?Set TASK_INSTRUCTION to a natural-language instruction}"
task_name_q=$(printf '%q' "${TASK_NAME}")
task_instruction_q=$(printf '%q' "${TASK_INSTRUCTION}")

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

# CAN is configured once after boot with configure_can_interfaces.sh. Never run
# the delivered per-interface watchdogs here: they can kill unrelated slcand
# processes, including a live body can5 link.
for interface in can1 can3 can5; do
  if ! ip link show "${interface}" 2>/dev/null | grep -q "UP"; then
    echo "Refused: ${interface} is not UP. Run tools/configure_can_interfaces.sh before starting body/arms."
    exit 1
  fi
done
# Body is deliberately not started or restarted here. It must already be
# running from a safe-low-position bringup.
source /opt/ros/jazzy/setup.bash
source "${repo_root}/custom_sdk/LIFT/body/ROS2/install/setup.bash"
if ! ros2 service list | grep -qx '/lift_height_status'; then
  echo "Refused: body is not already running with /lift_height_status."
  echo "Start/rebuild body only after lowering the platform to a safe low position."
  exit 1
fi

# Lift
gnome-terminal --title="lift" -x $shell_type -i -c "cd ../../LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller v2_pos_control.launch.py; $shell_exec"
sleep 1

# Realsense
gnome-terminal --title="realsense" -x $shell_type -i -c "cd ${workspace}; cd ../realsense; ./realsense.sh; $shell_exec"
sleep 3

# VR	
gnome-terminal --title="vr" -x $shell_type -i -c "cd ../../LIFT/ARX_VR_SDK/ROS2; ./ARX_VR.sh; $shell_exec"
sleep 1

# Collect
gnome-terminal --title="collect" -x $shell_type -i -c "source ${repo_root}/custom_sdk/LIFT/body/ROS2/install/setup.bash; cd ${repo_root}/act; conda activate act; python collect.py --task ${task_name_q} --task_instruction ${task_instruction_q}; $shell_exec"
