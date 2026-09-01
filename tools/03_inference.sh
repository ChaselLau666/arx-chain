#!/bin/bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${repo_root}/tools"
: "${TASK_INSTRUCTION:?Set TASK_INSTRUCTION to a natural-language instruction}"
MODEL_SERVER_URL=${MODEL_SERVER_URL:-http://192.168.31.83:8000}
task_instruction_q=$(printf '%q' "${TASK_INSTRUCTION}")
model_server_url_q=$(printf '%q' "${MODEL_SERVER_URL}")

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

for interface in can1 can3 can5; do
  if ! ip link show "${interface}" 2>/dev/null | grep -q "UP"; then
    echo "Refused: ${interface} is not UP. Run tools/configure_can_interfaces.sh before starting body/arms."
    exit 1
  fi
done
# Never restart body from inference. This check is read-only.
source /opt/ros/jazzy/setup.bash
source "${repo_root}/custom_sdk/LIFT/body/ROS2/install/setup.bash"
if ! ros2 service list | grep -qx '/lift_height_status'; then
  echo "Refused: body is not already running with /lift_height_status."
  exit 1
fi

# Lift
gnome-terminal --title="lift" -x $shell_type -i -c "cd ../../LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller open_double_arm.launch.py; $shell_exec"
sleep 1

# Realsense
gnome-terminal --title="realsense" -x $shell_type -i -c "cd ${workspace}; cd ../realsense; ./realsense.sh; $shell_exec"
sleep 3

# Remote inference defaults to dry-run and creates no body publisher.
gnome-terminal --title="inference" -x $shell_type -i -c "source ${repo_root}/custom_sdk/LIFT/body/ROS2/install/setup.bash; cd ${repo_root}/act; conda activate act; python remote_inference_client.py --server-url ${model_server_url_q} --task-instruction ${task_instruction_q}; $shell_exec"
