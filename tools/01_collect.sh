#!/bin/bash

workspace=$(pwd)
: "${LIFT_HEIGHT:?Set LIFT_HEIGHT to the desired lift command in [0, 20]}"
lift_height_q=$(printf '%q' "${LIFT_HEIGHT}")

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

# CAN
gnome-terminal -t "can1" -x bash -c "cd ${workspace}; cd ../../LIFT/ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
sleep 0.3
gnome-terminal -t "can3" -x bash -c "cd ${workspace}; cd ../../LIFT/ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
sleep 0.3
gnome-terminal -t "can5" -x bash -c "cd ${workspace}; cd ../../LIFT/ARX_CAN/arx_can; ./arx_can5.sh; exec bash;"
sleep 0.3

# Body
gnome-terminal --title="body" -x $shell_type -i -c "cd ../../LIFT/body/ROS2; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; $shell_exec"
sleep 1

# Set fixed height before VR starts, so body never briefly follows the raw VR
# height during this collection session.
source /opt/ros/jazzy/setup.bash
source ../../LIFT/body/ROS2/install/setup.bash
height_set=false
for _ in $(seq 1 20); do
  if ros2 param set /lift fixed_height "${LIFT_HEIGHT}"; then
    height_set=true
    break
  fi
  sleep 0.5
done
if [[ "${height_set}" != true ]]; then
  echo "Refused: could not set /lift fixed_height"
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
gnome-terminal --title="collect" -x $shell_type -i -c "cd ${workspace}; cd ../act; conda activate act; python collect.py --episode_idx -1 --height ${lift_height_q}; $shell_exec"
