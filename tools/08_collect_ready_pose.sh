#!/bin/bash
# Collection that parks both arms at a fixed ready pose before every episode.
#
# Same shape as 01_collect.sh - a gnome-terminal per component, the vendor's own
# CAN and body scripts, the same collector - with two functional differences.
#
# The arms are started with ros2 run instead of v2_pos_control.launch.py, so that
# go_home_position can be passed to them. That parameter is the whole reason this
# script exists: X5Controller declares it and hands it to the SDK at construction
# (X5Controller.cpp:16 and :30), so /arx_joy [0, 1] walks the arm there between
# episodes. The vendor's v2_pos_control.yaml does not carry it, and the launch
# file hardcodes its own params_file - the params_file launch argument it
# declares is never read - so there is no way to supply it through that path
# without editing the installed package.
#
# And a vr_pose_filter sits between the VR serial node and each arm. It is not
# here for the smoothing: it is here because the headset sends an ABSOLUTE pose
# and is never told that anything else moved the arm. Point an arm at the raw
# stream and the first VR frame after parking commands it to wherever the hand
# is, which is why parking alone looks like the arm "comes straight back". The
# filter mutes itself while /arx_joy is asking for GO_HOME, then anchors on the
# arm's own reported pose, and carries that offset on every frame afterwards -
# the same rebase Human DAgger does on every takeover. A still hand holds the
# arm where it was parked; a moving one carries on from there.
#
# SMOOTH_TAU=0 turns the smoothing off but keeps the filter, and so keeps the
# rebase. Removing the filter entirely is what SKIP_FILTER=1 does, and it gives
# up the parking with it.

set -Eeuo pipefail

# Absolute, from this file's location rather than $PWD. 01_collect.sh reaches the
# vendor trees with ../../LIFT, which only resolves when it is run from tools/.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

: "${LIFT_HEIGHT:?Set LIFT_HEIGHT to the desired lift command in [0, 20]}"
: "${TASK_NAME:?Set TASK_NAME, for example pickplace_right_to_bowl}"

# Where each arm parks. Measured on this robot; the vendor's own v2_collect.yaml
# carries [0, 0.948, 0.858, -0.573, 0, 0], the same pose within 0.2 deg. These
# are the only copy of the numbers: collect.py reads them back off the arms
# rather than keeping a second set in step.
READY_POSE_L=${READY_POSE_L:-[-0.0002, 0.9447, 0.8597, -0.5755, 0.0006, -0.0013]}
READY_POSE_R=${READY_POSE_R:-[-0.0002, 0.9466, 0.8604, -0.5724, -0.0002, -0.0006]}

# Time constant of the pose low-pass, seconds. 0 forwards poses unsmoothed but
# still rebased, which is the point of keeping the node in the path.
SMOOTH_TAU=${SMOOTH_TAU:-0.05}
# How long each /arx_joy message keeps the filters quiet. collect.py republishes
# every 50 ms while homing, so this only has to outlast one gap.
HOME_MUTE=${HOME_MUTE:-0.5}
# Arms straight onto the raw VR stream, as 01_collect.sh has them. The ready pose
# cannot be held that way, so the parking is turned off with it.
SKIP_FILTER=${SKIP_FILTER:-0}

LIFT_WS=${LIFT_WS:-/home/arx/LIFT/body/ROS2}
X5_WS=${X5_WS:-/home/arx/LIFT/ARX_X5/ROS2/X5_ws}
VR_WS=${VR_WS:-/home/arx/LIFT/ARX_VR_SDK/ROS2}
CAN_DIR=${CAN_DIR:-/home/arx/LIFT/ARX_CAN/arx_can}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
ACT_ENV=${ACT_ENV:-act}

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

die() { echo "Refused: $*" >&2; exit 1; }

# Checked and rewritten before anything powers up. Two ways this bites: a short
# array leaves the arm homing to whatever the SDK makes of a partial pose, and an
# all-integer one is rejected outright, because go_home_position is declared
# DOUBLE_ARRAY and rclpy refuses an INTEGER_ARRAY override rather than widening
# it. Writing 15 for a joint is a reasonable thing to do, so the value is
# normalised to DOUBLE literals rather than refused.
normalise_pose() {
    local name=$1 pose=$2 value out=()
    [[ "$pose" == \[*\] ]] || die "${name} must be a bracketed list, got: ${pose}"
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        [[ -z "$value" ]] && continue
        [[ "$value" =~ ^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]] \
            || die "${name} has a non-numeric joint value: ${value}"
        [[ "$value" == *.* || "$value" == *e* || "$value" == *E* ]] || value="${value}.0"
        out+=("$value")
    done < <(tr ',' '\n' <<< "${pose//[\[\]]/}")
    (( ${#out[@]} == 6 )) || die "${name} needs 6 joint values, found ${#out[@]}: ${pose}"
    local joined; printf -v joined '%s, ' "${out[@]}"
    printf '[%s]' "${joined%, }"
}
READY_POSE_L=$(normalise_pose READY_POSE_L "${READY_POSE_L}")
READY_POSE_R=$(normalise_pose READY_POSE_R "${READY_POSE_R}")

# Which topic each arm ends up subscribed to, and so where collect.py has to
# publish to command a pose. Reaching an arm means publishing where it listens.
if (( SKIP_FILTER )); then
    ARM_POSE_L=/ARX_VR_L
    ARM_POSE_R=/ARX_VR_R
else
    ARM_POSE_L=/ARX_VR_L_filtered
    ARM_POSE_R=/ARX_VR_R_filtered
fi

# The VR workspace ships a stale arm_control carrying only PosCmd, so it must be
# sourced before X5: whichever is sourced last wins, and X5Controller aborts at
# startup with an undefined JointControl typesupport symbol if it resolves
# arm_control against the VR copy.
arm_env="source /opt/ros/jazzy/setup.bash; source ${VR_WS}/install/setup.bash; source ${LIFT_WS}/install/setup.bash; source ${X5_WS}/install/setup.bash"

echo "Ready pose collection on $(hostname -s), ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
echo "  left  ${READY_POSE_L}"
echo "  right ${READY_POSE_R}"
if (( SKIP_FILTER )); then
    echo "  SKIP_FILTER=1: arms take the raw VR stream and the ready pose is NOT held"
else
    echo "  arms follow ${ARM_POSE_L} / ${ARM_POSE_R}, tau=${SMOOTH_TAU}s, home mute ${HOME_MUTE}s"
fi

# CAN
gnome-terminal -t "can1" -x bash -c "cd ${CAN_DIR}; ./arx_can1.sh; exec bash;"
sleep 0.3
gnome-terminal -t "can3" -x bash -c "cd ${CAN_DIR}; ./arx_can3.sh; exec bash;"
sleep 0.3
gnome-terminal -t "can5" -x bash -c "cd ${CAN_DIR}; ./arx_can5.sh; exec bash;"
sleep 0.3

# Body
gnome-terminal --title="body" -x $shell_type -i -c "cd ${LIFT_WS}; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; $shell_exec"
sleep 1

# Set fixed height before VR starts, so body never briefly follows the raw VR
# height during this collection session.
set +u
source /opt/ros/jazzy/setup.bash
source "${LIFT_WS}/install/setup.bash"
set -u
height_set=false
for _ in $(seq 1 20); do
  if ros2 param set /lift fixed_height "${LIFT_HEIGHT}"; then
    height_set=true
    break
  fi
  sleep 0.5
done
if [[ "${height_set}" != true ]]; then
  die "could not set /lift fixed_height"
fi

# Arms. Started with ros2 run so go_home_position can be given: everything else
# here reproduces v2_pos_control.yaml, and the node name has to be remapped
# because the constructor hardcodes Node("x5_controller_node"). collect.py reads
# go_home_position back from these two node names.
echo "WARNING: the arms power up now and walk to the ready pose. Stand clear."
gnome-terminal --title="arm_l" -x $shell_type -i -c "${arm_env}; ros2 run arx_x5_controller X5Controller --ros-args -r __node:=vr_arm_l -p arm_can_id:=can1 -p arm_control_type:=vr_slave -p arm_end_type:=2 -p arm_pub_topic_name:=arm_l_status -p arm_sub_topic_name:=${ARM_POSE_L#/} -p go_home_position:='${READY_POSE_L}'; $shell_exec"
sleep 0.5
gnome-terminal --title="arm_r" -x $shell_type -i -c "${arm_env}; ros2 run arx_x5_controller X5Controller --ros-args -r __node:=vr_arm_r -p arm_can_id:=can3 -p arm_control_type:=vr_slave -p arm_end_type:=2 -p arm_pub_topic_name:=arm_r_status -p arm_sub_topic_name:=${ARM_POSE_R#/} -p go_home_position:='${READY_POSE_R}'; $shell_exec"
sleep 1

# Realsense
gnome-terminal --title="realsense" -x $shell_type -i -c "cd ${repo_root}/realsense; ./realsense.sh; $shell_exec"
sleep 3

# VR
gnome-terminal --title="vr" -x $shell_type -i -c "cd ${VR_WS}; ./ARX_VR.sh; $shell_exec"
sleep 1

# Pose filters, one per side. Each reads the arm on its own side, because the
# offset it carries is that arm's, and the two arms are parked independently.
if (( ! SKIP_FILTER )); then
    gnome-terminal --title="filter_l" -x $shell_type -i -c "${arm_env}; ${ACT_PYTHON} ${repo_root}/act/vr_pose_filter.py --in-topic /ARX_VR_L --out-topic ${ARM_POSE_L} --node-name vr_pose_filter_l --arm-status-topic /arm_l_status_full --tau ${SMOOTH_TAU} --home-mute ${HOME_MUTE}; $shell_exec"
    sleep 0.5
    gnome-terminal --title="filter_r" -x $shell_type -i -c "${arm_env}; ${ACT_PYTHON} ${repo_root}/act/vr_pose_filter.py --in-topic /ARX_VR_R --out-topic ${ARM_POSE_R} --node-name vr_pose_filter_r --arm-status-topic /arm_r_status_full --tau ${SMOOTH_TAU} --home-mute ${HOME_MUTE}; $shell_exec"
    sleep 1
fi

# Collect. --ready_pose is what turns the parking on, and --ready_pose_topics
# tells it where the arms are actually listening.
lift_height_q=$(printf '%q' "${LIFT_HEIGHT}")
task_name_q=$(printf '%q' "${TASK_NAME}")
ready_args="--ready_pose --ready_pose_topics ${ARM_POSE_L} ${ARM_POSE_R}"
(( SKIP_FILTER )) && ready_args=""
gnome-terminal --title="collect" -x $shell_type -i -c "cd ${repo_root}/act; conda activate ${ACT_ENV}; python collect.py --episode_idx -1 ${ready_args} --height ${lift_height_q} --task ${task_name_q}; $shell_exec"
