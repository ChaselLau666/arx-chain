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

# Diagnostics. DIVE_PROBE=1 records everything that could move an arm, into one
# directory, from before anything powers up. It only adds a read-only subscriber
# and tees the terminals that already existed, so a normal run (the default, 0)
# behaves exactly as it did.
DIVE_PROBE=${DIVE_PROBE:-0}
PROBE_DIR=${PROBE_DIR:-${repo_root}/diagnostics/dive_$(date +%Y%m%d_%H%M%S)}

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

die() { echo "Refused: $*" >&2; exit 1; }

# Each returns nothing unless DIVE_PROBE=1, so the command strings below are
# unchanged on a normal run.
probe_pre() { (( DIVE_PROBE )) && printf 'stdbuf -oL -eL '; return 0; }
probe_py()  { (( DIVE_PROBE )) && printf -- '-u '; return 0; }
probe_log() { (( DIVE_PROBE )) && printf ' 2>&1 | tee -a %s/%s.log' "${PROBE_DIR}" "$1"; return 0; }

# What the run started from. A dive that only happens on one robot, or after one
# commit, is answered here rather than by memory.
#
# Run in the background and with every ros2 call bounded: the introspection below
# costs about six seconds, and spending that inline would push every later stage
# of the startup back by the same amount. Timing is part of what is being
# investigated, so the recorder must not be what changes it.
probe_snapshot() {
    (( DIVE_PROBE )) || return 0
    local file="${PROBE_DIR}/snapshot_$1.txt"
    {
        echo "=== $1 @ $(date '+%F %T.%3N') on $(hostname -s) ==="
        echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
        echo "LIFT_HEIGHT=${LIFT_HEIGHT} -> ${LIFT_HEIGHT_ROS:-<not normalised yet>}"
        echo "READY_POSE_L=${READY_POSE_L}"
        echo "READY_POSE_R=${READY_POSE_R}"
        echo "SKIP_FILTER=${SKIP_FILTER}  SMOOTH_TAU=${SMOOTH_TAU}  HOME_MUTE=${HOME_MUTE}"
        echo "ARM_POSE_L=${ARM_POSE_L}  ARM_POSE_R=${ARM_POSE_R}"
        echo "--- git ---"
        git -C "${repo_root}" log --oneline -1 2>&1
        git -C "${repo_root}" status --short 2>&1
        echo "--- CAN links ---"
        ip -br link show 2>&1 | grep -i can || echo "no can interfaces"
        echo "--- ros2 nodes ---"
        timeout 5 ros2 node list 2>&1 || true
        echo "--- ros2 topics ---"
        timeout 5 ros2 topic list 2>&1 || true
        echo "--- go_home_position as the controllers actually have it ---"
        for node in /vr_arm_l /vr_arm_r; do
            echo -n "${node}: "; timeout 5 ros2 param get "${node}" go_home_position 2>&1 || true
        done
        echo "--- /lift fixed_height ---"
        timeout 5 ros2 param get /lift fixed_height 2>&1 || true
    } > "${file}" 2>&1 &
    echo "  snapshot (in the background): ${file}"
}

normalise_lift_height() {
    local value=$1
    [[ "$value" =~ ^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]] \
        || die "LIFT_HEIGHT must be numeric and in [0, 20], got: ${value}"
    awk -v value="$value" 'BEGIN {
        if (value < 0.0 || value > 20.0) exit 1
        printf "%.12f", value + 0.0
    }' || die "LIFT_HEIGHT must be numeric and in [0, 20], got: ${value}"
}

# `ros2 param set` infers an integer from a bare value such as 15, while the
# SDK declares fixed_height as DOUBLE. Use one validated DOUBLE literal for
# both the ROS parameter and the collector metadata.

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
LIFT_HEIGHT_ROS=$(normalise_lift_height "$LIFT_HEIGHT")

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

# Probe first, so it is already listening when the arms get power. It creates
# subscriptions and no publishers, so it cannot be what moves an arm.
if (( DIVE_PROBE )); then
    mkdir -p "${PROBE_DIR}"
    echo "DIVE_PROBE=1: recording to ${PROBE_DIR}"
    probe_snapshot 01_before_can
    gnome-terminal --title="probe" -x $shell_type -i -c "${arm_env}; ${ACT_PYTHON} -u ${repo_root}/act/dive_probe.py --out-dir ${PROBE_DIR} --arm-cmd-topics ${ARM_POSE_L} ${ARM_POSE_R}$(probe_log probe); $shell_exec"
    sleep 1
fi

# CAN
gnome-terminal -t "can1" -x bash -c "cd ${CAN_DIR}; $(probe_pre)./arx_can1.sh$(probe_log can1); exec bash;"
sleep 0.3
gnome-terminal -t "can3" -x bash -c "cd ${CAN_DIR}; $(probe_pre)./arx_can3.sh$(probe_log can3); exec bash;"
sleep 0.3
gnome-terminal -t "can5" -x bash -c "cd ${CAN_DIR}; $(probe_pre)./arx_can5.sh$(probe_log can5); exec bash;"
sleep 0.3

# Body
gnome-terminal --title="body" -x $shell_type -i -c "cd ${LIFT_WS}; source install/setup.bash; $(probe_pre)ros2 launch arx_lift_controller lift.launch.py$(probe_log body); $shell_exec"
sleep 1

# Set fixed height before VR starts, so body never briefly follows the raw VR
# height during this collection session.
set +u
source /opt/ros/jazzy/setup.bash
source "${LIFT_WS}/install/setup.bash"
set -u
height_set=false
readback=""
for _ in $(seq 1 20); do
  ros2 param set /lift fixed_height "${LIFT_HEIGHT_ROS}" >/dev/null 2>&1 || true
  readback=$(ros2 param get /lift fixed_height 2>/dev/null \
             | sed -n 's/^Double value is: //p') || readback=""
  if [[ -n "$readback" ]] && awk -v a="$readback" -v b="${LIFT_HEIGHT_ROS}" \
      'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 1e-6) }'; then
    height_set=true
    break
  fi
  sleep 0.5
done
if [[ "${height_set}" != true ]]; then
  echo "Last value read back from /lift: ${readback:-<none>}" >&2
  die "/lift did not accept fixed_height=${LIFT_HEIGHT_ROS}; verify that the patched SDK is installed and restart body"
fi
echo "/lift fixed_height verified at ${LIFT_HEIGHT_ROS}"

# Arms. Started with ros2 run so go_home_position can be given: everything else
# here reproduces v2_pos_control.yaml, and the node name has to be remapped
# because the constructor hardcodes Node("x5_controller_node"). collect.py reads
# go_home_position back from these two node names.
echo "WARNING: the arms power up now and walk to the ready pose. Stand clear."
gnome-terminal --title="arm_l" -x $shell_type -i -c "${arm_env}; $(probe_pre)ros2 run arx_x5_controller X5Controller --ros-args -r __node:=vr_arm_l -p arm_can_id:=can1 -p arm_control_type:=vr_slave -p arm_end_type:=2 -p arm_pub_topic_name:=arm_l_status -p arm_sub_topic_name:=${ARM_POSE_L#/} -p go_home_position:='${READY_POSE_L}'$(probe_log arm_l); $shell_exec"
sleep 0.5
gnome-terminal --title="arm_r" -x $shell_type -i -c "${arm_env}; $(probe_pre)ros2 run arx_x5_controller X5Controller --ros-args -r __node:=vr_arm_r -p arm_can_id:=can3 -p arm_control_type:=vr_slave -p arm_end_type:=2 -p arm_pub_topic_name:=arm_r_status -p arm_sub_topic_name:=${ARM_POSE_R#/} -p go_home_position:='${READY_POSE_R}'$(probe_log arm_r); $shell_exec"
sleep 1
probe_snapshot 02_after_arms

# Realsense
gnome-terminal --title="realsense" -x $shell_type -i -c "cd ${repo_root}/realsense; $(probe_pre)./realsense.sh$(probe_log realsense); $shell_exec"
sleep 3

# VR
gnome-terminal --title="vr" -x $shell_type -i -c "cd ${VR_WS}; $(probe_pre)./ARX_VR.sh$(probe_log vr); $shell_exec"
sleep 1

# Pose filters, one per side. Each reads the arm on its own side, because the
# offset it carries is that arm's, and the two arms are parked independently.
if (( ! SKIP_FILTER )); then
    gnome-terminal --title="filter_l" -x $shell_type -i -c "${arm_env}; ${ACT_PYTHON} $(probe_py)${repo_root}/act/vr_pose_filter.py --in-topic /ARX_VR_L --out-topic ${ARM_POSE_L} --node-name vr_pose_filter_l --arm-status-topic /arm_l_status_full --tau ${SMOOTH_TAU} --home-mute ${HOME_MUTE}$(probe_log filter_l); $shell_exec"
    sleep 0.5
    gnome-terminal --title="filter_r" -x $shell_type -i -c "${arm_env}; ${ACT_PYTHON} $(probe_py)${repo_root}/act/vr_pose_filter.py --in-topic /ARX_VR_R --out-topic ${ARM_POSE_R} --node-name vr_pose_filter_r --arm-status-topic /arm_r_status_full --tau ${SMOOTH_TAU} --home-mute ${HOME_MUTE}$(probe_log filter_r); $shell_exec"
    sleep 1
fi
probe_snapshot 03_after_filters

# Collect. --ready_pose is what turns the parking on, and --ready_pose_topics
# tells it where the arms are actually listening.
lift_height_q=$(printf '%q' "${LIFT_HEIGHT_ROS}")
task_name_q=$(printf '%q' "${TASK_NAME}")
ready_args="--ready_pose --ready_pose_topics ${ARM_POSE_L} ${ARM_POSE_R}"
(( SKIP_FILTER )) && ready_args=""
gnome-terminal --title="collect" -x $shell_type -i -c "cd ${repo_root}/act; conda activate ${ACT_ENV}; python $(probe_py)collect.py --episode_idx -1 ${ready_args} --poscmd-topics ${ARM_POSE_L} ${ARM_POSE_R} --height ${lift_height_q} --task ${task_name_q}$(probe_log collect); $shell_exec"
