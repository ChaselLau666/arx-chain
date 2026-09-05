#!/bin/bash
set -Eeuo pipefail

# Collection with the VR pose stream low-passed before it reaches the arms.
#
# Like 01_collect.sh this brings up CAN and body itself, so it is a drop-in
# replacement for it. The functional difference is where the arms get commands:
# a filter node sits between the VR serial node and the arm controllers, so the
# poses that drive inverse kinematics are smooth. Nothing in the ARX SDK is
# modified - the arms are started with ros2 run and told which topic to listen
# to through the arm_sub_topic_name parameter they already expose.
#
# Filtering here rather than on the recorded joint angles is what teleop-app
# does, and for the same reason: the pose is what feeds IK.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# The domain is this robot's identity and must come from the machine
# (/etc/environment). Defaulting it once drove another robot's arms on a shared
# LAN, so this refuses rather than guesses.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment (ark-1=62, ark-2=63); refusing to guess which robot to talk to}"
: "${LIFT_HEIGHT:?Set LIFT_HEIGHT to the fixed lift command in [0, 20]}"
: "${TASK_NAME:?Set TASK_NAME, for example pickplace_right_to_bowl}"

SMOOTH_TAU=${SMOOTH_TAU:-0.05}
# CAN and body are started when absent, matching 01_collect.sh. Set this to 1
# for the earlier behaviour, where either one missing refuses the run.
SKIP_AUTOSTART=${SKIP_AUTOSTART:-0}
# Records without cameras, to exercise the collection path on a robot that
# has none. The episodes carry no images and are written elsewhere; see the
# collector invocation at the end of this file.
SKIP_CAMERAS=${SKIP_CAMERAS:-0}
# Starts nothing and reuses whatever is already running, for when the
# collector alone needs restarting. Verified, not assumed; see below.
COLLECTOR_ONLY=${COLLECTOR_ONLY:-0}
# Where each arm parks. X5Controller hands this to the SDK at construction,
# so the arm walks here as it comes up and every episode starts from the same
# place, and publishing [0, 1] on /arx_joy sends it back - that subscription
# is created outside the mode branches, so it works in vr_slave too.
# Measured on this robot; the vendor's own v2_collect.yaml carries
# [0, 0.948, 0.858, -0.573, 0, 0], which is the same pose within 0.2 deg.
READY_POSE_L=${READY_POSE_L:-[-0.0002, 0.9447, 0.8597, -0.5755, 0.0006, -0.0013]}
READY_POSE_R=${READY_POSE_R:-[-0.0002, 0.9466, 0.8604, -0.5724, -0.0002, -0.0006]}
LOG_DIR=${LOG_DIR:-$HOME/collect_logs}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
LIFT_WS=/home/arx/LIFT/body/ROS2
X5_WS=/home/arx/LIFT/ARX_X5/ROS2/X5_ws
VR_WS=/home/arx/LIFT/ARX_VR_SDK/ROS2
# The realsense build lives in the main checkout; this worktree only carries
# the tracked sources, so point at the built one rather than a missing install.
REALSENSE_WS=${REALSENSE_WS:-/home/arx/ROS2_LIFT_Play/realsense}
CAMERA_PROFILE=${CAMERA_PROFILE:-640x480x90}
FILTERED_L=/ARX_VR_L_filtered
FILTERED_R=/ARX_VR_R_filtered
# Which topic each arm ends up subscribed to. collect.py commands the ready pose
# on these, and reaching an arm means publishing where that arm is listening.
if [[ "${SMOOTH_TAU:-}" == "0" || "${SMOOTH_TAU:-}" == "0.0" ]]; then
    ARM_POSE_L=/ARX_VR_L
    ARM_POSE_R=/ARX_VR_R
else
    ARM_POSE_L=${FILTERED_L}
    ARM_POSE_R=${FILTERED_R}
fi

mkdir -p "$LOG_DIR"
pids=()

die() { echo "Refused: $*" >&2; exit 1; }

cleanup_on_error() {
    local status=$?
    if (( status != 0 )) && (( ${#pids[@]} )); then
        echo "Startup failed; stopping the components this script started." >&2
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_on_error EXIT

start_component() {
    local name=$1; shift
    local log="${LOG_DIR}/${name}.log"
    echo "  starting ${name} (log: ${log})"
    setsid "$@" > "$log" 2>&1 < /dev/null &
    pids+=("$!")
}

wait_for_topic() {
    local topic=$1 timeout=${2:-20}
    for _ in $(seq 1 $((timeout * 2))); do
        ros2 topic list 2>/dev/null | grep -qx "$topic" && return 0
        sleep 0.5
    done
    die "${topic} did not appear within ${timeout}s; see ${LOG_DIR}"
}

set +u
source /opt/ros/jazzy/setup.bash
# The VR workspace ships a stale arm_control carrying only PosCmd, so it must
# be sourced before X5: whichever is sourced last wins, and X5Controller aborts
# at startup with an undefined JointControl typesupport symbol if it resolves
# arm_control against the VR copy.
source "${VR_WS}/install/setup.bash"
source "${LIFT_WS}/install/setup.bash"
source "${X5_WS}/install/setup.bash"
# Cameras are the one component with no fallback: with no realsense build
# there is no rs_launch.py to run and no image topics for the collector.
# Checked here so the failure names the cause, rather than set -e aborting
# on a bare "No such file or directory" from the source below.
if (( ! SKIP_CAMERAS )); then
    [[ -f "${REALSENSE_WS}/install/setup.bash" ]] || die \
        "realsense is not built at ${REALSENSE_WS}; run: cd ${REALSENSE_WS} && colcon build"
    source "${REALSENSE_WS}/install/setup.bash"
fi
set -u

# --- preconditions, all refused rather than repaired -------------------------

if (( COLLECTOR_ONLY )); then
    # Reuse a stack that is already up. The usual case is a collector that died
    # or was stopped while every other component kept running, where restarting
    # the lot costs another lift homing cycle for nothing. Nothing is started
    # here, so what the collector reads is checked rather than assumed.
    echo "COLLECTOR_ONLY=1: reusing the running stack, starting only the collector."
    required=(/ARX_VR_L /arm_l_status_full /arm_r_status_full /body_information)
    if (( ! SKIP_CAMERAS )); then
        for camera in camera_h camera_l camera_r; do
            required+=("/camera/${camera}/color/image_rect_raw/compressed")
        done
    fi
    for topic in "${required[@]}"; do
        ros2 topic list 2>/dev/null | grep -qx "$topic" \
            || die "${topic} is missing, so the stack is not up; rerun without COLLECTOR_ONLY=1"
    done
    ros2 node list 2>/dev/null | grep -qx '/lift' \
        || die "/lift is not running; rerun without COLLECTOR_ONLY=1"
    echo "  every topic the collector reads is present"
else
    # Anchored on the executable path, not on any mention of the name: a plain
    # substring also matches the shell that happens to have the word in its command
    # line, which refuses startup for no reason.
    for pattern in '/arx_x5_controller/X5Controller( |$)' \
                   '/serial_port_node( |$)' \
                   '/act/collect\.py( |$)' \
                   '/act/vr_pose_filter\.py( |$)'; do
        if conflicting=$(pgrep -f -- "$pattern" 2>/dev/null); then
            echo "Conflicting processes for ${pattern}:" >&2
            ps -o pid,args -p "$(tr '\n' ',' <<< "$conflicting" | sed 's/,$//')" >&2 || true
            die "another control stack is already running"
        fi
    done

    declare -A CAN_DEVICE=( [can1]=/dev/arxcan1 [can3]=/dev/arxcan3 [can5]=/dev/arxcan5 )

    can_is_up() { ip link show "$1" 2>/dev/null | grep -q 'UP'; }

    # Deliberately not arx_can1.sh and friends: each of those is a watchdog loop
    # whose repair path runs `pkill -9 slcand`, taking down the daemon behind every
    # other interface too, and whose success path spins without ever sleeping. This
    # does the one-shot bring-up those scripts do, and nothing else.
    bring_up_can() {
        # Two statements, not one: local expands all its arguments before it
        # assigns any of them, so ${CAN_DEVICE[$iface]} on the same line reads an
        # iface that is still unset and yields an empty device path.
        local iface=$1
        local dev=${CAN_DEVICE[$iface]}
        if ! ip link show "$iface" >/dev/null 2>&1; then
            [[ -e "$dev" ]] || die "${dev} is missing; the CAN adapter for ${iface} is unplugged"
            echo "  ${iface}: starting slcand on ${dev}"
            sudo slcand -o -f -s8 "$dev" "$iface" || die "slcand failed for ${iface}"
            for _ in $(seq 1 20); do
                ip link show "$iface" >/dev/null 2>&1 && break
                sleep 0.25
            done
        fi
        sudo ip link set "$iface" up || die "could not bring ${iface} up"
        can_is_up "$iface" || die "${iface} is still not UP after bring-up"
    }

    for interface in can1 can3 can5; do
        can_is_up "$interface" && continue
        (( SKIP_AUTOSTART )) && die "${interface} is not UP"
        bring_up_can "$interface"
    done

    lift_is_up() { ros2 node list 2>/dev/null | grep -qx '/lift'; }

    if ! lift_is_up; then
        (( SKIP_AUTOSTART )) && die "/lift is not running"
        # The lift motor is uncalibrated at power-on and homes itself before it obeys
        # fixed_height, so the platform can travel on its own here. 01_collect.sh
        # starts body exactly the same way; this only says so out loud first.
        echo "WARNING: body starts now and the lift may home itself. Stand clear."
        start_component body ros2 launch arx_lift_controller lift.launch.py
        for _ in $(seq 1 60); do
            lift_is_up && break
            sleep 0.5
        done
        lift_is_up || die "/lift did not appear within 30s; see ${LOG_DIR}/body.log"
        wait_for_topic /body_information 20
    fi

fi

# --- pin the lift before VR starts ------------------------------------------
# The body must never briefly follow a raw VR height during collection.

# Both `param set` and `param get` exit 0 even for a parameter the node never
# declared: an unpatched body answers "Setting parameter failed: Invalid access
# to undeclared parameter(s)" and still reports success. Trusting the exit code
# let collection start with the platform free to follow the VR stick, which is
# the exact failure this guard exists to prevent. Read the value back instead.
height_set=false
for _ in $(seq 1 20); do
    ros2 param set /lift fixed_height "${LIFT_HEIGHT}" >/dev/null 2>&1 || true
    readback=$(ros2 param get /lift fixed_height 2>/dev/null \
               | sed -n 's/^Double value is: //p')
    if [[ -n "$readback" ]] && awk -v a="$readback" -v b="${LIFT_HEIGHT}" \
        'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 1e-6) }'; then
        height_set=true
        break
    fi
    sleep 0.5
done
if [[ "${height_set}" != true ]]; then
    echo "Last value read back from /lift: ${readback:-<none>}" >&2
    die "/lift did not accept fixed_height=${LIFT_HEIGHT}. If the body is running,\
 its SDK is probably unpatched: run tools/apply_lift_fixed_height_patch.sh and\
 restart body."
fi
echo "  /lift fixed_height set to ${LIFT_HEIGHT}"

if (( ! COLLECTOR_ONLY )); then
    # --- arms, pointed at the filtered pose stream ------------------------------

    echo "WARNING: the arms power up now and may home themselves. Stand clear."

    start_arm() {
        local node=$1 can=$2 pub=$3 sub=$4 home=$5
        start_component "$node" \
            ros2 run arx_x5_controller X5Controller --ros-args \
            -r __node:="$node" \
            -p arm_can_id:="$can" \
            -p arm_control_type:=vr_slave \
            -p arm_end_type:=2 \
            -p arm_pub_topic_name:="$pub" \
            -p arm_sub_topic_name:="$sub" \
            -p go_home_position:="$home"
    }

    if [[ "${ARM_POSE_L}" == "/ARX_VR_L" ]]; then
        echo "  SMOOTH_TAU=0: arms take the raw VR stream, matching 01_collect.sh"
    fi
    start_arm vr_arm_l can1 arm_l_status "${ARM_POSE_L}" "${READY_POSE_L}"
    start_arm vr_arm_r can3 arm_r_status "${ARM_POSE_R}" "${READY_POSE_R}"
    wait_for_topic /arm_l_status_full 25
    wait_for_topic /arm_r_status_full 25

    # --- cameras ----------------------------------------------------------------

    if (( SKIP_CAMERAS )); then
        echo "  SKIP_CAMERAS=1: no cameras started; the episodes will carry no images"
    else
        # realsense.sh opens a gnome-terminal per camera, which cannot work once this
        # script is detached from a display. The serials still come from that file, so
        # it stays the one place they are configured.
        declare -A CAMERA_SERIAL
        while read -r name serial; do
            [[ -n "$name" ]] && CAMERA_SERIAL["$name"]="$serial"
        done < <(sed -n 's/^ *\[\([a-z_]*\)\]="\([0-9]*\)".*/\1 \2/p' \
                 "${repo_root}/realsense/realsense.sh")
        (( ${#CAMERA_SERIAL[@]} == 3 )) \
            || die "expected 3 camera serials in realsense/realsense.sh, found ${#CAMERA_SERIAL[@]}"

        for camera in camera_h camera_l camera_r; do
            serial=${CAMERA_SERIAL[$camera]:-}
            [[ -n "$serial" ]] || die "no serial configured for ${camera}"
            start_component "$camera" \
                ros2 launch realsense2_camera rs_launch.py \
                camera_name:="$camera" \
                depth_module.color_profile:="${CAMERA_PROFILE}" \
                depth_module.depth_profile:="${CAMERA_PROFILE}" \
                serial_no:="_${serial}"
        done
        for camera in camera_h camera_l camera_r; do
            wait_for_topic "/camera/${camera}/color/image_rect_raw/compressed" 40
        done
    fi

    # --- VR, then the filter that feeds the arms --------------------------------

    start_component vr_serial ros2 run serial_port serial_port_node
    wait_for_topic /ARX_VR_L 25
    wait_for_topic /ARX_VR_R 25

    if [[ "${SMOOTH_TAU}" != "0" && "${SMOOTH_TAU}" != "0.0" ]]; then
        start_component vr_filter_l "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
            --in-topic /ARX_VR_L --out-topic "${FILTERED_L}" \
            --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_l
        start_component vr_filter_r "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
            --in-topic /ARX_VR_R --out-topic "${FILTERED_R}" \
            --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_r
        wait_for_topic "${FILTERED_L}" 20
        wait_for_topic "${FILTERED_R}" 20
        echo "  VR poses are low-passed at tau=${SMOOTH_TAU}s before reaching the arms"
    fi

fi

trap - EXIT
echo
echo "Stack is up. Components log to ${LOG_DIR}."
echo "Starting the collector in this terminal; Ctrl+C reaches it directly."
echo

cd "${repo_root}/act"
set +u
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate act
set -u
collect_args=(--episode_idx -1 --height "${LIFT_HEIGHT}" --task "${TASK_NAME}"
              --ready_pose_topics "${ARM_POSE_L}" "${ARM_POSE_R}")
if (( SKIP_CAMERAS )); then
    # --camera_names with no values leaves the list empty, which switches off
    # the per-camera checks in get_observation. Written outside datasets/ so an
    # image-less episode is never picked up as training data by mistake.
    pipeline_dir="${repo_root}/act/datasets_pipeline_test"
    mkdir -p "$pipeline_dir"
    collect_args+=(--camera_names --datasets "$pipeline_dir")
fi
exec python collect.py "${collect_args[@]}" "$@"
