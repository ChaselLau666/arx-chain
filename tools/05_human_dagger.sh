#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

source "${repo_root}/realsense/camera_serials.sh"
load_camera_serials dagger

# The ROS domain is this robot's identity and must come from the machine
# (/etc/environment), never from the repo: hardcoding it once drove another
# robot's lift on a shared LAN.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. The robot identity lives in /etc/environment (ark-1=62, ark-2=63); refusing to guess which robot to talk to}"

: "${TASK_NAME:?Set TASK_NAME, for example pickplace_right_to_bowl}"
: "${LIFT_HEIGHT:?Set LIFT_HEIGHT to the fixed lift command in [0, 20]}"

# POLICY_BACKEND selects the policy worker: "act" (local checkpoint, default)
# or "tau0vla" (remote inference over the dedicated direct link).
POLICY_BACKEND=${POLICY_BACKEND:-act}
case "$POLICY_BACKEND" in
  act)
    : "${CKPT_DIR:?Set CKPT_DIR to the directory containing the checkpoint and stats}"
    ;;
  tau0vla)
    : "${MODEL_SERVER_URL:?Set MODEL_SERVER_URL for the tau0vla backend}"
    : "${TASK_INSTRUCTION:?Set TASK_INSTRUCTION for the tau0vla backend}"
    ;;
  *)
    echo "Unknown POLICY_BACKEND: ${POLICY_BACKEND} (expected act or tau0vla)" >&2
    exit 1
    ;;
esac

CKPT_NAME=${CKPT_NAME:-policy_best.ckpt}
STATS_NAME=${STATS_NAME:-dataset_stats.pkl}
DAGGER_ROUND=${DAGGER_ROUND:-0}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-800}
CONFIG_PATH=${HUMAN_DAGGER_CONFIG:-"${repo_root}/act/data/human_dagger.yaml"}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
# Robot-local wall time, with nanoseconds to distinguish rapid restarts.
DATASET_DIR=${HUMAN_DAGGER_DATASET_DIR:-"${repo_root}/dagger_datasets_$(date +%Y%m%d_%H%M%S_%N)"}
MIN_FREE_GIB=${HUMAN_DAGGER_MIN_FREE_GIB:-5}

lift_ws=/home/arx/LIFT/body/ROS2
x5_ws=/home/arx/LIFT/ARX_X5/ROS2/X5_ws
vr_ws=/home/arx/LIFT/ARX_VR_SDK/ROS2
realsense_ws="${repo_root}/realsense"

runtime_root=${HUMAN_DAGGER_RUNTIME_DIR:-"${XDG_RUNTIME_DIR:-/tmp}/human_dagger-${UID}"}
active_manifest="${runtime_root}/active.manifest"
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
session_dir="${runtime_root}/${session_id}"
manifest="${session_dir}/pids.tsv"
log_dir="${session_dir}/logs"
session_active=false
frontend_pid=''
frontend_ticks=''

die() {
  echo "Refused: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file does not exist: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

is_number_in_range() {
  awk -v value="$1" -v low="$2" -v high="$3" \
    'BEGIN { exit !(value ~ /^[-+]?[0-9]+([.][0-9]+)?$/ && value >= low && value <= high) }'
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

proc_start_ticks() {
  local pid=$1 stat rest
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat < "/proc/${pid}/stat" || return 1
  rest=${stat#*) }
  # starttime is field 22; `rest` starts at field 3.
  set -- ${rest}
  [[ $# -ge 20 ]] || return 1
  printf '%s\n' "${20}"
}

manifest_entry_is_live() {
  local pid=$1 expected_ticks=$2 current_ticks
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  current_ticks=$(proc_start_ticks "$pid" 2>/dev/null) || return 1
  [[ "$current_ticks" == "$expected_ticks" ]]
}

manifest_has_live_processes() {
  local candidate=$1 label pid ticks
  [[ -f "$candidate" ]] || return 1
  while IFS=$'\t' read -r label pid ticks; do
    [[ -n "$label" && "$label" != \#* ]] || continue
    if manifest_entry_is_live "$pid" "$ticks"; then
      return 0
    fi
  done < "$candidate"
  return 1
}

request_hold_best_effort() {
  command -v ros2 >/dev/null 2>&1 || return 0
  if timeout 2 ros2 service list 2>/dev/null | grep -qx '/human_dagger/request_hold'; then
    timeout 3 ros2 service call \
      /human_dagger/request_hold std_srvs/srv/Trigger '{}' >/dev/null 2>&1 || true
  fi
}

on_exit() {
  local code=$?
  trap - EXIT INT TERM
  if [[ "$session_active" == true ]]; then
    request_hold_best_effort
    {
      printf '# frontend_exit_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf '# frontend_exit_code=%s\n' "$code"
    } >> "$manifest"
    echo
    echo "Human DAgger frontend exited (code ${code}); HOLD was requested if its service was reachable."
    echo "The hardware stack is intentionally still running so the lift is never stopped at height."
    echo "Use ${repo_root}/tools/04_safe_shutdown.sh to lower and stop this exact session."
    echo "Session manifest: ${manifest}"
  fi
  exit "$code"
}

on_signal() {
  local signal_name=$1
  request_hold_best_effort
  if [[ -n "$frontend_pid" ]] && manifest_entry_is_live "$frontend_pid" "$frontend_ticks"; then
    kill -s "$signal_name" "$frontend_pid" 2>/dev/null || true
  fi
}

trap on_exit EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

require_command awk
is_number_in_range "$LIFT_HEIGHT" 0 20 || die "LIFT_HEIGHT must be numeric and in [0, 20]"
# `ros2 param set` infers an integer type from a bare value such as `15`, but
# /lift.fixed_height is a DOUBLE parameter. Reuse one canonical DOUBLE literal
# for the ROS command, the Python runtime and the recorded metadata.
LIFT_HEIGHT_ROS=$(awk -v value="$LIFT_HEIGHT" 'BEGIN { printf "%.12f", value + 0.0 }')
is_nonnegative_integer "$DAGGER_ROUND" || die "DAGGER_ROUND must be a non-negative integer"
is_nonnegative_integer "$MAX_TIMESTEPS" || die "MAX_TIMESTEPS must be a non-negative integer"
(( MAX_TIMESTEPS >= 2 )) || die "MAX_TIMESTEPS must be at least 2"
is_nonnegative_integer "$MIN_FREE_GIB" || die "HUMAN_DAGGER_MIN_FREE_GIB must be a non-negative integer"
(( MIN_FREE_GIB >= 1 )) || die "HUMAN_DAGGER_MIN_FREE_GIB must be at least 1"

[[ -t 0 && -t 1 ]] || die "run this script in an interactive terminal on the robot desktop"
if [[ -n "${SSH_CONNECTION:-}" || -n "${SSH_TTY:-}" ]]; then
  if [[ "${HUMAN_DAGGER_ALLOW_SSH:-0}" != 1 ]]; then
    die "keyboard takeover must run in a local robot terminal, not over SSH (set HUMAN_DAGGER_ALLOW_SSH=1 to override; keep the physical e-stop within reach)"
  fi
  echo "WARNING: running keyboard takeover over SSH. If this connection drops,"
  echo "WARNING: nobody can take over from the policy. Keep the physical"
  echo "WARNING: emergency stop within reach at all times."
fi

require_command df
require_command find
require_command ip
require_command mktemp
require_command pgrep
require_command setsid
require_command timeout

require_file /opt/ros/jazzy/setup.bash
require_file "${lift_ws}/install/setup.bash"
require_file "${x5_ws}/install/setup.bash"
require_file "${vr_ws}/install/setup.bash"
require_file "${realsense_ws}/install/setup.bash"
require_file "$CONFIG_PATH"
require_file "$ACT_PYTHON"
require_file "${repo_root}/act/human_dagger.py"

ckpt_dir_abs=""
if [[ "$POLICY_BACKEND" == act ]]; then
  [[ -d "$CKPT_DIR" ]] || die "CKPT_DIR is not a directory: ${CKPT_DIR}"
  ckpt_dir_abs="$(cd "$CKPT_DIR" && pwd)"
  require_file "${ckpt_dir_abs}/${CKPT_NAME}"
  require_file "${ckpt_dir_abs}/${STATS_NAME}"
else
  require_file "${repo_root}/act/human_dagger_tau0vla_policy.py"
  require_file "${repo_root}/act/tau0vla_protocol.py"

  # Same direct-link gating as 03_tau0vla_inference.sh: the reviewed model
  # server lives on a dedicated Ethernet segment; Wi-Fi is a diagnostic
  # fallback that must be requested explicitly.
  DIRECT_SERVER_IP=${DIRECT_SERVER_IP:-192.168.50.2}
  DIRECT_INTERFACE=${DIRECT_INTERFACE:-enp130s0}
  DIRECT_CLIENT_IP=${DIRECT_CLIENT_IP:-192.168.50.1}
  ALLOW_NON_DIRECT_MODEL_SERVER=${ALLOW_NON_DIRECT_MODEL_SERVER:-0}
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
    die "Tau0VLA server is not ready at ${MODEL_SERVER_URL}"
  fi
fi

# Refuse all known competing arm owners and duplicate sensor/control stacks.
conflict_patterns=(
  '[v]2_pos_control'
  '[v]2_joint_control'
  '[o]pen_double_arm'
  '[/]arx_x5_controller/[X]5Controller'
  '[r]os2 run arx_x5_controller X5Controller'
  '[h]uman_dagger.py'
  '[/]arx_lift_controller/[l]ift_controller'
  '[s]erial_port_node'
  '[r]ealsense2_camera_node'
)
for pattern in "${conflict_patterns[@]}"; do
  if pids=$(pgrep -f -- "$pattern" 2>/dev/null); then
    echo "Conflicting process pattern: ${pattern}" >&2
    ps -o pid,ppid,pgid,args -p "$(tr '\n' ',' <<< "$pids" | sed 's/,$//')" >&2 || true
    die "another control or sensor stack is already running"
  fi
done

if [[ -n "${HUMAN_DAGGER_DATASET_DIR:-}" ]]; then
  mkdir -p "$DATASET_DIR"
else
  # Never silently reuse a default session directory if its timestamp collides.
  mkdir "$DATASET_DIR" || die "could not create a new dataset directory: ${DATASET_DIR}"
fi
echo "DAgger dataset directory: ${DATASET_DIR}"
[[ -w "$DATASET_DIR" ]] || die "dataset directory is not writable: ${DATASET_DIR}"
write_probe=$(mktemp "${DATASET_DIR}/.human_dagger_write_test.XXXXXX") || \
  die "could not create a write probe in ${DATASET_DIR}"
rm -f -- "$write_probe"
free_kib=$(df -Pk "$DATASET_DIR" | awk 'NR == 2 { print $4 }')
[[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not determine free space for ${DATASET_DIR}"
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
(( free_kib >= required_kib )) || \
  die "dataset filesystem has less than ${MIN_FREE_GIB} GiB free (${free_kib} KiB available)"
if find "$DATASET_DIR" -maxdepth 1 -type f -name '*.partial.hdf5' -print -quit | grep -q .; then
  die "stale partial HDF5 exists in ${DATASET_DIR}; inspect and move it to quarantine before starting"
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${vr_ws}/install/setup.bash"
# shellcheck disable=SC1091
source "${lift_ws}/install/setup.bash"
# Source x5_ws after vr_ws: the VR workspace carries a stale arm_control
# (PosCmd only) whose typesupport lacks JointControl; X5Controller must
# resolve arm_control against the full X5 build or it aborts with a
# symbol lookup error at startup.
# shellcheck disable=SC1091
source "${x5_ws}/install/setup.bash"
# shellcheck disable=SC1091
source "${realsense_ws}/install/setup.bash"
set -u
require_command ros2

if [[ "$POLICY_BACKEND" == act ]]; then
  if ! "$ACT_PYTHON" -c \
    'import cv2, h5py, numpy, rclpy, scipy, torch, yaml' >/dev/null 2>&1; then
    die "ACT_PYTHON is missing a required runtime module (cv2/h5py/numpy/rclpy/scipy/torch/yaml)"
  fi
else
  # Inference is remote: torch is not needed locally, requests is.
  if ! "$ACT_PYTHON" -c \
    'import cv2, h5py, numpy, rclpy, requests, scipy, yaml' >/dev/null 2>&1; then
    die "ACT_PYTHON is missing a required runtime module (cv2/h5py/numpy/rclpy/requests/scipy/yaml)"
  fi
fi

if [[ "$POLICY_BACKEND" == act ]]; then
  echo "Validating ACT architecture, checkpoint, statistics, and CUDA before hardware startup..."
  if ! "$ACT_PYTHON" "${repo_root}/act/human_dagger_policy.py" \
    --preflight \
    --ckpt-dir "$ckpt_dir_abs" \
    --ckpt-name "$CKPT_NAME" \
    --stats-name "$STATS_NAME"; then
    die "policy preflight failed; no CAN/body/arm component was started"
  fi
else
  echo "Validating the Tau0VLA server contract before hardware startup..."
  if ! "$ACT_PYTHON" "${repo_root}/act/human_dagger_tau0vla_policy.py" \
    --preflight \
    --server-url "$MODEL_SERVER_URL" \
    --task-instruction "$TASK_INSTRUCTION"; then
    die "tau0vla preflight failed; no CAN/body/arm component was started"
  fi
fi

# Do not start the vendor arx_can*.sh watchdog loops here. Each loop busy-spins
# and can globally kill every slcand process when one interface drops. Human
# DAgger accepts only transports that the operator has already configured UP.
can_is_up() {
  # SLCAN commonly reports operstate UNKNOWN even when its administrative
  # IFF_UP flag is set. Inspect interface flags rather than `ip -br` column 2.
  ip -o link show dev "$1" 2>/dev/null \
    | awk -F'[<>]' '$2 ~ /(^|,)UP(,|$)/ { found=1 } END { exit !found }'
}
for interface in can1 can3 can5; do
  can_is_up "$interface" || \
    die "${interface} is not UP; configure CAN safely before starting Human DAgger"
  ip -br link show "$interface"
done

# Only create an active-session manifest after every non-hardware preflight has
# passed. A dependency/checkpoint/disk refusal must not look like a live robot
# stack to the shutdown tooling.
mkdir -p "$runtime_root"
chmod 700 "$runtime_root"
if [[ -e "$active_manifest" || -L "$active_manifest" ]]; then
  previous_manifest=$(readlink "$active_manifest" 2>/dev/null || printf '%s' "$active_manifest")
  if manifest_has_live_processes "$previous_manifest"; then
    die "a tracked Human DAgger session is still active: ${previous_manifest}"
  fi
  rm -f -- "$active_manifest"
fi

mkdir -p "$log_dir"
chmod 700 "$session_dir" "$log_dir"
{
  echo '# human_dagger_session_v1'
  printf '# session_id=%s\n' "$session_id"
  printf '# repo_root=%s\n' "$repo_root"
  printf '# dataset_dir=%s\n' "$DATASET_DIR"
  printf '# started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '# hostname=%s\n' "$(hostname)"
  printf '# ros_domain_id=%s\n' "$ROS_DOMAIN_ID"
  printf '# columns=label<TAB>pid<TAB>linux_proc_start_ticks\n'
} > "$manifest"
ln -s "$manifest" "$active_manifest"
session_active=true

start_component() {
  local label=$1 pid ticks log_file
  shift
  log_file="${log_dir}/${label}.log"
  echo "Starting ${label}; log: ${log_file}"
  setsid "$@" </dev/null >>"$log_file" 2>&1 &
  pid=$!
  sleep 0.3
  if ! kill -0 "$pid" 2>/dev/null; then
    tail -n 30 "$log_file" >&2 || true
    die "${label} exited during startup"
  fi
  ticks=$(proc_start_ticks "$pid") || die "could not identify ${label} process ${pid}"
  printf '%s\t%s\t%s\n' "$label" "$pid" "$ticks" >> "$manifest"
}

wait_for_node() {
  local node=$1 timeout_seconds=$2 deadline
  deadline=$((SECONDS + timeout_seconds))
  until ros2 node list 2>/dev/null | grep -qx "$node"; do
    (( SECONDS < deadline )) || die "ROS node did not appear: ${node}"
    sleep 0.25
  done
}

wait_for_topic() {
  local topic=$1 timeout_seconds=$2 deadline
  deadline=$((SECONDS + timeout_seconds))
  echo "Waiting for fresh data on ${topic}"
  until timeout 2 ros2 topic echo --once "$topic" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || die "no message received on ${topic}"
    sleep 0.2
  done
}

wait_for_service() {
  local service=$1 timeout_seconds=$2 deadline
  deadline=$((SECONDS + timeout_seconds))
  until ros2 service list 2>/dev/null | grep -qx "$service"; do
    (( SECONDS < deadline )) || die "ROS service did not appear: ${service}"
    sleep 0.25
  done
}

start_frontend() {
  echo "Starting Human DAgger coordinator/UI in this terminal."
  local -a backend_args
  if [[ "$POLICY_BACKEND" == act ]]; then
    backend_args=(
      --ckpt-dir "$ckpt_dir_abs"
      --ckpt-name "$CKPT_NAME"
      --stats-name "$STATS_NAME"
    )
  else
    backend_args=(
      --policy-backend tau0vla
      --model-server-url "$MODEL_SERVER_URL"
      --task-instruction "$TASK_INSTRUCTION"
      --replan-steps "${REPLAN_STEPS:-auto}"
      --chunk-blend-steps "${CHUNK_BLEND_STEPS:-6}"
      --gripper-blend-steps "${GRIPPER_BLEND_STEPS:-0}"
      --gripper-debounce-frames "${GRIPPER_DEBOUNCE_FRAMES:-12}"
      --gripper-low-threshold "${GRIPPER_LOW_THRESHOLD:--2.1}"
      --gripper-high-threshold "${GRIPPER_HIGH_THRESHOLD:--1.05}"
      --gripper-low-value "${GRIPPER_LOW_VALUE:--3.384}"
      --gripper-high-value "${GRIPPER_HIGH_VALUE:-0.0}"
      --arm-ema-alpha "${ARM_EMA_ALPHA:-0.6}"
      --gripper-ema-alpha "${GRIPPER_EMA_ALPHA:-0.6}"
    )
  fi
  "$ACT_PYTHON" "${repo_root}/act/human_dagger.py" \
    --config "$CONFIG_PATH" \
    --datasets "$DATASET_DIR" \
    --task "$TASK_NAME" \
    --height "$LIFT_HEIGHT_ROS" \
    "${backend_args[@]}" \
    --dagger-round "$DAGGER_ROUND" \
    --episode-idx -1 \
    --max-timesteps "$MAX_TIMESTEPS" \
    --session-manifest "$manifest" </dev/tty &
  frontend_pid=$!
  sleep 0.3
  kill -0 "$frontend_pid" 2>/dev/null || die "Human DAgger frontend exited during startup"
  frontend_ticks=$(proc_start_ticks "$frontend_pid") || \
    die "could not identify Human DAgger frontend process ${frontend_pid}"
  printf '%s\t%s\t%s\n' frontend "$frontend_pid" "$frontend_ticks" >> "$manifest"
}

start_component body \
  ros2 run arx_lift_controller lift_controller --ros-args \
  -r __node:=lift \
  -r /ARX_VR_L:=/human_dagger/isolated/body_vr_disabled \
  -r /body_control:=/human_dagger/body/control \
  -r /joy:=/human_dagger/isolated/body_joy_disabled \
  -p robot_type:=0 \
  -p "fixed_height:=${LIFT_HEIGHT_ROS}"
wait_for_node /lift 20

height_set=false
for _ in $(seq 1 20); do
  if ros2 param set /lift fixed_height "$LIFT_HEIGHT_ROS" >/dev/null; then
    height_set=true
    break
  fi
  sleep 0.5
done
[[ "$height_set" == true ]] || die "could not set /lift fixed_height"
echo "/lift fixed_height set to ${LIFT_HEIGHT_ROS}"
wait_for_topic /body_information 15

# Bring up the sole command publisher before either X5 normal controller. It
# cannot publish until both feedback streams exist, but once they do its first
# command is a measured POSITION_CONTROL HOLD.
start_frontend

# Whole-session CPU profile of the frontend tree (supervisor + control/policy/
# recorder children). py-spy carries cap_sys_ptrace, so no sudo is needed; it
# exits by itself when the frontend does and only then writes the file.
# Collapsed-stack lines, most-sampled first:
#   sort -t" " -k2 -rn logs/pyspy_raw.txt | head
# Disable with PROFILE_DAGGER=0.
if [[ "${PROFILE_DAGGER:-1}" == 1 ]] \
  && PYSPY_BIN=$(command -v /home/arx/miniconda3/envs/act/bin/py-spy); then
  setsid "$PYSPY_BIN" record --pid "$frontend_pid" --subprocesses \
    --rate 50 --format raw -o "${log_dir}/pyspy_raw.txt" \
    </dev/null >>"${log_dir}/pyspy.log" 2>&1 &
  echo "Profiler attached (py-spy); stacks will land in ${log_dir}/pyspy_raw.txt"
fi

wait_for_node /human_dagger_control 30
wait_for_service /human_dagger/request_hold 10

start_component arm_left \
  ros2 run arx_x5_controller X5Controller --ros-args \
  -r __node:=human_dagger_arm_left \
  -r /arx_joy:=/human_dagger/isolated/arm/left/joy_disabled \
  -p arm_can_id:=can1 \
  -p arm_control_type:=normal \
  -p arm_end_type:=2 \
  -p arm_pub_topic_name:=/human_dagger/arm/left/status \
  -p arm_sub_topic_name:=/human_dagger/arm/left/command

start_component arm_right \
  ros2 run arx_x5_controller X5Controller --ros-args \
  -r __node:=human_dagger_arm_right \
  -r /arx_joy:=/human_dagger/isolated/arm/right/joy_disabled \
  -p arm_can_id:=can3 \
  -p arm_control_type:=normal \
  -p arm_end_type:=2 \
  -p arm_pub_topic_name:=/human_dagger/arm/right/status \
  -p arm_sub_topic_name:=/human_dagger/arm/right/command

COLOR_PROFILE=${COLOR_PROFILE:-640x480x90}
DEPTH_PROFILE=${DEPTH_PROFILE:-640x480x90}
# Nothing consumes depth (use_depth_image: false; the frontend and recorder
# subscribe to color only). Publishing it costs USB bandwidth -- fatal for a
# camera that renegotiated down to USB2 -- and CPU on all three nodes.
ENABLE_DEPTH=${ENABLE_DEPTH:-false}

start_camera() {
  local name=$1 raw_serial=$2 serial
  serial=${raw_serial#_}
  [[ "$serial" =~ ^[0-9]+$ ]] || die "invalid ${name} serial: ${raw_serial}"
  start_component "$name" \
    ros2 launch realsense2_camera rs_launch.py \
    camera_name:="$name" \
    depth_module.color_profile:="$COLOR_PROFILE" \
    depth_module.depth_profile:="$DEPTH_PROFILE" \
    enable_depth:="$ENABLE_DEPTH" \
    serial_no:="_${serial}"
}

start_camera camera_h "$CAMERA_H_SERIAL"
start_camera camera_l "$CAMERA_L_SERIAL"
start_camera camera_r "$CAMERA_R_SERIAL"

start_component vr_serial \
  ros2 run serial_port serial_port_node --ros-args \
  -r /ARX_VR_L:=/human_dagger/vr/left_raw \
  -r /ARX_VR_R:=/human_dagger/vr/right_raw

wait_for_topic /human_dagger/arm/left/status 20
wait_for_topic /human_dagger/arm/right/status 20
wait_for_topic /camera/camera_h/color/image_rect_raw/compressed 30
wait_for_topic /camera/camera_l/color/image_rect_raw/compressed 30
wait_for_topic /camera/camera_r/color/image_rect_raw/compressed 30
wait_for_topic /human_dagger/vr/left_raw 20
wait_for_topic /human_dagger/vr/right_raw 20

echo
echo "Preflight passed. Keep the physical emergency stop within reach."
echo "Controls: R=start, Space=human, P=policy, E=end; review with S/D/Q."
echo "Support-process logs: ${log_dir}"
echo

set +e
wait "$frontend_pid"
frontend_code=$?
frontend_pid=''
frontend_ticks=''
set -e
exit "$frontend_code"
