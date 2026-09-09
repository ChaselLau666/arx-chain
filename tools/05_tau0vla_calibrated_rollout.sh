#!/bin/bash
# One command per calibrated rollout: idempotent stack bring-up, mandatory full
# gripper calibration, fixed training-pose setup, policy run, and guarded return.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/tau0vla_robot_profile.sh"
load_tau0vla_robot_profile
: "${MODEL_PROFILE:=all-blue-feedback}"
: "${MODEL_SERVER_URL:=http://192.168.50.2:8000}"
: "${LOG_DIR:=/home/arx/logs/tau0vla-calibrated}"
: "${REPLAN_STEPS:=15}"
: "${CHUNK_BLEND_STEPS:=6}"
: "${GRIPPER_BLEND_STEPS:=6}"
: "${ARM_EMA_ALPHA:=0.6}"
: "${GRIPPER_EMA_ALPHA:=1.0}"
: "${MAX_RESPONSE_AGE_MS:=500}"
: "${LIFT_HEIGHT:=12.5}"
load_tau0vla_model_profile

case "${1:-}" in
  --check)
    ;;
  --execute)
    policy_mode=(--execute)
    max_steps=${MAX_STEPS:-3600}
    ;;
  --dry-run)
    policy_mode=()
    max_steps=${MAX_STEPS:-900}
    ;;
  *)
    echo "Usage: MODEL_PROFILE=all-blue-feedback $0 --check|--dry-run|--execute" >&2
    exit 1
    ;;
esac

export MODEL_SERVER_URL MODEL_PROFILE
# Must complete before CAN, lift, arm, camera, or calibration launchers.
check_tau0vla_server
echo "Robot $(hostname -s), ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, lift=${LIFT_HEIGHT}"
echo "Cameras head=${CAMERA_H_SERIAL}, left=${CAMERA_L_SERIAL}, right=${CAMERA_R_SERIAL}"
if [[ "$1" == --check ]]; then
  echo "CHECK_COMPLETE: no hardware started, no calibration or motion performed."
  exit 0
fi

if [[ "$1" == --dry-run ]]; then
  # A dry-run only observes an already running stack and never creates calibration.
  calibration=${CALIBRATION_FILE:-}
  if [[ -z "${calibration}" || ! -f "${calibration}" ]]; then
    echo "Refused: --dry-run requires CALIBRATION_FILE pointing to an existing valid calibration." >&2
    echo "No hardware, calibration, or robot motion was started." >&2
    exit 1
  fi
fi
# Hold across exec into the client. A dry-run also creates a model session, so it
# must never invalidate a concurrently executing rollout's session.
if ! command -v flock >/dev/null; then
  echo "Refused: flock is required for the per-robot rollout lock." >&2
  exit 1
fi
exec 9>"/tmp/tau0vla-rollout-${UID}-${ROS_DOMAIN_ID}.lock"
if ! flock -n 9; then
  echo "Refused: another rollout/check with a model session is already active." >&2
  exit 1
fi
if pgrep -f '[t]au0vla_.*client.py|[t]au0vla_calibrate_gripper.py|[p]ython.*tau0vla_return_' >/dev/null; then
  echo "Refused: policy, calibration, or return process is still active." >&2
  exit 1
fi
if [[ "$1" == --execute ]]; then
  "${script_dir}/00_tau0vla_calibrated_up.sh" --auto-confirm
fi
source_tau0vla_ros

mkdir -p "${LOG_DIR}"
stamp=$(date +%Y%m%d_%H%M%S)
if [[ "$1" == --execute ]]; then
  calibration="${LOG_DIR}/calibration_${stamp}.json"
fi
calibration_log="${LOG_DIR}/calibration_${stamp}.log"
client_log="${LOG_DIR}/client_${stamp}.log"
trace="${LOG_DIR}/trace_${stamp}.jsonl"

cd "${repo_root}/act"
set -o pipefail
if [[ "$1" == --execute ]]; then
  "${TAU0VLA_PYTHON}" tau0vla_calibrate_gripper.py --execute --auto-confirm --output "${calibration}" \
    2>&1 | tee -a "${calibration_log}"
  test -f "${calibration}"
fi

echo "Calibration: ${calibration}"
echo "Client log: ${client_log}"
echo "Trace: ${trace}"
exec "${TAU0VLA_PYTHON}" tau0vla_calibrated_client.py \
  --server-url "${MODEL_SERVER_URL}" \
  --experiment "${experiment}" \
  --protocol-version "${protocol_version}" \
  --task-instruction "${task}" \
  --calibration-file "${calibration}" \
  --expected-height "${LIFT_HEIGHT}" \
  --replan-steps "${REPLAN_STEPS}" \
  --chunk-blend-steps "${CHUNK_BLEND_STEPS}" \
  --gripper-blend-steps "${GRIPPER_BLEND_STEPS}" \
  --arm-ema-alpha "${ARM_EMA_ALPHA}" \
  --gripper-ema-alpha "${GRIPPER_EMA_ALPHA}" \
  --max-response-age-ms "${MAX_RESPONSE_AGE_MS}" \
  --trace-path "${trace}" \
  --log-path "${client_log}" \
  --max-steps "${max_steps}" \
  --auto-confirm \
  "${policy_mode[@]}"
