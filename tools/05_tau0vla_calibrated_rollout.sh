#!/bin/bash
# One command per calibrated rollout: idempotent stack bring-up, mandatory full
# gripper calibration, fixed training-pose setup, policy run, and guarded return.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
: "${ROS_DOMAIN_ID:?Set ROS_DOMAIN_ID=63 for ark-2}"
: "${MODEL_PROFILE:=blue-feedback}"
: "${MODEL_SERVER_URL:=http://192.168.50.2:8001}"
: "${LOG_DIR:=/home/arx/logs/tau0vla-calibrated}"
: "${REPLAN_STEPS:=15}"
: "${CHUNK_BLEND_STEPS:=6}"
: "${GRIPPER_BLEND_STEPS:=6}"
: "${ARM_EMA_ALPHA:=0.6}"
: "${GRIPPER_EMA_ALPHA:=1.0}"
: "${MAX_RESPONSE_AGE_MS:=500}"

case "${MODEL_PROFILE}" in
  blue-feedback)
    experiment=joint-feedback
    expected_route=arx-lift2s-0907-blue-joint-feedback-ft
    task='Pick up the blue box and place it in its designated position on the board.'
    ;;
  t-feedback)
    experiment=joint-feedback
    expected_route=arx-lift2s-0907-t-joint-feedback-ft
    task='Pick up the T-shaped part and place it in its designated position on the board.'
    ;;
  blue-vr)
    experiment=joint-vr
    expected_route=arx-lift2s-0907-blue-joint-vr-ft
    task='Pick up the blue box and place it in its designated position on the board.'
    ;;
  t-vr)
    experiment=joint-vr
    expected_route=arx-lift2s-0907-t-joint-vr-ft
    task='Pick up the T-shaped part and place it in its designated position on the board.'
    ;;
  *)
    echo "Unknown MODEL_PROFILE=${MODEL_PROFILE}; use blue-feedback, t-feedback, blue-vr or t-vr." >&2
    exit 1
    ;;
esac

case "${1:-}" in
  --execute)
    policy_mode=(--execute)
    max_steps=${MAX_STEPS:-3600}
    ;;
  --dry-run)
    policy_mode=()
    max_steps=${MAX_STEPS:-900}
    ;;
  *)
    echo "Usage: MODEL_PROFILE=blue-feedback $0 --dry-run|--execute" >&2
    exit 1
    ;;
esac

export MODEL_SERVER_URL
"${script_dir}/00_tau0vla_calibrated_up.sh"

health=$(curl --fail --silent --show-error --noproxy '*' --max-time 5 \
  "${MODEL_SERVER_URL}/health")
actual_route=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("route", ""))' <<<"${health}")
actual_experiment=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("experiment", ""))' <<<"${health}")
if [[ "${actual_route}" != "${expected_route}" || "${actual_experiment}" != "${experiment}" ]]; then
  echo "Refused: MODEL_PROFILE=${MODEL_PROFILE} expects ${expected_route}/${experiment}," >&2
  echo "but ${MODEL_SERVER_URL} serves ${actual_route}/${actual_experiment}." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
stamp=$(date +%Y%m%d_%H%M%S)
calibration="${LOG_DIR}/calibration_${stamp}.json"
calibration_log="${LOG_DIR}/calibration_${stamp}.log"
client_log="${LOG_DIR}/client_${stamp}.log"
trace="${LOG_DIR}/trace_${stamp}.jsonl"

set +u
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate act
set -u

cd "${repo_root}/act"
set -o pipefail
python tau0vla_calibrate_gripper.py --execute --output "${calibration}" \
  2>&1 | tee -a "${calibration_log}"
test -f "${calibration}"

echo "Calibration: ${calibration}"
echo "Client log: ${client_log}"
echo "Trace: ${trace}"
exec python tau0vla_calibrated_client.py \
  --server-url "${MODEL_SERVER_URL}" \
  --experiment "${experiment}" \
  --task-instruction "${task}" \
  --calibration-file "${calibration}" \
  --replan-steps "${REPLAN_STEPS}" \
  --chunk-blend-steps "${CHUNK_BLEND_STEPS}" \
  --gripper-blend-steps "${GRIPPER_BLEND_STEPS}" \
  --arm-ema-alpha "${ARM_EMA_ALPHA}" \
  --gripper-ema-alpha "${GRIPPER_EMA_ALPHA}" \
  --max-response-age-ms "${MAX_RESPONSE_AGE_MS}" \
  --trace-path "${trace}" \
  --log-path "${client_log}" \
  --max-steps "${max_steps}" \
  "${policy_mode[@]}"
