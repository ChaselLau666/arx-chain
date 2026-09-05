#!/bin/bash
set -Eeuo pipefail

# Record the VR stream and the arms to a file, for reading back afterwards.
#
# Sources the workspaces that carry the message definitions, so it can be run
# from any terminal without knowing which ones those are - the "message type
# arm_control/msg/PosCmd is invalid" error is what that knowledge costs.
#
#   ./vr_record.sh          record 30 seconds
#   ./vr_record.sh 60       record 60 seconds
#
# Ctrl+C stops early and still saves.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}

: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment; refusing to guess which robot to listen to}"

set +u
source /opt/ros/jazzy/setup.bash
# VR before X5: the VR workspace ships a stale arm_control carrying only PosCmd,
# and whichever is sourced last wins. See 06_collect_filtered.sh.
source /home/arx/LIFT/ARX_VR_SDK/ROS2/install/setup.bash
source /home/arx/LIFT/ARX_X5/ROS2/X5_ws/install/setup.bash
set -u

exec "$ACT_PYTHON" "${repo_root}/act/vr_record.py" "$@"
