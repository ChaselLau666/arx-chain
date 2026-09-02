#!/usr/bin/env bash
# Fully pinned one-click Human DAgger launcher for ark-2 pickplace.
# Every knob is hardcoded with absolute paths; nothing is read from the
# caller environment except ROS_DOMAIN_ID (robot identity, /etc/environment).
set -Eeuo pipefail

export TASK_NAME=pickplace_dagger
export LIFT_HEIGHT=15.5
export CKPT_DIR=/home/arx/ROS2_LIFT_Play/act/weights/act_ep000_024_seed0_epochs25000
export CKPT_NAME=policy_best.ckpt          # epoch 10716, val_loss 0.0256
export STATS_NAME=dataset_stats.pkl
export ACT_PYTHON=/home/arx/miniconda3/envs/act/bin/python
export DAGGER_ROUND=0
export MAX_TIMESTEPS=800
export HUMAN_DAGGER_CONFIG=/home/arx/ROS2_LIFT_Play/act/data/human_dagger.yaml
export HUMAN_DAGGER_DATASET_DIR=/home/arx/ROS2_LIFT_Play/act/dagger_datasets

exec /home/arx/ROS2_LIFT_Play/tools/05_human_dagger.sh
