#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
mkdir -p /data/gb/outputs/online_core
tmux new-session -d -s online_lasher_baseline_20261002 \
  'bash /data/gb/GOLA/scripts/run_online_and_score.sh lasher baseline 0 /data/wangwj/dataset/LasHeR/testingset /data/gb/outputs/online_core/lasher_baseline > /data/gb/outputs/online_core/lasher_baseline.controller.log 2>&1'
tmux new-session -d -s online_lasher_c1_20261002 \
  'bash /data/gb/GOLA/scripts/run_online_and_score.sh lasher c1 1 /data/wangwj/dataset/LasHeR/testingset /data/gb/outputs/online_core/lasher_c1 > /data/gb/outputs/online_core/lasher_c1.controller.log 2>&1'
tmux new-session -d -s online_rgbt234_baseline_20261002 \
  'bash /data/gb/GOLA/scripts/run_online_and_score.sh rgbt234 baseline 2 /data/zhouy/DATASET/RGB-T234 /data/gb/outputs/online_core/rgbt234_baseline > /data/gb/outputs/online_core/rgbt234_baseline.controller.log 2>&1'
tmux new-session -d -s online_rgbt234_c1_20261002 \
  'bash /data/gb/GOLA/scripts/run_online_and_score.sh rgbt234 c1 3 /data/zhouy/DATASET/RGB-T234 /data/gb/outputs/online_core/rgbt234_c1 > /data/gb/outputs/online_core/rgbt234_c1.controller.log 2>&1'
date -Iseconds
tmux ls
