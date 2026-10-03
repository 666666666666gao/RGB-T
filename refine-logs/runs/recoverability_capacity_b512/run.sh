#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
nvidia-smi -i 1 --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 1 > /data/gb/outputs/recoverability_pairwise_selector_own_b512_capacity_m0_20261004/gpu_memory.csv &
monitor_pid=$!
bash scripts/run_temporal.sh 1 train_recoverability --train /data/gb/outputs/recoverability_train_s42_20261003 /data/gb/outputs/recoverability_train_s43_20261003 /data/gb/outputs/recoverability_train_s44_20261003 /data/gb/outputs/recoverability_own_policy_train_20261003 --prefer-last-prefix --validation /data/gb/outputs/recoverability_own_policy_validation_20261003 --batch-size 512 --epochs 1 --seed 42 --search-supervision selector --action-ranking pairwise --output /data/gb/outputs/recoverability_pairwise_selector_own_b512_capacity_m0_20261004
kill "$monitor_pid"
date -Iseconds > /data/gb/outputs/recoverability_pairwise_selector_own_b512_capacity_m0_20261004/job_completed.txt
