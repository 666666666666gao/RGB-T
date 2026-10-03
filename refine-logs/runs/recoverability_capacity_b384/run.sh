#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
bash scripts/run_temporal.sh 1 train_recoverability --train /data/gb/outputs/recoverability_train_s42_20261003 /data/gb/outputs/recoverability_train_s43_20261003 /data/gb/outputs/recoverability_train_s44_20261003 /data/gb/outputs/recoverability_own_policy_train_20261003 --prefer-last-prefix --validation /data/gb/outputs/recoverability_own_policy_validation_20261003 --batch-size 384 --epochs 1 --seed 42 --search-supervision selector --action-ranking pairwise --output /data/gb/outputs/recoverability_pairwise_selector_own_b384_capacity_m0_20261004
date -Iseconds > /data/gb/outputs/recoverability_pairwise_selector_own_b384_capacity_m0_20261004/job_completed.txt
