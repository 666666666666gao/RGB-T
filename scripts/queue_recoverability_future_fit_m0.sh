#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
cache=/data/gb/outputs/recoverability_future_policy_merged_20261004
test ! -e "$cache"
for policy in c1 own; do
  for partition in train validation; do
    for shard in 0 1 2 3; do
      marker="/data/gb/outputs/recoverability_future_${policy}_${partition}_shard${shard}_20261004/job_completed.txt"
      while [[ ! -f "$marker" ]]; do
        sleep 240
      done
    done
  done
done
export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/data/gb/envs/gola/lib PYTHONPATH=/data/gb/GOLA OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.merge_recoverability_future_caches --output "$cache"
bash scripts/run_recoverability_future_fit.sh 0 c1 m0 > /data/gb/setup/future_policy_fit_c1_m0_20261004.log 2>&1 &
c1_pid=$!
bash scripts/run_recoverability_future_fit.sh 2 own m0 > /data/gb/setup/future_policy_fit_own_m0_20261004.log 2>&1 &
own_pid=$!
wait "$c1_pid"
wait "$own_pid"
date -Iseconds > /data/gb/setup/future_policy_both_fit_m0_completed_20261004.txt
