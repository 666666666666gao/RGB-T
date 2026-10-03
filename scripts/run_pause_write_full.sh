#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
root=/data/gb/outputs/abc_internal_validation_v1
bash scripts/run_temporal.sh 3 evaluate_online \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --variant c1 --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
  --pause-after-correction 3 --record-mechanisms \
  --output "$root/c1_pause3/predictions"
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.compare_paused_updates \
  --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json \
  --baseline "$root/baseline/predictions" --c1 "$root/c1/predictions" \
  --paused "$root/c1_pause3/predictions" --output "$root/paused_updates_report"
