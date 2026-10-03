#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
partition="$2"
run="/data/gb/outputs/recoverability_own_policy_${partition}_20261003"
bash scripts/run_temporal.sh "$gpu" collect_recoverability --partition "$partition" \
  --prefix-model /data/gb/outputs/recoverability_abc_s42_20261003/best.pth \
  --jobs-file "/data/gb/outputs/recoverability_own_policy_jobs_20261003/${partition}.json" \
  --max-prefix 1024 --batch-clips 16 --forward-batch 64 --seed 46 --output "$run"
date -Iseconds > "$run/job_completed.txt"
