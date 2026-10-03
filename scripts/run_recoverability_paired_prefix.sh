#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
session=recoverability_paired_c1_prefix_20261003
run="/data/gb/outputs/$session"
bash scripts/run_temporal.sh "$gpu" collect_recoverability \
  --partition train --jobs-file /data/gb/outputs/recoverability_own_policy_jobs_20261003/train.json \
  --serial-c1-prefix --max-prefix 1024 --batch-clips 16 --forward-batch 64 --seed 46 --output "$run"
date -Iseconds > "$run/job_completed.txt"
