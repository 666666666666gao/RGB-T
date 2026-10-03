#!/usr/bin/env bash
set -euo pipefail
policy="$1"
gpu="$2"
cd /data/gb/GOLA
root=/data/gb/outputs/abc_internal_validation_v1
bash scripts/run_temporal.sh "$gpu" evaluate_online \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --variant c1 --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
  --candidate-policy "$policy" --pause-after-correction 0 --record-mechanisms \
  --output "$root/c1_${policy}/predictions"
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.compare_candidate_policies \
  --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json \
  --baseline "$root/baseline/predictions" --c1 "$root/c1/predictions" \
  --candidate-run "$root/c1_${policy}/predictions" --candidate-policy "$policy" \
  --output "$root/c1_${policy}_report"
