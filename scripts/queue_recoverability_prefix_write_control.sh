#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
native=recoverability_native_lasher_s42_20261003
own=recoverability_prefix_own_s42_20261003
while [[ ! -f "/data/gb/outputs/$native/report_completed.txt" ]]; do
  if ! tmux has-session -t "$native" 2>/dev/null; then
    printf 'Formal LasHeR evaluation stopped without completion: %s\n' "$native" >&2
    exit 1
  fi
  sleep 240
done
test -f "/data/gb/outputs/$own/training_completed.txt"
run=/data/gb/outputs/recoverability_prefix_own_s42_unsafe_write_20261003
bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --model "/data/gb/outputs/$own/best.pth" --unsafe-writes \
  --validation-split /data/gb/outputs/c1_initial_seed42/split.json --output "$run/predictions"
while [[ ! -f "/data/gb/outputs/$own/job_completed.txt" ]]; do
  if ! tmux has-session -t "$own" 2>/dev/null; then
    printf 'Full own-prefix reference stopped without completion: %s\n' "$own" >&2
    exit 1
  fi
  sleep 240
done
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json --labels prefix_own_s42_unsafe_write \
  --runs "$run/predictions" --reference-labels baseline c1 prefix_own_s42 \
  --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
               /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
               "/data/gb/outputs/$own/predictions" --output "$run/report"
date -Iseconds > "$run/job_completed.txt"
