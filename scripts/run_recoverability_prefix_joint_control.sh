#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
own=/data/gb/outputs/recoverability_prefix_own_s42_20261003
test -f "$own/job_completed.txt"
run=/data/gb/outputs/recoverability_prefix_own_s42_no_search_raw_write_20261004
bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --model "$own/best.pth" --disable-search --unsafe-writes \
  --validation-split /data/gb/outputs/c1_initial_seed42/split.json --output "$run/predictions"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json --labels prefix_own_s42_no_search_raw_write \
  --runs "$run/predictions" --reference-labels baseline c1 prefix_own_s42 \
  --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
               /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
               "$own/predictions" --output "$run/report"
date -Iseconds > "$run/job_completed.txt"
