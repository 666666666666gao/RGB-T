#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
control="$2"
if [[ "$control" == no_search ]]; then
  flag=--disable-search
else
  flag=--unsafe-writes
fi
run="/data/gb/outputs/recoverability_${control}_s42_20261003"
bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --model /data/gb/outputs/recoverability_abc_s42_20261003/best.pth \
  --validation-split /data/gb/outputs/c1_initial_seed42/split.json "$flag" --output "$run/predictions"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json \
  --labels "$control" --runs "$run/predictions" --reference-labels baseline c1 \
  --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
  --output "$run/report"
date -Iseconds > "$run/report_completed.txt"
