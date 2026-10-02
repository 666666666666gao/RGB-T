#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
dataset="$1"
variant="$2"
gpu="$3"
root="$4"
run_output="$5"
export CUDA_VISIBLE_DEVICES="$gpu"
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
mkdir -p "$run_output"
bash scripts/run_online.sh --dataset "$dataset" --root "$root" --variant "$variant" \
  --output "$run_output/predictions" > "$run_output/inference.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u evaluation.py "$dataset" \
  --tracker_names "gola_$variant" --result_paths "$run_output/predictions" \
  > "$run_output/official_metrics.log" 2>&1
date -Iseconds > "$run_output/job_completed.txt"
