#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
variant="$2"
gpu="$3"
root="$4"
run_output="$5"
reference="$6"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
cd /data/gb/GOLA
mkdir -p "$run_output"
bash scripts/run_online.sh --dataset "$dataset" --root "$root" --variant "$variant" \
  --record-mechanisms --output "$run_output/predictions" > "$run_output/inference.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.collect_candidate_metrics \
  --dataset "$dataset" --data-root "$root" --variants "$variant" \
  --runs "$run_output/predictions" --reference-runs "$reference/predictions" \
  --output "$run_output/mechanisms" > "$run_output/mechanisms.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u evaluation.py "$dataset" \
  --tracker_names "gola_$variant" --result_paths "$run_output/predictions" \
  > "$run_output/official_metrics.log" 2>&1
date -Iseconds > "$run_output/job_completed.txt"
