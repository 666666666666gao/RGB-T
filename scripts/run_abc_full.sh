#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
dataset="$2"
data_root="$3"
model="$4"
output="$5"
variant="$6"
cd /data/gb/GOLA
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
mkdir -p "$output"
bash scripts/run_temporal.sh "$gpu" evaluate_abc \
  --dataset "$dataset" --root "$data_root" --model "$model" \
  --variant "$variant" --output "$output/predictions" \
  > "$output/inference.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.collect_core_metrics \
  --dataset "$dataset" --data-root "$data_root" --variants baseline "$variant" \
  --runs "/data/gb/outputs/online_core/${dataset}_baseline" "$output" \
  --output "$output/report" > "$output/native.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.collect_abc_metrics \
  --dataset "$dataset" --data-root "$data_root" --variants "$variant" \
  --runs "$output" --output "$output/report" > "$output/abc_metrics.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.plot_core_metrics \
  --report "$output/report/full_report.json" --output "$output/report" \
  > "$output/plot.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.paired_sequence_bootstrap \
  --reports "$output/report/full_report.json" --output "$output/report/paired_sequence_bootstrap.json" \
  > "$output/uncertainty.stdout.log" 2>&1
date -Iseconds > "$output/full_evaluation_completed.txt"
