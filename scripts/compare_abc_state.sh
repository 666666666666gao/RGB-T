#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
data_root="$2"
cd /data/gb/GOLA
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
main="/data/gb/outputs/abc_full_v1/${dataset}_abc"
box="/data/gb/outputs/abc_full_v1/${dataset}_abc_box_only"
output="/data/gb/outputs/abc_full_v1/state_compare/$dataset"
mkdir -p "$output"
while [ ! -f "$main/full_evaluation_completed.txt" ] || [ ! -f "$box/full_evaluation_completed.txt" ]; do
  sleep 240
done
/data/gb/envs/gola/bin/python -u -m research.collect_core_metrics \
  --dataset "$dataset" --data-root "$data_root" --variants abc_box_only abc \
  --runs "$box" "$main" --output "$output" > "$output/native.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.plot_core_metrics \
  --report "$output/full_report.json" --output "$output" > "$output/plot.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.paired_sequence_bootstrap \
  --reports "$output/full_report.json" --output "$output/paired_sequence_bootstrap.json" \
  > "$output/uncertainty.stdout.log" 2>&1
date -Iseconds > "$output/state_comparison_completed.txt"
