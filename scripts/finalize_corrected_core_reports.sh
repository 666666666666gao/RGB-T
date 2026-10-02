#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
data_root="$2"
cd /data/gb/GOLA
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
run="/data/gb/outputs/online_core_v2/${dataset}_c1"
output="/data/gb/outputs/full_reports_v2/$dataset"
mkdir -p "$output"
while [ ! -f "$run/job_completed.txt" ]; do
  if ! tmux has-session -t "online_${dataset}_c1_precision_v2" 2>/dev/null; then
    printf 'Inference/scoring terminated without completion: %s\n' "$run" >&2
    exit 1
  fi
  sleep 180
done
/data/gb/envs/gola/bin/python -u -m research.collect_core_metrics \
  --dataset "$dataset" --data-root "$data_root" --variants baseline c1 \
  --runs "/data/gb/outputs/online_core/${dataset}_baseline" "$run" \
  --output "$output" > "$output/collect_fixed.stdout.log" 2>&1
/data/gb/envs/gola/bin/python -u -m research.plot_core_metrics \
  --report "$output/full_report.json" --output "$output" \
  > "$output/plot_fixed.stdout.log" 2>&1
date -Iseconds > "$output/report_completed.txt"
