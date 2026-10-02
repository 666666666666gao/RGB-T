#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
output=/data/gb/outputs/full_reports/lasher
mkdir -p "$output"
while true; do
  ready=true
  for variant in baseline c1; do
    run="/data/gb/outputs/online_core/lasher_$variant"
    if [ ! -f "$run/job_completed.txt" ]; then
      if ! tmux has-session -t "online_lasher_${variant}_20261002" 2>/dev/null; then
        printf 'Inference/scoring terminated without completion: %s\n' "$run" >&2
        exit 1
      fi
      ready=false
    fi
  done
  if "$ready"; then break; fi
  sleep 180
done
/data/gb/envs/gola/bin/python -u -m research.collect_core_metrics \
  --dataset lasher --data-root /data/wangwj/dataset/LasHeR/testingset \
  --variants baseline c1 \
  --runs /data/gb/outputs/online_core/lasher_baseline /data/gb/outputs/online_core/lasher_c1 \
  --output "$output"
/data/gb/envs/gola/bin/python -u -m research.plot_core_metrics \
  --report "$output/full_report.json" --output "$output"
date -Iseconds > "$output/report_completed.txt"
