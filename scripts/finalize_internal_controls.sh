#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
root=/data/gb/outputs/abc_internal_validation_v1
while [[ ! -f "$root/baseline/predictions/inference_completion.json" || ! -f "$root/c1/predictions/inference_completion.json" ]]; do
  sleep 240
done
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.compare_internal_controls \
  --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json \
  --runs "$root/baseline" "$root/c1" "$root/initial" "$root/last" "$root/last_box_only" \
  --abc-report "$root/report/full_internal_report.json" \
  --output "$root/controls_report"
