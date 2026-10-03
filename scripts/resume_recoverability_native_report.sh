#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
dataset="$1"
if [[ "$dataset" == lasher ]]; then
  data=/data/wangwj/dataset/LasHeR/testingset
else
  data=/data/zhouy/DATASET/RGB-T234
fi
run="/data/gb/outputs/recoverability_native_${dataset}_s42_20261003"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics --dataset "$dataset" --root "$data" \
  --labels recoverability_s42 --runs "$run/predictions" --reference-labels baseline c1 \
  --references "/data/gb/outputs/online_core/${dataset}_baseline/predictions" "/data/gb/outputs/online_core_v2/${dataset}_c1/predictions" \
  --output "$run/mechanism_report"
test -f "$run/core_report/full_report.json"
for reference in baseline c1; do
  test -f "$run/${reference}_paired_report/paired_bootstrap.json"
  test -f "$run/${reference}_paired_report/official_curves.png"
done
date -Iseconds > "$run/report_completed.txt"
