#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
dataset="$2"
if [[ "$dataset" == lasher ]]; then
  data=/data/wangwj/dataset/LasHeR/testingset
else
  data=/data/zhouy/DATASET/RGB-T234
fi
run="/data/gb/outputs/recoverability_native_${dataset}_s42_20261003"
bash scripts/run_temporal.sh "$gpu" evaluate_recoverability --dataset "$dataset" --root "$data" \
  --model /data/gb/outputs/recoverability_abc_s42_20261003/best.pth --output "$run/predictions"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_core_metrics --dataset "$dataset" --data-root "$data" \
  --variants baseline c1 recoverability_s42 \
  --runs "/data/gb/outputs/online_core/${dataset}_baseline" "/data/gb/outputs/online_core_v2/${dataset}_c1" "$run" \
  --output "$run/core_report"
for reference in baseline c1; do
  if [[ "$reference" == baseline ]]; then
    reference_run="/data/gb/outputs/online_core/${dataset}_baseline"
  else
    reference_run="/data/gb/outputs/online_core_v2/${dataset}_c1"
  fi
  /data/gb/envs/gola/bin/python -u -m research.collect_core_metrics --dataset "$dataset" --data-root "$data" \
    --variants "$reference" recoverability_s42 --runs "$reference_run" "$run" --output "$run/${reference}_paired_report"
  /data/gb/envs/gola/bin/python -u -m research.paired_sequence_bootstrap \
    --reports "$run/${reference}_paired_report/full_report.json" --output "$run/${reference}_paired_report/paired_bootstrap.json"
  /data/gb/envs/gola/bin/python -u -m research.plot_core_metrics \
    --report "$run/${reference}_paired_report/full_report.json" --output "$run/${reference}_paired_report"
done
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics --dataset "$dataset" --root "$data" \
  --labels recoverability_s42 --runs "$run/predictions" --reference-labels baseline c1 \
  --references "/data/gb/outputs/online_core/${dataset}_baseline/predictions" "/data/gb/outputs/online_core_v2/${dataset}_c1/predictions" \
  --output "$run/mechanism_report"
date -Iseconds > "$run/report_completed.txt"
