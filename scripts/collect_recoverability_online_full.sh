#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
while [[ ! -f /data/gb/outputs/recoverability_online_s42_20261003/predictions/inference_completion.json || ! -f /data/gb/outputs/recoverability_online_s43_20261003/predictions/inference_completion.json || ! -f /data/gb/outputs/recoverability_online_s44_20261003/predictions/inference_completion.json || ! -f /data/gb/outputs/recoverability_online_s45_20261003/predictions/inference_completion.json ]]; do
  sleep 240
done
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json \
  --runs /data/gb/outputs/recoverability_online_s42_20261003/predictions /data/gb/outputs/recoverability_online_s43_20261003/predictions /data/gb/outputs/recoverability_online_s44_20261003/predictions /data/gb/outputs/recoverability_online_s45_20261003/predictions \
  --labels recoverability_s42 recoverability_s43 recoverability_s44 recoverability_s45 \
  --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
  --reference-labels baseline c1 \
  --output /data/gb/outputs/recoverability_online_full_report_20261003
