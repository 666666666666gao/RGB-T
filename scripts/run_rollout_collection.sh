#!/usr/bin/env bash
set -eu
gpu="$1"
partition="$2"
output="$3"
clips="$4"
max_history="$5"
horizon="$6"
batch_clips="$7"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export TORCH_HOME=/data/gb/cache/torch
export XDG_CACHE_HOME=/data/gb/cache
export TMPDIR=/data/gb/cache/tmp
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /data/gb/GOLA
mkdir -p "$output"
/data/gb/envs/gola/bin/python -u -m research.collect_rollouts \
  --partition "$partition" --output "$output" --clips "$clips" \
  --batch-clips "$batch_clips" --min-history 1 --max-history "$max_history" \
  --horizon "$horizon" > "$output/stdout.log" 2>&1
