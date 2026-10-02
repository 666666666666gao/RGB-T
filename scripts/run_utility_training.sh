#!/usr/bin/env bash
set -eu
gpu="$1"
train="$2"
validation="$3"
output="$4"
epochs="$5"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /data/gb/GOLA
mkdir -p "$output"
/data/gb/envs/gola/bin/python -u -m research.train_branch_utility \
  --train "$train" --validation "$validation" --output "$output" \
  --epochs "$epochs" --batch-size 32 > "$output/stdout.log" 2>&1
