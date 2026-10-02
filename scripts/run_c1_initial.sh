#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export TORCH_HOME=/data/gb/cache/torch
export XDG_CACHE_HOME=/data/gb/cache
export TMPDIR=/data/gb/cache/tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /data/gb/envs/gola/bin/python -u -m research.train_candidate \
  --output /data/gb/outputs/c1_initial_seed42 \
  --epochs 3 --steps-per-epoch 512 --validation-steps 32 \
  --batch-size 8 --workers 4 --log-every 20
