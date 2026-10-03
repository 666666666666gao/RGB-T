#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
entry="$2"
shift 2
cd /data/gb/GOLA
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export TORCH_HOME=/data/gb/cache/torch
export XDG_CACHE_HOME=/data/gb/cache
export TMPDIR=/data/gb/cache/tmp
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /data/gb/envs/gola/bin/python -u -m "research.$entry" "$@"
