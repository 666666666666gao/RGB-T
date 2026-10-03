#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/gb/envs/gola/lib PYTHONPATH=/data/gb/GOLA TORCH_HOME=/data/gb/cache/torch XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /data/gb/envs/gola/bin/python -u /data/gb/setup/future_policy_review_source/check_recoverability_future_policy.py --candidate-dir /data/gb/setup/future_policy_review_source --old-tracker /data/gb/setup/future_policy_review_source/old_tracker.py --old-collector /data/gb/setup/future_policy_review_source/old_collector.py --model /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth --output /data/gb/outputs/future_policy_state_m0_v3_20261004
