#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/gb/envs/gola/lib PYTHONPATH=/data/gb/GOLA TORCH_HOME=/data/gb/cache/torch XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
for policy in c1 own; do
 /data/gb/envs/gola/bin/python -u /data/gb/setup/future_policy_review_source/run_isolated_recoverability_collection.py --source-directory /data/gb/setup/future_policy_review_source --partition train --jobs-file /data/gb/setup/future_policy_review_source/m0_jobs.json --prefix-model /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth --prefix-write-verification action --future-policy "$policy" --batch-clips 1 --forward-batch 1 --max-prefix 64 --seed 42 --output "/data/gb/outputs/future_policy_cli_m0_${policy}_20261004"
done
