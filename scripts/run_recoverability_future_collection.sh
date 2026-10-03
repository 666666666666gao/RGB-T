#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
policy="$2"
shard="$3"
[[ "$policy" == c1 || "$policy" == own ]]
[[ "$shard" =~ ^[0-3]$ ]]
source_dir=/data/gb/setup/future_policy_review_source
jobs_dir=/data/gb/setup/recoverability_future_policy_jobs_20261004
/data/gb/envs/gola/bin/python -c 'import json; assert json.load(open("/data/gb/setup/future_policy_m0_acceptance_20261004.json"))["status"] == "PASS"'
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/gb/envs/gola/lib PYTHONPATH=/data/gb/GOLA
export TORCH_HOME=/data/gb/cache/torch XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
for partition in train validation; do
  output="/data/gb/outputs/recoverability_future_${policy}_${partition}_shard${shard}_20261004"
  test ! -e "$output"
  /data/gb/envs/gola/bin/python -u "$source_dir/run_isolated_recoverability_collection.py" \
    --source-directory "$source_dir" --partition "$partition" \
    --jobs-file "$jobs_dir/${partition}_shard${shard}.json" \
    --prefix-model /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth \
    --prefix-write-verification action --future-policy "$policy" \
    --batch-clips 1 --forward-batch 1 --max-prefix 1024 --seed 42 --output "$output"
  date -Iseconds > "$output/job_completed.txt"
done
