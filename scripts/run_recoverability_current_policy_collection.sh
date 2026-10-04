#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
policy="$2"
stage="$3"
[[ "$policy" == c1 || "$policy" == own ]]
[[ "$stage" == m0 || "$stage" =~ ^[0-3]$ ]]
source_dir=/data/gb/setup/future_policy_review_source
jobs_dir=/data/gb/setup/recoverability_future_policy_jobs_20261004
review_dir=/data/gb/setup/recoverability_current_policy_recollection
prefix_model=/data/gb/outputs/recoverability_warm_budgeted_action_own_b384_full_20261004/best.pth
/data/gb/envs/gola/bin/python - "$stage" "$review_dir" "$prefix_model" <<'PY'
import json
import sys
from pathlib import Path

stage, root, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
source = json.loads((root / 'source_review.json').read_text())
assert source['status'] == 'PASS_SOURCE_ONLY'
assert source['current_prefix_model'] == prefix
if stage != 'm0':
    receipt = json.loads((root / 'actual_m0_acceptance.json').read_text())
    assert receipt['status'] == 'PASS_ACTUAL_MATCHED_M0'
    assert receipt['prefix_model'] == prefix and receipt['prefix_checkpoint_epoch'] == 25
    assert receipt['jobs'] == json.loads((root / 'm0_jobs.json').read_text())['jobs']
    assert receipt['completed_policies'] == ['c1', 'own']
    assert receipt['nonfuture_arrays_bitwise_equal'] is True
    assert receipt['scope'] == 'MATCHED_FULL_COLLECTION_ONLY'
PY
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/gb/envs/gola/lib PYTHONPATH=/data/gb/GOLA
export TORCH_HOME=/data/gb/cache/torch XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
partitions=(train validation)
if [[ "$stage" == m0 ]]; then partitions=(train); fi
for partition in "${partitions[@]}"; do
  if [[ "$stage" == m0 ]]; then
    jobs_file="$review_dir/m0_jobs.json"
    output="/data/gb/outputs/recoverability_current_policy_m0_20261004/$policy"
  else
    jobs_file="$jobs_dir/${partition}_shard${stage}.json"
    output="/data/gb/outputs/recoverability_current_policy_${policy}_${partition}_shard${stage}_20261004"
  fi
  test ! -e "$output"
  /data/gb/envs/gola/bin/python -u "$source_dir/run_isolated_recoverability_collection.py" \
    --source-directory "$source_dir" --partition "$partition" --jobs-file "$jobs_file" \
    --prefix-model "$prefix_model" --prefix-write-verification action --future-policy "$policy" \
    --batch-clips 1 --forward-batch 1 --max-prefix 1024 --seed 42 --output "$output"
  date -Iseconds > "$output/job_completed.txt"
done
