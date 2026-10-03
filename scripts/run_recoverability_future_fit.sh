#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
policy="$2"
stage="$3"
[[ "$policy" == c1 || "$policy" == own ]]
[[ "$stage" == m0 || "$stage" == full ]]
cache=/data/gb/outputs/recoverability_future_policy_merged_20261004
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/outputs/recoverability_future_policy_merged_20261004/paired_cache_gate.json")); assert d["status"] == "PASS" and d["all_non_future_arrays_exact"]; assert d["matched_partitions"] == {"train":902,"validation":128}'
epochs=3
if [[ "$stage" == full ]]; then
  /data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/future_policy_fit_m0_acceptance_20261004.json")); assert d["status"] == "PASS" and d["both_teacher_arms_passed"] and d["batch_size"] == 384 and d["optimizer_steps_per_arm"] == 9'
  epochs=60
fi
run="/data/gb/outputs/recoverability_future_${policy}_b384_${stage}_20261004"
test ! -e "$run"
bash scripts/run_temporal.sh "$gpu" train_recoverability \
  --train "$cache/$policy/train" --validation "$cache/$policy/validation" \
  --batch-size 384 --epochs "$epochs" --seed 42 --search-supervision oracle \
  --action-ranking pairwise --write-verification action --output "$run"
date -Iseconds > "$run/training_completed.txt"
if [[ "$stage" == full ]]; then
  bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --model "$run/best.pth" --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
    --write-verification action --seed 42 --output "$run/predictions"
  export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/data/gb/envs/gola/lib PYTHONPATH=/data/gb/GOLA OMP_NUM_THREADS=4
  /data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --split /data/gb/outputs/c1_initial_seed42/split.json --labels "future_${policy}_b384" \
    --runs "$run/predictions" --reference-labels baseline c1 write_pair_reference_own_b384 \
    --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
                 /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
                 /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions \
    --output "$run/report"
fi
date -Iseconds > "$run/job_completed.txt"
