#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
mode="$2"
stage="$3"
prefix="$4"
[[ "$mode" == oracle || "$mode" == selector ]]
[[ "$stage" == m0 || "$stage" == full ]]
[[ "$prefix" == own || "$prefix" == c1 ]]
prefix_root=/data/gb/outputs/recoverability_own_policy_train_20261003
if [[ "$prefix" == c1 ]]; then
  prefix_root=/data/gb/outputs/recoverability_paired_c1_prefix_20261003
fi
epochs=60
if [[ "$stage" == m0 ]]; then
  epochs=1
fi
run="/data/gb/outputs/recoverability_pairwise_${mode}_${prefix}_b256_${stage}_20261004"
test ! -e "$run"
test -f "/data/gb/outputs/recoverability_search_${mode}_${prefix}_b256_full_20261004/job_completed.txt"
test -f "$prefix_root/job_completed.txt"
test -f /data/gb/outputs/recoverability_own_policy_validation_20261003/job_completed.txt
bash scripts/run_temporal.sh "$gpu" train_recoverability \
  --train /data/gb/outputs/recoverability_train_s42_20261003 /data/gb/outputs/recoverability_train_s43_20261003 \
          /data/gb/outputs/recoverability_train_s44_20261003 "$prefix_root" \
  --prefer-last-prefix --validation /data/gb/outputs/recoverability_own_policy_validation_20261003 \
  --batch-size 256 --epochs "$epochs" --seed 42 --search-supervision "$mode" --action-ranking pairwise --output "$run"
date -Iseconds > "$run/training_completed.txt"
if [[ "$stage" == full ]]; then
  bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --model "$run/best.pth" --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
    --seed 42 --output "$run/predictions"
  export CUDA_VISIBLE_DEVICES=
  export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
  export PYTHONPATH=/data/gb/GOLA
  export OMP_NUM_THREADS=4
  /data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --split /data/gb/outputs/c1_initial_seed42/split.json --labels "pairwise_${mode}_${prefix}_b256" \
    --runs "$run/predictions" --reference-labels baseline c1 "search_${mode}_${prefix}_b256" \
    --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
                 /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
                 "/data/gb/outputs/recoverability_search_${mode}_${prefix}_b256_full_20261004/predictions" \
    --output "$run/report"
fi
date -Iseconds > "$run/job_completed.txt"
