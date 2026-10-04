#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
arm="$2"
stage="$3"
[[ "$arm" == pairwise_lr4 || "$arm" == pairwise_lr5 || "$arm" == budgeted_lr4 || "$arm" == budgeted_lr5 ]]
[[ "$stage" == m0 || "$stage" == full ]]
ranking=budgeted
if [[ "$arm" == pairwise_lr4 || "$arm" == pairwise_lr5 ]]; then
  ranking=pairwise
fi
lr=1e-4
if [[ "$arm" == pairwise_lr5 || "$arm" == budgeted_lr5 ]]; then
  lr=1e-5
fi
cache=/data/gb/outputs/recoverability_best_policy_merged_20261004
private=/data/gb/experiments/recoverability_best_policy_fit_20261004
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/outputs/recoverability_best_policy_merged_20261004/best_policy_cache_cpu_gate.json")); assert d["status"] == "PASS" and d["matched_partitions"] == {"train":902,"validation":128} and d["all_ordered_jobs_exact"] and d["all_GT_current_and_history_replayed"] and d["all_schema_and_sources_verified"] and d["prefix_epoch"] == d["future_epoch"] == 4'
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/best_policy_fit_source_review_20261004.json")); assert d["status"] == "PASS" and d["review_independence"] == "same-family" and d["acceptance_status"] == "provisional"'
epochs=3
if [[ "$stage" == full ]]; then
  /data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/best_policy_fit_m0_acceptance_20261004.json")); assert d["status"] == "PASS" and d["all_four_lr_ranking_arms_passed"] and d["batch_size"] == 416 and d["optimizer_steps_per_arm"] == 9 and d["epochs_per_arm"] == 3'
  epochs=60
fi
run="/data/gb/outputs/recoverability_best_policy_${arm}_${stage}_20261004"
test ! -e "$run"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH="$private:/data/gb/GOLA" OMP_NUM_THREADS=4
export TORCH_HOME=/data/gb/cache/torch CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp
cd "$private"
/data/gb/envs/gola/bin/python -c 'from pathlib import Path; import research.train_recoverability as t, research.recoverability_modules as m; root=Path("/data/gb/experiments/recoverability_best_policy_fit_20261004/research"); assert Path(t.__file__).parent == Path(m.__file__).parent == root; print("TRAINING_SOURCE", t.__file__, m.__file__, flush=True)'
/data/gb/envs/gola/bin/python -u -m research.train_recoverability \
  --train "$cache/own/train" --validation "$cache/own/validation" \
  --batch-size 416 --epochs "$epochs" --lr "$lr" --seed 42 --search-supervision oracle \
  --action-ranking "$ranking" --write-verification action \
  --init-checkpoint /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth \
  --output "$run"
date -Iseconds > "$run/training_completed.txt"
date -Iseconds > "$run/job_completed.txt"
