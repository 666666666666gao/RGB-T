#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
policy="$2"
stage="$3"
[[ "$policy" == c1 || "$policy" == own ]]
[[ "$stage" == m0 || "$stage" == full ]]
cache=/data/gb/outputs/recoverability_current_policy_merged_20261004
private_source=/data/gb/experiments/recoverability_current_policy_fit_20261004
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/outputs/recoverability_current_policy_merged_20261004/paired_cache_gate.json")); assert d["status"] == "PASS" and d["all_non_future_arrays_exact"] and d["matched_partitions"] == {"train":902,"validation":128}'
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/current_policy_fit_source_review_20261004.json")); assert d["status"] == "PASS" and d["review_independence"] == "same-family" and d["acceptance_status"] == "provisional"'
epochs=3
if [[ "$stage" == full ]]; then
  /data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/current_policy_fit_m0_acceptance_20261004.json")); assert d["status"] == "PASS" and d["both_teacher_arms_passed"] and d["batch_size"] == 384 and d["optimizer_steps_per_arm"] == 9'
  epochs=60
fi
run="/data/gb/outputs/recoverability_current_policy_fit_${policy}_b384_${stage}_20261004"
test ! -e "$run"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH="$private_source:/data/gb/GOLA" OMP_NUM_THREADS=4
export TORCH_HOME=/data/gb/cache/torch CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp
cd "$private_source"
/data/gb/envs/gola/bin/python -c 'from pathlib import Path; import research.train_recoverability as t, research.recoverability_modules as m; root=Path("/data/gb/experiments/recoverability_current_policy_fit_20261004/research"); assert Path(t.__file__).parent == Path(m.__file__).parent == root; print("TRAINING_SOURCE", t.__file__, m.__file__, flush=True)'
/data/gb/envs/gola/bin/python -u -m research.train_recoverability \
  --train "$cache/$policy/train" --validation "$cache/$policy/validation" \
  --batch-size 384 --epochs "$epochs" --seed 42 --search-supervision oracle \
  --action-ranking budgeted --write-verification action --init-checkpoint /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth --output "$run"
date -Iseconds > "$run/training_completed.txt"
if [[ "$stage" == full ]]; then
  cd /data/gb/GOLA
  bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --model "$run/best.pth" --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
    --write-verification action --seed 42 --output "$run/predictions"
  export CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/gb/GOLA
  /data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --split /data/gb/outputs/c1_initial_seed42/split.json --labels "current_policy_fit_${policy}_b384" \
    --runs "$run/predictions" --reference-labels baseline c1 write_pair_reference_own_b384 "warm_budgeted_action_${policy}_b384" \
    --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
                 /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
                 /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions \
                 "/data/gb/outputs/recoverability_warm_budgeted_action_${policy}_b384_full_20261004/predictions" \
    --output "$run/report"
fi
date -Iseconds > "$run/job_completed.txt"
