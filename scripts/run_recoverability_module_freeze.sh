#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
variant="$2"
stage="$3"
case "$variant" in
  bc) frozen=(A);;
  ac) frozen=(B);;
  ab) frozen=(C);;
  c) frozen=(A B);;
  *) exit 2;;
esac
[[ "$stage" == m0 || "$stage" == full ]]
cache=/data/gb/outputs/recoverability_current_policy_merged_20261004
private_source=/data/gb/experiments/recoverability_module_freeze_20261004
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/outputs/recoverability_current_policy_merged_20261004/paired_cache_gate.json")); assert d["status"] == "PASS" and d["all_non_future_arrays_exact"] and d["matched_partitions"] == {"train":902,"validation":128}'
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/module_freeze_runner_audit_source_review_20261004.json")); assert d["status"] == "PASS" and d["review_independence"] == "same-family" and d["acceptance_status"] == "provisional"'
epochs=3
if [[ "$stage" == full ]]; then
  /data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/module_freeze_m0_acceptance_20261004.json")); assert d["status"] == "PASS" and d["all_four_arms_passed"] and d["batch_size"] == 416 and d["optimizer_steps_per_arm"] == 9 and d["epochs_per_arm"] == 3'
  epochs=60
fi
run="/data/gb/outputs/recoverability_module_freeze_${variant}_b416_${stage}_20261004"
test ! -e "$run"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH="$private_source:/data/gb/GOLA" OMP_NUM_THREADS=4
export TORCH_HOME=/data/gb/cache/torch CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp
cd "$private_source"
/data/gb/envs/gola/bin/python -c 'from pathlib import Path; import research.train_recoverability as t, research.recoverability_modules as m; root=Path("/data/gb/experiments/recoverability_module_freeze_20261004/research"); assert Path(t.__file__).parent == Path(m.__file__).parent == root; print("TRAINING_SOURCE", t.__file__, m.__file__, flush=True)'
/data/gb/envs/gola/bin/python -u -m research.train_recoverability \
  --train "$cache/own/train" --validation "$cache/own/validation" \
  --batch-size 416 --epochs "$epochs" --seed 42 --search-supervision oracle \
  --action-ranking budgeted --write-verification action --frozen-modules "${frozen[@]}" \
  --init-checkpoint /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth \
  --output "$run"
date -Iseconds > "$run/training_completed.txt"
date -Iseconds > "$run/job_completed.txt"
