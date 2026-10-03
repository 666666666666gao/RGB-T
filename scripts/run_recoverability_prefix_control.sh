#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
arm="$2"
seed="$3"
if [[ "$arm" == c1 ]]; then
  prefix_session=recoverability_paired_c1_prefix_20261003
elif [[ "$arm" == own ]]; then
  prefix_session=recoverability_own_policy_train_20261003
else
  printf 'Unknown prefix-control arm: %s\n' "$arm" >&2
  exit 1
fi
while [[ ! -f "/data/gb/outputs/$prefix_session/job_completed.txt" ]]; do
  if ! tmux has-session -t "$prefix_session" 2>/dev/null; then
    printf 'Prefix collection stopped without completion: %s\n' "$prefix_session" >&2
    exit 1
  fi
  sleep 240
done
test -f /data/gb/outputs/recoverability_own_policy_validation_20261003/job_completed.txt
run="/data/gb/outputs/recoverability_prefix_${arm}_s${seed}_20261003"
bash scripts/run_temporal.sh "$gpu" train_recoverability \
  --train /data/gb/outputs/recoverability_train_s42_20261003 /data/gb/outputs/recoverability_train_s43_20261003 \
          /data/gb/outputs/recoverability_train_s44_20261003 "/data/gb/outputs/$prefix_session" \
  --prefer-last-prefix --validation /data/gb/outputs/recoverability_own_policy_validation_20261003 \
  --batch-size 128 --epochs 30 --seed "$seed" --output "$run"
date -Iseconds > "$run/training_completed.txt"
bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --model "$run/best.pth" --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
  --seed "$seed" --output "$run/predictions"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --split /data/gb/outputs/c1_initial_seed42/split.json --labels "prefix_${arm}_s${seed}" \
  --runs "$run/predictions" --reference-labels baseline c1 \
  --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
               /data/gb/outputs/abc_internal_validation_v1/c1/predictions \
  --output "$run/report"
date -Iseconds > "$run/job_completed.txt"
