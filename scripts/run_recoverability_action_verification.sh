#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
arm="$2"
stage="$3"
[[ "$arm" == oracle_c1 || "$arm" == oracle_own || "$arm" == selector_own ]]
[[ "$stage" == m0 || "$stage" == full ]]
reference="/data/gb/outputs/recoverability_pairwise_${arm}_b256_full_20261004"
test -f "$reference/job_completed.txt"
run="/data/gb/outputs/recoverability_action_verify_${arm}_${stage}_20261004"
test ! -e "$run"
smoke=()
if [[ "$stage" == m0 ]]; then
  smoke=(--limit-sequences 2 --max-frames 64)
fi
if [[ "$stage" == m0 && "$arm" == oracle_c1 ]]; then
  bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --model "$reference/best.pth" --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
    --write-verification action --zero-init --parity-check --seed 42 "${smoke[@]}" \
    --output "$run/initialization_parity"
fi
bash scripts/run_temporal.sh "$gpu" evaluate_recoverability \
  --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
  --model "$reference/best.pth" --validation-split /data/gb/outputs/c1_initial_seed42/split.json \
  --write-verification action --seed 42 "${smoke[@]}" --output "$run/predictions"
if [[ "$stage" == full ]]; then
  export CUDA_VISIBLE_DEVICES=
  export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
  export PYTHONPATH=/data/gb/GOLA
  export OMP_NUM_THREADS=4
  /data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics \
    --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset \
    --split /data/gb/outputs/c1_initial_seed42/split.json --labels "action_verify_$arm" \
    --runs "$run/predictions" --reference-labels baseline c1 "pairwise_${arm}_b256" \
    --references /data/gb/outputs/abc_internal_validation_v1/baseline/predictions \
                 /data/gb/outputs/abc_internal_validation_v1/c1/predictions "$reference/predictions" \
    --output "$run/report"
fi
date -Iseconds > "$run/job_completed.txt"
