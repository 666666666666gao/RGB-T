#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export OMP_NUM_THREADS=4
for arm in oracle_c1 oracle_own selector_own; do
 train=/data/gb/outputs/recoverability_own_policy_train_20261003
 if [[ "$arm" == oracle_c1 ]]; then train=/data/gb/outputs/recoverability_paired_c1_prefix_20261003; fi
 /data/gb/envs/gola/bin/python -u scripts/diagnose_recoverability_write_decisions.py --model /data/gb/outputs/recoverability_pairwise_${arm}_b256_full_20261004/best.pth --cache "$train" --partition train --output /data/gb/outputs/recoverability_action_verification_cache_20261004/${arm}_train.json
 /data/gb/envs/gola/bin/python -u scripts/diagnose_recoverability_write_decisions.py --model /data/gb/outputs/recoverability_pairwise_${arm}_b256_full_20261004/best.pth --cache /data/gb/outputs/recoverability_own_policy_validation_20261003 --partition validation --output /data/gb/outputs/recoverability_action_verification_cache_20261004/${arm}_validation.json
done
date -Is > /data/gb/outputs/recoverability_action_verification_cache_20261004/job_completed.txt
