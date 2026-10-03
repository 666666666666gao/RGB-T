#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
bash scripts/run_temporal.sh 3 train_recoverability \
  --train /data/gb/outputs/recoverability_m0_20261003/train \
  --validation /data/gb/outputs/recoverability_m0_20261003/validation \
  --output /data/gb/outputs/recoverability_fit_m0_20261003 \
  --epochs 2 --batch-size 2 --seed 42
bash scripts/run_temporal.sh 3 train_recoverability \
  --train /data/gb/outputs/recoverability_fit_capacity_20261003/train \
  --validation /data/gb/outputs/recoverability_fit_capacity_20261003/validation \
  --output /data/gb/outputs/recoverability_fit_capacity64_20261003 \
  --epochs 1 --batch-size 64 --seed 42
