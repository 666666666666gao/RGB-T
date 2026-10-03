#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
batch="$1"
for gpu in 0 1 2 3; do
  seed=$((42 + gpu))
  run="recoverability_abc_s${seed}_20261003"
  tmux new-session -d -s "$run" "bash /data/gb/GOLA/scripts/run_temporal.sh $gpu train_recoverability --train /data/gb/outputs/recoverability_train_s42_20261003 /data/gb/outputs/recoverability_train_s43_20261003 /data/gb/outputs/recoverability_train_s44_20261003 --validation /data/gb/outputs/recoverability_validation_s100042_20261003 --output /data/gb/outputs/$run --epochs 30 --batch-size $batch --seed $seed > /data/gb/setup/$run.log 2>&1"
done
