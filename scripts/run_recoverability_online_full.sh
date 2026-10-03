#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
for gpu in 0 1 2 3; do
  seed=$((42 + gpu))
  name="recoverability_online_s${seed}_20261003"
  tmux new-session -d -s "$name" "bash /data/gb/GOLA/scripts/run_temporal.sh $gpu evaluate_recoverability --dataset lasher --root /data/wangwj/dataset/LasHeR/traingset --model /data/gb/outputs/recoverability_abc_s${seed}_20261003/best.pth --validation-split /data/gb/outputs/c1_initial_seed42/split.json --output /data/gb/outputs/$name/predictions > /data/gb/setup/$name.log 2>&1"
done
