#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
partition="$2"
clips="$3"
seed="$4"
cd /data/gb/GOLA
output="/data/gb/outputs/recoverability_${partition}_s${seed}_20261003"
exec bash scripts/run_temporal.sh "$gpu" collect_recoverability \
  --partition "$partition" --clips "$clips" --seed "$seed" \
  --batch-clips 32 --forward-batch 256 --max-prefix 1024 --output "$output"
