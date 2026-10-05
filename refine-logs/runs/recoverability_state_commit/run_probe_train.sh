#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
jobs="$2"
output="$3"
[[ "$gpu" == 0 || "$gpu" == 3 ]]
folder=/data/gb/GOLA/refine-logs/runs/recoverability_state_commit
bash /data/gb/GOLA/scripts/run_temporal.sh "$gpu" probe_geometry_commit \
  --root /data/wangwj/dataset/LasHeR \
  --cache trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np \
  --split /data/gb/outputs/c1_initial_seed42/split.json --jobs-file "$jobs" \
  --model /data/gb/outputs/recoverability_train_events_aug_budgeted_full_20261005/best.pth \
  --pretrained /data/gb/GOLA/pretrained_models/gola_b224.bin \
  --c1-head /data/gb/outputs/c1_initial_seed42/best.pth \
  --motion-run /data/gb/outputs/abc_joint_v1_seed42 \
  --write-verification action --horizons 3 32 --seed 42 --output "$output"
