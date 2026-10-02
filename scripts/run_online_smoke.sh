#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export CUDA_VISIBLE_DEVICES=1
mkdir -p /data/gb/outputs/online_smoke
for variant in baseline c1; do
  bash scripts/run_online.sh --dataset lasher \
    --root /data/wangwj/dataset/LasHeR/testingset --variant "$variant" \
    --output "/data/gb/outputs/online_smoke/lasher_$variant" \
    --limit-sequences 2 --max-frames 64 \
    > "/data/gb/outputs/online_smoke/lasher_$variant.stdout.log" 2>&1
  bash scripts/run_online.sh --dataset rgbt234 \
    --root /data/zhouy/DATASET/RGB-T234 --variant "$variant" \
    --output "/data/gb/outputs/online_smoke/rgbt234_$variant" \
    --limit-sequences 2 --max-frames 64 \
    > "/data/gb/outputs/online_smoke/rgbt234_$variant.stdout.log" 2>&1
done
date -Iseconds > /data/gb/outputs/online_smoke/all_smokes_completed.txt
