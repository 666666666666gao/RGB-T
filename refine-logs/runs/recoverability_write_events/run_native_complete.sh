#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
[[ "$gpu" == 0 || "$gpu" == 1 || "$gpu" == 2 || "$gpu" == 3 ]]
model=$(/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/write_events_native_selection_20261005.json")); assert d["status"] == "PASS" and d["all_four_full_fits_CPU_passed"] and d["completed_epochs"] in (60, 84) and d["optimizer_steps"] == 480; print(d["checkpoint"])')
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/GOLA/refine-logs/runs/recoverability_write_events/evaluation_source_review.json")); assert d["status"] == "PASS" and d["review_independence"] == "same-family" and d["acceptance_status"] == "provisional"'
lasher_offsets=(0 62 123 184)
lasher_counts=(62 61 61 61)
rgbt234_offsets=(0 59 118 176)
rgbt234_counts=(59 59 58 58)
for dataset in lasher rgbt234; do
  if [[ "$dataset" == lasher ]]; then
    data=/data/wangwj/dataset/LasHeR/testingset
    offset="${lasher_offsets[$gpu]}"
    count="${lasher_counts[$gpu]}"
  else
    data=/data/zhouy/DATASET/RGB-T234
    offset="${rgbt234_offsets[$gpu]}"
    count="${rgbt234_counts[$gpu]}"
  fi
  run="/data/gb/outputs/recoverability_write_events_native_full_20261005/$dataset/shards/gpu$gpu"
  test ! -e "$run"
  bash /data/gb/GOLA/scripts/run_temporal.sh "$gpu" evaluate_recoverability \
    --dataset "$dataset" --root "$data" --model "$model" --write-verification action \
    --seed 42 --sequence-offset "$offset" --limit-sequences "$count" --max-frames 0 \
    --output "$run/predictions"
  date -Iseconds > "$run/inference_completed.txt"
done
