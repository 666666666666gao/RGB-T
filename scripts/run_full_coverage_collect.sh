#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
[[ "$gpu" == 0 || "$gpu" == 1 || "$gpu" == 2 || "$gpu" == 3 ]]
cd /data/gb/GOLA
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/full_coverage_collect_source_review_20261004.json")); assert d["status"] == "PASS"'
parent=$(/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/full_coverage_jobs_cpu_acceptance_20261004.json")); assert d["status"] == "PASS" and d["all_train_video_names_exact"] and d["all_validation_video_names_exact"]; print(d["teacher"])')
for partition in train validation; do
  run="/data/gb/outputs/recoverability_full_coverage_collect_${partition}_gpu${gpu}_20261004"
  test ! -e "$run"
  bash scripts/run_temporal.sh "$gpu" collect_recoverability \
    --root /data/wangwj/dataset/LasHeR --partition "$partition" --clips 16 \
    --batch-clips 16 --forward-batch 64 --max-prefix 1024 --seed 42 \
    --prefix-model "$parent" --future-policy own --prefix-write-verification action \
    --jobs-file "/data/gb/setup/full_coverage_jobs_${partition}_gpu${gpu}_20261004.json" --output "$run"
  date -Iseconds > "$run/job_completed.txt"
done
