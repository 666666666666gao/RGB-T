#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
phase="$2"
[[ "$gpu" == 0 || "$gpu" == 1 || "$gpu" == 2 || "$gpu" == 3 ]]
[[ "$phase" == sanity || "$phase" == full ]]
cd /data/gb/GOLA
if [[ "$phase" == sanity ]]; then
  partitions=(sanity)
else
  /data/gb/envs/gola/bin/python -c 'import json; assert json.load(open("/data/gb/setup/best_native_policy_sanity_cpu_acceptance_20261005.json"))["status"] == "PASS"'
  partitions=(train validation)
fi
for part in "${partitions[@]}"; do
  partition="$part"
  [[ "$part" != sanity ]] || partition=train
  run="/data/gb/outputs/recoverability_best_native_policy_collect_${part}_gpu${gpu}_20261005"
  test ! -e "$run"
  bash scripts/run_temporal.sh "$gpu" collect_recoverability \
    --root /data/wangwj/dataset/LasHeR --partition "$partition" --clips 16 \
    --batch-clips 16 --forward-batch 64 --max-prefix 1024 --seed 42 \
    --prefix-model /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth \
    --future-policy own --prefix-write-verification action \
    --jobs-file "/data/gb/setup/best_native_policy_jobs_${part}_gpu${gpu}_20261005.json" --output "$run"
  date -Iseconds > "$run/job_completed.txt"
done
