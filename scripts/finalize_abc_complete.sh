#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
root=/data/gb/outputs/abc_full_v1
while [ ! -f "$root/lasher_abc/full_evaluation_completed.txt" ] || \
      [ ! -f "$root/lasher_abc_box_only/full_evaluation_completed.txt" ] || \
      [ ! -f "$root/rgbt234_abc/full_evaluation_completed.txt" ] || \
      [ ! -f "$root/rgbt234_abc_box_only/full_evaluation_completed.txt" ] || \
      [ ! -f "$root/state_compare/lasher/state_comparison_completed.txt" ] || \
      [ ! -f "$root/state_compare/rgbt234/state_comparison_completed.txt" ]; do
  sleep 240
done
/data/gb/envs/gola/bin/python -u -m research.merge_abc_complete \
  --full-root "$root" --training-run /data/gb/outputs/abc_joint_v1_seed42 \
  --parameter-report /data/gb/GOLA/refine-logs/runs/model_parameter_counts.json \
  --output "$root/complete_core"
