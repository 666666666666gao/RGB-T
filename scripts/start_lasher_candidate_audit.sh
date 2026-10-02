#!/usr/bin/env bash
set -euo pipefail
variant="$1"
gpu="$2"
reference="/data/gb/outputs/online_core/lasher_$variant"
while [ ! -f "$reference/job_completed.txt" ]; do
  if ! tmux has-session -t "online_lasher_${variant}_20261002" 2>/dev/null; then
    printf 'Original LasHeR task terminated without completion: %s\n' "$reference" >&2
    exit 1
  fi
  sleep 180
done
bash /data/gb/GOLA/scripts/run_candidate_audit.sh lasher "$variant" "$gpu" \
  /data/wangwj/dataset/LasHeR/testingset \
  "/data/gb/outputs/candidate_audit/lasher_$variant" "$reference"
