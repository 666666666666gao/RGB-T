#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
arm="$2"
reference="recoverability_pairwise_${arm}_b256_full_20261004"
while [[ ! -f "/data/gb/outputs/$reference/job_completed.txt" ]]; do
  if ! tmux has-session -t "$reference" 2>/dev/null; then
    printf 'Reference stopped without completion: %s\n' "$reference" >&2
    exit 1
  fi
  sleep 240
done
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$memory" -lt 500 ]]
bash scripts/run_recoverability_action_verification.sh "$gpu" "$arm" m0
