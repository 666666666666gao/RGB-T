#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
mode="$2"
prefix="$3"
reference="recoverability_search_${mode}_${prefix}_b256_full_20261004"
while [[ ! -f "/data/gb/outputs/$reference/job_completed.txt" ]]; do
  if ! tmux has-session -t "$reference" 2>/dev/null; then
    printf 'Reference stopped without completion: %s\n' "$reference" >&2
    exit 1
  fi
  sleep 240
done
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$memory" -lt 500 ]]
bash scripts/run_recoverability_pairwise.sh "$gpu" "$mode" m0 "$prefix"
