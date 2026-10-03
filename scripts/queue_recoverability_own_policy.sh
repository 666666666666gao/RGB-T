#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
for control in no_search unsafe_writes; do
  session="recoverability_${control}_s42_20261003"
  while [[ ! -f "/data/gb/outputs/$session/report_completed.txt" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      printf 'Control stopped without complete report: %s\n' "$session" >&2
      exit 1
    fi
    sleep 240
  done
done
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA
export TORCH_HOME=/data/gb/cache/torch
export XDG_CACHE_HOME=/data/gb/cache
export TMPDIR=/data/gb/cache/tmp
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/data/gb/envs/gola/bin/python -u scripts/check_recoverability_prefix.py
/data/gb/envs/gola/bin/python -c 'import json; from pathlib import Path; receipt=json.loads(Path("/data/gb/outputs/recoverability_own_policy_m0_20261003/receipt.json").read_text()); assert receipt["completed"]'
for spec in '2 train' '3 validation'; do
  read -r gpu partition <<< "$spec"
  session="recoverability_own_policy_${partition}_20261003"
  tmux new-session -d -s "$session" "cd /data/gb/GOLA && bash scripts/run_recoverability_own_policy_collect.sh $gpu $partition > /data/gb/setup/$session.log 2>&1"
done
date -Iseconds
