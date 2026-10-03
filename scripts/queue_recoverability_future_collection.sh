#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
wait_marker="$2"
shard="$3"
shift 3
while ! test -f "$wait_marker"; do
  sleep 240
done
for policy in "$@"; do
  bash /data/gb/setup/future_policy_review_source/run_recoverability_future_collection.sh "$gpu" "$policy" "$shard"
done
