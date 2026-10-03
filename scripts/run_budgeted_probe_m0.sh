#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
control="$2"
output="$3"
shift 3
cd /data/gb/GOLA
receipt="/data/gb/outputs/abc_internal_validation_v1/$control/predictions/inference_completion.json"
while [[ ! -f "$receipt" ]]; do
  sleep 240
done
/data/gb/envs/gola/bin/python - "$receipt" <<'PY'
import json
import sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
assert receipt['completed'] and receipt['smoke_only']
assert receipt['sequences'] == 98 and receipt['frames'] == 49418
assert receipt['candidate_timeline_recorded']
PY
exec bash scripts/run_temporal.sh "$gpu" audit_dense_candidates \
  --search-training-run /data/gb/outputs/abc_joint_v1_seed42 \
  --output "$output" --limit 2 "$@"
