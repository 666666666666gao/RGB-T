#!/usr/bin/env bash
set -euo pipefail
cd /data/gb/GOLA
gpu="$1"
reference=recoverability_prefix_own_s42_20261003
while [[ ! -f "/data/gb/outputs/$reference/job_completed.txt" ]]; do
  if ! tmux has-session -t "$reference" 2>/dev/null; then
    printf 'Full own-prefix evaluation stopped without completion: %s\n' "$reference" >&2
    exit 1
  fi
  sleep 240
done
run=/data/gb/outputs/recoverability_prefix_capacity_b256_20261003
bash scripts/run_temporal.sh "$gpu" train_recoverability \
  --train /data/gb/outputs/recoverability_train_s42_20261003 /data/gb/outputs/recoverability_train_s43_20261003 \
          /data/gb/outputs/recoverability_train_s44_20261003 /data/gb/outputs/recoverability_own_policy_train_20261003 \
  --prefer-last-prefix --validation /data/gb/outputs/recoverability_own_policy_validation_20261003 \
  --batch-size 256 --epochs 1 --seed 42 --output "$run"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/data/gb/envs/gola/lib
/data/gb/envs/gola/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data/gb/outputs/recoverability_prefix_capacity_b256_20261003')
config = json.loads((root / 'config.json').read_text())
receipt = json.loads((root / 'completion.json').read_text())
assert config['train_clips'] == 902 and config['validation_clips'] == 128
assert config['batch_size'] == 256 and receipt['epochs'] == 1 and receipt['optimizer_steps'] == 4
assert receipt['completed'] and all(receipt['modules_changed'].values())
assert receipt['frozen_c1_gradients_absent'] and receipt['strict_reload_metrics_equal']
weight = root / 'best.pth'
assert weight.resolve().parent == root.resolve()
size = weight.stat().st_size
weight.unlink()  # User requested removal of unnecessary experimental weights.
cleanup = {'completed': True, 'deleted_weights': [str(weight)], 'deleted_bytes': size,
           'scope': 'One-epoch real B256 capacity check, not a performance candidate or matched-prefix result.',
           'peak_cuda_mib': receipt['peak_cuda_mib'], 'optimizer_steps': receipt['optimizer_steps'],
           'logs_and_metrics_retained': True, 'other_experiment_weights_touched': False}
(root / 'capacity_cleanup.json').write_text(json.dumps(cleanup, indent=2))
print(json.dumps(cleanup), flush=True)
PY
date -Iseconds > "$run/job_completed.txt"
