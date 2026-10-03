#!/usr/bin/env bash
# One fixed collection -> training -> strict-audit chain; no retries.
set -eu
cd /data/gb/GOLA
train42=/data/gb/outputs/abc_temporal_train_s42_20261003
train43=/data/gb/outputs/abc_temporal_train_s43_20261003
train44=/data/gb/outputs/abc_temporal_train_s44_20261003
validation=/data/gb/outputs/abc_temporal_val_s100042_20261003
output=/data/gb/outputs/abc_joint_v1_seed42
while [ ! -f "$train42/completion.json" ] || [ ! -f "$train43/completion.json" ] || [ ! -f "$train44/completion.json" ] || [ ! -f "$validation/completion.json" ]; do
    sleep 240
done
bash scripts/run_temporal.sh 1 train_temporal --train "$train42" "$train43" "$train44" \
    --validation "$validation" --output "$output" --epochs 30 --batch-size 64
bash scripts/run_temporal.sh 1 audit_temporal_training --run "$output" --output "$output/reload_and_calibration.json"
printf 'completed\n' > "$output/training_and_audit_completed.txt"
