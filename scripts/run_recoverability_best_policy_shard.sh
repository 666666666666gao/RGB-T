#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
shard="$2"
[[ "$shard" == train0 || "$shard" == train1 || "$shard" == train2 || "$shard" == validation ]]
partition=train
if [[ "$shard" == validation ]]; then
  partition=validation
fi
cd /data/gb/GOLA
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/best_policy_shard_source_review_20261004.json")); assert d["status"] == "PASS" and d["review_independence"] == "same-family" and d["acceptance_status"] == "provisional"'
/data/gb/envs/gola/bin/python -c 'import json; d=json.load(open("/data/gb/setup/best_policy_probe_actual_cpu_acceptance_20261004.json")); assert d["status"] == "PASS" and d["both_partitions_passed"] and d["prefix_epoch"] == 4 and d["future_teacher_is_global_best"] and d["clips_per_partition"] == 16'
run="/data/gb/outputs/recoverability_best_policy_collect_${shard}_rest_20261004"
test ! -e "$run"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export OMP_NUM_THREADS=4 TORCH_HOME=/data/gb/cache/torch CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME=/data/gb/cache TMPDIR=/data/gb/cache/tmp
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability \
  --root /data/wangwj/dataset/LasHeR \
  --partition "$partition" --clips 16 --batch-clips 16 --forward-batch 64 --max-prefix 1024 --seed 42 \
  --prefix-model /data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth \
  --future-policy own --prefix-write-verification action \
  --jobs-file "/data/gb/setup/best_policy_jobs_${shard}_rest_20261004.json" \
  --output "$run"
date -Iseconds > "$run/job_completed.txt"
