#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
[[ "$dataset" == lasher || "$dataset" == rgbt234 ]]
run="/data/gb/outputs/recoverability_write_events_native_full_20261005/$dataset"
if [[ "$dataset" == lasher ]]; then
  data=/data/wangwj/dataset/LasHeR/testingset
else
  data=/data/zhouy/DATASET/RGB-T234
fi
cd /data/gb/GOLA
export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/data/gb/envs/gola/lib
export PYTHONPATH=/data/gb/GOLA OMP_NUM_THREADS=4
/data/gb/envs/gola/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS" and d["all_four_actual_shards_completed"] and d["all_sequences_full_frames"]' "$run/full_inference_merge_acceptance.json"
/data/gb/envs/gola/bin/python -u -m research.collect_core_metrics --dataset "$dataset" --data-root "$data" \
  --variants baseline c1 old4 write_events_complete \
  --runs "/data/gb/outputs/online_core/${dataset}_baseline" "/data/gb/outputs/online_core_v2/${dataset}_c1" "/data/gb/outputs/recoverability_write_pair_native_${dataset}_own_b384_full_20261004" "$run" \
  --output "$run/core_report"
for reference in baseline c1 old4; do
  if [[ "$reference" == baseline ]]; then
    reference_run="/data/gb/outputs/online_core/${dataset}_baseline"
  elif [[ "$reference" == old4 ]]; then
    reference_run="/data/gb/outputs/recoverability_write_pair_native_${dataset}_own_b384_full_20261004"
  else
    reference_run="/data/gb/outputs/online_core_v2/${dataset}_c1"
  fi
  /data/gb/envs/gola/bin/python -u -m research.collect_core_metrics --dataset "$dataset" --data-root "$data" \
    --variants "$reference" write_events_complete --runs "$reference_run" "$run" --output "$run/${reference}_paired_report"
  /data/gb/envs/gola/bin/python -u -m research.paired_sequence_bootstrap \
    --reports "$run/${reference}_paired_report/full_report.json" --output "$run/${reference}_paired_report/paired_bootstrap.json"
  /data/gb/envs/gola/bin/python -u -m research.plot_core_metrics \
    --report "$run/${reference}_paired_report/full_report.json" --output "$run/${reference}_paired_report"
done
/data/gb/envs/gola/bin/python -u -m research.collect_recoverability_metrics --dataset "$dataset" --root "$data" \
  --labels write_events_complete --runs "$run/predictions" --reference-labels baseline c1 old4 \
  --references "/data/gb/outputs/online_core/${dataset}_baseline/predictions" "/data/gb/outputs/online_core_v2/${dataset}_c1/predictions" "/data/gb/outputs/recoverability_write_pair_native_${dataset}_own_b384_full_20261004/predictions" \
  --output "$run/mechanism_report"
date -Iseconds > "$run/report_completed.txt"
