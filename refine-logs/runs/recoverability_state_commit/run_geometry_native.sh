#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
[[ "$gpu" == 0 || "$gpu" == 1 || "$gpu" == 2 || "$gpu" == 3 ]]
cd /data/gb/GOLA
folder=/data/gb/GOLA/refine-logs/runs/recoverability_state_commit
parent=$(/data/gb/envs/gola/bin/python -c 'import json;d=json.load(open("/data/gb/GOLA/refine-logs/runs/recoverability_state_commit/selected_geometry_native_checkpoint.json"));assert d["all_four60fits_and98videos_complete"];print(d["parent"])')
commit=$(/data/gb/envs/gola/bin/python -c 'import json;d=json.load(open("/data/gb/GOLA/refine-logs/runs/recoverability_state_commit/selected_geometry_native_checkpoint.json"));print(d["commit_model"])')
for dataset in lasher rgbt234; do
  mapfile -t partition < <(/data/gb/envs/gola/bin/python -c 'import json,sys;d=json.load(open("/data/gb/GOLA/refine-logs/runs/recoverability_state_commit/prepared_geometry_native_frame_shards.json"))["datasets"][sys.argv[1]];s=d["shards"][int(sys.argv[2])];print(d["root"]);print(s["offset"]);print(s["count"])' "$dataset" "$gpu")
  run="/data/gb/outputs/recoverability_geometry_commit_native_20261005/$dataset/shards/gpu$gpu"
  test ! -e "$run"
  bash scripts/run_temporal.sh "$gpu" evaluate_recoverability --dataset "$dataset" --root "${partition[0]}" \
    --model "$parent" --commit-model "$commit" --pretrained /data/gb/GOLA/pretrained_models/gola_b224.bin \
    --c1-head /data/gb/outputs/c1_initial_seed42/best.pth --motion-run /data/gb/outputs/abc_joint_v1_seed42 \
    --write-verification action --seed 42 --sequence-offset "${partition[1]}" --limit-sequences "${partition[2]}" \
    --max-frames 0 --output "$run/predictions"
  date -Iseconds > "$run/inference_completed.txt"
done
