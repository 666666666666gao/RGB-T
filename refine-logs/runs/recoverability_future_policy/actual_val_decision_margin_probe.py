"""Authorized CPU-only conditional VAL probe; no backbone, optimizer or CUDA."""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
sys.path.insert(0, "/data/gb/GOLA")
import numpy as np
import torch
from research.recoverability_modules import (
    DECISION_FIELDS, RecoverabilityModules, action_utility,
    search_supervision_targets, select_actions,
)
from research.train_recoverability import LABEL_FIELDS, load_data, forward

VAL = Path("/data/gb/outputs/recoverability_future_policy_merged_20261004/own/validation")
C1 = Path("/data/gb/outputs/c1_initial_seed42/best.pth")
OUT = Path("/data/gb/setup/actual_val_decision_margin_probe_20261004.json")
MODELS = {
    "old_own4": (Path("/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth"), 4,
                 "a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72"),
    "new_own55": (Path("/data/gb/outputs/recoverability_future_own_b384_full_20261004/best.pth"), 55,
                  "284928fa3f097b633842f531ad520f8415fe9238f23ea6a99b192bcd53c95c82"),
}
SOURCE_HASHES = {
    "research/recoverability_modules.py": "371269db053cdbd38b510557e213f1acfb9c6427e6eabd2df5685c59bdfbb158",
    "research/train_recoverability.py": "4b1327a541d426aeade308e4fc3d7c8e3110e35dae4612c11b93029783fd4e93",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def distribution(value):
    a = value.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "quantiles_0_25_50_75_100": np.percentile(a, [0, 25, 50, 75, 100]).tolist()}

def scalar(value):
    return value.item() if isinstance(value, torch.Tensor) else value

def selection_row(chosen, index):
    return {k: scalar(v[index]) for k, v in chosen.items()}

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.set_grad_enabled(False)
assert all(sha(Path("/data/gb/GOLA") / p) == h for p, h in SOURCE_HASHES.items())
started = time.perf_counter()
data, configs, names, jobs = load_data([str(VAL)], "validation", torch.device("cpu"))
cfg = configs[0]
assert len(jobs) == len(data["valid"]) == 128
assert set(data) == set(DECISION_FIELDS + LABEL_FIELDS)
assert all(v.device.type == "cpu" for v in data.values())
assert data["history_instance_descriptors"].shape == (128, 1024, 2, 768)
assert cfg["future_policy_mode"] == "own" and cfg["prefix_checkpoint_epoch"] == 4
assert cfg["prefix_write_verification"] == "action"
c1 = torch.load(C1, map_location="cpu", weights_only=False)
rows = torch.arange(128)
keep = data["original_choice"].long()
utility = action_utility(data)
reference = utility[rows, 0, keep, 0]
extra_cost = torch.zeros_like(utility)
extra_cost[:, 1:] = .01
true_net = utility - reference[:, None, None, None] - extra_cost
legal = data["action_valid"]
correct = data["valid"] & (data["current_iou"] >= .5)
local_recall = correct[:, 0].any(-1)
all_recall = correct.flatten(1).any(-1)
missing = ~local_recall & all_recall
missing_indices = torch.nonzero(missing).flatten().tolist()
oracle_best = true_net.masked_fill(~legal, -torch.inf).flatten(1).max(-1).values
positive = oracle_best > .03
target_summary = {
    "queries": 128, "valid_actions": int(legal.sum()),
    "queries_with_legal_true_net_gain_gt_003": int(positive.sum()),
    "queries_with_legal_true_net_gain_gt_005": int((oracle_best > .05).sum()),
    "local_recall_queries": int(local_recall.sum()),
    "all_region_recall_queries": int(all_recall.sum()),
    "missing_correct_candidate_reintroduced_queries": int(missing.sum()),
    "missing_query_ids": [{"index": i, "sequence": jobs[i][0], "query_frame": jobs[i][1]} for i in missing_indices],
    "utility": ".7 current GT IoU + .3 mean3 future GT IoU - .1 wrong_update_fraction",
    "true_net": "utility(action)-utility(original keep regular)-.01 only for extra-region actions",
    "teacher_scope": "Frozen own4 continuation labels, conditional on this cache's identical prefix/query states",
}
all_results = {}
output_fields = ("scores", "advantage", "quality_logits", "harm_logits", "write_risk_logits",
                 "target_probability", "region_advantage", "region_success_logits", "absence_logit")
for label, (path, epoch, expected_sha) in MODELS.items():
    assert sha(path) == expected_sha
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == epoch and checkpoint["args"]["threshold"] == .03
    assert checkpoint["args"]["c1_head"] == str(C1)
    model = RecoverabilityModules(c1).cpu()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(torch.equal(model.state_dict()["c1." + k], v) for k, v in c1["head"].items())
    captured = []
    def action_hook(module, inputs, output):
        captured.append(output.detach().clone())
    hook = model.action.register_forward_hook(action_hook)
    batches = {k: [] for k in output_fields}
    raw_batches, timings = [], []
    with torch.no_grad():
        for start in range(0, 128, 32):
            before = time.perf_counter()
            batch = {k: v[start:start + 32] for k, v in data.items()}
            captured.clear()
            out = forward(model, batch)
            assert len(captured) == 1 and captured[0].shape == (32, 35, 7)
            raw_batches.append(captured[0].reshape(32, 7, 5, 7))
            for key in output_fields:
                assert torch.isfinite(out[key]).all(), key
                batches[key].append(out[key].detach().clone())
            duration = time.perf_counter() - before
            timings.append(duration)
            batch_keep_tanh = raw_batches[-1][torch.arange(32), 0, keep[start:start + 32], 1].tanh()
            print("BATCH " + json.dumps({"model": label, "start": start, "queries": 32,
                                         "history": 1024, "seconds": duration,
                                         "keep_tanh_min": float(batch_keep_tanh.min()),
                                         "keep_tanh_max": float(batch_keep_tanh.max()),
                                         "keep_tanh_ge_099": int((batch_keep_tanh >= .99).sum())}), flush=True)
    hook.remove()
    output = {k: torch.cat(v, 0) for k, v in batches.items()}
    raw = torch.cat(raw_batches, 0)
    raw_tanh = raw[..., 1:3].tanh()
    keep_logit = raw[rows, 0, keep, 1]
    keep_tanh = raw_tanh[rows, 0, keep, 0]
    advantage = raw_tanh - keep_tanh[:, None, None, None]
    assert torch.equal(advantage, output["advantage"])
    harm = raw[..., 3:5].sigmoid()
    keep_harm = harm[rows, 0, keep, 0]
    harm_penalty = .1 * (harm - keep_harm[:, None, None, None]).clamp(min=0)
    risk = raw[..., 5].sigmoid() * (data["raw_score"] > .84)
    keep_risk = risk[rows, 0, keep]
    action_risk = torch.stack((risk, torch.zeros_like(risk)), -1)
    risk_term = .025 * (action_risk - keep_risk[:, None, None, None])
    reconstructed_score = advantage - harm_penalty
    reconstructed_score -= risk_term
    reconstructed_score[:, 1:] -= .01
    assert torch.equal(reconstructed_score, output["scores"])
    assert (output["scores"][rows, 0, keep, 0] == 0).all()
    # Any tanh head output is <=1. Harm cannot add score; action risk >=0.
    local_score_ceiling = 1 - keep_tanh + .025 * keep_risk
    extra_score_ceiling = local_score_ceiling - .01
    chosen = select_actions(output, data, threshold=.03, write_verification="action")
    forced = dict(output)
    forced["region_advantage"] = torch.full_like(output["region_advantage"], -1)
    forced["region_success_logits"] = torch.zeros_like(output["region_success_logits"])
    forced["absence_logit"] = torch.full_like(output["absence_logit"], 20)
    local = select_actions(forced, data, threshold=.03, write_verification="action")
    assert not local["search_triggered"].any() and (local["region"] == 0).all()
    local_utility = utility.flatten(1)[rows, local["flat_action"]]
    selector_gain, selector_success = search_supervision_targets(
        output, data, mode="selector", threshold=.03, write_verification="action")
    forced_choices, forced_details = {}, {}
    for region in range(1, 7):
        forced["region_advantage"] = torch.full_like(output["region_advantage"], -1)
        forced["region_advantage"][:, region - 1] = 1
        selection = select_actions(forced, data, threshold=.03, write_verification="action")
        assert selection["search_triggered"].all() and (selection["searched_region"] == region).all()
        assert ((selection["region"] == 0) | (selection["region"] == region)).all()
        u = utility.flatten(1)[rows, selection["flat_action"]]
        has_candidate = data["valid"][:, region].any(-1)
        exact_helper_gain = torch.where(has_candidate, u - local_utility, 0)
        assert torch.equal(exact_helper_gain, selector_gain[:, region - 1])
        forced_choices[region] = selection
        forced_details[region] = {
            "selected_utility": u, "net_gain_vs_keep": u - reference - .01,
            "net_gain_vs_no_search_selector": u - local_utility - .01,
            "source_selector_gross_gain": exact_helper_gain, "has_candidate": has_candidate}
    quality = output["quality_logits"].sigmoid()
    chosen_utility = utility.flatten(1)[rows, chosen["flat_action"]] - .01 * chosen["search_triggered"]
    chosen_current = data["current_iou"][rows, chosen["region"], chosen["candidate"]]
    changed = (chosen["region"] != 0) | (chosen["candidate"] != keep)
    available = torch.zeros_like(data["valid"])
    available[:, 0] = data["valid"][:, 0]
    available[rows, chosen["searched_region"]] = data["valid"][rows, chosen["searched_region"]] & chosen["search_triggered"][:, None]
    available_actions = legal & available[..., None]
    best_available = output["scores"].masked_fill(~available_actions, -torch.inf).flatten(1).max(-1).values
    query_records = []
    for i in range(128):
        query_records.append({
            "index": i, "sequence": jobs[i][0], "query_frame": jobs[i][1], "keep_candidate": int(keep[i]),
            "keep_raw_logit": float(keep_logit[i]), "keep_tanh_reference": float(keep_tanh[i]),
            "keep_tanh_slope": float(1 - keep_tanh[i].square()), "keep_risk": float(keep_risk[i]),
            "optimistic_local_score_ceiling": float(local_score_ceiling[i]),
            "optimistic_extra_score_ceiling": float(extra_score_ceiling[i]),
            "oracle_true_net_gain": float(oracle_best[i]), "reintroduced_missing_correct": bool(missing[i]),
            "selected": selection_row(chosen, i), "no_search_selected": selection_row(local, i),
            "selected_current_gt_iou": float(chosen_current[i]),
            "selected_gt_utility_after_search_cost": float(chosen_utility[i]),
            "selected_gt_net_gain_vs_keep": float(chosen_utility[i] - reference[i]),
            "available_best_score_minus_threshold": float(best_available[i] - .03),
            "forced_regions": {str(region): {
                "chosen": selection_row(forced_choices[region], i),
                **{key: scalar(v[i]) for key, v in forced_details[region].items()}}
                for region in range(1, 7)}
        })
    details = []
    for i in missing_indices:
        correct_actions = correct[i, ..., None] & legal[i]
        correct_extra = correct_actions.clone()
        correct_extra[0] = False
        assert correct_extra.any()
        candidate_records = []
        for region in range(7):
            for candidate in range(5):
                valid = bool(data["valid"][i, region, candidate])
                rivals = data["valid"][i].clone()
                rivals[:] = False
                rivals[0] = data["valid"][i, 0]
                rivals[region] = data["valid"][i, region]
                predicted_q = quality[i, region, candidate]
                rank = int((quality[i][rivals] > predicted_q).sum()) + 1 if valid else None
                candidate_records.append({
                    "region": region, "source": cfg["regions"][region], "candidate": candidate,
                    "valid": valid, "action_legal_regular_pause": legal[i, region, candidate].tolist(),
                    "image_box_recorded": data["image_boxes"][i, region, candidate].tolist(),
                    "raw_score": float(data["raw_score"][i, region, candidate]),
                    "current_gt_iou": float(data["current_iou"][i, region, candidate]),
                    "gt_correct": bool(correct[i, region, candidate]),
                    "future_gt_iou_regular_pause": data["future_iou"][i, region, candidate].tolist(),
                    "wrong_update_fraction_regular_pause": data["wrong_update_fraction"][i, region, candidate].tolist(),
                    "true_utility_regular_pause": utility[i, region, candidate].tolist(),
                    "true_net_gain_vs_keep_regular_pause": true_net[i, region, candidate].tolist(),
                    "raw_action_head7": raw[i, region, candidate].tolist(),
                    "raw_tanh_regular_pause": raw_tanh[i, region, candidate].tolist(),
                    "predicted_advantage_regular_pause": advantage[i, region, candidate].tolist(),
                    "final_scores_regular_pause": output["scores"][i, region, candidate].tolist(),
                    "harm_penalty_regular_pause": harm_penalty[i, region, candidate].tolist(),
                    "relative_risk_term_regular_pause": risk_term[i, region, candidate].tolist(),
                    "estimated_current_quality": float(predicted_q),
                    "quality_rank_original_plus_this_region": rank,
                    "quality_gap_vs_keep": float(predicted_q - quality[i, 0, keep[i]]),
                    "actual_selector_selected_candidate": bool(chosen["region"][i] == region and chosen["candidate"][i] == candidate),
                    "forced_own_region_selected_candidate": None if region == 0 else bool(forced_choices[region]["region"][i] == region and forced_choices[region]["candidate"][i] == candidate),
                })
        best_correct_true = true_net[i][correct_extra].max()
        best_correct_adv = advantage[i][correct_extra].max()
        best_correct_score = output["scores"][i][correct_extra].max()
        details.append({
            **query_records[i],
            "best_correct_extra_true_net_gain": float(best_correct_true),
            "any_correct_extra_true_net_gt_003": bool(best_correct_true > .03),
            "best_correct_extra_predicted_advantage": float(best_correct_adv),
            "best_correct_extra_final_score": float(best_correct_score),
            "any_correct_extra_score_gt_003": bool(best_correct_score > .03),
            "tanh_reference_alone_caps_extra_below_threshold": bool(extra_score_ceiling[i] <= .03),
            "candidates": candidate_records,
        })
    gpu_comparison = {"available": False, "reason": "Old checkpoint was validated on a different old cache; no same-current-cache GPU artifact assumed."}
    if label == "new_own55":
        gpu_path = path.parent / "best_validation.npz"
        with np.load(gpu_path, allow_pickle=False) as archive:
            gpu = {k: archive[k] for k in archive.files}
        mismatch = {}
        for key in ("flat_action", "region", "candidate", "pause", "searched_region", "search_triggered"):
            cpu_value = chosen[key].numpy()
            mismatch[key] = np.flatnonzero(cpu_value != gpu[key]).tolist()
        gpu_comparison = {
            "available": True, "gpu_artifact": str(gpu_path), "sha256": sha(gpu_path),
            "cpu_batch_size": 32, "gpu_original_batch_size": checkpoint["args"]["batch_size"],
            "decision_mismatch_indices": mismatch,
            "all_recorded_choices_exact": not any(mismatch.values()),
            "selected_current_max_abs_difference": float(np.abs(chosen_current.numpy() - gpu["current"]).max()),
            "selected_utility_max_abs_difference": float(np.abs(chosen_utility.numpy() - gpu["utility"]).max()),
            "cpu_min_abs_available_score_to_003": float((best_available - .03).abs().min()),
            "scope": "CPU float32 batch32 versus original GPU validation; raw GPU action-head7 was not recorded, so logits are CPU diagnostic values rather than asserted bitwise GPU values.",
        }
    ceil_blocks = extra_score_ceiling <= .03
    hypothesis = {
        "keep_tanh_reference": distribution(keep_tanh),
        "keep_raw_logit": distribution(keep_logit),
        "keep_tanh_slope": distribution(1 - keep_tanh.square()),
        "near_positive_one_counts": {str(t): int((keep_tanh >= t).sum()) for t in (.9, .95, .96, .99, .999, .9999)},
        "exact_positive_one_count": int((keep_tanh == 1).sum()),
        "optimistic_extra_ceiling_distribution": distribution(extra_score_ceiling),
        "all_query_extra_ceiling_at_or_below_003": int(ceil_blocks.sum()),
        "oracle_positive_queries_extra_ceiling_at_or_below_003": int((positive & ceil_blocks).sum()),
        "missing3_queries_extra_ceiling_at_or_below_003": int((missing & ceil_blocks).sum()),
        "interpretation": "A ceiling at/below threshold is a mathematical restriction for that fixed state. A ceiling above threshold means this saturation ceiling alone does not explain that failure; actual score gaps, harm/risk and search availability remain distinct."
    }
    if int((missing & ceil_blocks).sum()) == 0:
        hypothesis["missing_query_saturation_explanation"] = "REJECTED_AS_A_SUFFICIENT_EXPLANATION: none of the missing-candidate queries is capped by this keep-tanh ceiling."
    else:
        hypothesis["missing_query_saturation_explanation"] = "SUPPORTED_ONLY_FOR_REPORTED_CAPPED_QUERIES; not asserted as sole cause or online performance explanation."
    if float(keep_tanh.max()) < .9:
        hypothesis["positive_reference_near_one_saturation"] = "REJECTED: all references are below .9; recorded tanh values do not exhibit the proposed positive saturation."
    else:
        hypothesis["positive_reference_near_one_saturation"] = "See actual threshold counts and state-specific ceiling; proximity alone is not a causal attribution."
    all_results[label] = {
        "checkpoint": str(path), "checkpoint_sha256": expected_sha, "epoch": epoch,
        "loaded_parameters": sum(p.numel() for p in model.parameters()),
        "C1_tensors_frozen_and_equal": True, "batches": 4, "batch_size": 32,
        "history_frames_per_forward": 1024, "batch_seconds": timings,
        "raw_action_hook_calls": len(raw_batches),
        "scores_reconstructed_exactly_from_hook": True,
        "selection_summary": {
            "changed_location": int(changed.sum()), "pause": int(chosen["pause"].sum()),
            "search_triggered": int(chosen["search_triggered"].sum()),
            "selected_extra": int((chosen["region"] != 0).sum()),
            "selected_correct_after_missing_recall": int((missing & (chosen_current >= .5)).sum()),
            "oracle_positive_queries_intervened": int((positive & (chosen["flat_action"] != keep * 2)).sum()),
            "mean_gt_net_gain_vs_keep": float((chosen_utility - reference).mean()),
        },
        "saturation_hypothesis": hypothesis,
        "missing_candidate_query_details": details,
        "all128_query_records": query_records,
        "CPU_GPU_boundary": gpu_comparison,
    }
    assert sha(path) == expected_sha
    del model, output, raw, batches, raw_batches
assert not torch.cuda.is_initialized()
report = {
    "status": "PASS_DIAGNOSTIC_COMPLETED", "review_independence": "same-family",
    "acceptance_status": "provisional", "reviewer_model": "gpt-6-astra",
    "reviewer_reasoning_effort": "max",
    "observed_cst": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "script_path": str(Path(__file__)), "script_sha256": sha(__file__),
    "source_sha256": SOURCE_HASHES, "validation_root": str(VAL),
    "fields_loaded": list(DECISION_FIELDS + LABEL_FIELDS),
    "data_sha_scope": "Previously independently verified full cache; no repeated16-shard read",
    "target_summary": target_summary, "models": all_results,
    "theoretical_bound": "tanh(action)<=1 gives relative advantage<=1-tanh(keep). Thus extra score<=1-tanh(keep)+.025*actual_keep_risk-.01 because harm>=0 and action_risk>=0. This bound uses recorded CPU head values, not an assumed saturation.",
    "forced_search_semantics": "Exact source selector-teacher intervention: absence20, successlogits0, region_advantage=-1 everywhere then1 for each of6 regions. No NN is rerun for forced regions. Report both source gross marginal gain against the local selector and actual GT utility minus.01 once per forced search versus keep; forced search pays even if the selected action stays original.",
    "comparison_boundary": "Both checkpoints evaluated on identical current own4-prefix/own4-future VAL states. Their original training caches differ; this is a fixed-state diagnostic, not a matched causal estimate of teacher training, an online rollout, or native accuracy.",
    "quality_boundary": "Quality ranks and GT-correct comparisons are retrospective only. No quality-rerank policy or structural change is executed.",
    "execution": {"CUDA_VISIBLE_DEVICES": "", "cuda_initialized": False, "device": "cpu",
                  "torch_threads": 4, "small_model_forward_batches": 8,
                  "decision_queries_per_model": 128, "visual_backbone_instances": 0,
                  "visual_forwards": 0, "optimizer_updates": 0, "backward_calls": 0,
                  "train_embeddings_read": False, "full16_shards_reread": False,
                  "process_signals": 0, "production_source_changes": 0,
                  "elapsed_seconds": time.perf_counter() - started},
}
with OUT.open("x") as stream:
    stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print("PROBE_RESULT " + json.dumps({
    "status": report["status"], "path": str(OUT), "sha256": sha(OUT),
    "target_summary": target_summary,
    "model_summaries": {k: {"selection": v["selection_summary"],
                           "saturation": v["saturation_hypothesis"],
                           "cpu_gpu": v["CPU_GPU_boundary"],
                           "batch_seconds": v["batch_seconds"]}
                        for k, v in all_results.items()},
    "elapsed_seconds": report["execution"]["elapsed_seconds"],
}), flush=True)
