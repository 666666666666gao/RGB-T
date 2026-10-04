"""Independent CPU audit of completed budgeted-action internal full98 outputs."""
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
START = time.perf_counter()
SOURCE = Path("/data/gb/GOLA")
GTROOT = Path("/data/wangwj/dataset/LasHeR/traingset")
SPLIT = Path("/data/gb/outputs/c1_initial_seed42/split.json")
OUT = Path("/data/gb/setup/budgeted_action_actual_full98_cpu_audit_20261004.json")
SOURCES = {
    "research/collect_recoverability_metrics.py": "e3898e3d97494f36778b7d3ed93f94e18afd9ebc59adc8b3ba0bf26f07a1b553",
    "research/collect_core_metrics.py": "fee14cc7192c5382b89ccbc8068bf56d5c1167a72ae91864f414de0299305eda",
    "research/collect_candidate_metrics.py": "ae88efa107728894b2322c3e9da50b72e1eefcd31e587dba2e5088255dd2c600",
    "research/collect_abc_metrics.py": "cc75ef35f74c0c34080964787328c45db495431ab6fe2b498ce16d70c0d1aea9",
    "research/evaluate_recoverability.py": "08a1d2c548cf68410a769cf7eaf72f5727b16df6a409f14d55c8b7e91a75b7a8",
    "research/recoverability_tracker.py": "60287e5b6610866309ac40c7e72778a55674004924dc3e5a6559627de53f966e",
}
ROOTS = {
    "baseline": Path("/data/gb/outputs/abc_internal_validation_v1/baseline/predictions"),
    "c1": Path("/data/gb/outputs/abc_internal_validation_v1/c1/predictions"),
    "write_pair_reference_own_b384": Path("/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions"),
    "future_c1_b384": Path("/data/gb/outputs/recoverability_future_c1_b384_full_20261004/predictions"),
    "future_own_b384": Path("/data/gb/outputs/recoverability_future_own_b384_full_20261004/predictions"),
}
ROOTS.update({
    "budgeted_action_c1_b384": Path("/data/gb/outputs/recoverability_budgeted_action_c1_b384_full_20261004/predictions"),
    "budgeted_action_own_b384": Path("/data/gb/outputs/recoverability_budgeted_action_own_b384_full_20261004/predictions"),
})
NEW = ("budgeted_action_c1_b384", "budgeted_action_own_b384")
REFERENCES = {
    "budgeted_action_c1_b384": ["baseline", "c1", "write_pair_reference_own_b384", "future_c1_b384"],
    "budgeted_action_own_b384": ["baseline", "c1", "write_pair_reference_own_b384", "future_own_b384"],
}
CHECKS = {"integer_exact": 0, "float_compared": 0, "max_float_absolute_difference": 0.}
MANIFEST = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def record(path):
    digest = sha(path)
    MANIFEST.append((str(path), digest))
    return digest

def close(actual, expected, context):
    if isinstance(actual, (int, np.integer)):
        assert int(actual) == expected, (context, actual, expected)
        CHECKS["integer_exact"] += 1
    elif actual is None:
        assert expected is None, context
    else:
        gap = abs(float(actual) - float(expected))
        assert gap <= 1e-12, (context, actual, expected, gap)
        CHECKS["float_compared"] += 1
        CHECKS["max_float_absolute_difference"] = max(CHECKS["max_float_absolute_difference"], gap)

def xywh(box):
    out = np.asarray(box).copy()
    out[..., 2:] -= out[..., :2]
    return out

def overlap(box, gt):
    box = np.asarray(box, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    size_b, size_g = np.maximum(box[..., 2:], 0), np.maximum(gt[..., 2:], 0)
    low = np.maximum(box[..., :2], gt[..., :2])
    high = np.minimum(box[..., :2] + size_b, gt[..., :2] + size_g)
    intersection = np.maximum(high - low, 0).prod(-1)
    union = size_b.prod(-1) + size_g.prod(-1) - intersection
    return intersection / np.maximum(union, 1e-8)

def known(gt):
    return np.isfinite(gt).all(-1) & (gt[..., 2:] > 0).all(-1)

def failure_events(q, valid):
    events, active = [], None
    low, high = [], []
    for frame in range(1, len(q)):
        if not valid[frame]:
            low, high = [], []
            continue
        if active is None:
            low = low + [frame] if q[frame] < .2 else []
            if len(low) == 3:
                active = {"start_frame_zero_based": low[0], "confirmed_frame_zero_based": frame,
                          "recovered": False, "recovery_start_frame_zero_based": None, "delay_frames": None}
                high = []
        else:
            high = high + [frame] if q[frame] >= .5 else []
            if len(high) == 3:
                active.update(recovered=True, recovery_start_frame_zero_based=high[0],
                              delay_frames=high[0] - active["start_frame_zero_based"])
                events.append(active)
                active, low, high = None, [], []
    if active is not None:
        events.append(active)
    return events

def mechanism(t, prediction, gt):
    n = len(prediction) - 1
    assert all(len(v) == n and np.isfinite(v).all() for v in t.values())
    box = xywh(t["boxes_xyxy"]).reshape(n, 35, 4)
    valid = t["valid"].reshape(n, 35)
    rows = np.arange(n)
    choice, keep = t["choice"], t["original_choice"]
    assert (keep >= 0).all() and (keep < 5).all()
    assert (choice >= 0).all() and (choice < 35).all()
    assert valid[rows, choice].all() and valid[rows, keep].all()
    assert (valid.sum(1) <= 10).all()
    assert np.array_equal(t["region"], choice // 5)
    assert np.allclose(box[rows, choice], prediction[1:], atol=.000501, rtol=0)
    extra_regions = t["valid"][:, 1:].any(-1)
    assert (extra_regions.sum(-1) <= 1).all()
    assert np.array_equal(t["extra_executed"], extra_regions.any(-1))
    assert (t["extra_executed"] <= t["search_requested"]).all()
    assert (t["searched_region"][~t["search_requested"]] == 0).all()
    requested_regions = t["searched_region"][t["search_requested"]]
    assert ((requested_regions >= 1) & (requested_regions <= 6)).all()
    assert extra_regions[rows[t["extra_executed"]], t["searched_region"][t["extra_executed"]] - 1].all()
    q = overlap(box, gt[1:, None])
    good_gt = known(gt[1:])
    selected, original = q[rows, choice], q[rows, keep]
    local = np.where(valid[:, :5], q[:, :5], -1).max(-1) >= .5
    recalled = np.where(valid, q, -1).max(-1) >= .5
    changed = choice != keep
    failed = good_gt & (selected < .2)
    writes, pause = t["template_updated"], t["pause"]
    raw = t["raw_score"].reshape(n, 35)[rows, choice]
    assert (~pause | (raw > .84)).all()
    assert np.array_equal(writes, (raw > .84) & ~pause)
    frames = np.arange(1, n + 1)
    assert ((t["prior_template_frame"] >= 0) & (t["prior_template_frame"] < frames)).all()
    assert t["prior_template_frame"][0] == 0
    assert np.array_equal(t["prior_template_frame"][1:], t["template_source_frame"][:-1])
    assert np.array_equal(t["prior_template_box"][1:], t["template_source_box"][:-1])
    assert np.array_equal(t["template_source_frame"], np.where(writes, frames, t["prior_template_frame"]))
    assert np.array_equal(t["template_source_box"][writes], t["boxes_xyxy"].reshape(n, 35, 4)[rows, choice][writes])
    assert np.array_equal(t["template_source_box"][~writes], t["prior_template_box"][~writes])
    rates = t["memory_write_rates"]
    assert rates.shape == (n, 4, 2) and ((rates >= 0) & (rates <= 1)).all()
    assert np.array_equal(rates[:, :2].sum(1) > 0, t["memory_target_commit"])
    assert not rates[~writes, :2].any()
    prior_gt = gt[t["prior_template_frame"]]
    prior_q = overlap(xywh(t["prior_template_box"]), prior_gt)
    active_known = good_gt & known(prior_gt)
    c = {
        "tracking_frames": n, "valid_tracking_frames": int(good_gt.sum()),
        "candidate_evaluations": int(valid.sum()),
        "local_candidate_recalled_frames": int((local & good_gt).sum()),
        "budget_candidate_recalled_frames": int((recalled & good_gt).sum()),
        "local_missing_frames": int((~local & good_gt).sum()),
        "missing_correct_candidates_reintroduced": int((~local & recalled & good_gt).sum()),
        "reintroduced_candidates_selected_correctly": int((~local & recalled & good_gt & (selected >= .5)).sum()),
        "c1_same_state_failed_frames": int((good_gt & (original < .2)).sum()),
        "c1_same_state_correct_frames": int((good_gt & (original >= .5)).sum()),
        "selected_failed_frames": int(failed.sum()),
        "changed_candidate_indices": int(changed.sum()),
        "valid_changed_candidate_indices": int((changed & good_gt).sum()),
        "same_state_rescues": int((good_gt & (original < .2) & (selected >= .5)).sum()),
        "same_state_harms": int((good_gt & (original >= .5) & (selected < .2)).sum()),
        "same_state_iou_improvements": int((good_gt & (selected > original + 1e-8)).sum()),
        "same_state_iou_degradations": int((good_gt & (selected < original - 1e-8)).sum()),
        "requested_extra_searches": int(t["search_requested"].sum()),
        "extra_visual_forwards": int(t["extra_executed"].sum()),
        "searches_without_new_correct_candidate": int((t["extra_executed"] & good_gt & ~(~local & recalled)).sum()),
        "requested_extra_area_sum_pixels_squared": float(t["requested_extra_area"].sum()),
        "template_updates": int(writes.sum()), "known_template_updates": int((writes & good_gt).sum()),
        "wrong_template_updates_localization_proxy": int((writes & failed).sum()),
        "paused_query_writes": int(pause.sum()), "known_paused_query_writes": int((pause & good_gt).sum()),
        "wrong_query_writes_prevented_localization_proxy": int((pause & failed).sum()),
        "correct_query_writes_prevented": int((pause & good_gt & (selected >= .5)).sum()),
        "known_active_template_frames": int(active_known.sum()),
        "wrong_active_template_frames_localization_proxy": int((active_known & (prior_q < .2)).sum()),
    }
    supplemental = {"chosen_extra_region_frames": int((choice >= 5).sum()),
                    "chosen_extra_gt_correct_frames": int((good_gt & (choice >= 5) & (selected >= .5)).sum()),
                    "selected_regular_actions": int((~pause).sum()), "selected_pause_actions": int(pause.sum()),
                    "skipped_empty_extra_regions": int((t["search_requested"] & ~t["extra_executed"]).sum())}
    return c, supplemental

assert all(sha(SOURCE / name) == digest for name, digest in SOURCES.items())
split = json.loads(SPLIT.read_text())
assert len(split["train"]) == 881 and len(split["validation"]) == 98
assert len(set(split["train"])) == 881 and len(set(split["validation"])) == 98
assert not set(split["train"]) & set(split["validation"])
names = sorted(split["validation"])
assert set(names) <= {p.name for p in GTROOT.iterdir() if p.is_dir()}
record(SPLIT)
reports, csv_rows, source_artifacts = {}, {}, {}
for label in NEW:
    p = ROOTS[label].parent
    assert (p / "job_completed.txt").exists() and (p / "report/recoverability_metrics_completed.txt").exists()
    report_path = p / "report/full_recoverability_report.json"
    report = json.loads(report_path.read_text())
    assert report["completed"] and report["sequences"] == 98
    assert report["scope"] == "full original98 internal videos" and not report["native_accuracy_completed"]
    assert report["args"]["dataset"] == "lasher" and report["args"]["root"] == str(GTROOT)
    assert report["args"]["split"] == str(SPLIT) and report["args"]["labels"] == [label]
    assert report["args"]["reference_labels"] == REFERENCES[label]
    assert report["args"]["references"] == [str(ROOTS[k]) for k in REFERENCES[label]]
    reports[label] = report
    csv_path = p / "report/per_sequence.csv"
    with csv_path.open() as f:
        cr = list(csv.DictReader(f))
    assert len(cr) == 5 * 98
    csv_rows[label] = {(r["variant"], r["sequence"]): r for r in cr}
    assert len(csv_rows[label]) == len(cr)
    source_artifacts[label] = {str(q): record(q) for q in
        (p / "job_completed.txt", p / "report/recoverability_metrics_completed.txt", report_path, csv_path)}
gt_cache = {name: np.loadtxt(GTROOT / name / "init.txt", delimiter=",", dtype=np.float64, ndmin=2) for name in names}
for name in names:
    record(GTROOT / name / "init.txt")
assert sum(map(len, gt_cache.values())) == 49418
results, means, per_sequence = {}, {}, {}
for label, path in ROOTS.items():
    config_path, receipt_path = path / "inference_config.json", path / "inference_completion.json"
    config = json.loads(config_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert config["root"] == str(GTROOT) and config["dataset"] == "lasher"
    assert config["validation_split"] == str(SPLIT) and config["limit_sequences"] == config["max_frames"] == config.get("sequence_offset", 0) == 0
    assert receipt["completed"] and receipt["sequences"] == 98 and receipt["frames"] == 49418
    assert receipt["smoke_only"] is True and len(receipt["records"]) == 98
    recs = {r["sequence"]: r for r in receipt["records"]}
    assert set(recs) == set(names) and {p.stem for p in path.glob("*.txt")} == set(names)
    if label in NEW:
        assert config["write_verification"] == "action" and config["policy"] == "learned"
        assert not config["disable_search"] and not config["unsafe_writes"] and config["threshold"] == .03
        assert config["model"] == str(path.parent / "best.pth")
        assert config["head_epoch"] == 25
        assert {p.name[:-len("_recoverability_decisions.npz")] for p in path.glob("*_recoverability_decisions.npz")} == set(names)
    records, all_events, counter_rows, supplements, latency_rows = [], [], [], [], []
    total_frames, valid_frames, quality_sum = 0, 0, 0.
    for name in names:
        gt = gt_cache[name]
        pred_path = path / (name + ".txt")
        pred = np.loadtxt(pred_path, dtype=np.float64, ndmin=2)
        record(pred_path)
        assert pred.shape == gt.shape == (len(gt), 4) and np.isfinite(pred).all()
        assert recs[name]["frames"] == len(gt)
        q = overlap(pred, gt)
        valid = known(gt); valid[0] = False
        assert valid.any() and np.isfinite(q[valid]).all()
        ev = failure_events(q, valid)
        all_events.extend({"sequence": name, **e} for e in ev)
        row = {"sequence": name, "frames": len(gt), "valid_frames": int(valid.sum()),
               "mean_valid_iou": float(q[valid].mean()), "failed_frames": int((valid & (q < .2)).sum()),
               "failure_events": len(ev), "recovered_events": sum(e["recovered"] for e in ev)}
        records.append(row)
        total_frames += len(gt); valid_frames += int(valid.sum()); quality_sum += float(q[valid].sum())
        if label in NEW:
            diag_path = path / (name + "_recoverability_decisions.npz")
            with np.load(diag_path, allow_pickle=False) as z:
                t = {k: z[k] for k in z.files}
            record(diag_path)
            counters, supplement = mechanism(t, pred, gt)
            stats = recs[name]["stats"]
            for key in ("template_updates", "extra_visual_forwards", "changed_candidate_indices", "paused_query_writes"):
                assert counters[key] == stats[key], (label, name, key)
            assert counters["requested_extra_searches"] == stats["extra_searches_requested"]
            assert supplement["skipped_empty_extra_regions"] == stats["skipped_empty_extra_regions"]
            counter_rows.append(counters); supplements.append(supplement)
        for report_label, table in csv_rows.items():
            if label not in reports[report_label]["variants"]:
                continue
            csvrow = table[(label, name)]
            for key, val in row.items():
                if key != "sequence":
                    close(val, float(csvrow[key]), (label, name, key))
            if label in NEW:
                for key, val in counters.items():
                    close(val, float(csvrow[key]), (label, name, key))
        latency_path = path / (name + "_latency.npy")
        latency = np.load(latency_path, allow_pickle=False)
        record(latency_path)
        assert latency.shape == (len(gt) - 1,) and np.isfinite(latency).all() and (latency > 0).all()
        latency_rows.append(latency)
    means[label] = np.asarray([r["mean_valid_iou"] for r in records])
    per_sequence[label] = records
    delays = [e["delay_frames"] for e in all_events if e["recovered"]]
    value = {"frames": total_frames, "valid_frames": valid_frames,
             "sequence_mean_iou": float(means[label].mean()), "frame_mean_iou": quality_sum / valid_frames,
             "recovery_rate": len(delays) / len(all_events) if all_events else None,
             "median_recovery_delay_frames": float(np.median(delays)) if delays else None}
    latency = np.concatenate(latency_rows)
    efficiency = {"instrumented_tracking_fps": len(latency) / float(latency.sum()),
                  "latency_p50_ms": float(np.percentile(latency, 50) * 1000),
                  "latency_p95_ms": float(np.percentile(latency, 95) * 1000)}
    for report_label, report in reports.items():
        if label not in report["variants"]:
            continue
        reported = report["variants"][label]
        for key, val in value.items():
            close(val, reported[key], (label, key))
        assert all_events == reported["failure_events"], label
        for key, val in efficiency.items():
            close(val, reported["efficiency"][key], (label, key))
    assert value["frames"] == 49418 and value["valid_frames"] == 49233
    value.update(sequences=98, failed_frames=sum(r["failed_frames"] for r in records),
                 failure_event_count=len(all_events), recovered_event_count=len(delays),
                 unrecovered_event_count=len(all_events) - len(delays),
                 per_sequence_metrics_exact_to_report_csv=True, full_failure_event_records_exact=True,
                 efficiency=efficiency, inference_config_sha256=record(config_path),
                 inference_completion_sha256=record(receipt_path),
                 predictions_root=str(path))
    if counter_rows:
        totals = {key: sum(c[key] for c in counter_rows) for key in counter_rows[0]}
        for key, val in totals.items():
            close(val, reports[label]["variants"][label]["counters"][key], (label, key))
        value["independently_recomputed_mechanism_counters"] = totals
        value["additional_action_counts"] = {key: sum(c[key] for c in supplements) for key in supplements[0]}
        value["all98_mechanism_arrays_finite_and_shape_checked"] = True
        value["all49320_selected_boxes_and_actual_template_write_lineage_verified"] = True
    results[label] = value
    print("AUDITED " + json.dumps({"label": label, "frames": total_frames, "mean_iou": value["sequence_mean_iou"],
                                  "failure_events": len(all_events), "seconds_so_far": time.perf_counter() - START}), flush=True)

resamples = np.random.default_rng(42).integers(0, 98, size=(5000, 98))
paired = {}
for label in NEW:
    for reference in REFERENCES[label]:
        delta = (means[label] - means[reference]) * 100
        d = {"mean_iou_delta_percentage_points": float(delta.mean()),
             "paired_sequence_95_ci": np.percentile(delta[resamples].mean(1), [2.5, 97.5]).tolist(),
             "improved_sequences": int((delta > 0).sum()), "worsened_sequences": int((delta < 0).sum()),
             "tied_sequences": int((delta == 0).sum()), "bootstrap_resamples": 5000}
        expected = reports[label]["paired"][label + "_vs_" + reference]
        for key, val in d.items():
            if isinstance(val, list):
                for x, y in zip(val, expected[key]):
                    close(x, y, (label, reference, key))
            else:
                close(val, expected[key], (label, reference, key))
        paired[label + "_vs_" + reference] = d
assert all(sha(SOURCE / name) == digest for name, digest in SOURCES.items())
audit = {
    "status": "PASS", "review_independence": "same-family", "acceptance_status": "provisional",
    "reviewer_model": "gpt-6-astra", "reviewer_reasoning_effort": "max",
    "observed_cst": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "script_path": str(Path(__file__)), "script_sha256": sha(__file__),
    "source_sha256": SOURCES, "split_path": str(SPLIT), "split_sha256": sha(SPLIT),
    "scope": {"dataset": "LasHeR traingset internal held-out98", "split_train_sequences": 881,
              "split_validation_sequences": 98, "frames_per_variant": 49418, "valid_noninitial_gt_frames": 49233,
              "first_frame_excluded": True, "unknown_gt_frames_excluded": True,
              "formal_native_test_datasets_read": False, "native_official_accuracy_claimed": False},
    "method": "Reviewer independent NumPy IoU, state-machine failure/recovery, mechanism masks and lineage, and seed42 paired5000 sequence bootstrap; existing collector functions were read, not invoked for the audited metrics.",
    "definitions": {"failure": "3 consecutive valid post-init GT frames IoU<.2; unknown GT resets confirmation streak",
                    "recovery": "3 consecutive valid frames IoU>=.5; delay from failure start to recovery start",
                    "writes": "actual template_updated == selected raw_score>.84 AND not pause in action verification",
                    "recall": "actual valid executed proposals at IoU>=.5; selected-correct additionally uses chosen proposal",
                    "harm_rescue": "same-state selected versus original C1 proposal, not independent causal rollout"},
    "input_files_hashed": len(MANIFEST),
    "input_manifest_sha256": hashlib.sha256(json.dumps(sorted(MANIFEST), separators=(",", ":")).encode()).hexdigest(),
    "actual_files_read": {"gt_txt": 98, "prediction_txt": 686, "new_mechanism_npz": 196, "latency_npy": 686},
    "report_artifacts_sha256": source_artifacts,
    "checks": CHECKS,
    "checker_corrections": [],
    "variants": results, "paired": paired,
    "sequence_summary_digest_sha256": hashlib.sha256(json.dumps(per_sequence, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    "limitations": ["Internal development98, not 245-sequence formal native test or training-seed uncertainty.",
                    "Stored candidate/template quality is a GT localization proxy, not semantic identity labels.",
                    "Memory EMA mass, quality-calibration bins and motion forecast calibration were not independently recomputed in this focused audit; all stored diagnostic arrays were read and checked finite.",
                    "Both checkpoints are selected by the pre-existing validation rule; these two completed fits do not establish a unique failure cause."],
    "execution": {"device": "CPU", "CUDA_VISIBLE_DEVICES": "", "torch_imported": False,
                  "NN_forward": 0, "backward": 0, "optimizer": 0, "GPU_queries": 0, "process_signals": 0,
                  "production_source_modified": False, "elapsed_seconds": time.perf_counter() - START},
}
with OUT.open("x") as f:
    json.dump(audit, f, indent=2, allow_nan=False)
print("AUDIT_COMPLETE " + json.dumps({"status": audit["status"], "path": str(OUT), "sha256": sha(OUT),
                                     "checks": CHECKS, "paired": paired,
                                     "elapsed_seconds": audit["execution"]["elapsed_seconds"]}), flush=True)
