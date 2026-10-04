"""Actual terminal B384 budgeted-action M0 CPU acceptance; no neural forwards."""
import ast
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
PRIVATE = Path("/data/gb/experiments/recoverability_budgeted_action_20261004")
MAIN = Path("/data/gb/GOLA")
assert Path.cwd() == PRIVATE
sys.path.insert(0, str(PRIVATE))
import numpy as np
import torch
import research.recoverability_modules as modules
import research.train_recoverability as trainer

assert Path(modules.__file__).parent == Path(trainer.__file__).parent == PRIVATE / "research"
assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.set_grad_enabled(False)
START = time.perf_counter()
CACHE = Path("/data/gb/outputs/recoverability_future_policy_merged_20261004")
PRIOR = Path("/data/gb/setup/future_policy_actual_m0_cpu_audit_20261004.json")
C1 = Path("/data/gb/outputs/c1_initial_seed42/best.pth")
OUT = Path("/data/gb/setup/budgeted_action_actual_m0_cpu_audit_20261004.json")
SOURCE_HASHES = {
    str(PRIVATE / "research/recoverability_modules.py"): "082945bc2742c046dc75439ec41e2d37e209974d4afc72a8f1cbb5b096824aaa",
    str(PRIVATE / "research/train_recoverability.py"): "557ed8c019ab61e4f1dc2aa34ed305418a1c3b8b560e3ff2ab6f59bfacfe2544",
    str(MAIN / "research/recoverability_modules.py"): "371269db053cdbd38b510557e213f1acfb9c6427e6eabd2df5685c59bdfbb158",
    str(MAIN / "research/train_recoverability.py"): "4b1327a541d426aeade308e4fc3d7c8e3110e35dae4612c11b93029783fd4e93",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def definitions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

assert all(sha(p) == h for p, h in SOURCE_HASHES.items())
a, b = definitions(MAIN / "research/recoverability_modules.py"), definitions(PRIVATE / "research/recoverability_modules.py")
assert set(b) - set(a) == {"budgeted_winner_loss"} and not set(a) - set(b)
assert [k for k in a if a[k] != b[k]] == ["objective"]
a, b = definitions(MAIN / "research/train_recoverability.py"), definitions(PRIVATE / "research/train_recoverability.py")
assert set(a) == set(b) and [k for k in a if a[k] != b[k]] == ["arguments"]
prior = read(PRIOR)
assert sha(PRIOR) == "8e9fc10a3cf99dc19e32a512556d5b9481d5d8d9700ec9a46bc38029cce739e0"
assert prior["status"] == "PASS" and prior["all16_real_source_shards_checked"]
assert prior["all4_merged_caches_equal_ordered_sources"] and prior["all28_non_future_arrays_exact_on_both_partitions"]
gate_path = CACHE / "paired_cache_gate.json"
gate = read(gate_path)
assert sha(gate_path) == prior["paired_cache_gate_sha256"]
assert gate["status"] == "PASS" and gate["all_non_future_arrays_exact"]
assert gate["matched_partitions"] == {"train": 902, "validation": 128}
c1 = torch.load(C1, map_location="cpu", weights_only=False)
assert sha(C1) == prior["frozen_C1_source_sha256"]
cache_evidence = {}
for policy in ("c1", "own"):
    for partition, count in (("train", 902), ("validation", 128)):
        root = CACHE / policy / partition
        old = prior["cache_checks"][policy + "_" + partition]
        config, receipt = read(root / "config.json"), read(root / "completion.json")
        assert sha(root / "config.json") == old["merged_config_sha256"]
        assert sha(root / "completion.json") == old["merged_completion_sha256"]
        assert receipt["completed"] and receipt["clips"] == config["clips"] == count
        assert receipt["partition"] == config["partition"] == partition
        assert config["future_policy_mode"] == policy and config["prefix_checkpoint_epoch"] == 4
        assert config["prefix_write_verification"] == "action" and config["max_prefix"] == 1024
        for source, digest in old["source_receipt_sha256"].items():
            assert sha(source) == digest
        with zipfile.ZipFile(root / "samples.npz") as archive:
            members = {i.filename: [i.CRC, i.file_size, i.compress_size] for i in archive.infolist()}
        assert len(members) == 30
        cache_evidence[policy + "_" + partition] = {
            "clips": count, "valid_actions": receipt["valid_actions"],
            "config_sha256": sha(root / "config.json"), "completion_sha256": sha(root / "completion.json"),
            "all_source_receipt_hashes_still_exact": True,
            "npz_archive_directory_sha256": hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest(),
            "npz_archive_member_count": len(members),
        }
        cache_evidence[policy + "_" + partition]["_members"] = members
for partition in ("train", "validation"):
    left, right = [cache_evidence[p + "_" + partition]["_members"] for p in ("c1", "own")]
    for key in left:
        if key not in ("future_iou.npy", "wrong_update_fraction.npy"):
            assert left[key] == right[key], (partition, key)
for entry in cache_evidence.values():
    del entry["_members"]

arms, states, small = {}, {}, {}
for policy in ("c1", "own"):
    root = Path("/data/gb/outputs/recoverability_budgeted_action_" + policy + "_b384_m0_20261004")
    required = ("config.json", "completion.json", "train.jsonl", "metrics.json", "best.pth",
                "best_validation.npz", "training_completed.txt", "job_completed.txt")
    assert all((root / n).is_file() for n in required)
    config, receipt, metrics = read(root / "config.json"), read(root / "completion.json"), read(root / "metrics.json")
    logs = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    assert receipt["completed"] and receipt["epochs"] == config["epochs"] == 3
    assert config["batch_size"] == 384 and config["seed"] == 42 and config["train_clips"] == 902 and config["validation_clips"] == 128
    assert config["lr"] == config["weight_decay"] == 1e-4 and config["threshold"] == .03
    assert config["action_ranking"] == "budgeted" and config["search_supervision"] == "oracle"
    assert config["write_verification"] == "action" and config["write_pair_calibration"] is False
    assert config["frozen_c1"] and config["initial_all_choices_match_c1"] and not config["prefer_last_prefix"]
    assert config["train"] == [str(CACHE / policy / "train")] and config["validation"] == str(CACHE / policy / "validation")
    assert config["c1_head"] == str(C1) and config["output"] == str(root)
    control = Path("/data/gb/outputs/recoverability_future_" + policy + "_b384_full_20261004")
    control_cfg, control_receipt = read(control / "config.json"), read(control / "completion.json")
    assert (control / "job_completed.txt").is_file()
    assert control_receipt["completed"] and control_receipt["epochs"] == 60 and control_receipt["optimizer_steps"] == 180
    assert control_cfg["action_ranking"] == "pairwise"
    assert set(config) == set(control_cfg)
    differences = [k for k in config if config[k] != control_cfg[k]]
    assert set(differences) == {"output", "epochs", "action_ranking"}, differences
    assert config["source_configs"] == [read(CACHE / policy / p / "config.json") for p in ("train", "validation")]
    train_jobs, val_jobs = config["train_jobs"], config["validation_jobs"]
    assert len(train_jobs) == len({tuple(j) for j in train_jobs}) == 902
    assert len(val_jobs) == len({tuple(j) for j in val_jobs}) == 128
    split = read(config["source_configs"][0]["split"])
    assert len(split["train"]) == 881 and len(split["validation"]) == 98
    assert {j[0] for j in train_jobs} <= set(split["train"])
    assert {j[0] for j in val_jobs} <= set(split["validation"])
    assert not {j[0] for j in train_jobs} & {j[0] for j in val_jobs}
    assert receipt["optimizer_steps"] == 3 * math.ceil(902 / 384) == 9
    triplets = [[r["epoch"], r["step"], r["optimizer_steps"]] for r in logs]
    assert triplets == [[1,1,1], [1,3,3], [2,1,4], [2,3,6], [3,1,7], [3,3,9]]
    assert all(np.isfinite(r["loss"]) and all(np.isfinite(v) for v in r["parts"].values()) for r in logs)
    assert all(set(r["module_gradient_norms"]) == {"A", "B", "C"} for r in logs)
    assert all(np.isfinite(v) and v > 0 for r in logs for v in r["module_gradient_norms"].values())
    assert receipt["modules_changed"] == {"A": True, "B": True, "C": True}
    assert all(np.isfinite(v) and v > 0 for v in receipt["max_module_gradient_norms"].values())
    assert all(receipt["max_module_gradient_norms"][k] >= max(r["module_gradient_norms"][k] for r in logs) for k in ("A","B","C"))
    assert receipt["frozen_c1_gradients_absent"] and receipt["strict_reload_metrics_equal"]
    assert receipt["retained_weights"] == ["best.pth"] and not receipt["official_tracking_accuracy"]
    assert [m["epoch"] for m in metrics] == [0,1,2,3]
    assert all(all(np.isfinite(v) for v in m.values()) for m in metrics)
    best_epoch = max(range(4), key=lambda e: metrics[e]["utility"])
    assert best_epoch == receipt["best_epoch"]
    checkpoint = torch.load(root / "best.pth", map_location="cpu", weights_only=False)
    assert checkpoint["module"] == "ABC_recoverability" and checkpoint["epoch"] == best_epoch
    assert checkpoint["validation"] == receipt["best_validation"] == {k:v for k,v in metrics[best_epoch].items() if k != "epoch"}
    assert receipt["last_validation"] == {k:v for k,v in metrics[-1].items() if k != "epoch"}
    assert all(config[k] == v for k,v in checkpoint["args"].items())
    torch.manual_seed(42)
    model = modules.RecoverabilityModules(c1).cpu().eval()
    if best_epoch == 0:
        assert all(torch.equal(model.state_dict()[k], v) for k,v in checkpoint["model"].items())
    loaded = model.load_state_dict(checkpoint["model"], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    assert all(not p.requires_grad and p.grad is None for p in model.c1.parameters())
    assert all(torch.equal(checkpoint["model"]["c1." + k], v) for k,v in c1["head"].items())
    counts = {key:sum(p.numel() for p in m.parameters()) for key,m in {"A":model.memory, "B":model.search, "C":model.action}.items()}
    assert counts == config["trainable_parameters"] == {"A":109187,"B":117133,"C":86791}
    state_shape = {k:[list(v.shape),str(v.dtype)] for k,v in checkpoint["model"].items()}
    old = torch.load(control / "best.pth",map_location="cpu",weights_only=False)
    assert {k:[list(v.shape),str(v.dtype)] for k,v in old["model"].items()} == state_shape
    assert old["epoch"] == control_receipt["best_epoch"]
    with np.load(root / "best_validation.npz",allow_pickle=False) as z:
        values = {k:z[k] for k in z.files}
    assert all(v.shape == (128,) and np.isfinite(v).all() for v in values.values())
    fields = ("current_iou","future_iou","wrong_update_fraction","valid","action_valid","original_choice","raw_score")
    with np.load(CACHE / policy / "validation/samples.npz", allow_pickle=False) as z:
        data = {k:torch.from_numpy(z[k].copy()) for k in fields}
    small[policy] = data
    assert data["valid"].shape == (128,7,5) and data["action_valid"].shape == (128,7,5,2)
    assert torch.equal(data["action_valid"][...,0],data["valid"])
    assert torch.equal(data["action_valid"][...,1],data["valid"] & (data["raw_score"] > .84))
    assert int(data["action_valid"].sum()) == 4774 == receipt["best_validation"]["valid_actions"]
    flat = torch.from_numpy(values["flat_action"])
    rows = torch.arange(128)
    region, candidate, pause = flat // 10, flat % 10 // 2, flat % 2
    assert torch.equal(region,torch.from_numpy(values["region"]))
    assert torch.equal(candidate,torch.from_numpy(values["candidate"]))
    assert torch.equal(pause.bool(),torch.from_numpy(values["pause"]))
    assert data["action_valid"][rows,region,candidate,pause].all()
    searched = torch.from_numpy(values["searched_region"])
    search = torch.from_numpy(values["search_triggered"])
    assert ((searched >= 1) & (searched <= 6)).all()
    assert ((region == 0) | (search & (region == searched))).all()
    keep = data["original_choice"].long()
    if best_epoch == 0:
        assert torch.equal(flat,keep*2) and not search.any() and not pause.any()
    utility = modules.action_utility(data)
    selected_u = utility[rows,region,candidate,pause]
    selected_current = data["current_iou"][rows,region,candidate].float()
    keep_u = utility[rows,0,keep,0]
    keep_current = data["current_iou"][rows,0,keep].float()
    orig_recall = data["current_iou"][:,0].masked_fill(~data["valid"][:,0],-1).max(-1).values >= .5
    available = data["valid"].clone(); available[:,1:] = False
    available[rows,searched] = data["valid"][rows,searched] & search[:,None]
    budget_recall = data["current_iou"].masked_fill(~available,-1).flatten(1).max(-1).values >= .5
    expected = {
        "current":selected_current,"c1_current":keep_current,
        "utility_before_search_cost":selected_u,"utility":selected_u-.01*search,"c1_utility":keep_u,
        "flat_action":flat,"region":region,"candidate":candidate,"pause":pause.bool(),
        "searched_region":searched,"search_triggered":search,
        "changed_location":(region != 0)|(candidate != keep),
        "rescued":(keep_current < .2)&(selected_current >= .5),
        "harmed":(keep_current >= .5)&(selected_current < .2),
        "failed":selected_current < .2,"c1_failed":keep_current < .2,
        "original_recall":orig_recall,"budget_recall":budget_recall,"missing_candidate_found":~orig_recall & budget_recall,
        "wrong_update_fraction":data["wrong_update_fraction"][rows,region,candidate,pause],
        "c1_wrong_update_fraction":data["wrong_update_fraction"][rows,0,keep,0],
    }
    assert set(expected) == set(values) and len(values) == 21
    max_row_error, max_mean_error = 0., 0.
    for k, v in expected.items():
        array = v.numpy()
        error = float(np.abs(array.astype(float)-values[k].astype(float)).max())
        if k in ("utility_before_search_cost","utility","c1_utility"):
            assert error <= np.finfo(np.float32).eps, (policy,k,error)
        else:
            assert np.array_equal(array,values[k]), (policy,k,error)
        max_row_error = max(max_row_error,error)
        if k in ("flat_action","region","candidate","searched_region"):
            continue
        if values[k].dtype == np.bool_:
            assert int(values[k].sum()) == receipt["best_validation"][k]
        else:
            mean64 = float(values[k].astype(np.float64).mean())
            gap = abs(mean64-receipt["best_validation"][k])
            bound = 2*float(abs(np.spacing(np.float32(mean64))))
            assert gap <= bound, (policy,k,gap,bound)
            max_mean_error = max(max_mean_error,gap)
    for label in ("current","utility"):
        assert receipt["best_validation"][label+"_gain_against_c1"] == receipt["best_validation"][label]-receipt["best_validation"]["c1_"+label]
    states[policy] = checkpoint["model"]
    arms[policy] = {
        "status":"PASS", "root":str(root), "actual_job_completed":(root/"job_completed.txt").read_text().strip(),
        "artifact_sha256":{n:sha(root/n) for n in required},
        "paired_control_root":str(control), "control_config_sha256":sha(control/"config.json"),
        "control_completion_sha256":sha(control/"completion.json"),
        "config_differences_vs_completed_control":differences,
        "same902_train128_validation_jobs_source_configs_exact":True,
        "epochs":3,"actual_optimizer_steps":9,"logged_step_triplets":triplets,
        "logging_explanation":"Only first and last of3updates/epoch are logged;6rows record9actualupdates.",
        "all_logged_loss_parts_and_ABC_gradients_finite_positive":True,
        "runtime_ABC_parameter_changes":receipt["modules_changed"],
        "max_module_gradient_norms":receipt["max_module_gradient_norms"],
        "runtime_C1_gradients_absent":True,
        "CPU_strict_state_dict_load":"PASS",
        "checkpoint_parameter_state_keys":len(checkpoint["model"]),
        "model_state_schema_equal_completed_pairwise_control":True,
        "frozen_C1_tensors_exact":len(c1["head"]),
        "frozen_C1_tensor_elements":sum(v.numel() for v in c1["head"].values()),
        "ABC_parameter_counts":counts,
        "production_strict_reload_metrics_checkpoint_receipt_epoch_exact":True,
        "all21_best_validation_fields_reconciled_to_actual_cache_labels":True,
        "CPU_utility_row_max_abs_error":max_row_error,
        "GPU_summary_vs_float64_NPZ_mean_max_abs_error":max_mean_error,
        "best_epoch":best_epoch,"best0_equals_seed42_initial_state":best_epoch==0,
        "best_validation":receipt["best_validation"],
        "epoch_losses":[m["loss"] for m in metrics],
        "peak_cuda_mib_recorded_not_requeried":receipt["peak_cuda_mib"],
        "learning_gain_claimed":False,
    }
    print("ARM_AUDIT "+json.dumps({"policy":policy,"status":"PASS","best_epoch":best_epoch,
                                    "row_float_error":max_row_error,"mean_float_error":max_mean_error}),flush=True)
    del model,checkpoint,old

assert all(torch.equal(small["c1"][k],small["own"][k]) for k in small["c1"] if k not in ("future_iou","wrong_update_fraction"))
assert all(torch.equal(states["c1"][k],states["own"][k]) for k in states["c1"])
assert not torch.cuda.is_initialized()
assert all(sha(p) == h for p,h in SOURCE_HASHES.items())
audit = {
    "status":"PASS","review_independence":"same-family","acceptance_status":"provisional",
    "reviewer_model":"gpt-6-astra","reviewer_reasoning_effort":"max",
    "reviewed_at_cst":datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "both_teacher_arms_passed":True,"batch_size":384,"optimizer_steps_per_arm":9,"epochs_per_arm":3,
    "source_sha256":SOURCE_HASHES,"script_sha256":sha(__file__),
    "actual_import_paths":[modules.__file__,trainer.__file__],
    "unchanged_private_model_classes_and_deployment_AST":True,
    "previous_full_cache_audit":{"path":str(PRIOR),"sha256":sha(PRIOR),
        "actual_all16_to4_full_array_equality_and28_nonfuture_exact_already_passed":True,
        "scope":"Reused prior independent actual full-array audit; this run rechecked merged/source receipt and config hashes,30-entry ZIP metadata and same data paths against completed controls. No TRAIN embeddings or16shard arrays reread."},
    "paired_cache_gate":{"path":str(gate_path),"sha256":sha(gate_path),"status":gate["status"],"matched_partitions":gate["matched_partitions"]},
    "cache_metadata_rechecks":cache_evidence,
    "frozen_C1_source":str(C1),"frozen_C1_source_sha256":sha(C1),
    "arms":arms,"both_best0_model_states_exact":True,
    "strict_reload_boundary":"Actual GPU trainer strict-load and exact validation-dict equality are recorded and crosschecked against retained checkpoint and epoch metrics; reviewer independently strict-loads all parameters on CPU without a neural forward.",
    "ABC_update_evidence_boundary":"Actual runtime before/after comparisons assertABCchanged and logs record finite positive gradients; onlybest0isretained, so finalepoch3weightsare not independently compared.",
    "CPU_float_boundary":"All discrete decisions and directly indexed labels exact. Utility CPU/GPU arithmetic may differ by float32 roundoff; observed errors recorded. GPU summary compared to float64 mean within2float32ULP. This never relaxes production strict reload metric equality.",
    "scope":"Readiness for separately authorized matched60epoch180update full runs, not evidence of learned accuracy or online improvement.",
    "blocking_findings":[],
    "checker_corrections":[{"attempt":1,"failure":"FileNotFoundError loading /data/gb/outputs/recoverability_future_c1_b384_m0_20261004/best.pth","cause":"Historical temporary M0 checkpoint is no longer present; it is not a required artifact of this new M0 acceptance","correction":"Compare state schema to the explicitly designated completed corresponding pairwise full control; seed42 fresh CPU initialization independently proves best0 state identity","prior_script_local":"actual_m0_cpu_audit_v1.py","initial_attempt_emitted_acceptance":False,"production_changes":False}],
    "reviewer_execution":{"CPU_only":True,"CUDA_VISIBLE_DEVICES":"","cuda_initialized":False,
        "NN_forward":0,"optimizer_steps":0,"backward":0,"TRAIN_embeddings_read":False,
        "GPU_queries":0,"process_signals":0,"production_source_modified":False,
        "elapsed_seconds":time.perf_counter()-START},
}
with OUT.open("x") as f:
    json.dump(audit,f,indent=2,allow_nan=False)
print("M0_AUDIT_COMPLETE "+json.dumps({"status":"PASS","path":str(OUT),"sha256":sha(OUT),
                                      "seconds":audit["reviewer_execution"]["elapsed_seconds"]}),flush=True)
