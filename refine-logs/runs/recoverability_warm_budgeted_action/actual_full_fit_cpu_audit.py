"""Audit the completed 60-epoch warm fits on CPU; preserve all training and evaluation jobs."""
import ast
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PRIVATE = Path("/data/gb/experiments/recoverability_warm_budgeted_action_20261004")
MAIN = Path("/data/gb/GOLA")
OUTPUTS = Path("/data/gb/outputs")
CACHE = OUTPUTS / "recoverability_future_policy_merged_20261004"
PARENT = OUTPUTS / "recoverability_write_pair_reference_own_b384_full_20261004/best.pth"
C1 = OUTPUTS / "c1_initial_seed42/best.pth"
REVIEW = Path("/data/gb/setup/warm_budgeted_action_source_review_20261004.json")
ACCEPTANCE = Path("/data/gb/setup/warm_budgeted_action_fit_m0_acceptance_20261004.json")
assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and Path.cwd() == PRIVATE
AUDIT = Path("/data/gb/setup/warm_budgeted_action_actual_full_fit_cpu_audit_20261004.json")
assert ACCEPTANCE.is_file()
sys.path.insert(0, str(PRIVATE))
import numpy as np
import torch
import research.recoverability_modules as modules
import research.train_recoverability as trainer

torch.set_num_threads(4)
torch.set_grad_enabled(False)
assert not torch.cuda.is_initialized()
assert Path(modules.__file__).parent == Path(trainer.__file__).parent == PRIVATE / "research"
START = time.perf_counter()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


m0_gate = read(ACCEPTANCE)
assert m0_gate["status"] == "PASS" and m0_gate["both_teacher_arms_passed"]
assert m0_gate["batch_size"] == 384 and m0_gate["optimizer_steps_per_arm"] == 9
assert m0_gate["epochs_per_arm"] == 3
review = read(REVIEW)
assert review["status"] == "PASS" and review["review_independence"] == "same-family"
assert review["acceptance_status"] == "provisional"
verified = review["verification"]["local_remote_source_bytes_equal"]
SOURCE_HASHES = {
    str(PRIVATE / "research/train_recoverability.py"): verified["trainer_sha256"],
    str(PRIVATE / "research/recoverability_modules.py"): verified["modules_sha256"],
    str(PRIVATE / "scripts/run_recoverability_warm_budgeted_action_fit.sh"): verified["runner_sha256"],
    str(MAIN / "research/train_recoverability.py"): "4b1327a541d426aeade308e4fc3d7c8e3110e35dae4612c11b93029783fd4e93",
    str(MAIN / "research/recoverability_modules.py"): "371269db053cdbd38b510557e213f1acfb9c6427e6eabd2df5685c59bdfbb158",
}
assert all(sha(path) == digest for path, digest in SOURCE_HASHES.items())
gate = read(CACHE / "paired_cache_gate.json")
assert gate["status"] == "PASS" and gate["all_non_future_arrays_exact"]
assert gate["matched_partitions"] == {"train": 902, "validation": 128}
parent_sha = sha(PARENT)
assert parent_sha == "a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72"
parent = torch.load(PARENT, map_location="cpu", weights_only=False)
c1 = torch.load(C1, map_location="cpu", weights_only=False)
assert parent["module"] == "ABC_recoverability" and parent["epoch"] == 4
assert len(parent["model"]) == 25 and len(c1["head"]) == 8
assert all(torch.equal(parent["model"]["c1." + k], v) for k, v in c1["head"].items())
runner = PRIVATE / "scripts/run_recoverability_warm_budgeted_action_fit.sh"
guard_line = next(line.strip() for line in runner.read_text().splitlines()
                  if line.strip().startswith("/data/gb/envs/gola/bin/python -c")
                  and str(ACCEPTANCE) in line)
guard_code = shlex.split(guard_line)[2]
opened = subprocess.run([sys.executable, "-c", guard_code], capture_output=True, text=True)
assert opened.returncode == 0, opened.stderr


def replay(model, data):
    return trainer.evaluate(model, data, 384, .03, details=True,
                            search_supervision="oracle", action_ranking="budgeted",
                            write_pair_calibration=False, write_verification="action")


def metric_errors(actual, expected):
    assert set(actual) == set(expected)
    errors = {k: abs(actual[k] - expected[k]) for k in actual}
    assert all(value <= 2e-6 for value in errors.values()), errors
    return errors


def state_changes(state):
    result = {}
    for group, prefix in (("A", "memory."), ("B", "search."), ("C", "action.")):
        keys = [key for key in parent["model"] if key.startswith(prefix)]
        result[group] = {
            "changed_tensors": sum(not torch.equal(state[key], parent["model"][key]) for key in keys),
            "total_tensors": len(keys),
            "changed_elements": sum(int((state[key] != parent["model"][key]).sum()) for key in keys),
            "max_abs_change": max(float((state[key] - parent["model"][key]).abs().max()) for key in keys),
        }
    return result


arms = {}
for policy in ("c1", "own"):
    root = OUTPUTS / ("recoverability_warm_budgeted_action_" + policy + "_b384_full_20261004")
    required = ("config.json", "completion.json", "metrics.json", "train.jsonl",
                "best.pth", "best_validation.npz", "training_completed.txt")
    assert all((root / name).is_file() for name in required)
    cfg, receipt, metrics = [read(root / name) for name in ("config.json", "completion.json", "metrics.json")]
    assert cfg["module"] == "ABC_recoverability" and receipt["completed"]
    assert cfg["epochs"] == receipt["epochs"] == 60 and cfg["batch_size"] == 384
    assert cfg["seed"] == 42 and cfg["train_clips"] == 902 and cfg["validation_clips"] == 128
    assert cfg["lr"] == cfg["weight_decay"] == 1e-4 and cfg["threshold"] == .03
    assert cfg["search_supervision"] == "oracle" and cfg["action_ranking"] == "budgeted"
    assert cfg["write_verification"] == "action" and cfg["write_pair_calibration"] is False
    assert not cfg["prefer_last_prefix"] and cfg["frozen_c1"]
    assert cfg["c1_head"] == str(C1) and cfg["init_checkpoint"] == str(PARENT)
    assert cfg["initial_checkpoint_epoch"] == 4 and type(cfg["initial_all_choices_match_c1"]) is bool
    assert cfg["output"] == str(root)
    assert cfg["train"] == [str(CACHE / policy / "train")]
    assert cfg["validation"] == str(CACHE / policy / "validation")
    source_configs = [read(CACHE / policy / part / "config.json") for part in ("train", "validation")]
    assert cfg["source_configs"] == source_configs
    split = read(source_configs[0]["split"])
    assert len(split["train"]) == 881 and len(split["validation"]) == 98
    assert not set(split["train"]) & set(split["validation"])
    for part, count, source in zip(("train", "validation"), (902, 128), source_configs):
        cache_done = read(CACHE / policy / part / "completion.json")
        assert source["partition"] == cache_done["partition"] == part
        assert source["clips"] == cache_done["clips"] == count and cache_done["completed"]
        assert not cache_done["decision_input_contains_future"] and source["head"] == str(C1)
        assert source["future_policy_mode"] == policy and source["prefix_checkpoint_epoch"] == 4
        assert source["prefix_write_verification"] == "action" and source["max_prefix"] == 1024
        jobs = cfg[part + "_jobs"]
        assert len(jobs) == len({tuple(job) for job in jobs}) == count
        assert {job[0] for job in jobs} <= set(split[part])
    logs = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    triplets = [[row["epoch"], row["step"], row["optimizer_steps"]] for row in logs]
    assert triplets == [[e, s, (e - 1) * 3 + s] for e in range(1, 61) for s in (1, 3)]
    assert receipt["optimizer_steps"] == 60 * math.ceil(902 / 384) == 180
    assert all(np.isfinite(row["loss"]) and all(np.isfinite(v) for v in row["parts"].values()) for row in logs)
    assert all(set(row["module_gradient_norms"]) == {"A", "B", "C"} for row in logs)
    assert all(np.isfinite(v) and v > 0 for row in logs for v in row["module_gradient_norms"].values())
    assert receipt["modules_changed"] == {"A": True, "B": True, "C": True}
    assert all(np.isfinite(v) and v >= max(row["module_gradient_norms"][key] for row in logs)
               for key, v in receipt["max_module_gradient_norms"].items())
    assert receipt["frozen_c1_gradients_absent"] and receipt["strict_reload_metrics_equal"]
    assert receipt["retained_weights"] == ["best.pth"] and not receipt["official_tracking_accuracy"]
    assert sorted(p.name for p in root.glob("*.pth")) == ["best.pth"]
    assert [row["epoch"] for row in metrics] == list(range(61))
    assert all(np.isfinite(v) for row in metrics for v in row.values())
    best_epoch = max(range(61), key=lambda epoch: metrics[epoch]["utility"])
    ck = torch.load(root / "best.pth", map_location="cpu", weights_only=False)
    assert ck["module"] == "ABC_recoverability" and ck["epoch"] == receipt["best_epoch"] == best_epoch
    assert ck["validation"] == receipt["best_validation"] == {k: v for k, v in metrics[best_epoch].items() if k != "epoch"}
    assert receipt["last_validation"] == {k: v for k, v in metrics[60].items() if k != "epoch"}
    assert all(cfg[key] == value for key, value in ck["args"].items())
    assert ck["selection_policy"] == cfg["checkpoint_selection"]
    assert len(ck["model"]) == 25 and set(ck["model"]) == set(parent["model"])
    assert all(torch.equal(ck["model"]["c1." + key], value) for key, value in c1["head"].items())
    changes = state_changes(ck["model"])
    if best_epoch == 0:
        assert all(torch.equal(value, parent["model"][key]) for key, value in ck["model"].items())
    else:
        assert all(value["changed_tensors"] > 0 for value in changes.values())
    model = modules.RecoverabilityModules(c1).cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    assert all(torch.equal(value, parent["model"][key]) for key, value in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.c1.parameters())
    model.train()
    assert not model.c1.training
    model.eval()
    counts = {name: sum(p.numel() for p in block.parameters()) for name, block in
              (("A", model.memory), ("B", model.search), ("C", model.action))}
    assert counts == cfg["trainable_parameters"] == {"A": 109187, "B": 117133, "C": 86791}
    data, _, _, jobs = trainer.load_data([cfg["validation"]], "validation", torch.device("cpu"))
    assert [list(job) for job in jobs] == cfg["validation_jobs"]
    initial_metrics, initial_values = replay(model, data)
    initial_error = metric_errors(initial_metrics, {k: v for k, v in metrics[0].items() if k != "epoch"})
    val_parity = (np.array_equal(initial_values["flat_action"], data["original_choice"].numpy() * 2)
                  and not initial_values["search_triggered"].any())
    assert not cfg["initial_all_choices_match_c1"] or val_parity
    model.load_state_dict(ck["model"], strict=True)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    if best_epoch == 0:
        measured, values = initial_metrics, initial_values
    else:
        measured, values = replay(model, data)
    best_error = metric_errors(measured, ck["validation"])
    with np.load(root / "best_validation.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    assert set(saved) == set(values) and len(saved) == 21
    field_errors = {}
    for key, value in values.items():
        assert value.shape == saved[key].shape == (128,) and np.isfinite(saved[key]).all()
        field_errors[key] = float(np.abs(value.astype(float) - saved[key].astype(float)).max())
        if value.dtype.kind in "biu":
            assert np.array_equal(value, saved[key]), (policy, key)
        else:
            assert field_errors[key] <= np.finfo(np.float32).eps, (policy, key, field_errors[key])
    arms[policy] = {
        "status": "PASS", "root": str(root),
        "training_completed_at": (root / "training_completed.txt").read_text().strip(),
        "epochs": 60, "optimizer_steps": 180, "logged_step_triplets": triplets,
        "actual_module_updates_vs_loaded_parent": receipt["modules_changed"],
        "terminal_update_evidence": "Unchanged audited trainer snapshots ABC after parent load; its terminal changed flags are true, with 180 AdamW updates and finite positive A/B/C gradients on all120 logged first/last steps. Terminal weights are not separately retained.",
        "max_module_gradient_norms": receipt["max_module_gradient_norms"],
        "C1_frozen_runtime_and_all8_retained_tensors_exact": True,
        "ABC_parameter_counts": counts, "strict25_tensor_CPU_load": True,
        "config_initial_checkpoint_epoch": cfg["initial_checkpoint_epoch"],
        "config_full_TRAIN_VAL_initial_parity": cfg["initial_all_choices_match_c1"],
        "CPU_initial_VAL128_parity": bool(val_parity),
        "epoch0_metrics_reproduced_from_parent_on_VAL128": True,
        "initial_metric_max_abs_errors": initial_error,
        "best_epoch": best_epoch, "best0_is_parent_not_new_learning": best_epoch == 0,
        "selected_best_ABC_changes_vs_parent": changes,
        "all61_epoch_metrics_and_strict_best_selection_exact": True,
        "all21_cached_fields_CPU_reproduced": True,
        "per_field_max_abs_errors": field_errors, "best_metric_max_abs_errors": best_error,
        "best_validation": ck["validation"],
        "epoch_utility": [row["utility"] for row in metrics],
        "artifact_sha256": {name: sha(root / name) for name in required},
        "learning_gain_claimed": False,
    }
    print("ARM_PASS " + policy + " best_epoch=" + str(best_epoch), file=sys.stderr, flush=True)
    del model, data, ck, initial_values, values, saved

assert set(arms) == {"c1", "own"} and all(row["status"] == "PASS" for row in arms.values())
assert all(sha(path) == digest for path, digest in SOURCE_HASHES.items())
assert sha(PARENT) == parent_sha and not torch.cuda.is_initialized()
record = {
    "status": "PASS", "both_teacher_arms_passed": True,
    "batch_size": 384, "optimizer_steps_per_arm": 180, "epochs_per_arm": 60,
    "review_independence": "same-family", "acceptance_status": "provisional",
    "reviewer_invocation": review["reviewer_invocation"],
    "accepted_at_cst": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "scope": "Actual completed 60-epoch/180-update warm-fit CPU acceptance. Full98 online results and the full-native five-metric +2pp goal require separate audits. No online or native accuracy claim.",
    "source_review": str(REVIEW), "source_sha256": SOURCE_HASHES,
    "actual_import_paths": [trainer.__file__, modules.__file__],
    "parent_checkpoint": str(PARENT), "parent_sha256_before_and_after": parent_sha, "parent_epoch": 4,
    "actual_roots": {key: value["root"] for key, value in arms.items()},
    "best_epochs": {key: value["best_epoch"] for key, value in arms.items()},
    "arms": arms, "learning_gain_claimed": False,
    "existing_M0_gate": {"status": "PASS", "both_teacher_arms_passed": True,
                         "batch_size": 384, "optimizer_steps_per_arm": 9,
                         "acceptance_path": str(ACCEPTANCE), "actual_full_guard_passed": True},
    "limitations": [
        "Epoch0 and selected-best numerical replay covers all128 VAL queries per teacher. The config's complete TRAIN+VAL initial parity is recorded; TRAIN embeddings were not reread.",
        "Terminal ABC changes are evidenced by the source-linked runtime receipt and real logged optimization; selected-best parameter differences are independently computed from retained tensors.",
        "GPU numerical outputs are compared on CPU using the existing 2e-6 scalar-metric tolerance and float32-epsilon row tolerance; all discrete fields must match exactly."
    ],
    "execution": {"CUDA_initialized": False, "GPU_queries": 0, "optimizer_steps": 0,
                  "TRAIN_embeddings_read": False, "jobs_started": 0, "weights_modified": False,
                  "deleted_or_other_M0_weights_read": False, "production_source_modified": False,
                  "runtime_seconds": round(time.perf_counter() - START, 3)}
}
AUDIT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
print(json.dumps(record, indent=2, allow_nan=False))

