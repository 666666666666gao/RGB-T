"""Literal source gates and stdlib fixtures; never import the production auditor."""
import argparse
import ast
import copy
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

from trainer_source_checks import assignment, evaluate, function, run_nodes


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
AUDIT_PATH = RUN / "actual_module_freeze_fit_cpu_audit.py"
RUNNER_PATH = ROOT / "scripts/run_recoverability_module_freeze.sh"


def rejected(call):
    try:
        call()
    except (AssertionError, FileNotFoundError):
        return True
    raise AssertionError("Invalid fixture was unexpectedly accepted")


def guard(code, payload):
    def fixture_open(path):
        if payload is None:
            raise FileNotFoundError(path)
        return io.StringIO(json.dumps(payload))
    exec(code, {"open": fixture_open})


class Tensor:
    """Only arithmetic for checking the auditor's state-difference helper."""
    def __init__(self, values):
        self.values = values

    def __ne__(self, other):
        return Tensor([left != right for left, right in zip(self.values, other.values)])

    def __sub__(self, other):
        return Tensor([left - right for left, right in zip(self.values, other.values)])

    def sum(self):
        return sum(self.values)

    def abs(self):
        return Tensor([abs(value) for value in self.values])

    def max(self):
        return max(self.values)


def main():
    source, runner = AUDIT_PATH.read_text(), RUNNER_PATH.read_text()
    tree = ast.parse(source, filename=str(AUDIT_PATH))
    compile(tree, str(AUDIT_PATH), "exec")
    trainer = ast.parse((ROOT / "research/train_recoverability.py").read_text())
    bash = r"E:\dcda6-main\Git\bin\bash.exe"
    syntax = subprocess.run([bash, "-n", str(RUNNER_PATH)], text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr

    frozen = ast.literal_eval(assignment(tree.body, "FROZEN").value)
    shell_frozen = {match.group(1): match.group(2).split() for match in re.finditer(r"^\s+(bc|ac|ab|c)\) frozen=\(([^)]*)\);;", runner, re.MULTILINE)}
    assert frozen == shell_frozen == {"bc": ["A"], "ac": ["B"], "ab": ["C"], "c": ["A", "B"]}
    assert ast.literal_eval(assignment(tree.body, "BATCH").value) == 416
    assert math.ceil(902 / 416) == 3
    assert "set -euo pipefail" in runner and '[[ "$stage" == m0 || "$stage" == full ]]' in runner
    assert 'test ! -e "$run"' in runner and 'cd "$private_source"' in runner
    assert runner.index('cd "$private_source"') < runner.index('print("TRAINING_SOURCE"') < runner.index("python -u -m research.train_recoverability")
    assert runner.index("python -u -m research.train_recoverability") < runner.index('date -Iseconds > "$run/training_completed.txt"') < runner.index('date -Iseconds > "$run/job_completed.txt"')
    assert "evaluate_recoverability" not in runner

    gates = [shlex.split(line.strip())[2] for line in runner.splitlines()
             if line.strip().startswith("/data/gb/envs/gola/bin/python -c") and 'import json;' in line]
    assert len(gates) == 3
    payloads = [
        {"status": "PASS", "all_non_future_arrays_exact": True, "matched_partitions": {"train": 902, "validation": 128}},
        {"status": "PASS", "review_independence": "same-family", "acceptance_status": "provisional"},
        {"status": "PASS", "all_four_arms_passed": True, "batch_size": 416, "optimizer_steps_per_arm": 9, "epochs_per_arm": 3},
    ]
    guard_cases = []
    for index, (code, payload) in enumerate(zip(gates, payloads)):
        guard(code, payload)
        rejected(lambda: guard(code, None))
        for key, value in payload.items():
            invalid = copy.deepcopy(payload)
            invalid[key] = False if isinstance(value, bool) else (value + 1 if isinstance(value, int) else "INVALID")
            rejected(lambda: guard(code, invalid))
        guard_cases.append({"gate": index, "valid_passed": True, "missing_receipt_rejected": True, "invalid_fields_rejected": list(payload)})
    assert runner.index("module_freeze_m0_acceptance_20261004.json") < runner.index("python -u -m research.train_recoverability")

    parser_ns = {"argparse": argparse, "__doc__": "Source parser fixture"}
    run_nodes([function(trainer, "arguments")], parser_ns)
    command = next(line for line in runner.replace("\\\n", " ").splitlines() if line.startswith("/data/gb/envs/gola/bin/python -u -m "))
    tokens = shlex.split(command)[4:]
    command_cases = []
    for variant, groups in frozen.items():
        for stage, epochs in (("m0", 3), ("full", 60)):
            output = f"/data/gb/outputs/recoverability_module_freeze_{variant}_b416_{stage}_20261004"
            expanded = []
            for token in tokens:
                if token == "${frozen[@]}":
                    expanded.extend(groups)
                else:
                    expanded.append(token.replace("$cache", "/data/gb/outputs/recoverability_current_policy_merged_20261004").replace("$epochs", str(epochs)).replace("$run", output))
            sys.argv = ["fixture", *expanded]
            cfg = vars(parser_ns["arguments"]())
            assert cfg["batch_size"] == 416 and cfg["epochs"] == epochs and cfg["frozen_modules"] == groups
            assert cfg["train"] == ["/data/gb/outputs/recoverability_current_policy_merged_20261004/own/train"]
            assert cfg["validation"] == "/data/gb/outputs/recoverability_current_policy_merged_20261004/own/validation"
            assert cfg["lr"] == cfg["weight_decay"] == 1e-4 and cfg["threshold"] == .03 and cfg["seed"] == 42
            assert (cfg["search_supervision"], cfg["action_ranking"], cfg["write_verification"]) == ("oracle", "budgeted", "action")
            assert cfg["write_pair_calibration"] is False and cfg["prefer_last_prefix"] is False
            assert cfg["init_checkpoint"] == "/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth"
            assert cfg["output"] == output
            command_cases.append({"variant": variant, "stage": stage, "epochs": epochs, "steps": epochs * 3})

    namespace = {}
    run_nodes([function(tree, "metric_errors")], namespace)
    namespace["metric_errors"]({"utility": .5, "clips": 128}, {"utility": .500001, "clips": 128})
    rejected(lambda: namespace["metric_errors"]({"utility": .5}, {"utility": .50001}))
    rejected(lambda: namespace["metric_errors"]({"utility": float("nan")}, {"utility": .5}))
    rejected(lambda: namespace["metric_errors"]({"utility": .5}, {"other": .5}))

    arm_loop = next(node for node in tree.body if isinstance(node, ast.For) and ast.unparse(node.iter) == "FROZEN.items()")
    checks = [node for node in arm_loop.body if isinstance(node, ast.Assert) and 142 <= node.lineno <= 154]
    gradient_loop = next(node for node in arm_loop.body if isinstance(node, ast.For) and ast.unparse(node.iter) == "logs")
    assert len(checks) == 7
    log_cases = []
    for variant, groups in frozen.items():
        active = {name: name not in groups for name in ("A", "B", "C")}
        for epochs in (3, 60):
            logs = [{"epoch": epoch, "step": step, "optimizer_steps": (epoch - 1) * 3 + step,
                     "loss": .2, "parts": {"part": .2},
                     "module_gradient_norms": {name: (.25 if value else 0.) for name, value in active.items()}}
                    for epoch in range(1, epochs + 1) for step in (1, 3)]
            receipt = {"optimizer_steps": epochs * 3, "modules_changed": active.copy(),
                       "frozen_module_weights_unchanged": {name: True for name in groups},
                       "max_module_gradient_norms": {name: (.5 if value else 0.) for name, value in active.items()}}
            ns = {"logs": logs, "triplets": [[row["epoch"], row["step"], row["optimizer_steps"]] for row in logs],
                  "receipt": receipt, "EPOCHS": epochs, "BATCH": 416, "math": math,
                  "np": SimpleNamespace(isfinite=math.isfinite), "active": active, "frozen": groups}
            run_nodes([*checks, gradient_loop], ns)
            ns["triplets"][0][2] = 0
            rejected(lambda: run_nodes(checks, ns))
            ns["triplets"][0][2] = 1
            logs[0]["module_gradient_norms"][groups[0]] = .1
            rejected(lambda: run_nodes([gradient_loop], ns))
            logs[0]["module_gradient_norms"][groups[0]] = 0.
            receipt["modules_changed"][groups[0]] = True
            rejected(lambda: run_nodes(checks, ns))
            log_cases.append({"variant": variant, "epochs": epochs, "log_rows": len(logs), "invalid_step_gradient_change_rejected": True})

    parent = {prefix + str(index): Tensor([float(index), float(index + 1)])
              for prefix, count in (("memory.", 9), ("search.", 4), ("action.", 4)) for index in range(count)}
    state_ns = {"parent": {"model": parent}, "torch": SimpleNamespace(equal=lambda left, right: left.values == right.values)}
    run_nodes([function(tree, "state_changes")], state_ns)
    state_assert = next(node for node in ast.walk(arm_loop) if isinstance(node, ast.Assert) and isinstance(node.test, ast.IfExp))
    state_cases = []
    for variant, groups in frozen.items():
        for best_epoch in (0, 2):
            state = copy.deepcopy(parent)
            if best_epoch:
                for name, prefix in (("A", "memory."), ("B", "search."), ("C", "action.")):
                    if name not in groups:
                        state[prefix + "0"].values[0] += .125
            changes = state_ns["state_changes"](state)
            for name, change in changes.items():
                ns = {"name": name, "change": change, "frozen": groups, "best_epoch": best_epoch}
                run_nodes([state_assert], ns)
                wrong = copy.deepcopy(change)
                wrong["changed_tensors"] = 1 if name in groups or best_epoch == 0 else 0
                rejected(lambda: run_nodes([state_assert], dict(ns, change=wrong)))
            state_cases.append({"variant": variant, "best_epoch": best_epoch, "correct_passed_wrong_rejected": True})

    reference = json.loads((ROOT / "refine-logs/runs/recoverability_current_policy_fit/full_fit/oracle_own/config.json").read_text())
    assert reference["batch_size"] == 384 and reference["trainable_parameters"] == {"A": 109187, "B": 117133, "C": 86791}
    for cfg in reference["source_configs"]:
        assert cfg["future_policy_mode"] == "own" and cfg["prefix_checkpoint_epoch"] == 25
        assert cfg["prefix_write_verification"] == "action" and cfg["max_prefix"] == 1024
    old_receipt = json.loads((ROOT / "refine-logs/runs/recoverability_current_policy_fit/actual_oracle_full_fit_acceptance.json").read_text())
    assert old_receipt["parent_sha256_before_and_after"] in source
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    report = {"status": "PASS", "scope": "Source syntax, literal guards, stdlib scalar/config/log fixtures only",
              "bash_syntax": {"status": "PASS", "executable": bash}, "auditor_python_compile": "PASS",
              "guard_cases": guard_cases, "runner_command_cases": command_cases,
              "audit_log_cases": log_cases, "selected_checkpoint_cases": state_cases,
              "metric_error_invalid_values_rejected": True, "existing_parent_and_cache_metadata_match": True,
              "new_batch_size": 416, "joint_ABC_reference_batch_size": 384,
              "torch_imported": False, "numpy_imported": False, "neural_execution": False,
              "ssh_used": False, "gpu_used": False, "runtime_certification": False,
              "limitations": ["Fake values check audit acceptance logic; no real new checkpoints or outputs were audited.",
                              "No M0/full run was launched by these fixtures.",
                              "B416 differs from B384 joint-ABC, so the comparison is not an isolated module ablation."]}
    (RUN / "runner_audit_source_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": len(guard_cases), "commands": len(command_cases),
                      "log_cases": len(log_cases), "selected_checkpoint_cases": len(state_cases),
                      "neural_execution": False}, indent=2))


if __name__ == "__main__":
    main()
