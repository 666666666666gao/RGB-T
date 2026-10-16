# GPU3-first scheduling source review — initial, preserved

Verdict: **PASS_SOURCE_ONLY_WITH_NONBLOCKING_NOTE**. Blocking findings: **0**. Nonblocking findings: **1**.

Scope: **SOURCE_ONLY**. Requested model: `gpt-6-astra`; requested reasoning effort: `max`. Actual route/model/effort are **UNATTESTED** by this reviewer. Review independence: **same-family**; acceptance: **provisional**. This is a fresh source review, not native qualification, training completion, or scientific acceptance.

Reviewed changes are `cadtrack_abc/dense_main_pipeline.py`, `cadtrack_abc/continue_dense_main.py`, the new `cadtrack_abc/dense_training_worker.py`, and the subsequently supplied `launch_gpu3_first_gate_20261016_remote.py`. The preserved original controllers, supplied scope, worker/admission test sources and reports, and the unchanged qualification/training entrypoints were read as supporting evidence. Project and Torch modules were not imported or executed. No server/GPU/process/temperature/power query, deployment, NN launch, or source edit was performed.

## N1 — Invalid probe receipt leaves a stopped worker labelled running

**Nonblocking**, at `cadtrack_abc/dense_training_worker.py:75–78`.

If the probe subprocess exits 0 but its receipt fails either existing qualification predicate, the assertions stop the worker before `full20`. However, `run('probe', ...)` has left the top-level worker state at `PROBE_RUNNING`; no terminal worker status is saved on this validation failure. The existing `check_gpu3_first_worker_20261016.py:29–41` invalid-probe scenario exercises this exact path. Its saved fixture, `gpu3_first_worker_cpu_cases_20261016/invalid_probe/worker/worker_state.json`, contains `status: PROBE_RUNNING` and a completed probe phase with `actual_exit_code: 0`, with no `full20` phase.

The gate is fail-closed: this cannot admit a failed probe to full training. The parent records the worker's nonzero exit after its branch joins finish. Therefore this is a stale status/diagnostic defect, not a scheduling or scientific correctness blocker.

Minimal correction: replace these two assertions with the same predicates in an explicit validation-failure branch that saves `FAILED_PROBE_QUALIFICATION` and returns 1. Preserve the probe subprocess's real exit code 0. Extend the existing invalid-probe fixture assertion to require that return code and state, and still require no `full20` launch. No generic exception handler, retry, fallback, or scheduling framework is needed.

## Correct behavior confirmed from source

- Admission preserves the actual R2 exit-0, pair-completion, and all-20-order checks. GPU3's memory determines initial admission; GPU2 may remain occupied. The CPU main and CPU branch workers inherit an empty CUDA device mask.
- Native qualification runs on physical GPU3 and must complete before branch workers start. The unchanged native, motion, and continuation entrypoints accept physical devices 2/3 and select local CUDA device 0, so the device remapping is compatible.
- The raw branch waits only for GPU3; the control branch waits only for GPU2. Each uses a `<500 MiB` threshold and 300-second waits. Each branch launches its own probe on its own device, checks the existing 2-update, raw-clip-count, and changed-parameter predicates, then launches its own full command on that same device. Nonzero probe or full-child exits are saved and returned. No process is preempted or signalled.
- The original nested `training_commands` and `run` functions are unchanged. Full commands retain 20 epochs, 20,000 clips per epoch, 8 query steps, 4 workers, the chosen batch, and the original learning rates. The prepared batch-128 run therefore still requires 400,000 raw clips, 3,140 updates, and 3,200,000 query observations per branch.
- Probe/full outputs are separate, and each full command creates a fresh process. Both training entrypoints reload the locked R2 HA parent and create fresh optimizers/schedulers. Probe weights are not used as full-training initialization. Scheduling does not change architecture, losses, teachers, sampling seeds, or optimizer construction.
- The main controller joins both workers before validating both probe receipts and both full-training receipts. Existing 20-epoch actual-order equality, matching parent checks, full-98 selection, all-three-benchmark evaluation, fresh evidence collection/calibration, fresh consequence collection/decision training, and final evaluation remain in order. A read-only comparison found every statement from the `raw_fit, control_fit` assignment through the end unchanged.
- The added launch helper requires the old zero-NN queue's recorded `-15` exit, both old PIDs absent, the old R3 output absent, actual R2 exit 0, reviewed source identities, the warm environment, and disk budget. It creates a unique CPU gate root with a 180-second not-before time and one detached wrapper. The wrapper preserves the actual queue return code. Five-second startup verification checks the expected queue module/root and wrapper path in their own `/proc` command lines. These are expected-module/path substring checks, not equality of the whole argument vector. No change is required for this cooperating-operator launch path.

## Evidence and limits

The supplied worker report contains 5 PASS host cases and the admission report contains 5 PASS host cases. Their subprocesses, GPU responses, and sleeps are mocked; they are not runtime NN evidence and were not rerun by this reviewer. The supplied source-budget report records unchanged ASTs for `run` and `training_commands` and unchanged active-R2 sources. Independent read-only text comparisons here confirmed both functions and all downstream main statements unchanged. Current scope-listed source identities matched the supplied scope.

An optional reviewer AST command could not start the host `python` shim (`No pyvenv.cfg file`). It executed no Python/project code and is not counted as a passing parse check. The review relies on direct source inspection, original-source comparison, read-only text comparison, and the explicitly attributed existing host checks.

The reported R2 completion, former GPU2 occupant, GPU3 availability, and intentional old-gate retirement were supplied context; this review did not re-query them. The reviewed launch code checks the required receipts again when actually invoked. R3 native qualification, per-branch real probes, complete training, formal metrics, and module evidence are still unexecuted/unestablished by this review. **All 7 metrics +2 and all 3 module-effectiveness evidence remain UNMET.** Negative H/HA results are scientific outcomes and are not scheduling defects.

This initial report and its accompanying JSON must remain preserved if N1 is fixed and a bounded follow-up review is requested.
