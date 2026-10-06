# Current relation-policy collection source review

- **Status:** PASS
- **Scope:** CURRENT_RELATION_POLICY_COLLECTION_SOURCE
- **Time:** 2026-10-06T22:09:39.105175+08:00
- **Reviewer request:** gpt-6-astra / max, fresh `fork_turns=none`, confirmed by parent.
- **Attribution:** Backend model/effort identity was not independently attested. Same-family review; acceptance provisional; `runtime_verified: false`.

No concrete blocking source defect was found. The source may proceed through its declared TRAIN and validation sanity gates; successful GPU execution and exact runtime prefix parity are still pending.

## Checked

- **actual_source_data_contract — PASS.** Archived fit_full/relations_lr5/config.json has 9 TRAIN source configs totaling 2576 rows and 2576 unique sequence/query pairs, 881 TRAIN names; validation has 196 unique pairs over 98 names, disjoint from TRAIN. All source query frames are <=818. VAL groups are 49 each; sanity jobs are girlintrees/16, whitecarafterbike/21, whiteblcakwoman/25, girlbike/27.

- **fixed_selected_policy_load_and_call — PASS.** research/collect_recoverability.py:98-100,302-303,360-367 passes explicit search_value, keeps weighted as default, instantiates the existing candidate-relations family before strict checkpoint loading, and freezes it. The controller requests restored epoch5 with gross/action/own. Archived selected full98 inference_config agrees with family, epoch5, threshold .03, learned policy, search enabled and action writes.

- **causal_inputs_and_branch_state — PASS.** collect_recoverability.py:107-130 calls the selected tracker through query-1. Lines194-215 materialize all current decision arrays before GT-label/continuation work and future decode at256-279. Future-own shadows copy stats and an 8-entry deque; accept_observation uses Branch.copy and replaces memory with functional updates. No shared branch/deque or in-place persistent-memory mutation was found on these paths.

- **real_npz_schema_and_prefix_parity — PASS.** Read actual cli_m0/own/samples.npz: original_choice int64[2], current_iou float32[2,7,5], history_frames int64[2,64], history_valid bool[2,64], history_boxes float32[2,64,4], future_iou float32[2,7,5,2,3], action_valid bool[2,7,5,2]. Current-IoU advanced indexing succeeds. Evaluator emits decisions for frames1..N-1 and archives boxes_xyxy[region,choice] after output-box correction; controller prepends native init.txt XYXY and casts to float32 before comparing every causal history frame exactly. This is policy-prefix consistency, not GT tracking accuracy.

- **sampling_and_sanity_order — PASS.** Controller samples seed42 TRAIN128 failure(<.2),64 ambiguity([.2,.5)),64 normal(>=.5) from prior TRAIN keep IoU, checks available counts and distinct pairs, and reuses all196 developer jobs. Outer stage loop runs both TRAIN and VAL one-query/card sanities before either full collection. Completion and config checks enforce model/family/epoch/gross/action/own and exact requested jobs.

- **resources_and_protected_paths — PASS.** Controller requires >20GiB measured /data/gb free space and records free bytes plus the20GiB budget; checks only GPU index/memory/utilization for four idle cards. New output root is exclusive. Production paths read model/data but contain no optimizer, checkpoint write/retirement, TEST inference, dataset move, or power/temperature operation. Offline1024 history arrays do not change deployed4-slot modality memory or8-observation motion history.

- **deployment_and_passive_observer_source — PASS.** Inspected private deploy and observer helpers without executing them. Deployer requires scoped PASS, stages explicit owned paths with an empty-index check, archives those paths, checks three existing source hashes, compiles both Python source files, and starts one detached collection controller at an exclusive output path. Observer reads original PID/progress/receipts only, first180s then240s or measured near-ETA, and does not run models or retries.

- **local_static_checks — PASS.** AST parsed both collection sources, private deploy/observer helpers, and their embedded remote Python. CRLF-aware git diff --check passed. Local NumPy/config-only checks passed via cached offline uv Python. No collection, NN forward, optimizer, deployment, SSH, or TEST access was executed during this review.

## Findings and limits

Blocking findings: none. Non-blocking defects: none. No production code was changed by this reviewer.

- This is source review with local existing-artifact checks; current remote state, model bytes, GPU collection, exact runtime prefix parity, resource peak and completion are not verified here.
- Archived restoration receipt establishes original developer98 prediction/decision parity to its declared scope, not original checkpoint-byte identity.
- Full-stratum availability is checked by the controller before any child starts; this reviewer did not reread the nine remote samples.npz archives.
- Developer196 is reused, and strata use prior TRAIN keep IoU; sampled counts are not newly established independent recovery events.
- Future mixed-policy training requires the already-declared loader treatment of distinct policies; it is outside this data-generation review and has not been launched.
- No new tracking efficacy, all-five+2 success, or three-positive-seed requirement is asserted.

The local default `python` command was unusable (`No pyvenv.cfg file`); the completed syntax/NumPy reads used the already-cached offline uv interpreter. The initial ordinary whitespace check treated existing CRLF line endings as trailing whitespace; the deployer's configured CRLF-aware check passed. Neither issue required a production change.

No deployment, remote call, NN forward, training, TEST access, checkpoint deletion, power/temperature operation or external messaging was performed. Historical artifacts remain intact. See the adjacent JSON for exact reviewed files and primary artifacts.
