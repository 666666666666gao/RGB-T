# PASS — frame-suffix and collection-resume delta

Fresh secondary source review against `451a243c8d6d114c47d9c7b8a68a65889ce756d5`, limited to `research/collect_event_rollouts.py` and `scripts/run_current_event_rollout_training.py`.

**Same-family / provisional; requested Astra/max, actual backend and effort UNATTESTED.** The machine scope remains `COMPLETE_CURRENT_EVENT_ROLLOUT_PIPELINE_SOURCE` for the existing runner/native admission. This is a narrow delta review, not a new full-pipeline audit.

No remaining blocking or non-blocking findings.

One initial blocker was found, reported, fixed by the parent, and rechecked: the first resume draft compared JSON `regions` (list) to runtime `REGIONS` (tuple), so every resume would fail. The current comparison JSON-normalizes the new config and excludes only `resume`. Executing the exact comparison against all four archived configs now succeeds; changing the seed, partition, checkpoint, schema or query is rejected.

The all-881 CPU frame audit found exactly three native/cache length mismatches: `ab_boyfromtrees` 276/209, `ab_righthandfoamboard` 169/157 and `rightofthe4girls` 933/925. Each cache is the exact native image-path prefix. Their query maxima are 1, 141 and 775, respectively, leaving all H3 frames inside the cache. The saved all-16,452-query GT audit also passes. Accepting the longer native suffix preserves the exact per-frame box, choice, pause and write assertions. AST comparison confirms this is the only semantic change inside `collect_sequence`.

Resume validates the persisted config and closed JSONL rows against their requested query indices and full causal prefix indices. Both NPZ files precede the closed row. Only those closed videos are skipped; an incomplete video is recomputed. Progress counts and unique membership are checked, and the complete downstream schema/causality validator remains before training.

The queue resumes only `COLLECT_FULL_TRAIN`, uses the saved shards and arms, preserves the actual four prefix sanity runs, long32 backward witness and eight completed sanity optimizer steps, and passes `--resume` only to TRAIN. The saved plan contains 16,452 unique TRAIN queries over 881 videos and 1,824 DEV queries over 98 disjoint videos. The original four 24-epoch, batch32, seed42 reference/budgeted arms at 1e-5/3e-6 remain. AST equality verifies every postcollection statement through eight full98 evaluations, protected-parent selection, the same checkpoint on both native datasets, and own-weight cleanup is unchanged.

Validation: both files parsed/compiled without imports; actual config-expression regression checks; exact AST comparisons; saved plan uniqueness/disjointness and affected H3 boundaries; four archived completed sanity fit receipts; `git diff --check`. All passed. No NN, optimizer, remote query, deployment, process control or production-output mutation was performed by the reviewer. New NPZ collection and GPU resume execution are not claimed.

Execution still follows the parent's stated recovery procedure: stop only the known owned original processes/observer, archive original owner/wave logs and four shard config/progress/JSONL files before any restart, deploy the two reviewed files, resume the same root explicitly and observe only the new owner. Archiving matters because the unchanged wave helper opens stage logs with `w`. This operational procedure was described by the parent and was not executed or attested here.

Source hashes and detailed evidence are in the companion JSON.
