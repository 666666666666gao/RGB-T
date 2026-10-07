# PASS — native initialization delta

Narrow source review against `131e261046fe7ae4cff5f69da3228ca99fadca45`: only the first-frame initialization change in `research/collect_event_rollouts.py`. No blocking or non-blocking findings remain.

**Same-family / provisional; requested Astra/max, actual backend and effort UNATTESTED.** The existing runner scope stays `COMPLETE_CURRENT_EVENT_ROLLOUT_PIPELINE_SOURCE`; this is not a new whole-pipeline audit.

The existing CPU audit covers all 881 TRAIN and 98 DEV sequences. Its only initializer mismatch is TRAIN `rightgreen`: native XYXY `[899, 298, 961, 488]` versus cached `[899, 298, 960, 488]`. This establishes a one-pixel input mismatch, not a neural rollout failure or accuracy result.

The final collector reads only the first `init.txt` line through the same `np.fromstring(..., sep=',')` assignment, `(4,)` shape assertion and XYWH-to-XYXY conversion used by `evaluate_recoverability.py:267–270`. AST equality verifies those operations directly. The initial `np.loadtxt(...)[0]` draft was changed during review to match the native parser literally.

That same raw initializer reaches both the tracker and the context's frame0 history box. The tracker copies it into its initial motion history and template-source box. Only permitted frame0 GT initializes state. Current and H3 supervision still use the cached boxes; no later annotation is read by the initializer and no future label enters decision input.

The entire collector AST matches the committed base after replacing only the initializer statements with the prior assignment. Thus per-frame parity checks, label generation, storage and previously reviewed resume rules are unchanged. The queue source is unchanged, preserving the existing sanity/capacity gates, four full24-epoch fits, full98 selection, both native datasets and cleanup order.

The other 978 audited initializer values are identical. The parent reports `rightgreen` is still unclosed and that 287 completed videos / 4,207 queries remain preserved as of 00:55:33 CST. That runtime state was not queried by this reviewer. The stated recovery procedure preserves all further closed outputs and archives logs before explicitly resuming the same root after the owned-process stop.

Validation passed: AST parse/compile, exact native-parser AST comparison, whole-file semantic isolation of this delta, unchanged queue text, and inspection of the all979 CPU audit. The local NumPy availability probe reported `ModuleNotFoundError`; no parser runtime execution is claimed and no environment was changed. No remote query, NN, optimizer, deployment, process control or production-output mutation was performed.

Exact source hashes and structured evidence are in the companion JSON.
