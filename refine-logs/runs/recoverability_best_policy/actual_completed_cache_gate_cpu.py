"""Bind the two real completed TRAIN-data CPU merge receipts before any fitting."""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
base, setup = Path('/data/gb/outputs/recoverability_best_policy_merged_20261004'), Path('/data/gb/setup')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'best_policy_merge_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
parent = Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
cfgs, receipts = {}, {}
for part, count in (('train', 902), ('validation', 128)):
    root = base / 'own' / part
    cfg, done, accepted = read(root / 'config.json'), read(root / 'completion.json'), read(root / 'partition_cpu_acceptance.json')
    assert (root / 'job_completed.txt').is_file() and done['completed'] and done['strict_npz_reload_equal']
    assert accepted['status'] == 'PASS' and accepted['partition'] == cfg['partition'] == done['partition'] == part
    assert cfg['clips'] == done['clips'] == accepted['clips'] == len(cfg['jobs']) == count
    assert accepted['all_ordered_jobs_exact'] and accepted['all_arrays_strict_reload_exact'] and accepted['causal_prefix_and_copy_before_future']
    assert accepted['prefix_epoch'] == accepted['future_epoch'] == cfg['prefix_checkpoint_epoch'] == cfg['future_checkpoint_epoch'] == 4
    assert cfg['prefix_model'] == str(parent) and cfg['future_policy_mode'] == 'own' and cfg['prefix_write_verification'] == 'action'
    assert not done['decision_input_contains_future'] and not done['official_tracking_accuracy']
    assert all(sha(root / name) == digest for name, digest in accepted['artifact_sha256'].items())
    assert all(row['GT_current_iou_max_error'] <= 1e-4 and row['GT_history_iou_max_error'] <= 1e-4 and row['all_valid_past_frames_exact']
               and all(sha(Path(row['root']) / name) == digest for name, digest in row['artifact_sha256'].items())
               for row in accepted['source_GT_evidence'])
    original = read(Path('/data/gb/outputs/recoverability_current_policy_merged_20261004/own') / part / 'config.json')
    assert cfg['jobs'] == original['jobs'] and len({(job['sequence'], job['query_frame']) for job in cfg['jobs']}) == count
    cfgs[part], receipts[part] = cfg, accepted
common = ('prefix_model', 'prefix_checkpoint_epoch', 'prefix_policy', 'prefix_write_verification', 'prefix_execution',
          'future_policy', 'future_policy_mode', 'future_checkpoint_epoch', 'future_execution', 'future_horizon',
          'root', 'cache', 'head', 'pretrained', 'motion_run', 'split', 'regions', 'max_prefix', 'batch_clips', 'forward_batch', 'seed')
assert all(cfgs['train'][key] == cfgs['validation'][key] for key in common)
assert receipts['train']['decision_fields'] == receipts['validation']['decision_fields']
assert receipts['train']['schema'].keys() == receipts['validation']['schema'].keys()
assert all(receipts['train']['schema'][key]['dtype'] == receipts['validation']['schema'][key]['dtype']
           and receipts['train']['schema'][key]['shape'][1:] == receipts['validation']['schema'][key]['shape'][1:]
           for key in receipts['train']['schema'])
split = read(cfgs['train']['split'])
assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
assert all({job['sequence'] for job in cfgs[part]['jobs']} <= set(split[part]) for part in cfgs)
assert not {job['sequence'] for job in cfgs['train']['jobs']} & {job['sequence'] for job in cfgs['validation']['jobs']}
record = {'status': 'PASS', 'matched_partitions': {'train': 902, 'validation': 128},
          'all_ordered_jobs_exact': True, 'all_GT_current_and_history_replayed': True,
          'all_schema_and_sources_verified': True, 'prefix_epoch': 4, 'future_epoch': 4,
          'teacher_checkpoint_sha256': sha(parent), 'partition_acceptance_sha256':
          {part: sha(base / 'own' / part / 'partition_cpu_acceptance.json') for part in cfgs},
          'completed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'execution': {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'native_test_reads': False},
          'training_started': False, 'official_goal_completed': False,
          'next_gate': 'Real B416 3epoch9step sanity and CPU acceptance before newfullfit',
          'future_label_scope': 'Source-linked actual frozenown-policy teacher labels; no independent future-neural rerun.'}
(base / 'best_policy_cache_cpu_gate.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
