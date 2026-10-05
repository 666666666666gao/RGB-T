"""Predeclare persistent/recovering/normal TRAIN events from closed old4 prefixes."""
import datetime
import json
from pathlib import Path
import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_state_commit'
inventory = json.loads((FOLDER / 'actual_persistent_train_event_inventory.json').read_text())
config = json.loads(Path(inventory['source'], 'config.json').read_text())
split = json.loads(Path(config['split']).read_text())
dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(config['root'], str(ROOT / config['cache']))
index = {dataset[i].get_name(): i for i in range(len(dataset))}
offsets = [-2, -1, 0, 1, 3]


def eligible(name, q):
    seq = dataset[index[name]]
    if q < 2 or q + 32 >= len(seq):
        return False
    boxes = np.stack([seq[t].get_bounding_box() for t in [0] + list(range(q, q + 33))])
    return bool(np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all())


qualified = [e for e in inventory['event_candidates'] if
             all(eligible(e['sequence'], e['query_frame'] + offset) for offset in offsets)]
old = json.loads((FOLDER / 'prepared_geometry_commit_train_jobs_cpu.json').read_text())['jobs']
normal = [j for j in old if ':normal:' in j['event_id']]
assert len(normal) == 8
jobs = [j | {'event_offset': None, 'cached_event_role': 'normal',
             'sampling_role': 'normal reference rechecked under old4 in completed first48; not newly inferred'} for j in normal]
seen = {j['sequence'] for j in normal}
rng = np.random.default_rng(42)
selected = []
for role in ['persistent33', 'recovered32']:
    for written in [False, True]:
        candidates = [e for e in qualified if e['cached_role'] == role and e['cached_failure_start_written'] == written]
        rng.shuffle(candidates)
        picked = 0
        for e in candidates:
            if e['sequence'] in seen:
                continue
            selected.append(e)
            seen.add(e['sequence'])
            picked += 1
            if picked == 4:
                break
        assert picked == 4, (role, written, len(candidates))
for e in selected:
    for offset in offsets:
        jobs.append({'sequence': e['sequence'], 'query_frame': e['query_frame'] + offset,
                     'event_id': e['event_id'], 'source_event_frame': e['query_frame'], 'event_offset': offset,
                     'cached_event_role': e['cached_role'], 'cached_failure_start_written': e['cached_failure_start_written'],
                     'sampling_role': 'Closed old4 TRAIN prefix event label; actual new query outcome and eligibility require NN observation'})
assert len(jobs) == 88 and len({(j['sequence'], j['query_frame']) for j in jobs}) == 88
assert len({(j['sequence'], j['event_id']) for j in jobs}) == len(seen) == 24
assert all(j['sequence'] in split['train'] and eligible(j['sequence'], j['query_frame']) for j in jobs)
sanity = next(j for j in jobs if j['event_offset'] == 0 and j['cached_event_role'] == 'persistent33' and j['cached_failure_start_written'])
remaining = [j for j in jobs if (j['sequence'], j['query_frame']) != (sanity['sequence'], sanity['query_frame'])]
record = {'status': 'ACTUAL_TRAIN88_GT_ELIGIBLE_JOBS_PREDECLARED',
          'computed_at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'jobs': jobs, 'events': 24, 'queries': 88, 'sequences': 24,
          'source_prefix_inventory': 'actual_persistent_train_event_inventory.json',
          'eligible_event_candidates': len(qualified), 'selected_event_seeds': selected,
          'cached_role_event_counts': {'persistent33': 8, 'recovered32': 8, 'normal': 8},
          'event_offsets': offsets, 'past_written_failure_starts': 8,
          'root': config['root'], 'cache': config['cache'], 'split': config['split'],
          'actor': config['prefix_model'], 'source_actor_epoch': 4,
          'GT_role': 'TRAIN sampling eligibility and closed TRAIN labels only, never actor inputs',
          'NN_calls': 0, 'images_decoded': 0, 'native_TEST_read': False,
          'sanity': sanity, 'remaining_four_GPU_shards': [22, 22, 22, 21],
          'interpretation': 'Cached persistent/recovery strata do not guarantee runtime roles; verify actual old4 outcomes.'}
(FOLDER / 'prepared_velocity_train88_jobs.json').write_text(json.dumps(record, indent=2) + '\n')
for name, subset in [('sanity', [sanity])] + [('gpu' + str(i), remaining[i::4]) for i in range(4)]:
    (FOLDER / ('velocity_train_' + name + '_jobs.json')).write_text(json.dumps({'jobs': subset}, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k not in ['jobs', 'selected_event_seeds']}))
