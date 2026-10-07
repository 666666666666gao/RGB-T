"""Mini-batch causal histories from shared per-video event rollout artifacts.

Keep full histories on CPU, transfer only the requested batch to CUDA. Later
video observations are stored once but never sliced into an earlier query.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .recoverability_modules import DECISION_FIELDS
from .train_recoverability import LABEL_FIELDS


HISTORY_KEYS = tuple(key for key in DECISION_FIELDS + LABEL_FIELDS if key.startswith('history_'))
QUERY_KEYS = tuple(key for key in DECISION_FIELDS + LABEL_FIELDS if not key.startswith('history_'))
STATE_KEYS = ('online_identity_anchor', 'online_identity_memory', 'parent_flat_action', 'parent_search_triggered')


class EventRolloutData:
    def __init__(self, roots, partition):
        self.configs, self.jobs, self.prefixes = [], [], {}
        chunks = {key: [] for key in QUERY_KEYS + STATE_KEYS}
        for root in map(Path, roots):
            config = json.loads((root / 'config.json').read_text())
            done = json.loads((root / 'completion.json').read_text())
            assert config['storage_schema'] == 'causal_sequence_prefix_v1'
            assert config['partition'] == done['partition'] == partition and done['completed']
            assert not done['decision_input_contains_future'] and done['exact_actual_prefix_parity']
            assert config['prefix_model_family'] == 'ABC_candidate_relations' and config['prefix_checkpoint_epoch'] == 5
            assert config['prefix_search_value'] == 'gross' and config['prefix_write_verification'] == 'action'
            assert config['future_policy_mode'] == 'own' and config['future_horizon'] == 3
            requested = {(row['sequence'], row['query_frame']) for row in config['jobs']}
            loaded = set()
            for sequence in sorted({row['sequence'] for row in config['jobs']}):
                assert sequence not in self.prefixes, 'Shards must contain disjoint videos'
                with np.load(root / (sequence + '_prefix.npz')) as archive:
                    prefix = {key: torch.from_numpy(archive[key]) for key in HISTORY_KEYS}
                length = len(prefix['history_frames'])
                assert torch.equal(prefix['history_frames'], torch.arange(length)) and prefix['history_valid'].all()
                assert all(torch.isfinite(value).all() for value in prefix.values())
                self.prefixes[sequence] = prefix
                with np.load(root / (sequence + '_queries.npz')) as archive:
                    queries = archive['query_frames']
                    assert np.array_equal(queries, np.unique(queries)) and (queries >= 1).all() and queries.max() == length
                    assert archive['instance_features'].shape == (len(queries), 7, 5, 2, 768)
                    for key in chunks:
                        values = torch.from_numpy(archive[key])
                        assert len(values) == len(queries) and torch.isfinite(values).all()
                        chunks[key].append(values)
                self.jobs.extend((sequence, int(query)) for query in queries)
                loaded.update((sequence, int(query)) for query in queries)
            assert loaded == requested and len(loaded) == done['clips'] == config['clips']
            self.configs.append(config)
        assert self.jobs and len(set(self.jobs)) == len(self.jobs)
        self.queries = {key: torch.cat(value) for key, value in chunks.items()}
        assert self.queries['valid'][:, 0].any(1).all()
        assert torch.equal(self.queries['action_valid'][..., 0], self.queries['valid'])
        assert torch.equal(self.queries['action_valid'][..., 1], self.queries['valid'] & (self.queries['raw_score'] > .84))
        self.names = {name for name, _ in self.jobs}

    def __len__(self):
        return len(self.jobs)

    def batch(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long, device='cpu')
        selected = [self.jobs[index] for index in indices.tolist()]
        capacity = max(8, max(query for _, query in selected))
        data = {key: values[indices].to(device) for key, values in self.queries.items()}
        for key in HISTORY_KEYS:
            values = []
            for row, (name, query) in enumerate(selected):
                source = self.prefixes[name][key]
                shape = (capacity,) + source.shape[1:]
                value = torch.zeros(shape, dtype=source.dtype)
                if key == 'history_instance_descriptors':
                    value[:] = self.queries['anchor_features'][indices[row]].to(source.dtype)
                elif key == 'history_boxes':
                    value[:] = source[0]
                elif key == 'history_iou':
                    value.fill_(-1.)
                # This boundary is causal even when the shared source video has
                # observations after this query. No current/future frame enters.
                value[-query:] = source[:query]
                values.append(value)
            data[key] = torch.stack(values).to(device)
        assert (data['history_frames'][:, -1] < torch.tensor([query for _, query in selected], device=device)).all()
        assert data['history_valid'][:, -1].all()
        return data
