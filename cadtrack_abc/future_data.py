"""Matched full-prefix packet inputs; each batch keeps four contexts separate."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .counterfactual import ACTION_MODES
from .future_actions import action_features, context_baseline


class FuturePackets(Dataset):
    def __init__(self, index, manifest, split, teacher_sha, threshold, teacher_policy):
        self.root = Path(index).parent
        data = json.loads(Path(index).read_text())
        assert data['completed'] and data['teacher_sha256'] == teacher_sha
        self.records = data['records']
        self.allowed = set(manifest['train_sequences' if split == 'train' else 'validation_sequences'])
        self.names = {r['sequence'] for r in self.records}
        assert self.records and self.names <= self.allowed
        assert len({(r['sequence'], r['frame']) for r in self.records}) == len(self.records)
        self.split, self.teacher_sha, self.threshold = split, teacher_sha, threshold
        self.model_family = ('CADTrackABC-native-current-quality' if teacher_policy == 'native'
                             else 'CADTrackFuturePolicy-future-actions')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        packet = torch.load(self.root / record['file'], map_location='cpu', weights_only=False)
        metadata, decision, labels = packet['metadata'], packet['decision'], packet['labels']
        assert metadata['split'] == self.split and metadata['teacher_sha256'] == self.teacher_sha
        assert metadata['model_family'] == self.model_family
        assert metadata['threshold'] == self.threshold and metadata['full_predicted_prefix_from_frame0']
        assert (metadata['sequence'], metadata['frame']) == (record['sequence'], record['frame'])
        age = torch.tensor([metadata['frame'] - 1])
        assert int(decision['frames'][0, -1]) == int(age[0])
        contexts = []
        for context_index, context in enumerate(decision['contexts']):
            baseline = context_baseline(context, self.threshold)
            features = action_features(context, decision['motion'], age, baseline).squeeze(0)
            utilities = torch.stack([torch.stack([
                next(label['utility'] for label in labels if
                     (label['context'], label['candidate'], label['mode']) == (context_index, candidate, mode))
                for mode in ACTION_MODES]) for candidate in range(len(features))])
            contexts.append((features, baseline.squeeze(0), utilities))
        assert [len(c[0]) for c in contexts] == [5, 10, 10, 10]
        context = metadata['reference_action'][0]
        assert context in (0, 1)
        reference_candidate, reference_mode = metadata['reference_action'][1:]
        reference_utility = contexts[context][2][reference_candidate, ACTION_MODES.index(reference_mode)]
        return {'contexts': contexts, 'observed_context': context, 'reference_utility': reference_utility,
                'search_input': decision['search_input'].squeeze(0)}


def collate_packets(packets):
    return {'contexts': [tuple(torch.stack([p['contexts'][context][item] for p in packets])
                               for item in range(3)) for context in range(4)],
            'observed_context': torch.tensor([p['observed_context'] for p in packets]),
            'reference_utility': torch.stack([p['reference_utility'] for p in packets]),
            'search_input': torch.stack([p['search_input'] for p in packets])}
