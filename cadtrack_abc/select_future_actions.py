"""Select one checkpoint from complete98-video comparisons, then retain it."""
import argparse
import filecmp
import json
from pathlib import Path
import shutil

import torch

from .future_actions import FutureActionHead
from .search_actions import SearchActionHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', required=True, help='Directory containing fit_complete.json and scheduled candidates.')
    parser.add_argument('--policy', choices=['future_actions', 'search_actions'], default='future_actions')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--development', required=True, help='Contains candidate_epochXX/development_shard0,1.json.')
    args = parser.parse_args()
    fit = Path(args.fit).resolve()
    receipt = json.loads((fit / 'fit_complete.json').read_text())
    assert receipt['status'] == ('SEARCH_ACTION_FIT_COMPLETE' if args.policy == 'search_actions' else 'FUTURE_HEAD_FIT_COMPLETE')
    names = set(json.loads(Path(args.manifest).read_text())['validation_sequences'])
    assert len(names) == 98
    scores = []
    for candidate in receipt['candidates']:
        checkpoint = Path(candidate['checkpoint']).resolve()
        assert checkpoint.parent == fit and checkpoint.name == 'candidate_epoch' + str(candidate['epoch']).zfill(2) + '.pth'
        records = []
        for shard in range(2):
            source = Path(args.development) / checkpoint.stem / ('development_shard' + str(shard) + '.json')
            data = json.loads(source.read_text())
            assert data['completed'] and data['full_videos'] and not data['sanity']
            assert data['shard'] == shard and data['shards'] == 2 and data['epoch'] == candidate['epoch']
            assert Path(data['checkpoint']).resolve() == checkpoint and data['policy_family'] == args.policy
            records.extend(data['records'])
        assert len(records) == 98 and {r['sequence'] for r in records} == names
        score = sum(r['mean_iou'] for r in records) / len(records)
        scores.append(dict(candidate, sequence_mean_iou=score,
                           frame_mean_iou=sum(r['iou_sum'] for r in records) / sum(r['valid_frames'] for r in records)))
    selected = max(scores, key=lambda value: value['sequence_mean_iou'])
    selected_path = Path(selected['checkpoint']).resolve()
    best = fit / 'best.pth'
    shutil.copyfile(selected_path, best)
    assert filecmp.cmp(selected_path, best, shallow=False)
    payload = torch.load(best, map_location='cpu', weights_only=False)
    assert payload['policy_family'] == args.policy and payload['epoch'] == selected['epoch']
    FutureActionHead().load_state_dict(payload['future_action'], strict=True)
    if args.policy == 'search_actions':
        SearchActionHead().load_state_dict(payload['search_action'], strict=True)
    result = {'selected': selected, 'best_checkpoint': str(best), 'candidates': scores,
              'full98_development_selection_complete': True, 'formal_metrics_complete': False,
              'selection_metric': 'Sequence-equal mean IoU on repeated98 development videos, not official test.',
              'strict_future_head_reload': True, 'cleanup': 'Only listed own candidate_epoch weights; fixed teacher/pretrain untouched.'}
    (fit / 'selection.json').write_text(json.dumps(result, indent=2))
    for candidate in receipt['candidates']:
        Path(candidate['checkpoint']).unlink()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
