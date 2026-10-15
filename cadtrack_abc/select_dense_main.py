"""Select initialized and trained R3 checkpoints using six complete full98 runs."""
import argparse
import filecmp
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .baseline import CleanCADTrack
from .dense_main_model import DenseABCModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', required=True)
    parser.add_argument('--development', required=True)
    parser.add_argument('--split-manifest', required=True)
    args = parser.parse_args()
    fit = Path(args.fit).resolve()
    terminal = json.loads((fit/'fit_complete.json').read_text())
    manifest = json.loads(Path(args.split_manifest).read_text())
    assert terminal['status'] in ('DENSE_MOTION_TRAINING_COMPLETE','DENSE_CONSEQUENCE_TRAINING_COMPLETE')
    assert terminal['epochs'] == 20
    assert [c['epoch'] for c in terminal['candidates']] == [0, 1, 2, 5, 10, 20]
    names = set(manifest['validation_sequences'])
    assert len(names) == 98 and not names & set(manifest['train_sequences'])
    scores = []
    for candidate in terminal['candidates']:
        checkpoint = Path(candidate['checkpoint']).resolve()
        assert checkpoint.parent == fit
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        records = []
        for shard in range(2):
            path = Path(args.development)/checkpoint.stem/f'development_shard{shard}.json'
            result = json.loads(path.read_text())
            assert result['completed'] and result['full_videos'] and not result['sanity']
            assert not result['official_test_used'] and result['strict_dense_abc_load']
            assert result['shard'] == shard and result['shards'] == 2
            assert result['epoch'] == candidate['epoch']
            assert result['checkpoint_sha256'] == digest
            records.extend(result['records'])
        assert len(records) == 98 and {r['sequence'] for r in records} == names
        assert sum(r['frames'] for r in records) == 52200
        scores.append(dict(candidate, checkpoint_sha256=digest,
            sequence_mean_iou=sum(r['mean_iou'] for r in records)/98,
            frame_mean_iou=sum(r['iou_sum'] for r in records)/sum(r['valid_frames'] for r in records)))
    overall = max(scores, key=lambda r:r['sequence_mean_iou'])
    trained = max((r for r in scores if r['epoch'] > 0), key=lambda r:r['sequence_mean_iou'])
    source = torch.load(scores[0]['checkpoint'], map_location='cpu', weights_only=False)
    model = DenseABCModel(CleanCADTrack(source['config']['pretrained']), source['config']['intervention_threshold'])
    selected_paths = []
    for selected, filename in [(overall, 'best.pth'), (trained, 'trained_best.pth')]:
        path = fit/filename
        source = Path(selected['checkpoint'])
        shutil.copyfile(source, path)
        assert filecmp.cmp(source, path, shallow=False)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['epoch'] == selected['epoch'] and payload['model_family'] == 'CADTrackDenseABC-R3'
        model.load_trainable_state(payload['dense_abc'])
        selected_paths.append(str(path))
    result = dict(selected=overall, trained_selected=trained, candidates=scores,
        best_checkpoint=selected_paths[0], trained_best_checkpoint=selected_paths[1],
        full98_development_selection_complete=True, strict_dense_abc_reload=True,
        formal_metrics_complete=False, initialization_is_training_gain=False,
        selection_metric='Sequence-equal valid-box IoU on repeated98 full videos; first frame included; not official SR.',
        official_policy='Same preselected positive-epoch checkpoint/config on all three benchmarks, including negative results. Pure epoch0 uses existing references; no baseline LasHeR rerun.',
        cleanup='Deferred until complete formal evaluation and audit; preserve pretrain and required best dependencies.')
    (fit/'selection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
