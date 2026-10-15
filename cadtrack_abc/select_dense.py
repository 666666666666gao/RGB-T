"""Select initialized and trained R2 checkpoints using six complete full98 runs."""
import argparse
import filecmp
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .baseline import CleanCADTrack
from .dense_model import DenseCADTrack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', required=True)
    parser.add_argument('--development', required=True)
    args = parser.parse_args()
    fit = Path(args.fit).resolve()
    terminal = json.loads((fit/'fit_complete.json').read_text())
    manifest = json.loads((fit/'manifest.json').read_text())
    assert terminal['status'] == 'DENSE_TRAINING_COMPLETE' and not terminal['probe']
    assert terminal['epochs'] == 20 and terminal['raw_clips'] == 400000
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
            assert not result['official_test_used'] and result['strict_dense_load']
            assert result['shard'] == shard and result['shards'] == 2
            assert result['epoch'] == candidate['epoch']
            assert result['checkpoint_sha256'] == digest
            assert result['use_memory'] == terminal['use_memory']
            records.extend(result['records'])
        assert len(records) == 98 and {r['sequence'] for r in records} == names
        assert sum(r['frames'] for r in records) == 52200
        scores.append(dict(candidate, checkpoint_sha256=digest,
            sequence_mean_iou=sum(r['mean_iou'] for r in records)/98,
            frame_mean_iou=sum(r['iou_sum'] for r in records)/sum(r['valid_frames'] for r in records)))
    overall = max(scores, key=lambda r:r['sequence_mean_iou'])
    trained = max((r for r in scores if r['epoch'] > 0), key=lambda r:r['sequence_mean_iou'])
    model = DenseCADTrack(CleanCADTrack(manifest['config']['pretrained']), terminal['use_memory'])
    selected_paths = []
    for selected, filename in [(overall, 'best.pth'), (trained, 'trained_best.pth')]:
        path = fit/filename
        source = Path(selected['checkpoint'])
        shutil.copyfile(source, path)
        assert filecmp.cmp(source, path, shallow=False)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['epoch'] == selected['epoch'] and payload['model_family'] == 'CADTrackDense-R2'
        model.load_trainable_state(payload['dense'])
        selected_paths.append(str(path))
    result = dict(selected=overall, trained_selected=trained, candidates=scores,
        best_checkpoint=selected_paths[0], trained_best_checkpoint=selected_paths[1],
        full98_development_selection_complete=True, strict_dense_reload=True,
        formal_metrics_complete=False, initialization_is_training_gain=False,
        selection_metric='Sequence-equal valid-box IoU on repeated98 full videos; first frame included; not official SR.',
        official_policy='Same preselected positive-epoch checkpoint/config on all three benchmarks, including negative results. Pure epoch0 uses existing references; no baseline LasHeR rerun.',
        cleanup='Deferred until complete formal evaluation and audit; preserve pretrain and required best dependencies.')
    (fit/'selection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
