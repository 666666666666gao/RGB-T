import json
from pathlib import Path

project = Path('/home/gaob/CADTrack-ABC')
root = project / 'outputs/r1_main_full98_20261015/candidate_epoch02'
manifest = json.loads((project / 'outputs/cad_abc_20261013/manifest.json').read_text())
receipts = [json.loads((root / f'development_shard{shard}.json').read_text()) for shard in (0, 1)]
names = sorted(manifest['validation_sequences'])
for shard, receipt in enumerate(receipts):
    assert receipt['completed'] and receipt['full_videos'] and not receipt['sanity']
    assert receipt['epoch'] == 2 and receipt['policy_family'] == 'search_actions'
    assert receipt['shard'] == shard and receipt['shards'] == 2
    assert [record['sequence'] for record in receipt['records']] == names[shard::2]
records = [record for receipt in receipts for record in receipt['records']]
assert len(records) == 98 and len({record['sequence'] for record in records}) == 98
assert sum(record['frames'] for record in records) == 52200
result = dict(completed=True, checkpoint_epoch=2, policy='search_actions', sequences=98,
    frames=52200, valid_frames=sum(record['valid_frames'] for record in records),
    sequence_equal_mean_iou=sum(record['mean_iou'] for record in records) / 98,
    frame_equal_mean_iou=sum(record['iou_sum'] for record in records) /
        sum(record['valid_frames'] for record in records),
    extra_forwards=sum(record['extra_forwards'] for record in records),
    future_interventions=sum(record['future_interventions'] for record in records),
    shard_seconds=[receipt['seconds'] for receipt in receipts],
    original_native_ssh_exit_codes=[0, 0],
    scope='One of six complete repeated-development candidates; not selected final checkpoint or formal LasHeR SR.')
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
