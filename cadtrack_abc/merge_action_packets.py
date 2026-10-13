"""Merge the two successful collection shards into a matched full-split index."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collections', nargs=2, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    allowed = set(manifest['train_sequences' if args.split == 'train' else 'validation_sequences'])
    assert len(allowed) == (881 if args.split == 'train' else 98)
    shards = [(Path(folder).resolve(), json.loads((Path(folder) / 'collection_complete.json').read_text()))
              for folder in args.collections]
    records = []
    for folder, receipt in shards:
        assert receipt['completed']
        assert receipt['teacher_sha256'] == shards[0][1]['teacher_sha256']
        assert receipt['model_family'] == shards[0][1]['model_family']
        assert receipt['queries'] == len(receipt['records'])
        for original in receipt['records']:
            assert original['split'] == args.split and original['action_labels'] == 210
            assert original['teacher_sha256'] == receipt['teacher_sha256']
            packet = folder / original['file']
            assert packet.is_file()
            record = dict(original, file=os.path.relpath(packet, output.parent))
            records.append(record)
    assert {record['sequence'] for record in records} == allowed
    assert len({(record['sequence'], record['frame']) for record in records}) == len(records)
    result = {'completed': True, 'split': args.split, 'teacher_sha256': shards[0][1]['teacher_sha256'],
              'model_family': shards[0][1]['model_family'], 'queries': len(records),
              'sequences': len(allowed), 'records': sorted(records, key=lambda record: (record['sequence'], record['frame']))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(result, records=None)), flush=True)


if __name__ == '__main__':
    main()
