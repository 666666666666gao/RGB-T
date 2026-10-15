"""Full formal inference of one preselected positive-epoch R3 checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .dense_main_evaluation import load_dense_tracker, track_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['lasher','rgbt234','rgbt210'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, choices=[0,1], required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    tracker,payload = load_dense_tracker(args.checkpoint)
    assert payload['epoch'] > 0, 'Use existing pure reference for epoch0; never rerun pure LasHeR.'
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    sequences=sorted(path for path in Path(args.root).iterdir() if path.is_dir())
    assert len(sequences)=={'lasher':245,'rgbt234':234,'rgbt210':210}[args.dataset]
    records=[]
    for index,sequence in enumerate(sequences[args.shard::2],1):
        rgb,tir=sorted((sequence/'visible').iterdir()),sorted((sequence/'infrared').iterdir())
        annotation=sequence/('visible.txt' if args.dataset=='rgbt234' else 'init.txt')
        with annotation.open() as stream:
            initial=np.fromstring(stream.readline().strip(),sep=',')
        prediction,latency,diagnostics=track_video(tracker,rgb,tir,initial)
        np.savetxt(output/f'{sequence.name}.txt',prediction,delimiter='\t',fmt='%.3f')
        np.save(output/f'{sequence.name}_latency.npy',latency)
        (output/f'{sequence.name}_mechanisms.json').write_text(json.dumps(diagnostics))
        record=dict(sequence=sequence.name,frames=len(prediction),seconds=float(latency.sum()),
            extra_forwards=sum(d['extra_forwards'] for d in diagnostics),template_updates=sum(d['template_updated'] for d in diagnostics))
        records.append(record)
        print(json.dumps(dict(event='sequence_complete',shard=args.shard,index=index,**record)),flush=True)
    receipt=dict(completed=True,dataset=args.dataset,shard=args.shard,shards=2,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        model_family='CADTrackDenseABC-R3',policy_family='dense_ABC_R3_'+payload['stage'],
        stage=payload['stage'],future_actions_trained=payload['config']['future_actions_trained'],
        search_actions_trained=payload['config']['search_actions_trained'],
        epoch=payload['epoch'],records=records,sequences=len(records),frames=sum(r['frames'] for r in records),
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),strict_base_and_dense_abc_load=True,
        first_box_only_initialization=True,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
        components=dict(A=True,B=True,C=True),
        protocol='Native CAD crop/Center mapping; dense A and soft motion prior; local or one-extra candidate relations and graded submissions. Enabled components are not evidence of effectiveness.')
    (output/f'complete_shard{args.shard}.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
