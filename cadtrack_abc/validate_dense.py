"""Complete continuous98-video development selection for dense R2 models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import iou, xyxy
from .dense_evaluation import load_dense_tracker, track_video


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--shard',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    manifest=json.loads(Path(args.manifest).read_text())
    names=sorted(manifest['validation_sequences'])
    assert len(names)==len(set(names))==98
    assert not set(names)&set(manifest['train_sequences'])
    tracker,payload=load_dense_tracker(args.checkpoint)
    assert payload['use_memory']==manifest['config']['use_memory']
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    started=time.monotonic()
    for name in names[args.shard::2]:
        sequence=Path(payload['config']['train_root'])/name
        rgb,tir=sorted((sequence/'visible').iterdir()),sorted((sequence/'infrared').iterdir())
        labels=np.loadtxt(sequence/'init.txt',delimiter=',',ndmin=2)
        assert len(rgb)==len(tir)==len(labels)
        prediction,latency,diagnostics=track_video(tracker,rgb,tir,labels[0])
        valid=(labels[:,2:]>0).all(-1)
        overlaps=iou(xyxy(torch.tensor(prediction)),xyxy(torch.tensor(labels))).numpy()[valid]
        np.savetxt(output/f'{name}.txt',prediction,delimiter='\t',fmt='%.17g')
        np.save(output/f'{name}_latency.npy',latency)
        (output/f'{name}_mechanisms.json').write_text(json.dumps(diagnostics))
        record=dict(sequence=name,frames=len(prediction),valid_frames=int(valid.sum()),
            iou_sum=float(overlaps.sum()),mean_iou=float(overlaps.mean()),extra_forwards=0,
            template_updates=sum(d['template_updated'] for d in diagnostics))
        records.append(record)
        print(json.dumps(record),flush=True)
    receipt=dict(completed=True,full_videos=True,sanity=False,records=records,sequences=len(records),
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        epoch=payload['epoch'],model_family='CADTrackDense-R2',use_memory=payload['use_memory'],
        shard=args.shard,shards=2,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        seconds=time.monotonic()-started,strict_dense_load=True,first_gt_only_state_initialization=True,
        official_test_used=False,formal_metrics=None,
        scope='Repeated98 development videos, positive-width/height sequence IoU; not official SR. First frame included. Stored float64 predictions round-trip exactly.')
    (output/f'development_shard{args.shard}.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
