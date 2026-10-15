"""Train matched R2-H/R2-HA on raw causal clips; select only after full98."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline import CleanCADTrack
from .data import VideoClips, collate_clips
from .dense_model import DenseCADTrack


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--head-lr', type=float, default=1e-5)
    parser.add_argument('--prompt-lr', type=float, default=1e-4)
    parser.add_argument('--probe', action='store_true', help='Two actual updates only; never a formal training result.')
    args = parser.parse_args()
    source = json.loads(Path(args.config).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pretrained = Path(source['pretrained']).resolve()
    pretrained_sha = hashlib.sha256(pretrained.read_bytes()).hexdigest()
    assert pretrained_sha == 'cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731'
    names = sorted(p.name for p in Path(source['train_root']).iterdir() if p.is_dir())
    assert len(names) == 979
    validation = sorted(random.Random(20261013).sample(names, 98))
    training = sorted(set(names) - set(validation))
    assert len(training) == 881
    config = dict(pretrained=str(pretrained), pretrained_sha256=pretrained_sha,
        train_root=source['train_root'], seed=42, data_generator_seed=42,
        use_memory=args.memory, batch=args.batch, workers=args.workers,
        epochs=1 if args.probe else args.epochs,
        samples_per_epoch=args.batch * 2 if args.probe else args.samples,
        clip_steps=args.steps, head_lr=args.head_lr, prompt_lr=args.prompt_lr,
        lr_drop_epoch=10, lr_drop_gamma=.1, probe=args.probe,
        localization_weights=dict(focal=1., l1=5., giou=2.),
        backbone_mode='Frozen/eval', fusion_and_head_mode='Train including BatchNorm; both variants identical.',
        selection='Complete scheduled full98 evaluations after all prescribed training. No clip-best selection.')
    model = DenseCADTrack(CleanCADTrack(pretrained), args.memory).cuda()
    frozen_before = module_hash(model.base.backbone)
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    groups = [dict(params=list(model.base.decode_fuse_search.parameters()) + list(model.base.box_head.parameters()), lr=args.head_lr)]
    if args.memory:
        groups.append(dict(params=model.prompt.parameters(), lr=args.prompt_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    parameters = [value for value in model.parameters() if value.requires_grad]
    dataset = VideoClips(source['train_root'], training, config['samples_per_epoch'], 42, args.steps)
    data_generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=data_generator,
        num_workers=args.workers, collate_fn=collate_clips, persistent_workers=False, prefetch_factor=2)
    manifest = dict(config=config, model_family='CADTrackDense-R2', variant='HA' if args.memory else 'H',
        train_sequences=training, validation_sequences=validation, split_seed=20261013,
        pretrained_strict_tensors=model.base.loaded_tensors, pretrained_epoch=model.base.loaded_epoch,
        trainable_parameters={name:value.numel() for name,value in model.named_parameters() if value.requires_grad},
        frozen_backbone_sha256=frozen_before, torch_version=torch.__version__,
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], baseline_lasher_rerun=False,
        status='QUALIFICATION_RUNNING' if args.probe else 'TRAINING_RUNNING')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(event='dense_loaded',variant=manifest['variant'],
        trainable=sum(manifest['trainable_parameters'].values()),frozen=sum(p.numel() for p in model.base.backbone.parameters()),
        actual_epochs=config['epochs'],actual_samples_per_epoch=len(dataset),probe=args.probe)),flush=True)
    candidates, history = [], []
    updates = clips_seen = 0
    started = time.monotonic()

    def save_candidate(epoch):
        path = output/f'candidate_epoch{epoch:02d}.pth'
        payload = dict(model_family=manifest['model_family'],use_memory=args.memory,
            dense=model.trainable_state(),epoch=epoch,updates=updates,config=config)
        torch.save(payload,path)
        candidates.append(dict(epoch=epoch,checkpoint=str(path),updates=updates))

    save_candidate(0)
    for epoch in range(1, config['epochs'] + 1):
        dataset.epoch = epoch - 1
        model.train()
        order = hashlib.sha256()
        begin = time.monotonic()
        total_loss = 0.
        epoch_clips = 0
        for batch, clips in enumerate(loader, 1):
            for clip in clips:
                key = [clip['name'],clip['frames'][:args.steps + 1].tolist()]
                order.update((json.dumps(key,separators=(',',':'))+'\n').encode())
            optimizer.zero_grad(set_to_none=True)
            result = model(clips)
            loss = result['loss']
            assert bool(torch.isfinite(loss)), (epoch,batch,float(loss))
            loss.backward()
            assert all(p.grad is None for p in model.base.backbone.parameters())
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
            assert bool(torch.isfinite(norm)), (epoch,batch,float(norm))
            if batch <= 2 or batch % 25 == 0:
                gradients = {}
                for name,prefix in [('fusion','base.decode_fuse_search.'),('head','base.box_head.'),('A','prompt.')]:
                    if name == 'A' and not args.memory:
                        continue
                    squared = sum((p.grad.square().sum() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None),loss.new_zeros(()))
                    gradients[name] = float(squared.sqrt())
            optimizer.step()
            updates += 1
            clips_seen += len(clips)
            epoch_clips += len(clips)
            total_loss += float(loss.detach()) * len(clips)
            if batch <= 2 or batch % 25 == 0:
                elapsed = time.monotonic() - begin
                record = dict(event='dense_update',epoch=epoch,epochs=config['epochs'],batch=batch,
                    batches=len(loader),updates=updates,clips_seen=clips_seen,loss=float(loss.detach()),
                    parts={k:float(v) for k,v in result['parts'].items()},gradients=gradients,
                    clips_per_second=epoch_clips/elapsed,seconds_per_update=elapsed/batch,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    formal_test_metrics=None,probe=args.probe)
                (output/'progress.json').write_text(json.dumps(record,indent=2))
                print(json.dumps(record),flush=True)
        assert epoch_clips == len(dataset)
        scheduler.step()
        record = dict(epoch=epoch,updates=updates,clips=epoch_clips,
            mean_loss=total_loss/epoch_clips,clip_order_sha256=order.hexdigest(),
            epoch_seconds=time.monotonic()-begin,elapsed_seconds=time.monotonic()-started)
        history.append(record)
        (output/'history.json').write_text(json.dumps(history,indent=2))
        payload = dict(model_family=manifest['model_family'],use_memory=args.memory,dense=model.trainable_state(),
            epoch=epoch,updates=updates,config=config,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
        torch.save(payload,output/'last.pth')
        if epoch in (1,2,5,10,20):
            save_candidate(epoch)
        print(json.dumps(dict(event='dense_epoch_complete',**record)),flush=True)
    assert clips_seen == len(dataset) * config['epochs']
    frozen_after = module_hash(model.base.backbone)
    assert frozen_before == frozen_after
    changed = {}
    for name,prefix in [('fusion','base.decode_fuse_search.'),('head','base.box_head.'),('A','prompt.')]:
        if name == 'A' and not args.memory:
            continue
        changed[name] = sum(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters() if n.startswith(prefix))
    assert all(changed.values()),changed
    reloaded = torch.load(output/'last.pth',map_location='cpu',weights_only=False)
    clone = DenseCADTrack(CleanCADTrack(pretrained),args.memory)
    clone.load_trainable_state(reloaded['dense'])
    for name,state in clone.trainable_state().items():
        assert all(torch.equal(value,reloaded['dense'][name][key]) for key,value in state.items())
    terminal = dict(status='DENSE_QUALIFICATION_COMPLETE' if args.probe else 'DENSE_TRAINING_COMPLETE',
        model_family=manifest['model_family'],use_memory=args.memory,epochs=config['epochs'],updates=updates,
        raw_clips=clips_seen,query_frame_observations=clips_seen*args.steps,candidates=candidates,
        changed_parameter_tensors=changed,frozen_backbone_before=frozen_before,frozen_backbone_after=frozen_after,
        strict_dense_reload_passed=True,pretrained_sha256=pretrained_sha,
        full98_selection_complete=False,formal_test_metrics=None,
        elapsed_seconds=time.monotonic()-started,probe=args.probe)
    (output/'fit_complete.json').write_text(json.dumps(terminal,indent=2))
    print(json.dumps(terminal),flush=True)


if __name__ == '__main__':
    main()
