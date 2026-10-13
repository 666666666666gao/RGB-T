import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import VideoClips, collate_clips
from .model import CADTrackABC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    distributed = 'RANK' in os.environ
    if distributed:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.manual_seed(config['seed'])
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(config['train_root']).iterdir() if p.is_dir())
    split_rng = random.Random(20261013)
    validation = sorted(split_rng.sample(names, 98))
    training = sorted(set(names) - set(validation))
    assert len(training) == 881 and len(validation) == 98
    model = CADTrackABC(config['pretrained'], config['threshold']).cuda(local_rank)
    abc_parameters = {n: p.numel() for n, p in model.abc.named_parameters()}
    if rank == 0:
        manifest = dict(config, train_sequences=training, validation_sequences=validation,
                        trainable_parameters=abc_parameters, pretrained_loaded_epoch=model.base.loaded_epoch,
                        pretrained_strict_tensors=model.base.loaded_tensors,
                        base_parameters=sum(p.numel() for p in model.base.parameters()),
                        pytorch=torch.__version__, physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        baseline={'LasHeR_PR': 72.74, 'LasHeR_NPR': 69.16, 'LasHeR_SR': 57.84,
                                  'source': 'user_provided', 'baseline_rerun': False},
                        status='SANITY_RUNNING' if args.sanity else 'TRAINING_RUNNING')
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps({'event': 'loaded', 'strict_tensors': model.base.loaded_tensors,
                          'trainable': sum(abc_parameters.values()), 'frozen': manifest['base_parameters'],
                          'train_sequences': len(training), 'validation_sequences': len(validation)}, ensure_ascii=False), flush=True)
    before = {name: value.detach().clone() for name, value in model.abc.named_parameters()} if args.sanity else None
    network = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if distributed else model
    optimizer = torch.optim.AdamW(model.abc.parameters(), lr=config['lr'], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_drop_epoch'], gamma=.1)
    samples = config['batch_per_gpu'] * world * 2 if args.sanity else config['samples_per_epoch']
    train_data = VideoClips(config['train_root'], training, samples, config['seed'], config['clip_steps'])
    sampler = DistributedSampler(train_data, shuffle=True, seed=config['seed']) if distributed else None
    loader = DataLoader(train_data, batch_size=config['batch_per_gpu'], sampler=sampler,
                        shuffle=sampler is None, num_workers=config['workers'], collate_fn=collate_clips,
                        persistent_workers=False, prefetch_factor=2)
    val_samples = config['batch_per_gpu'] * world if args.sanity else config['val_samples']
    val_data = VideoClips(config['train_root'], validation, val_samples, config['seed'] + 999, config['clip_steps'])
    val_sampler = DistributedSampler(val_data, shuffle=False) if distributed else None
    val_loader = DataLoader(val_data, batch_size=config['batch_per_gpu'], sampler=val_sampler,
                            num_workers=config['workers'], collate_fn=collate_clips, prefetch_factor=2)
    epochs = 1 if args.sanity else config['epochs']
    best, updates = -1., 0
    total_start = time.monotonic()
    for epoch in range(epochs):
        train_data.epoch = epoch
        if sampler is not None: sampler.set_epoch(epoch)
        network.train()
        start = time.monotonic()
        epoch_loss, count = 0., 0
        for batch, clips in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            result = network(clips)
            loss = result['loss']
            assert torch.isfinite(loss), (epoch, batch, float(loss))
            loss.backward()
            gradients = {}
            for group, prefix in [('A', 'memory.'), ('B_motion', 'motion.'), ('B_search', 'search.'), ('C', 'action.'), ('relations', 'relations.')]:
                squared = sum((p.grad.square().sum() for n, p in model.abc.named_parameters() if n.startswith(prefix) and p.grad is not None), loss.new_zeros(()))
                gradients[group] = float(squared.sqrt())
            assert all(np.isfinite(list(gradients.values())))
            assert all(p.grad is None for p in model.base.parameters())
            torch.nn.utils.clip_grad_norm_(model.abc.parameters(), 1.)
            optimizer.step()
            updates += 1
            epoch_loss += float(loss.detach()) * len(clips)
            count += len(clips)
            if rank == 0 and (batch < 2 or (batch + 1) % 25 == 0):
                elapsed = time.monotonic() - start
                record = {'event': 'update', 'epoch': epoch + 1, 'epochs': epochs, 'batch': batch + 1,
                          'batches': len(loader), 'updates': updates, 'loss': float(loss.detach()),
                          'parts': {k: float(v) for k, v in result['parts'].items()}, 'gradients': gradients,
                          'seconds_per_update': elapsed / (batch + 1),
                          'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9}
                print(json.dumps(record), flush=True)
                (output / 'progress.json').write_text(json.dumps(record, indent=2))
        network.eval()
        statistics = torch.tensor([epoch_loss, count, 0., 0.], device=model.device, dtype=torch.float64)
        with torch.no_grad():
            for clips in val_loader:
                result = network(clips, compute_loss=False)
                statistics[2] += result['sequence_iou'].sum()
                statistics[3] += len(clips)
        if distributed: dist.all_reduce(statistics)
        validation_iou = float(statistics[2] / statistics[3])
        scheduler.step()
        if rank == 0:
            record = {'event': 'epoch_complete', 'epoch': epoch + 1, 'updates': updates,
                      'train_loss': float(statistics[0] / statistics[1]), 'validation_clip_iou': validation_iou,
                      'epoch_seconds': time.monotonic() - start, 'elapsed_seconds': time.monotonic() - total_start,
                      'formal_test_metrics': None}
            print(json.dumps(record), flush=True)
            with (output / 'history.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
            payload = {'abc': model.abc.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'updates': updates,
                       'config': config, 'validation_clip_iou': validation_iou}
            torch.save(payload, output / 'last.pth')
            if validation_iou > best:
                best = validation_iou
                torch.save(payload, output / 'best.pth')
            if args.sanity:
                changed = {group: sum(int(not torch.equal(before[n], p.detach())) for n, p in model.abc.named_parameters() if n.startswith(prefix))
                           for group, prefix in [('A', 'memory.'), ('B_motion', 'motion.'), ('B_search', 'search.'), ('C', 'action.'), ('relations', 'relations.')]}
                assert all(changed.values()), changed
                reloaded = torch.load(output / 'best.pth', map_location='cpu', weights_only=False)
                model.abc.load_state_dict(reloaded['abc'], strict=True)
                witness = {'status': 'SANITY_PASS', 'changed_parameter_tensors': changed,
                           'strict_base_tensors': model.base.loaded_tensors, 'strict_best_reload': True,
                           'optimizer_updates': updates, 'world_size': world,
                           'seconds_per_update': (time.monotonic() - start) / len(loader)}
                (output / 'sanity.json').write_text(json.dumps(witness, indent=2))
            else:
                remaining = (epochs - epoch - 1) * record['epoch_seconds']
                (output / 'progress.json').write_text(json.dumps(dict(record, estimated_remaining_seconds=remaining, best_validation_clip_iou=best), indent=2))
        if distributed: dist.barrier()
    if rank == 0:
        model.abc.load_state_dict(torch.load(output / 'best.pth', map_location='cpu', weights_only=False)['abc'], strict=True)
        terminal = {'status': 'SANITY_COMPLETE' if args.sanity else 'TRAINING_COMPLETE', 'epochs': epochs,
                    'updates': updates, 'best_validation_clip_iou': best,
                    'best_checkpoint': str(output / 'best.pth'), 'strict_reload_passed': True,
                    'formal_test_metrics': None, 'elapsed_seconds': time.monotonic() - total_start}
        (output / 'complete.json').write_text(json.dumps(terminal, indent=2))
        print(json.dumps(terminal), flush=True)
    if distributed: dist.destroy_process_group()


if __name__ == '__main__':
    main()
