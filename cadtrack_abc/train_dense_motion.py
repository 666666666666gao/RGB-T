"""R3 raw localization/motion stage; launch only after the R2 localizer lock.

This is not consequence-policy training. Future/search heads remain frozen and
their future-action completion flags stay false in every checkpoint/receipt.
"""
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
from .dense_main_model import DenseABCModel
from .train_dense import module_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--r2-pair-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--samples',type=int,default=20000)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--localizer-lr',type=float,default=1e-5)
    parser.add_argument('--temporal-lr',type=float,default=1e-4)
    parser.add_argument('--threshold',type=float,default=.03)
    parser.add_argument('--probe',action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    source = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    pretrained = Path(source['pretrained']).resolve()
    pretrained_sha = hashlib.sha256(pretrained.read_bytes()).hexdigest()
    assert pretrained_sha == 'cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731'
    pair_root = Path(args.r2_pair_root).resolve()
    pair = json.loads((pair_root/'pair_complete.json').read_text())
    assert pair['status'] == 'DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE'
    assert pair['all20_actual_clip_orders_equal']
    parent_run = pair_root/'HA'
    completed = json.loads((parent_run/'pipeline_complete.json').read_text())
    assert completed['full98_selection_complete'] and set(completed['reports']) == {'lasher','rgbt234','rgbt210'}
    selection = json.loads((parent_run/'fit/selection.json').read_text())
    assert selection['full98_development_selection_complete'] and selection['strict_dense_reload']
    parent_path = Path(selection['best_checkpoint']).resolve()
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert parent_sha == selection['selected']['checkpoint_sha256']
    parent = torch.load(parent_path,map_location='cpu',weights_only=False)
    assert parent['model_family'] == 'CADTrackDense-R2' and parent['use_memory']
    assert parent['config']['pretrained_sha256'] == pretrained_sha
    names = sorted(path.name for path in Path(source['train_root']).iterdir() if path.is_dir())
    assert len(names) == 979
    validation = sorted(random.Random(20261013).sample(names,98))
    training = sorted(set(names)-set(validation))
    config = dict(pretrained=str(pretrained),pretrained_sha256=pretrained_sha,
        parent_checkpoint=str(parent_path),parent_checkpoint_sha256=parent_sha,parent_epoch=parent['epoch'],
        parent_selection='R2-HA full98 overall best, including epoch0 if it wins. Never selected by formal test scores.',
        train_root=source['train_root'],seed=42,data_generator_seed=42,
        epochs=1 if args.probe else args.epochs,
        samples_per_epoch=args.batch*2 if args.probe else args.samples,
        batch=args.batch,workers=args.workers,clip_steps=args.steps,probe=args.probe,
        localizer_lr=args.localizer_lr,temporal_lr=args.temporal_lr,
        lr_drop_epoch=10,lr_drop_gamma=.1,intervention_threshold=args.threshold,
        stage='dense_motion',future_actions_trained=False,search_actions_trained=False,
        pretrained_parent_is_training_gain=False,
        future_label_scope='Motion likelihood only; future images never enter raw tracking state.')
    model = DenseABCModel(CleanCADTrack(pretrained),args.threshold).cuda()
    model.load_localizer_state(parent['dense'])
    model.decisions.requires_grad_(False).eval()
    groups = dict(localizer=list(model.base.decode_fuse_search.parameters())+
        list(model.base.box_head.parameters())+list(model.prompt.parameters()),
        motion=list(model.motion.parameters()),motion_prompt=list(model.motion_prompt.parameters()),
        candidate_evidence=list(model.candidate_evidence.parameters()))
    optimizer = torch.optim.AdamW([dict(params=values,lr=args.localizer_lr if name=='localizer' else args.temporal_lr)
        for name,values in groups.items()],weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
    parameters = [p for values in groups.values() for p in values]
    frozen_before = dict(backbone=module_hash(model.base.backbone),decisions=module_hash(model.decisions))
    before = {name:value.detach().cpu().clone() for name,value in model.named_parameters() if value.requires_grad}
    manifest = dict(config=config,model_family='CADTrackDenseABC-R3',stage='dense_motion',
        train_sequences=training,validation_sequences=validation,split_seed=20261013,
        pretrained_strict_tensors=model.base.loaded_tensors,pretrained_epoch=model.base.loaded_epoch,
        trainable_parameters={name:value.numel() for name,value in model.named_parameters() if value.requires_grad},
        frozen_before=frozen_before,parent_R2_checkpoint_sha256=parent_sha,
        selection='Prescribed complete training then full98 selection; final ABC requires later consequence-head training.',
        baseline_lasher_rerun=False,full_ABC_training_complete=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    dataset = VideoClips(source['train_root'],training,config['samples_per_epoch'],42,args.steps)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset,batch_size=args.batch,shuffle=True,generator=generator,
        num_workers=args.workers,collate_fn=collate_clips,persistent_workers=False,prefetch_factor=2)
    history,candidates = [],[]
    updates = clips_seen = 0
    started = time.monotonic()

    def save_candidate(epoch):
        path = output/f'candidate_epoch{epoch:02d}.pth'
        torch.save(dict(model_family='CADTrackDenseABC-R3',stage='dense_motion',dense_abc=model.trainable_state(),
            epoch=epoch,updates=updates,config=config),path)
        candidates.append(dict(epoch=epoch,updates=updates,checkpoint=str(path)))

    save_candidate(0)
    for epoch in range(1,config['epochs']+1):
        dataset.epoch = epoch-1
        model.train()
        model.decisions.eval()
        order = hashlib.sha256()
        begin = time.monotonic()
        total_loss = 0.
        count = 0
        for batch,clips in enumerate(loader,1):
            for clip in clips:
                key = [clip['name'],clip['frames'][:args.steps+1].tolist()]
                order.update((json.dumps(key,separators=(',',':'))+'\n').encode())
            optimizer.zero_grad(set_to_none=True)
            result = model(clips)
            loss = result['loss']
            assert bool(torch.isfinite(loss)),(epoch,batch,float(loss))
            loss.backward()
            assert all(p.grad is None for p in model.base.backbone.parameters())
            assert all(p.grad is None for p in model.decisions.parameters())
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
            assert bool(torch.isfinite(norm)),(epoch,batch,float(norm))
            log = batch <= 2 or batch % 25 == 0
            if log:
                gradients = {name:float(sum((p.grad.square().sum() for p in values if p.grad is not None),
                    loss.new_zeros(())).sqrt()) for name,values in groups.items()}
            optimizer.step()
            updates += 1
            clips_seen += len(clips)
            count += len(clips)
            total_loss += float(loss.detach())*len(clips)
            if log:
                elapsed = time.monotonic()-begin
                record = dict(stage='dense_motion',probe=args.probe,epoch=epoch,epochs=config['epochs'],
                    batch=batch,batches=len(loader),updates=updates,clips_seen=clips_seen,
                    loss=float(loss.detach()),parts={name:float(value) for name,value in result['parts'].items()},
                    gradients=gradients,clips_per_second=count/elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    future_actions_trained=False,search_actions_trained=False,formal_metrics=None)
                (output/'progress.json').write_text(json.dumps(record,indent=2))
                print(json.dumps(record),flush=True)
        assert count == len(dataset)
        scheduler.step()
        history.append(dict(epoch=epoch,updates=updates,clips=count,mean_loss=total_loss/count,
            clip_order_sha256=order.hexdigest(),epoch_seconds=time.monotonic()-begin))
        (output/'history.json').write_text(json.dumps(history,indent=2))
        torch.save(dict(model_family='CADTrackDenseABC-R3',stage='dense_motion',dense_abc=model.trainable_state(),
            epoch=epoch,updates=updates,config=config,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),output/'last.pth')
        if epoch in (1,2,5,10,20):
            save_candidate(epoch)
    assert clips_seen == len(dataset)*config['epochs']
    frozen_after = dict(backbone=module_hash(model.base.backbone),decisions=module_hash(model.decisions))
    assert frozen_before == frozen_after
    changed = {prefix:sum(not torch.equal(before[name],p.detach().cpu()) for name,p in model.named_parameters()
        if name.startswith(prefix) and name in before) for prefix in
        ('base.decode_fuse_search.','base.box_head.','prompt.','motion.','motion_prompt.','candidate_evidence.')}
    assert all(changed.values()),changed
    payload = torch.load(output/'last.pth',map_location='cpu',weights_only=False)
    clone = DenseABCModel(CleanCADTrack(pretrained),args.threshold)
    clone.load_trainable_state(payload['dense_abc'])
    reloaded = clone.trainable_state()
    for component,state in reloaded['localizer'].items():
        assert all(torch.equal(value,payload['dense_abc']['localizer'][component][name]) for name,value in state.items())
    for group in ('motion','motion_prompt','candidates','decisions'):
        assert all(torch.equal(value,payload['dense_abc'][group][name]) for name,value in reloaded[group].items())
    receipt = dict(status='DENSE_MOTION_QUALIFICATION_COMPLETE' if args.probe else 'DENSE_MOTION_TRAINING_COMPLETE',
        epochs=config['epochs'],updates=updates,raw_clips=clips_seen,query_frame_observations=clips_seen*args.steps,
        changed_parameter_tensors=changed,frozen_before=frozen_before,frozen_after=frozen_after,
        strict_dense_abc_reload=True,parent_checkpoint_sha256=parent_sha,candidates=candidates,
        elapsed_seconds=time.monotonic()-started,probe=args.probe,stage='dense_motion',
        future_actions_trained=False,search_actions_trained=False,full_ABC_training_complete=False,
        full98_selection_complete=False,formal_metrics=None,goal_achieved=False)
    (output/'fit_complete.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
