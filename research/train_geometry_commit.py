"""Fit the small commit head on completed matched TRAIN event consequences.

old4 A/B/C and C1 are frozen. Event-disjoint TRAIN-only held-out states choose
one best checkpoint; full video and native tests remain separate acceptance.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .geometry_commit import ACTION_NAMES, GeometryCommitHead, choose_commit, commit_features, selected_index
from .recoverability_modules import RecoverabilityModules


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--probe-root', required=True)
    p.add_argument('--jobs', required=True)
    p.add_argument('--parent', required=True)
    p.add_argument('--c1-head', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, choices=[3, 32], required=True)
    p.add_argument('--lost-penalty', type=float, choices=[0., .1], required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sanity', action='store_true')
    return p.parse_args()


@torch.no_grad()
def load_states(args, modules, device):
    jobs = json.loads(Path(args.jobs).read_text())['jobs']
    labels = {(j['sequence'], j['query_frame']): j for j in jobs}
    events = {}
    for j in jobs:
        events[j['event_id']] = j['cached_event_role']
    assert len(events) == 24 and len(labels) == 88
    rng = random.Random(args.seed)
    validation = set()
    for role in ['normal', 'persistent33', 'recovered32']:
        names = sorted(event for event, value in events.items() if value == role)
        assert len(names) == 8
        rng.shuffle(names)
        validation.update(names[:2])
    features, utility, keep, legal_write, event_ids, query_rows = [], [], [], [], [], []
    for part, count in [('sanity', 1), ('gpu0', 22), ('gpu1', 22), ('gpu2', 22), ('gpu3', 21)]:
        out = Path(args.probe_root) / part
        assert (out / 'COMPLETE').is_file()
        config = json.loads((out / 'config.json').read_text())
        assert config['model'] == args.parent and config['checkpoint_epoch'] == 4
        rows = json.loads((out / 'events.json').read_text())['results']
        assert len(rows) == count
        for index, row in enumerate(rows):
            j = labels[(row['sequence'], row['query_frame'])]
            assert j['event_id'] == row['event_id']
            with np.load(out / f'event_{index:04d}.npz') as arrays:
                data = {key[len('decision_'):]: torch.from_numpy(arrays[key].copy()).to(device)[None]
                        for key in arrays.files if key.startswith('decision_')}
                anchor = torch.from_numpy(arrays['pre_query_identity_anchor'].copy()).to(device)[None]
                memory = torch.from_numpy(arrays['pre_query_identity_memory'].copy()).to(device)[None]
                choice = selected_index(data, arrays['raw_boxes_xyxy'][0])
                features.append(commit_features(modules, data, anchor, memory, choice)[0])
                quality = np.stack([arrays[name + '_iou'][1:args.horizon + 1] for name in ACTION_NAMES])
                utility.append(quality.mean(1) - args.lost_penalty * (quality < .2).mean(1))
            keep.append(int(row['actual_policy_query_pause']))
            legal_write.append(row['query_raw_write_eligible'])
            event_ids.append(j['event_id'])
            query_rows.append({'sequence':j['sequence'], 'query_frame':j['query_frame'], 'event_id':j['event_id'],
                               'partition':'validation' if j['event_id'] in validation else 'train'})
    assert len(features) == len({(r['sequence'],r['query_frame']) for r in query_rows}) == 88
    counts = Counter(event_ids)
    valid = torch.tensor([event in validation for event in event_ids], device=device)
    data = {'features':torch.stack(features), 'utility':torch.tensor(np.stack(utility),dtype=torch.float32,device=device),
            'keep':torch.tensor(keep,device=device), 'raw_write_eligible':torch.tensor(legal_write,device=device),
            'weight':torch.tensor([1/counts[e] for e in event_ids],device=device), 'validation':valid}
    assert valid.sum() == 22 and (~valid).sum() == 66
    return data, query_rows


@torch.no_grad()
def evaluate(head, data, mask):
    scores = head(data['features'][mask])
    keep = data['keep'][mask]
    choice = choose_commit(scores, keep, data['raw_write_eligible'][mask])
    utility = data['utility'][mask]
    rows = torch.arange(len(choice),device=choice.device)
    weights = data['weight'][mask]
    delta = utility[rows, choice] - utility[rows, keep]
    return {'event_weighted_utility_gain':float((delta*weights).sum()/weights.sum()),
            'queries':len(choice), 'changed_actions':int((choice!=keep).sum()),
            'geometry_actions':int((choice>=2).sum()), 'query_gains':int((delta>1e-8).sum()),
            'query_losses':int((delta < -1e-8).sum())}


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    parent = torch.load(args.parent,map_location='cpu',weights_only=False)
    assert parent['module']=='ABC_recoverability' and parent['epoch']==4
    c1 = torch.load(args.c1_head,map_location='cpu',weights_only=False)
    modules = RecoverabilityModules(c1).to(device)
    modules.load_state_dict(parent['model'],strict=True)
    modules.eval().requires_grad_(False)
    frozen = {key:value.detach().cpu().clone() for key,value in modules.state_dict().items()}
    data, jobs = load_states(args,modules,device)
    head = GeometryCommitHead().to(device)
    initial = {key:value.detach().clone() for key,value in head.state_dict().items()}
    assert torch.equal(choose_commit(head(data['features']),data['keep'],data['raw_write_eligible']),data['keep'])
    out=Path(args.output)
    assert not out.exists()
    out.mkdir(parents=True)
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(vars(args)|{'train_events':18,'validation_events':6,
        'train_queries':66,'validation_queries':22,'frozen_parent':'old4 A/B/C and C1',
        'labels':'same-state, frozen-old4 future consequences; GT never enters commit_features',
        'selection':'strict maximum held-out equal-event utility gain; initialization eligible',
        'training_scope':'TRAIN88 commit-head prototype; not full881 or end-to-end training'},indent=2)+'\n')
    valid = data['validation']; train = ~valid
    records=[{'epoch':0,'validation':evaluate(head,data,valid),'train':evaluate(head,data,train)}]
    best=records[0]['validation']['event_weighted_utility_gain']
    def save(epoch):
        torch.save({'module':'geometry_commit','head':head.state_dict(),'epoch':epoch,'args':vars(args),
                    'parent_model':args.parent,'features':463,'threshold':.03},out/'best.pth')
    save(0)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.0001)
    features=data['features'][train]; utility=data['utility'][train]; keep=data['keep'][train]
    weight=data['weight'][train]; rows=torch.arange(len(keep),device=device)
    target=utility-utility[rows,keep,None]
    legal=torch.ones_like(target,dtype=torch.bool)
    legal[:,1]=legal[:,3]=data['raw_write_eligible'][train]
    pairs=legal[:,:,None]&legal[:,None,:]&(utility[:,:,None]-utility[:,None,:]>.03)
    for epoch in range(1,args.epochs+1):
        head.train(); optimizer.zero_grad(set_to_none=True)
        scores=head(features); delta=scores-scores[rows,keep,None]
        regression=(F.smooth_l1_loss(delta,target,reduction='none')*legal).sum(1)/legal.sum(1)
        gap=utility[:,:,None]-utility[:,None,:]
        ranking=(F.relu(gap+scores[:,None,:]-scores[:,:,None])*pairs).sum((1,2))/pairs.sum((1,2)).clamp(min=1)
        loss=((regression+.2*ranking)*weight).sum()/weight.sum()
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step(); head.eval()
        record={'epoch':epoch,'optimizer_updates':epoch,'loss':float(loss.detach()),
                'validation':evaluate(head,data,valid),'train':evaluate(head,data,train)}
        records.append(record)
        if record['validation']['event_weighted_utility_gain']>best:
            best=record['validation']['event_weighted_utility_gain']; save(epoch)
        print(json.dumps(record),flush=True)
    assert all(torch.equal(frozen[key],value.detach().cpu()) for key,value in modules.state_dict().items())
    changed={key:float((value.detach()-initial[key]).norm()) for key,value in head.state_dict().items()}
    assert all(value>0 for value in changed.values())
    chosen=torch.load(out/'best.pth',map_location=device,weights_only=False)
    head.load_state_dict(chosen['head'],strict=True)
    reloaded=evaluate(head,data,valid)
    assert reloaded==records[chosen['epoch']]['validation']
    (out/'history.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'complete':True,'epochs':args.epochs,'optimizer_updates':args.epochs,
        'selected_epoch':chosen['epoch'],'reloaded_validation':reloaded,'head_parameter_change_norms':changed,
        'frozen_parent_exact':True,'sanity_only':args.sanity,'native_accuracy':None},indent=2)+'\n')
    (out/'COMPLETE').write_text('COMPLETE\n')


if __name__=='__main__':
    main()
