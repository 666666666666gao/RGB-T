"""Train sequence-bootstrap residual calibration on actual epoch5 policy states.

This is a finite C calibration experiment, not a new full ABC fit or an online
policy. Held-out equal-count ranking is a cached diagnostic; no past video
outputs are changed. Labels only enter the loss and post-prediction reports.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .recoverability_modules import DECISION_FIELDS, RecoverabilityModules, action_utility, select_actions
from .train_recoverability import LABEL_FIELDS, load_data


PARENT = '/data/gb/outputs/candidate_relation_native_recovery_20261006/reconstructed_epoch5/best.pth'
COLLECTION = Path('/data/gb/outputs/current_relation_policy_collection_20261006')
C1 = '/data/gb/outputs/c1_initial_seed42/best.pth'


class ResidualCommittee(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Linear(64, 2)) for _ in range(3)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, features, reference_action):
        predictions = torch.stack([head(features) for head in self.heads])
        rows = torch.arange(len(reference_action), device=reference_action.device)
        reference = predictions.flatten(2)[:, rows, reference_action]
        return predictions - reference[:, :, None, None, None]


@torch.no_grad()
def features_and_labels(model, data, batch_size):
    hidden = []
    handle = model.action[1].register_forward_hook(lambda module, args, result: hidden.append(result.detach()))
    parts = []
    for start in range(0, len(data['valid']), batch_size):
        batch = {key: value[start:start+batch_size] for key, value in data.items()}
        output = model({key: batch[key] for key in DECISION_FIELDS})
        assert len(hidden) == 1
        embedding = hidden.pop().reshape(-1, 7, 5, 128)
        region = torch.eye(7, device=embedding.device)[None, :, None].expand(len(embedding), -1, 5, -1)
        cues = torch.cat((embedding, output['scores'], output['advantage'], output['harm_logits'].sigmoid(),
                          output['write_risk_logits'][..., None].sigmoid(), output['quality_logits'][..., None].sigmoid(),
                          output['future_quality'][..., None], batch['evidence'].float(), region,
                          batch['raw_score'].float()[..., None], batch['c1_quality'].float()[..., None]), -1)
        chosen = select_actions(output, batch, .03, 'action', 'gross')
        available = batch['action_valid'].clone()
        available[:, 1:] = False
        rows = torch.arange(len(embedding), device=embedding.device)
        available[rows, chosen['searched_region']] = batch['action_valid'][rows, chosen['searched_region']] & chosen['search_triggered'][:, None, None]
        keep = batch['original_choice'].long()
        utility = action_utility(batch)
        delta = utility - utility[rows, 0, keep, 0][:, None, None, None]
        parent_delta = delta.flatten(1)[rows, chosen['flat_action']]
        parent_score = output['scores'].flatten(1)[rows, chosen['flat_action']]
        action_delta = delta-parent_delta[:,None,None,None]
        reference = torch.zeros_like(delta, dtype=torch.bool)
        reference[rows, 0, keep, 0] = True
        parts.append(dict(features=cues, parent_scores=output['scores'], target=delta,
                          residual_target=action_delta-(output['scores']-parent_score[:,None,None,None]),
                          action_delta=action_delta, legal=available,
                          keep=keep, reference=reference, parent_action=chosen['flat_action'],
                          search_cost=.01*chosen['search_triggered'].float(),
                          current=batch['current_iou'].float(), future=batch['future_iou'].float().mean(-1)))
    handle.remove()
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


@torch.no_grad()
def evaluate(head, data):
    residual = head(data['features'], data['parent_action'])
    mean, uncertainty = residual.mean(0), residual.std(0, unbiased=False)
    legal = data['legal'] & ~data['reference']
    target = data['target']
    parent_count = int((data['parent_action'] != data['keep'] * 2).sum())
    rows = torch.arange(len(target), device=target.device)
    parent_value = target.flatten(1)[rows, data['parent_action']] - data['search_cost']
    report = dict(queries=len(target), parent_intervention_count=parent_count,
                  parent_cached_net_utility=float(parent_value.mean()),
                  residual_mse=float((residual-data['residual_target'][None]).square()[:, data['legal']].mean()),
                  uncertainty_mean=float(uncertainty[data['legal']].mean()),
                  labels_are_same_state_H3_not_complete_video=True,
                  equal_count_is_batch_rank_diagnostic_not_online_policy=True)
    # Both variants have exactly the parent's number of changed query actions.
    # The population rank uses scores only; true utility is read afterwards.
    for name, beta in (('mean_only', 0.), ('uncertainty', .5)):
        scores = data['parent_scores'] + .5 * mean - beta * uncertainty
        scores = scores-scores[rows,0,data['keep'],0][:,None,None,None]
        best_score, candidate = scores.masked_fill(~legal, -torch.inf).flatten(1).max(1)
        # Peak extraction can retain only the original winner. Such a query
        # cannot receive a non-keep action in this equal-count comparison.
        assert int(torch.isfinite(best_score).sum()) >= parent_count
        order = best_score.argsort(descending=True, stable=True)
        selected = data['keep'] * 2
        selected = selected.clone()
        selected[order[:parent_count]] = candidate[order[:parent_count]]
        assert int((selected != data['keep'] * 2).sum()) == parent_count
        utility = target.flatten(1)[rows, selected] - data['search_cost']
        parent_current = data['current'].flatten(1)[rows, data['parent_action']//2]
        selected_current = data['current'].flatten(1)[rows, selected // 2]
        gain = utility-parent_value
        report[name] = dict(interventions=parent_count, cached_net_utility=float(utility.mean()),
                            gain_vs_parent=float(gain.mean()),
                            beneficial_vs_parent=int((gain > .03).sum()), harmful_vs_parent=int((gain < -.03).sum()),
                            immediate_rescues_vs_parent=int(((parent_current < .2) & (selected_current >= .5)).sum()),
                            immediate_harms_vs_parent=int(((parent_current >= .5) & (selected_current < .2)).sum()))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--lr', required=True, type=float)
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--epochs', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    assert args.epochs > 0 and args.lr > 0 and args.batch_size > 0
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    device = torch.device('cuda:0'); started = time.perf_counter()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    train_roots = [str(COLLECTION/'train/full'/f'gpu{gpu}') for gpu in range(4)]
    validation_roots = [str(COLLECTION/'validation/full'/f'gpu{gpu}') for gpu in range(4)]
    train, train_configs, train_names, train_jobs = load_data(train_roots, 'train', device)
    validation, val_configs, val_names, val_jobs = load_data(validation_roots, 'validation', device)
    assert not train_names & val_names and len(train_jobs) == 256 and len(val_jobs) == 196
    for config in train_configs + val_configs:
        assert config['prefix_model'] == PARENT and config['prefix_model_family'] == 'ABC_candidate_relations'
        assert config['prefix_search_value'] == 'gross' and config['prefix_write_verification'] == 'action'
        assert config['future_policy_mode'] == 'own' and config['future_horizon'] == 3
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    c1 = torch.load(C1, map_location='cpu', weights_only=False)
    model = RecoverabilityModules(c1, candidate_relations=True).to(device)
    model.load_state_dict(parent['model'], strict=True); model.eval().requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    training = features_and_labels(model, train, 64)
    development = features_and_labels(model, validation, 64)
    del train, validation
    sequences = sorted(train_names)
    job_sequences = torch.tensor([sequences.index(job[0]) for job in train_jobs], device=device)
    rng = np.random.default_rng(42)
    counts = np.stack([np.bincount(rng.integers(len(sequences),size=len(sequences)),minlength=len(sequences)) for _ in range(3)])
    bootstrap = torch.as_tensor(counts, device=device).float()[:, job_sequences]
    head = ResidualCommittee(training['features'].shape[-1]).to(device)
    initial = {key: value.detach().clone() for key, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    config = vars(args) | dict(module='action_risk_residual_probe', parent_model=PARENT, seed=42,
        committee_heads=3, sequence_bootstrap=True, features=training['features'].shape[-1],
        train_queries=len(train_jobs), validation_queries=len(val_jobs), train_sequences=len(train_names),
        validation_sequences=len(val_names), train_jobs=train_jobs, validation_jobs=val_jobs,
        future_horizon=3, future_policy='own_epoch5_gross_action', main_ABC_frozen=True,
        all_881_raw_video_training=False, no_test_data=True, forward_has_no_GT_or_future=True,
        validation_reused_for_checkpoint_selection=True, scientific_scope='C risk residual feasibility; not deployed online or formal tracking accuracy')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    history=[];best=-math.inf;steps=0;max_gradient=0.
    for epoch in range(args.epochs+1):
        if epoch:
            head.train()
            for indices in torch.randperm(len(train_jobs),device=device).split(args.batch_size):
                predicted = head(training['features'][indices],training['parent_action'][indices])
                error = F.smooth_l1_loss(predicted,training['residual_target'][indices][None].expand_as(predicted),reduction='none',beta=.03)
                weight = training['legal'][indices].float()
                if args.balanced:
                    meaningful = training['action_delta'][indices].abs() > .03
                    weight = weight * torch.where(meaningful,4.,1.)
                # Retain sequence resampling multiplicity after normalizing actions.
                denominators = weight.flatten(1).sum(-1)
                per_query = (error*weight[None]).flatten(2).sum(-1) / denominators[None]
                query_weight = bootstrap[:,indices]
                loss = (per_query*query_weight).sum() / query_weight.sum()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(norm)
                max_gradient=max(max_gradient,float(norm));optimizer.step();steps+=1
        head.eval();train_report=evaluate(head,training);val_report=evaluate(head,development)
        row=dict(epoch=epoch,optimizer_steps=steps,train=train_report,validation=val_report)
        history.append(row);(out/'metrics.json').write_text(json.dumps(history,indent=2,allow_nan=False))
        checkpoint=dict(module='action_risk_residual_probe',head=head.state_dict(),epoch=epoch,args=config,parent_model=PARENT)
        score=val_report['uncertainty']['cached_net_utility']
        if score > best:
            best=score;torch.save(checkpoint,out/'best.pth')
        if epoch==args.epochs:torch.save(checkpoint,out/'last.pth')
        print(json.dumps(row),flush=True)
    assert steps==args.epochs*math.ceil(len(train_jobs)/args.batch_size) and max_gradient>0
    assert any(not torch.equal(value,initial[key]) for key,value in head.state_dict().items())
    assert all(torch.equal(value.cpu(),before[key]) for key,value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    endpoint=evaluate(head,development)
    for filename in ('best.pth','last.pth'):
        saved=torch.load(out/filename,map_location=device,weights_only=False)
        head.load_state_dict(saved['head'],strict=True)
        reloaded=evaluate(head,development)
        assert reloaded==history[saved['epoch']]['validation']
    (out/'completion.json').write_text(json.dumps(dict(completed=True,epochs=args.epochs,optimizer_steps=steps,
        committee_changed=True,max_committee_gradient_norm=max_gradient,main_ABC_changed=False,
        parent_parameters_exact=True,parent_gradients_absent=True,strict_best_last_reload_equal=True,
        best_epoch=torch.load(out/'best.pth',map_location='cpu',weights_only=False)['epoch'],
        last_validation=endpoint,elapsed_seconds=time.perf_counter()-started,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20),indent=2,allow_nan=False))


if __name__=='__main__':main()
