"""Train complete A/B/C from shared exact-prefix event rollout supervision.

The network and objective remain the parent recipe. CPU histories are assembled
per mini-batch. Independent linear memory updates use their parallel closed form
for training; online inference retains the original incremental state update.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .event_rollout_data import EventRolloutData
from .recoverability_modules import DECISION_FIELDS, RecoverabilityModules, action_utility, objective, select_actions


def forward(model, data):
    data = {key: data[key] for key in DECISION_FIELDS}
    memory_model = model.memory
    anchor = memory_model.encode(data['anchor_features'])
    past = memory_model.encode(data['history_instance_descriptors'])
    batch, length, sensors, hidden = past.shape
    logits = memory_model.gate_logits(past, data['history_evidence'].float(),
                                     data['history_quality'].float(), anchor[:, None])
    probability = logits.softmax(-1)
    observed = data['history_valid'] & (data['history_frames'] > 0)
    commit = observed[..., None] & data['history_write'][..., None] & (probability[..., 0] >= .5)
    rates = memory_model.write_rates(past.reshape(-1, sensors, hidden), probability.reshape(-1, sensors, 3),
        observed.flatten(), commit.reshape(-1, sensors)).reshape(batch, length, 4, sensors)
    # m_t=(1-r_t)m_(t-1)+r_t*d_t. Rates do not depend on m_(t-1).
    # Exclusive suffix products avoid division by zero for a complete overwrite.
    survival = (1 - rates).flip(1).cumprod(1).flip(1)
    suffix = torch.cat((survival[:, 1:], torch.ones_like(survival[:, :1])), 1)
    memory = memory_model.initialize(anchor) * survival[:, 0, :, :, None]
    memory = memory + (past[:, :, None] * (rates * suffix)[..., None]).sum(1)
    output = model.decide(data, anchor, memory)
    output.update(past_descriptors=past, past_gate_logits=logits)
    return output


@torch.no_grad()
def evaluate(model, dataset, args, device, initial_parity=False):
    model.eval()
    values, total_loss, max_memory_error, parity_changes = [], 0., 0., 0
    for indices in torch.arange(len(dataset)).split(args.batch_size):
        data = dataset.batch(indices, device)
        output = forward(model, data)
        loss, _ = objective(model, output, data, args.search_supervision, args.threshold,
                            args.action_ranking, False, 'action', 'gross')
        chosen = select_actions(output, data, args.threshold, 'action', 'gross')
        if initial_parity:
            teacher = model.decide({key:data[key] for key in DECISION_FIELDS}, data['online_identity_anchor'], data['online_identity_memory'])
            teacher_choice = select_actions(teacher, data, args.threshold, 'action', 'gross')
            assert torch.equal(teacher_choice['flat_action'], data['parent_flat_action'])
            assert torch.equal(teacher_choice['search_triggered'], data['parent_search_triggered'])
            delta = (output['memory'] - data['online_identity_memory']).abs().max()
            max_memory_error = max(max_memory_error, float(delta))
            parity_changes += int((chosen['flat_action'] != teacher_choice['flat_action']).sum())
            parity_changes += int((chosen['search_triggered'] != teacher_choice['search_triggered']).sum())
        utility = action_utility(data).flatten(1)
        rows = torch.arange(len(indices), device=device)
        parent = data['parent_flat_action'].long()
        actual = utility[rows, chosen['flat_action']] - .01 * chosen['search_triggered']
        reference = utility[rows, parent] - .01 * data['parent_search_triggered']
        current = data['current_iou'].flatten(1)
        actual_current = current[rows, chosen['flat_action'] // 2]
        reference_current = current[rows, parent // 2]
        values.append(torch.stack((actual, reference, actual_current, reference_current,
            (chosen['flat_action'] != parent).float(),
            ((reference_current >= .5) & (actual_current < .2)).float()), -1).cpu())
        total_loss += float(loss) * len(indices)
    result = torch.cat(values)
    metrics = dict(clips=len(dataset), loss=total_loss/len(dataset), utility=float(result[:, 0].mean()),
        parent_utility=float(result[:, 1].mean()), utility_gain=float((result[:, 0]-result[:, 1]).mean()),
        current_iou=float(result[:, 2].mean()), parent_current_iou=float(result[:, 3].mean()),
        changed_actions=int(result[:, 4].sum()), current_harms_against_parent=int(result[:, 5].sum()))
    if initial_parity:
        assert parity_changes == 0, ('Reconstructed cache must preserve parent decisions', parity_changes, max_memory_error)
        metrics.update(initial_parent_decision_changes=parity_changes, maximum_initial_memory_error=max_memory_error)
    assert all(np.isfinite(value) for value in metrics.values())
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', nargs='+', required=True)
    p.add_argument('--init-checkpoint', required=True)
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--threshold', type=float, default=.03)
    p.add_argument('--search-supervision', choices=('oracle', 'selector'), default='oracle')
    p.add_argument('--action-ranking', choices=('reference', 'budgeted'), default='reference')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    assert args.epochs > 0 and args.batch_size > 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    train = EventRolloutData(args.train, 'train')
    validation = EventRolloutData(args.validation, 'validation')
    assert not train.names & validation.names
    reference = train.configs[0]
    for config in train.configs + validation.configs:
        for key in ('root', 'cache', 'split', 'head', 'pretrained', 'motion_run', 'prefix_model', 'future_policy', 'future_horizon'):
            assert config[key] == reference[key], key
        assert config['head'] == args.c1_head and config['prefix_model'] == args.init_checkpoint
    split = json.loads(Path(reference['split']).read_text())
    assert train.names <= set(split['train']) and validation.names <= set(split['validation'])
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'ABC_candidate_relations' and checkpoint['epoch'] == 5
    model = RecoverabilityModules(c1, candidate_relations=True).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    assert all(torch.equal(value.cpu(), checkpoint['model'][key]) for key, value in model.state_dict().items())
    groups = dict(A=model.memory, B=model.search, C=torch.nn.ModuleList((model.action, model.relations)))
    initial = {name: {key: value.detach().clone() for key, value in module.state_dict().items()} for name, module in groups.items()}
    initial_c1 = {key: value.detach().clone() for key, value in model.c1.state_dict().items()}
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | dict(module='ABC_candidate_relations', train_clips=len(train), validation_clips=len(validation),
        train_jobs=train.jobs, validation_jobs=validation.jobs, source_configs=train.configs + validation.configs,
        trainable_parameters={name:sum(value.numel() for value in module.parameters()) for name,module in groups.items()},
        frozen_full_GOLA_C1_and_motion=True, full_histories_retained_on_CPU=True,
        checkpoint_selection='Largest repeated-developer H3 utility charging .01 per search request, including padding-only requests; not native TEST metrics',
        initialization='Exact pretrained E5/gross/action modules; initial cache decisions must match actual trajectories',
        history_training='Parallel equivalent linear-memory recurrence; original incremental online update unchanged',
        scope='CompleteABC new module optimization on causal event states, not88Mbackbone end-to-end training')
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    train_initial = evaluate(model, train, args, device, initial_parity=True)
    val_initial = evaluate(model, validation, args, device, initial_parity=True)
    # Subsequent evaluations have the same checkpoint-selection metric schema.
    records = [dict(epoch=0, **{key:value for key,value in val_initial.items() if not key.startswith(('initial_', 'maximum_initial_'))})]
    best = records[0]['utility']
    def save(epoch, metrics, filename):
        torch.save(dict(module='ABC_candidate_relations', model=model.state_dict(), epoch=epoch,
                        args=vars(args), validation=metrics, selection_policy=config['checkpoint_selection']), output/filename)
    save(0, records[0], 'best.pth')
    save(0, records[0], 'initial.pth')
    (output / 'initial_parity.json').write_text(json.dumps(dict(train=train_initial, validation=val_initial), indent=2))
    steps, started = 0, time.perf_counter()
    max_gradients = {name:0. for name in groups}
    with (output / 'train.jsonl').open('w') as stream:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for indices in torch.randperm(len(train)).split(args.batch_size):
                data = train.batch(indices, device)
                prediction = forward(model, data)
                loss, parts = objective(model, prediction, data, args.search_supervision, args.threshold,
                                        args.action_ranking, False, 'action', 'gross')
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms = {name:float(torch.stack([value.grad.detach().square().sum() for value in module.parameters() if value.grad is not None]).sum().sqrt())
                         for name,module in groups.items()}
                assert all(np.isfinite(value) for value in norms.values()) and all(value.grad is None for value in model.c1.parameters())
                for name,value in norms.items():max_gradients[name]=max(max_gradients[name],value)
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
                optimizer.step()
                steps += 1
                row = dict(epoch=epoch, optimizer_steps=steps, loss=float(loss.detach()), parts=parts,
                           module_gradient_norms=norms, peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20)
                stream.write(json.dumps(row)+'\n');stream.flush()
            metrics = evaluate(model, validation, args, device)
            records.append(dict(epoch=epoch, **metrics))
            if metrics['utility'] > best:
                best=metrics['utility'];save(epoch, metrics, 'best.pth')
            (output / 'metrics.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(dict(epoch=epoch, optimizer_steps=steps, validation=metrics)), flush=True)
    save(args.epochs, metrics, 'last.pth')
    changed={name:any(not torch.equal(initial[name][key],value) for key,value in module.state_dict().items()) for name,module in groups.items()}
    assert all(changed.values()) and all(value>0 for value in max_gradients.values())
    assert all(torch.equal(initial_c1[key],value) for key,value in model.c1.state_dict().items())
    reloads={}
    for filename in ('best.pth','last.pth'):
        saved=torch.load(output/filename,map_location=device,weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        measured=evaluate(model,validation,args,device)
        expected={key:value for key,value in saved['validation'].items() if key!='epoch'}
        assert measured==expected
        reloads[filename]=dict(epoch=saved['epoch'],validation=measured,strict_reload=True)
    receipt=dict(completed=True,epochs=args.epochs,optimizer_steps=steps,modules_changed=changed,
        max_module_gradient_norms=max_gradients,frozen_C1_weights_exact=True,GOLA_and_motion_new_gradients=False,
        peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20,elapsed_seconds=time.perf_counter()-started,
        checkpoints=reloads,native_TEST_metrics_new=False)
    (output/'completion.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':main()
