"""Check real shared-event causality, full-objective gradients and CUDA capacity."""
import argparse
import json
import time
from pathlib import Path

import torch

from .event_rollout_data import EventRolloutData, HISTORY_KEYS
from .recoverability_modules import DECISION_FIELDS, RecoverabilityModules, objective, select_actions
from .train_event_recoverability import forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', nargs='+', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    train = EventRolloutData(args.train, 'train')
    val = EventRolloutData(args.validation, 'validation')
    assert not train.names & val.names
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    model = RecoverabilityModules(c1, candidate_relations=True).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    witnesses = [i for i,(name,q) in enumerate(train.jobs) if q<=32 and
                 (train.prefixes[name]['history_write'][:q] & (train.prefixes[name]['history_frames'][:q]>0)).any()]
    assert witnesses, 'Include a real short query after an observed template write'
    short = min(witnesses, key=lambda i: train.jobs[i][1])
    long = max(range(len(train)), key=lambda i: train.jobs[i][1])
    assert train.jobs[short][1] <= 32, 'Include a real early query for the sequential gradient witness'
    data = train.batch([short], device)
    q = train.jobs[short][1]
    witness = dict(query=train.jobs[short], observed_history_writes=int(data['history_write'].sum()))
    # Directly check the real shared source boundary, including left padding.
    for key in HISTORY_KEYS:
        assert torch.equal(data[key][0, -q:].cpu(), train.prefixes[train.jobs[short][0]][key][:q])
    gradients, outputs, losses = [], [], []
    for implementation in (model, lambda row: forward(model, row)):
        model.zero_grad(set_to_none=True)
        output = implementation({key:data[key] for key in DECISION_FIELDS})
        probability=output['past_gate_logits'].softmax(-1)
        verified_write=data['history_valid'][...,None] & (data['history_frames'][...,None]>0) & data['history_write'][...,None] & (probability[...,0]>=.5)
        assert verified_write.any(), 'The real sequential witness must exercise an accepted memory write'
        witness['verified_modality_writes']=int(verified_write.sum())
        loss, _ = objective(model, output, data, 'oracle', .03, 'reference', False, 'action', 'gross')
        loss.backward()
        gradients.append({name:value.grad.detach().clone() for name,value in model.named_parameters() if value.grad is not None})
        outputs.append({key:output[key].detach().clone() for key in ('memory', 'advantage', 'harm_logits')})
        losses.append(float(loss.detach()))
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=3e-5, rtol=3e-4)
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=2e-6, rtol=2e-5)
    gradient_error = max(float((gradients[0][name]-gradients[1][name]).abs().max()) for name in gradients[0])
    del data, gradients, outputs, output, loss
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    # Repeat the actual longest collected state, not a short synthetic history.
    data = train.batch([long] * args.batch_size, device)
    output = forward(model, data)
    loss, _ = objective(model, output, data, 'oracle', .03, 'reference', False, 'action', 'gross')
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
    assert all(value.grad is None for value in model.c1.parameters())
    torch.cuda.synchronize(device)
    capacity = dict(batch=args.batch_size, actual_longest_query=train.jobs[long],
                    history_length=data['history_valid'].shape[1],
                    forward_backward_seconds=time.perf_counter()-started,
                    peak_allocated_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20,
                    peak_reserved_cuda_mib=torch.cuda.max_memory_reserved(device)/2**20)
    assert all(torch.equal(value.cpu(), checkpoint['model'][name]) for name,value in model.state_dict().items())
    receipt = dict(passed=True, actual_shared_prefix_boundary=True, gradient_witness=witness,full_objective_gradient_max_error=gradient_error,
                   sequential_parallel_losses=losses, capacity=capacity, optimizer_updates=0,
                   original_parent_weights_unchanged=True, native_TEST_accuracy=False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
