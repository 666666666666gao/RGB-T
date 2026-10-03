"""Strict initial/best/last reload and held-out ABC quality calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .temporal_modules import TemporalModules
from .train_temporal import load_data, forward, evaluate


def calibration(predicted, target):
    predicted, target = predicted.double().cpu().numpy(), target.double().cpu().numpy()
    bins = []
    for index in range(10):
        mask = (predicted >= index / 10) & ((predicted < (index + 1) / 10) if index < 9 else (predicted <= 1))
        bins.append({'bin': index, 'count': int(mask.sum()),
                     'prediction_sum': float(predicted[mask].sum()), 'target_sum': float(target[mask].sum())})
    assert sum(row['count'] for row in bins) == len(predicted)
    ece = sum(abs(row['prediction_sum'] - row['target_sum']) for row in bins) / len(predicted)
    return {'valid_actions': len(predicted), 'mse': float(((predicted - target) ** 2).mean()),
            'mae': float(np.abs(predicted - target).mean()), 'ece_10_bins': ece, 'bins': bins}


@torch.inference_mode()
def inspect_checkpoint(path, config, validation, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    c1 = torch.load(args['c1_head'], map_location='cpu', weights_only=False)
    model = TemporalModules(c1, args['slots'], args['motion_history'], args['modes'], checkpoint['horizon']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    assert all(torch.equal(value.cpu(), c1['head'][key]) for key, value in model.c1.state_dict().items())
    metrics = evaluate(model, validation, args['batch_size'], args['rank_gap'])
    assert metrics == checkpoint['validation'], 'Strict reload did not reproduce saved metrics'
    predictions, targets = {'current': [], 'future': []}, {'current': [], 'future': []}
    cost_error = []
    for start in range(0, len(validation['valid']), args['batch_size']):
        batch = {key: value[start:start + args['batch_size']] for key, value in validation.items()}
        output = forward(model, batch)
        valid = batch['valid']
        predictions['current'].append(output['current_logits'][valid].sigmoid())
        predictions['future'].append(output['future_logits'][valid].sigmoid())
        targets['current'].append(batch['current_iou'][valid])
        targets['future'].append(batch['future_iou'].mean(-1)[valid])
        cost_error.append((output['cost'][valid] - batch['wrong_update_fraction'][valid]).square())
        if checkpoint['epoch'] == 0:
            assert torch.equal(output['scores'], output['c1_scores'])
            choice = output['scores'].masked_fill(~valid, -torch.inf).argmax(1)
            assert torch.equal(choice, batch['original_choice'].long())
    result = {'epoch': checkpoint['epoch'], 'strict_reload_all_metric_differences': 0,
              'frozen_c1_state_exact': True, 'metrics': metrics,
              'calibration': {name: calibration(torch.cat(predictions[name]), torch.cat(targets[name]))
                              for name in predictions},
              'wrong_update_fraction_mse': float(torch.cat(cost_error).mean())}
    return result, checkpoint['model']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    root = Path(args.run)
    config = json.loads((root / 'config.json').read_text())
    receipt = json.loads((root / 'completion.json').read_text())
    records = json.loads((root / 'metrics.json').read_text())
    assert receipt['completed'] and not receipt['official_tracking_accuracy']
    assert [row['epoch'] for row in records] == list(range(config['epochs'] + 1))
    assert receipt['optimizer_steps'] == int(np.ceil(config['train_clips'] / config['batch_size'])) * config['epochs']
    validation, _, val_names = load_data([config['validation']], 'validation', device)
    train_names = {job['sequence'] for c in config['source_configs'] if c['partition'] == 'train' for job in c['jobs']}
    assert not train_names & val_names
    results, states = {}, {}
    for name in ('initial', 'best', 'last'):
        results[name], states[name] = inspect_checkpoint(root / (name + '.pth'), config, validation, device)
        epoch = results[name]['epoch']
        assert results[name]['metrics'] == {k: v for k, v in records[epoch].items() if k != 'epoch'}
    expected_best = max(records, key=lambda row: row['current'])['epoch']
    assert results['best']['epoch'] == expected_best
    assert results['initial']['epoch'] == 0 and results['last']['epoch'] == config['epochs']
    changed = {module: any(not torch.equal(value, states['last'][key])
                          for key, value in states['initial'].items() if key.startswith(prefix))
               for module, prefix in (('A', 'memory.'), ('B', 'motion.'), ('C', 'selector.'))}
    assert changed == receipt['modules_changed'] and all(changed.values())
    assert all(torch.equal(value, states['last'][key]) for key, value in states['initial'].items() if key.startswith('c1.'))
    sampling = []
    for c in config['source_configs']:
        jobs = c['jobs']
        sampling.append({'partition': c['partition'], 'sampling_seed': c['seed'], 'clips': len(jobs),
                          'unique_sequences': len({j['sequence'] for j in jobs}),
                          'unique_sequence_query_pairs': len({(j['sequence'], j['query_frame']) for j in jobs})})
    train_jobs = [job for c in config['source_configs'] if c['partition'] == 'train' for job in c['jobs']]
    pooled_unique = len({(job['sequence'], job['query_frame']) for job in train_jobs})
    report = {'completed': True, 'official_tracking_accuracy': False,
              'protocol': 'all held-out clips; fixed-C1 counterfactual future target; original disjoint TRAIN split',
              'checkpoints': results, 'expected_best_epoch': expected_best,
              'last_modules_changed': changed, 'initial_last_c1_state_exact': True,
              'train_validation_sequence_overlap': 0, 'sampling': sampling,
              'pooled_training_sampling': {'sampled_clips': len(train_jobs), 'unique_sequence_query_pairs': pooled_unique,
                                           'repeated_sequence_query_pairs': len(train_jobs) - pooled_unique},
              'optimizer_steps': receipt['optimizer_steps'], 'all_epochs_retained': len(records)}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'completed': True, 'best_epoch': expected_best,
                      'validation_clips': len(validation['valid']), 'all_strict_reloads_exact': True}))


if __name__ == '__main__':
    main()
