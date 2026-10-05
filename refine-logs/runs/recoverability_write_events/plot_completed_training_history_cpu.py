"""Plot the existing complete training logs on CPU; cached VAL is not benchmark accuracy."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
setup = Path('/data/gb/setup')
acceptance = json.loads((setup / 'write_events_b336_full_fit_cpu_acceptance_20261005.json').read_text())
assert acceptance['status'] == 'PASS' and acceptance['all_four_ABC_fits_passed']
assert all(info['optimizer_steps'] == 480 and info['epochs'] == 60 for info in acceptance['arms'].values())
target = Path('/data/gb/outputs/recoverability_write_events_training_history_20261005')
assert not target.exists()
target.mkdir()
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
summary = {}
for arm, info in acceptance['arms'].items():
    root = Path(info['output'])
    validation = json.loads((root / 'metrics.json').read_text())
    training = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    done = json.loads((root / 'completion.json').read_text())
    assert done['completed'] and done['epochs'] == info['epochs'] and done['optimizer_steps'] == 480
    assert [row['epoch'] for row in validation] == list(range(info['epochs'] + 1))
    assert len(training) == 2 * info['epochs'] and training[-1]['optimizer_steps'] == 480
    axes[0, 0].plot([row['optimizer_steps'] for row in training], [row['loss'] for row in training], label=arm, lw=1)
    epochs = [row['epoch'] * (480 // info['epochs']) for row in validation]
    axes[0, 1].plot(epochs, [row['utility'] for row in validation], label=arm, lw=1.5)
    axes[1, 0].plot(epochs, [100 * row['current_gain_against_c1'] for row in validation], label=arm, lw=1.5)
    axes[1, 1].plot(epochs, [row['changed_location'] for row in validation], label=arm, lw=1.5)
    summary[arm] = {'completed_epochs': info['epochs'], 'optimizer_steps': 480, 'logged_steps': len(training),
                    'cached_best_epoch': done['best_epoch'], 'initial_cached_VAL': validation[0],
                    'last_cached_VAL': validation[-1], 'modules_changed': done['modules_changed'],
                    'max_module_gradient_norms': done['max_module_gradient_norms']}
axes[0, 0].set(title='Loss at logged training steps (first and last each epoch)', xlabel='Optimizer updates', ylabel='Training objective')
axes[0, 1].set(title='Cached developer VAL utility (not native accuracy)', xlabel='Optimizer updates', ylabel='Utility')
axes[1, 0].set(title='Cached current-IoU gain versus C1 (not SR)', xlabel='Optimizer updates', ylabel='IoU difference, percentage points')
axes[1, 1].set(title='Changed locations in 196 cached VAL states', xlabel='Optimizer updates', ylabel='Number of states')
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('Four completed ABC fits: 480 updates each; GOLA-B and C1 frozen')
fig.savefig(target / 'training_history.png', dpi=180)
fig.savefig(target / 'training_history.pdf')
plt.close(fig)
(target / 'training_history_summary.json').write_text(json.dumps({'completed': True, 'arms': summary,
    'scope': 'Plots of already completed actual training logs and reused cached developer validation; no new neural forward, optimizer update or official benchmark score.',
    'training_loss_sampling': 'Only the actually logged first and last step per epoch (8 updates in every arm); not an epoch-mean estimate.'}, indent=2) + '\n')
print(json.dumps({'completed': True, 'output': str(target), 'actual_optimizer_steps_per_arm': 480,
                  'new_neural_calls': 0, 'new_optimizer_steps': 0}))
