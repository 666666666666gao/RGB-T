"""Read-only, standard-library-only R1 receipt audit. No target code execution."""
from pathlib import Path, PurePosixPath
import ast
import csv
import hashlib
import json
import math
import struct
from datetime import datetime, timezone

ROOT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013')
REPO = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/cadtrack_abc')
OUT = ROOT / 'r1_integrity_audit_20261015'
SNAP = ROOT / 'r1_integrity_inputs_20261015_0705'
inputs = {}
checks = []


def read(path):
    path = Path(path)
    data = path.read_bytes()
    inputs[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                         'lines': len(data.splitlines())}
    return data


def data(path):
    return json.loads(read(path))


def check(name, condition, detail=None):
    checks.append({'id': name, 'pass': bool(condition), 'detail': detail})


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


inventory = data(ROOT / 'r1_integrity_input_paths.json')
initial = json.loads((OUT / 'initial_input_inventory.json').read_text())
for row in initial:
    path = Path(row['path'])
    check('exists:' + str(path), path.is_file())
    if path.is_file():
        value = read(path)
        check('initial_hash_unchanged:' + str(path), hashlib.sha256(value).hexdigest() == row['sha256'])
        if path.suffix == '.json' or path.name.endswith('.stdout.txt'):
            json.loads(value)

manifest = data(ROOT / 'full_training_manifest.json')
check('manifest_second_copy_identical', read(ROOT / 'native_formal_results_20261014/manifest.json') == read(ROOT / 'full_training_manifest.json'))
names = sorted(manifest['validation_sequences'])
train_names = manifest['train_sequences']
check('manifest_exact_881_train_98_dev_disjoint', len(names) == len(set(names)) == 98 and
      len(train_names) == len(set(train_names)) == 881 and not set(names) & set(train_names))
sampling = [data(REPO / f'results/full_sampling_20261014/validation_shard{s}.json') for s in (0, 1)]
frame_inventory = {}
for s, item in enumerate(sampling):
    check(f'sampling_shard{s}_identity', item['split'] == 'validation' and item['shard'] == s and item['shards'] == 2)
    check(f'sampling_shard{s}_scope', item['program_complete'] and item['full_shard_queries_complete'] and
          item['full_sequence_rollouts'] and not item['sanity'] and not item['official_test_used'])
    check(f'sampling_shard{s}_exact_expected_names', [r['sequence'] for r in item['records']] == names[s::2])
    frame_inventory.update({r['sequence']: r['frames'] for r in item['records']})
check('independent_previous_sampling_inventory_52200_frames', len(frame_inventory) == 98 and sum(frame_inventory.values()) == 52200)

selection = data(SNAP / 'outputs/r1_matched_fits_20261015/fit_search_actions/selection.json')
fit_search = data(REPO / 'results/r1_actual_admission_20261015/fit_search_complete.json')
fit_future = data(REPO / 'results/r1_actual_admission_20261015/fit_future_complete.json')
terminal = data(REPO / 'results/r1_actual_admission_20261015/full_fit_terminal_verified.json')
epochs = [0, 1, 2, 5, 10, 20]
fit_remote = PurePosixPath('/home/gaob/CADTrack-ABC/outputs/r1_matched_fits_20261015/fit_search_actions')
check('selection_exact_scheduled_epochs', [r['epoch'] for r in selection['candidates']] == epochs)
check('fit_search_exact_scheduled_epochs', [r['epoch'] for r in fit_search['candidates']] == epochs)
aggregations, records_export = [], []
valid_counts_by_epoch = {}
for epoch in epochs:
    expected_checkpoint = str(fit_remote / f'candidate_epoch{epoch:02d}.pth')
    candidate = next(r for r in selection['candidates'] if r['epoch'] == epoch)
    check(f'epoch{epoch}_candidate_path', candidate['checkpoint'] == expected_checkpoint)
    receipts = [data(SNAP / f'outputs/r1_main_full98_20261015/candidate_epoch{epoch:02d}/development_shard{s}.json') for s in (0, 1)]
    rows = []
    shards = []
    for s, item in enumerate(receipts):
        check(f'epoch{epoch}_shard{s}_completion', item['completed'] is True and item['full_videos'] is True and item['sanity'] is False)
        check(f'epoch{epoch}_shard{s}_checkpoint_policy', item['checkpoint'] == expected_checkpoint and
              item['epoch'] == epoch and item['policy_family'] == 'search_actions')
        check(f'epoch{epoch}_shard{s}_shard_identity', item['shard'] == s and item['shards'] == 2 and item['sequences'] == 49)
        check(f'epoch{epoch}_shard{s}_exact_ordered_names', [r['sequence'] for r in item['records']] == names[s::2])
        check(f'epoch{epoch}_shard{s}_protocol', item['strict_abc_and_future_head_load'] is True and
              item['first_gt_only_state_initialization'] is True and item['official_test_used'] is False and item['formal_metrics'] is None)
        for r in item['records']:
            key = f'epoch{epoch}_shard{s}_{r["sequence"]}'
            check(key + '_exact_full_frame_count', r['frames'] == frame_inventory[r['sequence']])
            check(key + '_scope_bounds', 0 < r['valid_frames'] <= r['frames'] and
                  0 <= r['extra_forwards'] <= r['frames'] - 1 and 0 <= r['future_interventions'] <= r['frames'] - 1)
            check(key + '_finite_metric_bounds', math.isfinite(r['iou_sum']) and math.isfinite(r['mean_iou']) and
                  0 <= r['iou_sum'] <= r['valid_frames'] and 0 <= r['mean_iou'] <= 1)
            check(key + '_float32_mean_exact', f32(r['iou_sum'] / r['valid_frames']) == r['mean_iou'])
            records_export.append(dict(epoch=epoch, shard=s, **r))
        rows.extend(item['records'])
        shards.append({'shard': s, 'sequences': len(item['records']), 'frames': sum(r['frames'] for r in item['records']),
                       'valid_frames': sum(r['valid_frames'] for r in item['records']), 'seconds_receipt': item['seconds']})
    check(f'epoch{epoch}_exact_unique_98', len(rows) == len({r['sequence'] for r in rows}) == 98)
    check(f'epoch{epoch}_full_52200_frames', sum(r['frames'] for r in rows) == 52200)
    valid_counts_by_epoch[epoch] = {r['sequence']: r['valid_frames'] for r in rows}
    summary = {'epoch': epoch, 'checkpoint': expected_checkpoint,
               'sequence_mean_iou': sum(r['mean_iou'] for r in rows) / 98,
               'frame_mean_iou': sum(r['iou_sum'] for r in rows) / sum(r['valid_frames'] for r in rows),
               'iou_sum': sum(r['iou_sum'] for r in rows),
               'frames': sum(r['frames'] for r in rows), 'valid_frames': sum(r['valid_frames'] for r in rows),
               'extra_forwards': sum(r['extra_forwards'] for r in rows),
               'future_interventions': sum(r['future_interventions'] for r in rows), 'shards': shards}
    for metric in ['sequence_mean_iou', 'frame_mean_iou']:
        check(f'epoch{epoch}_{metric}_bit_exact_aggregate', summary[metric] == candidate[metric],
              {'recomputed': summary[metric], 'reported': candidate[metric], 'difference': summary[metric] - candidate[metric]})
    aggregations.append(summary)
check('valid_frame_mask_counts_same_every_epoch', all(value == valid_counts_by_epoch[0] for value in valid_counts_by_epoch.values()))
selected = max(aggregations, key=lambda x: x['sequence_mean_iou'])
trained = max((x for x in aggregations if x['epoch'] > 0), key=lambda x: x['sequence_mean_iou'])
check('selected_epoch10_recomputed', selected['epoch'] == trained['epoch'] == 10)
for role, calculated in [('selected', selected), ('trained_selected', trained)]:
    check(role + '_all_values_exact', all(selection[role][k] == calculated[k] for k in selection[role]))
check('one_selected_and_trained_model_path', selection['best_checkpoint'] == selection['trained_best_checkpoint'] == str(fit_remote / 'best.pth'))
check('selection_scope_complete_not_formal', selection['full98_development_selection_complete'] is True and selection['formal_metrics_complete'] is False)

head_count = 275 * 2 + (275 * 128 + 128) + (128 * 12 + 12)
search_head_count = 295 * 2 + (295 * 128 + 128) + (128 * 6 + 6)
for policy, receipt, expected in [('future_actions', fit_future, head_count), ('search_actions', fit_search, head_count + search_head_count)]:
    check(policy + '_parameters_arithmetic', receipt['trainable_parameters'] == expected,
          {'calculated': expected, 'reported': receipt['trainable_parameters']})
    check(policy + '_20x80_updates', receipt['epochs'] == 20 and receipt['updates'] == 20 * math.ceil(receipt['train_queries'] / 64) == 1600)
    check(policy + '_scope_seed42', receipt['train_queries'] == 5092 and receipt['train_sequences'] == 881 and
          receipt['training_hyperparameters'] == {'batch': 64, 'lr': .0001, 'seed': 42, 'teacher_policy': 'native'})
    check(policy + '_candidate_epochs', [r['epoch'] for r in receipt['candidates']] == epochs)
    job = next(j for j in terminal['jobs'] if j['policy'] == policy)
    check(policy + '_terminal_matches_receipt', job['completed_fit'] == receipt and job['completed_epochs'] == 20 and
          job['latest_epoch']['updates'] == 1600 and job['latest_epoch']['epoch'] == 20 and job['process_exists'] is False)
    check(policy + '_cached_validation_not_formal', job['latest_epoch']['formal_metrics'] is None and
          job['latest_epoch']['cached_validation']['formal_metrics'] is None and job['latest_epoch']['cached_validation']['queries'] == 605)
check('matched_fit_teacher_and_indices_identical', all(fit_search[k] == fit_future[k] for k in
      ['teacher_sha256', 'train_index', 'validation_index', 'training_hyperparameters', 'train_queries', 'train_sequences', 'epochs', 'updates']))

admission = data(SNAP / 'outputs/r1_main_reports_20261015/admission.json')
check('admission_stdout_json_exact', admission == data(ROOT / 'verify_r1_main_formal_admission_remote.stdout.txt'))
config_bytes = read(REPO / 'config.json')
check('config_admission_hash_LF_exact', hashlib.sha256(config_bytes.replace(b'\r\n', b'\n')).hexdigest() == admission['configuration_sha256'],
      {'local_raw_sha256': hashlib.sha256(config_bytes).hexdigest(), 'local_LF_sha256': hashlib.sha256(config_bytes.replace(b'\r\n', b'\n')).hexdigest(),
       'admitted_remote_sha256': admission['configuration_sha256'], 'difference_explanation': 'CRLF versus LF only; normalized bytes reproduce admitted digest.'})
for name, digest in admission['source_sha256'].items():
    check('admitted_source_hash:' + name, hashlib.sha256(read(REPO / name)).hexdigest() == digest)
check('admission_selected_policy_matches_selection', admission['selected_epoch'] == 10 and admission['checkpoint'] == selection['best_checkpoint'] and
      admission['main_full98_selection_complete'] and admission['main_inference_started'] and not admission['formal_metrics_complete'])
check('admission_two_workers_per_card', admission['concurrency_per_physical_gpu'] == {'2': 2, '3': 2} and
      all(sum(r['physical_gpu'] == gpu for r in admission['actual_gpu_jobs']) == 2 for gpu in (2, 3)))
check('admission_main_jobs_physical2', len(admission['main_jobs']) == 2 and all(r['physical_gpu'] == 2 for r in admission['main_jobs']))
for s, job in enumerate(admission['main_jobs']):
    command = job['command']
    check(f'admission_main_shard{s}_arguments', command[command.index('--shard') + 1] == str(s) and
          command[command.index('--policy') + 1] == 'search_actions' and command[command.index('--checkpoint') + 1] == selection['best_checkpoint'])
control_admission = data(ROOT / 'record_r1_control_epoch00_admission_remote.stdout.txt')
check('control_admission_two_epoch00_shards_gpu3', len(control_admission['launches']) == 2 and
      all(r['physical_gpu'] == 3 and r['epoch'] == 0 and r['shard'] == s and r['policy'] == 'future_actions'
          for s, r in enumerate(control_admission['launches'])))

dataset = data(SNAP / 'dataset/rgbt210_complete.json')
check('RGBT210_210_unique_104706_frames', dataset['completed'] and len(dataset['records']) == len({r['sequence'] for r in dataset['records']}) == dataset['sequences'] == 210 and
      sum(r['frames'] for r in dataset['records']) == dataset['frames'] == 104706)
check('RGBT210_installed_GT_all_receipts_match', all(r['same_as_installed_toolkit_gt'] and r['frames'] == r['installed_gt_frames'] for r in dataset['records']) and
      dataset['installed_toolkit_gt_differing_sequences'] == [] and dataset['installed_toolkit_gt_frames'] == 104706)
check('RGBT210_extraction_stdout_matches_inventory', data(ROOT / 'extract_verify_official_rgbt210_remote.stdout.txt') == {k:v for k,v in dataset.items() if k != 'records'})

plan = data(SNAP / 'plan_snapshot.json')
native_report = data(REPO / 'results/native_lasher_20261014/lasher_report/full_report.json')
pure_report = data(REPO / 'results/pure_cad_rgbt234_20261015/full_report.json')
check('historical_native_formal_claim_numbers_exist_match', plan['native_formal_round_closure']['metrics_percent'] == native_report['overall_metrics_percent'] and
      native_report['sequences'] == 245 and native_report['frames'] == 220703)
check('historical_pure234_claim_numbers_exist_match', plan['pure_cad_other_benchmark_preparation_20261015']['formal_metrics'] == pure_report['overall_metrics_percent'] and
      pure_report['sequences'] == 234 and pure_report['frames'] == 116649)
check('plan_completed_fits_match_primary_receipts', plan['r1_matched_pair_20261015']['completed_fits'] == {'future_actions': fit_future, 'search_actions': fit_search})

additional_sources = ['future_data.py', 'counterfactual.py', 'collect_actions.py', 'ported_modules.py', 'r1_pair_pipeline.py', 'batched_actions.py']
for name in additional_sources:
    read(REPO / name)
for path in list(inputs):
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(encoding='utf-8'), filename=path)
        check('ast_parse:' + path, True)

with (OUT / 'all_588_candidate_sequence_records.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records_export[0]))
    writer.writeheader()
    writer.writerows(records_export)

result = {'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL',
          'generated_at': datetime.now(timezone.utc).isoformat(),
          'runtime': 'E:/python.exe -X utf8 -B; Python standard library only; AST parsed, no imported project modules',
          'cpu_only': True, 'ssh_calls': 0, 'neural_forwards': 0, 'optimizer_updates': 0,
          'check_count': len(checks), 'failures': [c for c in checks if not c['pass']],
          'aggregations': aggregations, 'selected_epoch': selected['epoch'], 'trained_selected_epoch': trained['epoch'],
          'per_sequence_records_checked': len(records_export),
          'selection_delta_sequence_mean_iou_vs_epoch0': selected['sequence_mean_iou'] - aggregations[0]['sequence_mean_iou'],
          'parameter_counts': {'future_head': head_count, 'search_head': search_head_count, 'joint': head_count + search_head_count},
          'checks': checks,
          'limits': ['Recomputed receipt aggregates, not IoUs from raw prediction and GT arrays.',
                     'No checkpoint bytes or optimizer tensors loaded; remote checkpoint digests are recorded evidence.',
                     'Formal R1 three-benchmark metrics are not present in the supplied completed snapshot.',
                     'Static source inspection cannot certify future runtime or overall goal achievement.']}
(OUT / 'deterministic_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'audited_input_hashes.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['checks']}, ensure_ascii=False, indent=2))
