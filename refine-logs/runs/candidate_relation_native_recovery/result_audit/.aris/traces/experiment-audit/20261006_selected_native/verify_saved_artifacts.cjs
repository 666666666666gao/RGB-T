// Read saved JSON/CSV artifacts only. This does not invoke NN, scoring, or bootstrap.
const fs = require('fs');
const assert = require('assert');
const path = require('path');
const base = 'refine-logs/runs/candidate_relation_native_recovery';
const b = base + '/complete/training';
const old = 'refine-logs/runs/candidate_relation/complete_v2/training';
const trace = base + '/result_audit/.aris/traces/experiment-audit/20261006_selected_native';
const filesRead = [];
function text(p) { filesRead.push(p); return fs.readFileSync(p, 'utf8').replace(/^\uFEFF/, ''); }
function read(p) { return JSON.parse(text(p)); }
function csv(p) {
  const lines = text(p).trim().split(/\r?\n/);
  const split = line => line.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/).map(x => x.replace(/^"|"$/g, ''));
  const header = split(lines.shift());
  return lines.map(line => Object.fromEntries(split(line).map((v, i) => [header[i], v])));
}
const near = (a, b) => assert(Math.abs(a - b) < 1e-8, `${a} != ${b}`);
const selection = read(b + '/selected_model.json');
const complete = read(b + '/native/complete_metrics.json');
const history = read(old + '/fit_full/relations_lr5/metrics.json');
const rebuilt = read(b + '/reconstructed_epoch5/metrics.json');
assert.deepStrictEqual(rebuilt, history.slice(0, 6));
assert.deepStrictEqual(history.map(x => x.epoch), Array.from({length: 61}, (_, i) => i));
const originalConfig = read(old + '/fit_full/relations_lr5/config.json');
const rebuiltConfig = read(b + '/reconstructed_epoch5/config.json');
for (const k of Object.keys(originalConfig).filter(k => !['epochs', 'output'].includes(k))) assert.deepStrictEqual(rebuiltConfig[k], originalConfig[k], k);
const trainNames = new Set(rebuiltConfig.train_jobs.map(x => x[0]));
const valNames = new Set(rebuiltConfig.validation_jobs.map(x => x[0]));
assert.equal(trainNames.size, 881); assert.equal(valNames.size, 98);
assert(![...trainNames].some(x => valNames.has(x)));
const developer = read(old + '/full98_report/full_recoverability_report.json');
assert(developer.completed && developer.sequences === 98);
for (const [label, mean] of Object.entries(selection.sequence_mean_iou)) near(mean, developer.variants[label].sequence_mean_iou);
const candidates = Object.keys(selection.sequence_mean_iou).filter(x => /_(best|last)$/.test(x));
assert.equal(candidates.length, 8);
assert.equal(candidates.sort((a, z) => selection.sequence_mean_iou[z] - selection.sequence_mean_iou[a])[0], selection.selected_candidate);
assert.equal(selection.parent_model, complete.same_ABC_model_both_datasets);
assert(complete.completed && !complete.all_five_plus_two);
const summary = {created_at: new Date().toISOString(), scoring_rerun: false, nn_rerun: false, bootstrap_rerun: false,
  original_history_records: history.length, reconstructed_history_exact_original_prefix: true,
  reconstructed_config_matches_except_epochs_output: true, train_sequences: trainNames.size,
  developer_sequences: valNames.size, train_developer_disjoint: true, selected_developer_argmax_matches: true,
  seed: rebuiltConfig.seed, selected_epoch: selection.selected_epoch, datasets: {}, pairs: []};
for (const [dataset, n, frames, attrCount] of [['lasher', 245, 220703, 19], ['rgbt234', 234, 116649, 12]]) {
  const root = b + '/native/' + dataset;
  const plan = read(b + '/native/plan.json').datasets[dataset];
  const core = read(root + '/core_report/full_report.json');
  assert(core.all_actual_ground_truth_verified && core.sequences === n && core.frames === frames);
  const merged = read(root + '/predictions/inference_completion.json');
  const mergedConfig = read(root + '/predictions/inference_config.json');
  const records = [], shards = [];
  for (let gpu = 0; gpu < 4; gpu++) {
    const prefix = root + '/shards/gpu' + gpu + '/predictions/';
    const config = read(prefix + 'inference_config.json'), done = read(prefix + 'inference_completion.json');
    assert(done.completed && done.records.length === done.sequences);
    assert.equal(config.model, selection.parent_model); assert.equal(config.head_epoch, 5);
    assert.equal(config.seed, 42); assert.equal(config.model_training_seed, 42);
    assert.equal(config.max_frames, 0); assert.equal(config.validation_split, null);
    assert.equal(config.policy, 'learned'); assert.equal(config.search_value, 'gross');
    assert.equal(config.write_verification, 'action'); assert.equal(config.state_commit_model, null);
    for (const key of ['zero_init', 'parity_check', 'unsafe_writes', 'disable_search']) assert.equal(config[key], false);
    assert.deepStrictEqual(done.records.map(x => x.sequence), plan.names.slice(config.sequence_offset, config.sequence_offset + config.limit_sequences));
    assert.equal(done.frames, done.records.reduce((a, x) => a + x.frames, 0));
    for (const record of done.records) assert.equal(record.frames, plan.lengths[record.sequence]);
    records.push(...done.records);
    shards.push({gpu, sequences: done.sequences, frames: done.frames, max_frames: config.max_frames, offset: config.sequence_offset});
  }
  assert.deepStrictEqual(records.map(x => x.sequence), plan.names);
  assert.equal(new Set(records.map(x => x.sequence)).size, n);
  assert.equal(records.reduce((a, x) => a + x.frames, 0), frames);
  assert(merged.completed && !merged.smoke_only && merged.full_shard_union_verified);
  assert.deepStrictEqual(merged.records.map(x => [x.sequence, x.frames]), records.map(x => [x.sequence, x.frames]));
  assert.equal(mergedConfig.model, selection.parent_model);
  const rows = csv(root + '/core_report/per_sequence.csv');
  assert.equal(rows.length, n * 5);
  for (const [variant, value] of Object.entries(core.variants)) {
    assert.equal(Object.keys(value.attributes).length, attrCount);
    for (const [metric, score] of Object.entries(value.overall_metrics_percent)) {
      const curve = value.mean_curves[metric];
      assert(curve.thresholds.length === curve.values.length && curve.values.every(Number.isFinite));
      const aggregate = ['SR', 'MSR'].includes(metric) ? curve.values.reduce((a, x) => a + x, 0) / curve.values.length : curve.values[20];
      near(aggregate * 100, score);
      const selectedRows = rows.filter(x => x.variant === variant);
      assert.equal(selectedRows.length, n);
      assert.equal(new Set(selectedRows.map(x => x.sequence)).size, n);
      near(selectedRows.reduce((a, x) => a + Number(x[metric]), 0) / n, score);
    }
  }
  for (const item of complete.five_metrics.filter(x => x.dataset === dataset)) {
    near(item.percent, core.variants.selected_ABC.overall_metrics_percent[item.metric]);
    near(item.baseline_percent, core.variants.baseline.overall_metrics_percent[item.metric]);
    near(item.delta_vs_baseline_pp, item.percent - item.baseline_percent);
    near(item.target_percent, item.baseline_percent + 2);
    assert.equal(item.meets_plus_two, item.percent >= item.target_percent);
  }
  for (const reference of ['baseline', 'c1', 'old4', 'gross_parent']) {
    const folder = root + '/' + reference + '_paired_report';
    const pair = read(folder + '/full_report.json'), bootstrap = read(folder + '/paired_bootstrap.json');
    assert.deepStrictEqual(Object.keys(pair.variants), [reference, 'selected_ABC']);
    assert.deepStrictEqual(pair.variants[reference].overall_metrics_percent, core.variants[reference].overall_metrics_percent);
    assert.deepStrictEqual(pair.variants.selected_ABC.overall_metrics_percent, core.variants.selected_ABC.overall_metrics_percent);
    assert.equal(bootstrap.args.iterations, 5000); assert.equal(bootstrap.args.seed, 42);
    const data = bootstrap.datasets[dataset]; assert.equal(data.sequences, n);
    for (const [metric, value] of Object.entries(data.metrics)) {
      near(value.mean_delta_percentage_points, core.variants.selected_ABC.overall_metrics_percent[metric] - core.variants[reference].overall_metrics_percent[metric]);
      assert.equal(value.improved_sequences + value.worsened_sequences + value.tied_sequences, n);
      summary.pairs.push({dataset, reference, metric, ...value});
    }
    for (const f of ['overall.csv', 'per_sequence.csv', 'per_attribute.csv', 'official_curves.png', 'official_curves.pdf', 'attribute_deltas.png', 'attribute_deltas.pdf']) assert(fs.statSync(folder + '/' + f).size > 0);
  }
  const mechanism = read(root + '/mechanism_report/full_recoverability_report.json');
  assert(mechanism.completed && mechanism.sequences === n);
  assert.equal(mechanism.variants.selected_ABC.frames, frames);
  const selected = mechanism.variants.selected_ABC;
  assert.equal(selected.current_quality_calibration_bins.length, 10);
  assert.equal(selected.current_quality_calibration_bins.reduce((a, x) => a + x.count, 0), selected.counters.known_candidate_evaluations);
  for (const f of ['per_sequence.csv', 'interventions.csv', 'recoverability_metrics_completed.txt']) assert(fs.statSync(root + '/mechanism_report/' + f).size > 0);
  summary.datasets[dataset] = {sequences: n, frames, shards, union_exact: true,
    native_metric_curve_and_saved_sequence_means_match: true, attributes: attrCount,
    selected_metrics: core.variants.selected_ABC.overall_metrics_percent,
    mechanism_report_completed: mechanism.completed, mechanism_valid_frames: selected.valid_frames,
    optional_state_commit_counters_present: Object.keys(selected.counters).some(k => k.startsWith('state_'))};
}
const derivative = read(base + '/actual_native_complete_summary.json');
assert.deepStrictEqual(derivative.native_complete, complete);
assert.equal(derivative.formal_attribute_rows, 81); assert.equal(derivative.paired_comparison_rows, 20);
const fiveRows = csv(base + '/native_five_metrics.csv'); assert.equal(fiveRows.length, 5);
for (const row of fiveRows) { const actual = complete.five_metrics.find(x => x.dataset === row.dataset && x.metric === row.metric); for (const key of ['percent', 'baseline_percent', 'delta_vs_baseline_pp', 'target_percent']) near(Number(row[key]), actual[key]); }
const attributeRows = csv(base + '/native_all_attribute_metrics.csv'); assert.equal(attributeRows.length, 81);
for (const row of attributeRows) {
  const actual = read(b + '/native/' + row.dataset + '/core_report/full_report.json').variants.selected_ABC.attributes[row.attribute];
  near(Number(row.selected_ABC_percent), actual[row.metric]); assert.equal(Number(row.sequences), actual.sequences);
}
const pairRows = csv(base + '/native_four_reference_paired_metrics.csv'); assert.equal(pairRows.length, 20);
for (const row of pairRows) {
  const actual = summary.pairs.find(x => x.dataset === row.dataset && x.reference === row.reference && x.metric === row.metric);
  near(Number(row.mean_delta_percentage_points), actual.mean_delta_percentage_points);
  assert.deepStrictEqual(JSON.parse(row.percentile_95_interval_percentage_points), actual.percentile_95_interval_percentage_points);
}
summary.derivative_rows = {five_metrics: fiveRows.length, attribute_metrics: attributeRows.length, paired_metrics: pairRows.length};
summary.files_read = [...new Set(filesRead)];
summary.status = 'PASS';
fs.writeFileSync(trace + '/saved_artifact_verification.json', JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({status: summary.status, files_read: summary.files_read.length, datasets: summary.datasets, derivative_rows: summary.derivative_rows}));
