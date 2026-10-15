"""Independent CPU audit; original toolkit AST plus separately derived equations.

No project/neural imports, no weights deserialization, no raw-array output.
"""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.path.insert(0, 'D:/Program Files/UserCache/gb/uv/archive-v0/4TmToZ9eHxHD81sohMJS9')
import numpy as np
np._set_promotion_state('weak')

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'main_rgbt_audit_inputs_1150'
def readj(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def csvout(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

checks = []
def check(label, ok, detail=None):
    checks.append({'check': label, 'pass': bool(ok), 'details': detail})

def defs(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

# Execute the original, non-neural toolkit utility file unchanged. Extract exact
# scoring classes/functions to avoid importing the unrelated plotting package.
ns = {'np': np, 'os': os, 'BaseRGBTDataet': object, 'TrackerResult': object}
utils_path = ROOT / 'toolkit/rgbt/utils.py'
exec(compile(utils_path.read_text(encoding='utf-8-sig'), str(utils_path), 'exec'), ns)
defs(ROOT / 'toolkit/rgbt/metrics/base.py', ['Metric'], ns)
defs(ROOT / 'toolkit/rgbt/metrics/metrics.py', ['PR','SR','MPR','MSR'], ns)
defs(ROOT / 'toolkit/rgbt/dataset/basedataset.py', ['initial_gt_file','initial_result_file','TrackerResult'], ns)
check('numpy_version', np.__version__ == '1.26.4', np.__version__)

initial = readj(OUT / 'initial_input_hashes.json')
before = readj(OUT / 'input_hashes_before.json')
before_shas = {k:v['sha256'] for k,v in before.items()}
check('independent_before_hashes_match_initial', before_shas == initial,
      {'files': len(before), 'changed': [k for k in before_shas if before_shas[k] != initial.get(k)], 'missing': sorted(set(initial)-set(before))})
manifest = readj(ROOT / 'raw_input_manifest.json')
manifest_rows = {row['file']:row for row in manifest['files']}
check('manifest_file_hashes_and_sizes', all(before[k]['sha256']==v['sha256'] and before[k]['bytes']==v['bytes'] for k,v in manifest_rows.items()), {'rows':len(manifest_rows), 'not_self_listed':sorted(set(before)-set(manifest_rows))})
protocols = readj(ROOT / 'toolkit_protocols.json')

def independent_errors(pred, gt):
    # The documented source runtime uses NumPy2.2.6. NEP50 retains float32
    # scalar arithmetic and promotes comparison against float64 thresholds.
    centers_p = pred[:,:2] + (pred[:,2:]-np.float32(1))/np.float32(2)
    centers_g = gt[:,:2] + (gt[:,2:]-np.float32(1))/np.float32(2)
    center_error = np.sqrt(np.sum((centers_p-centers_g)**2, axis=1))
    right_p = pred[:,:2] + pred[:,2:]
    right_g = gt[:,:2] + gt[:,2:]
    overlap = np.maximum(np.float32(0), np.minimum(right_p,right_g)-np.maximum(pred[:,:2],gt[:,:2]))
    inter = overlap[:,0]*overlap[:,1]
    combined_area = pred[:,2]*pred[:,3] + gt[:,2]*gt[:,3]
    iou = inter/(combined_area-inter)
    return center_error, iou

all_runs = {}
seq_rows, curve_rows, attribute_rows, latency_rows = [], [], [], []
sequence_curve_rows = []
metric_functions = {'rgbt210':['PR','SR'], 'rgbt234':['MPR','MSR']}
for run, dataset in [('main/rgbt210','rgbt210'),('main/rgbt234','rgbt234'),('pure_rgbt234','rgbt234')]:
    prefix = run.replace('/', '_')
    report = readj(ROOT/'results'/run/'full_report.json')
    protocol = protocols[dataset]
    names = protocol['sequences']
    metrics = metric_functions[dataset]
    n_expected = 210 if dataset=='rgbt210' else 234
    gt_root = ROOT/'ground_truth'/dataset
    pred_root = ROOT/'predictions'/run
    check(prefix+'_sequence_inventory', len(names)==len(set(names))==n_expected and set(names)=={p.name for p in gt_root.iterdir() if p.is_dir()} and set(names)=={p.stem for p in pred_root.glob('*.txt')})
    attr_root = ROOT/'toolkit_attributes'/dataset
    attr_names = (attr_root/'SequencesName.txt').read_text(encoding='utf-8-sig').splitlines()
    # RGBT210 uses its attribute name list filtered by evaluator.ALL; RGBT234
    # zips the attribute flags directly with evaluator sequence order.
    reconstructed = {}
    for attr in protocol['attributes']:
        flags = np.loadtxt(attr_root/(attr+'.txt'))
        source_names = attr_names[:len(flags)] if dataset=='rgbt210' else names
        reconstructed[attr] = [name for flag,name in zip(flags,source_names) if flag and name in names]
    check(prefix+'_raw_attribute_membership', reconstructed==protocol['attributes'], {'attribute_name_rows':len(attr_names), 'attributes':len(reconstructed)})
    if dataset=='rgbt234':
        check(prefix+'_raw_sequence_order', attr_names==names)
    receipts = [readj(pred_root/f'complete_shard{i}.json') for i in range(2)]
    records = [row for receipt in receipts for row in receipt['records']]
    record_by_name = {row['sequence']:row for row in records}
    check(prefix+'_completion_receipts', len(records)==len(record_by_name)==n_expected and set(record_by_name)==set(names) and all(r['completed'] and r['dataset']==dataset and r['shard']==i and r['shards']==2 and r['epoch']==report['epoch'] and r['checkpoint']==report['checkpoint'] and r['sequences']==len(r['records']) and r['frames']==sum(x['frames'] for x in r['records']) for i,r in enumerate(receipts)))
    check(prefix+'_shard_membership', all([r['sequence'] for r in receipt['records']]==sorted(names)[i::2] for i,receipt in enumerate(receipts)))
    gt_v_name,gt_i_name = ('init.txt','init.txt') if dataset=='rgbt210' else ('visible.txt','infrared.txt')
    toolkit_gt = ns['initial_gt_file'](str(gt_root),names,gt_v_name,gt_i_name,lambda x:x)
    tracker = ns['TrackerResult']('audited',str(pred_root),names,'','ltwh')
    toolkit_curves = {}
    for metric in metrics:
        fun = ns[metric]()
        value, curves = fun(toolkit_gt,tracker,names)
        check(prefix+'_'+metric+'_thresholds', np.array_equal(fun.thr,np.asarray(protocol['thresholds'][metric])) and np.array_equal(fun.thr,np.asarray(report['mean_curves'][metric]['thresholds'])))
        toolkit_curves[metric] = curves
        check(prefix+'_'+metric+'_toolkit_overall', abs(value*100-report['overall_metrics_percent'][metric])<1e-12, {'recomputed_percent':value*100,'reported_percent':report['overall_metrics_percent'][metric]})
    totals = {'frames':0,'latencies':0,'raw_prediction_nonfinite':0,'raw_prediction_nonpositive_wh':0,'rounded_prediction_nonpositive_wh':0,'visible_gt_nonfinite':0,'visible_gt_nonpositive_wh':0,'infrared_gt_nonfinite':0,'infrared_gt_nonpositive_wh':0,'gt_visible_infrared_unequal_rows':0,'initial_prediction_matches_gt_3_decimal':0,'rounded_initial_prediction_differs_from_gt':0,'first_frame_metric_overwrite_sequences':0,'zero_area_union':0,'latency_nonpositive':0,'latency_nonfinite':0,'exact_center_threshold_equalities':0,'exact_iou_threshold_equalities':0}
    max_error = {'CLE':0.,'IoU':0.,'curves':0.,'reported_per_sequence':0.,'csv':0.,'attributes':0.,'mean_curves':0.,'latency_record_seconds':0.}
    independent_curves={metric:[] for metric in metrics}
    raw_csv = list(csv.DictReader((ROOT/'results'/run/'per_sequence.csv').open(encoding='utf-8-sig',newline='')))
    check(prefix+'_per_sequence_membership', [r['sequence'] for r in report['per_sequence']]==names and [r['sequence'] for r in raw_csv]==names)
    all_latency=[]
    for index,name in enumerate(names):
        gv=np.loadtxt(gt_root/name/gt_v_name,delimiter=',',dtype=np.float32,ndmin=2)
        gi=np.loadtxt(gt_root/name/gt_i_name,delimiter=',',dtype=np.float32,ndmin=2)
        raw=np.loadtxt(pred_root/(name+'.txt'),delimiter='\t',dtype=np.float64,ndmin=2)
        pred=np.asarray(raw,dtype=np.float32).round(0)
        count=len(gv)
        lat=np.load(pred_root/(name+'_latency.npy'),allow_pickle=False)
        check(prefix+'_'+name+'_shapes', gv.shape==gi.shape==raw.shape==(count,4) and lat.shape==(count-1,) and record_by_name[name]['frames']==count)
        totals['frames']+=count; totals['latencies']+=len(lat)
        for key,array in [('raw_prediction',raw),('rounded_prediction',pred),('visible_gt',gv),('infrared_gt',gi)]:
            if key+'_nonfinite' in totals: totals[key+'_nonfinite']+=int((~np.isfinite(array)).sum())
            totals[key+'_nonpositive_wh']+=int(np.sum(np.any(array[:,2:]<=0,axis=1)))
        totals['gt_visible_infrared_unequal_rows']+=int(np.sum(np.any(gv!=gi,axis=1)))
        totals['initial_prediction_matches_gt_3_decimal']+=int(np.array_equal(raw[0],np.round(gv[0].astype(np.float64),3)))
        totals['rounded_initial_prediction_differs_from_gt']+=int(not np.array_equal(pred[0],gv[0]))
        if dataset=='rgbt210':
            pred[0]=gv[0]
            totals['first_frame_metric_overwrite_sequences']+=1
        err_v,over_v=independent_errors(pred,gv)
        err_i,over_i=independent_errors(pred,gi)
        for metric,func,expected_v,expected_i in [('CLE',ns['CLE'],err_v,err_i),('IoU',ns['IoU'],over_v,over_i)]:
            actual_v=np.asarray([func(p,g) for p,g in zip(pred,gv)])
            actual_i=np.asarray([func(p,g) for p,g in zip(pred,gi)])
            error=max(float(np.max(np.abs(actual_v-expected_v))),float(np.max(np.abs(actual_i-expected_i))))
            max_error[metric]=max(max_error[metric],error)
        check(prefix+'_'+name+'_finite_underlying_errors', np.isfinite(err_v).all() and np.isfinite(err_i).all() and np.isfinite(over_v).all() and np.isfinite(over_i).all())
        err=err_v if dataset=='rgbt210' else np.minimum(err_v,err_i)
        over=over_v if dataset=='rgbt210' else np.maximum(over_v,over_i)
        scalar_row={'run':run,'dataset':dataset,'sequence':name,'frames':count}
        for metric in metrics:
            thresholds=np.asarray(protocol['thresholds'][metric])
            success=metric in ('SR','MSR')
            values=(over[:,None]>thresholds if success else err[:,None]<=thresholds).sum(0)/count
            totals['exact_iou_threshold_equalities' if success else 'exact_center_threshold_equalities']+=int(((over if success else err)[:,None]==thresholds).sum())
            independent_curves[metric].append(values)
            max_error['curves']=max(max_error['curves'],float(np.max(np.abs(values-toolkit_curves[metric][index]))))
            scalar=float(values.mean() if success else values[20])*100
            scalar_row[metric]=scalar
            max_error['reported_per_sequence']=max(max_error['reported_per_sequence'],abs(scalar-report['per_sequence'][index][metric]))
            max_error['csv']=max(max_error['csv'],abs(scalar-float(raw_csv[index][metric])))
            for threshold,value in zip(thresholds,values):
                sequence_curve_rows.append({'run':run,'sequence':name,'metric':metric,'threshold':float(threshold),'value':float(value)})
        seq_rows.append(scalar_row)
        totals['latency_nonpositive']+=int((lat<=0).sum())
        totals['latency_nonfinite']+=int((~np.isfinite(lat)).sum())
        seconds=float(lat.sum())
        seconds_error=abs(seconds-record_by_name[name]['seconds'])
        max_error['latency_record_seconds']=max(max_error['latency_record_seconds'],seconds_error)
        all_latency.append(lat)
        latency_rows.append({'run':run,'sequence':name,'samples':len(lat),'seconds':seconds,'fps':len(lat)/seconds,'latency_p50_ms':float(np.percentile(lat,50)*1000),'latency_p95_ms':float(np.percentile(lat,95)*1000),'record_seconds_difference':seconds_error})
    independent_curves={k:np.asarray(v) for k,v in independent_curves.items()}
    metric_values={}
    for metric,curves in independent_curves.items():
        mean=curves.mean(0)
        metric_values[metric]=float(curves.mean() if metric in ('SR','MSR') else mean[20])*100
        max_error['mean_curves']=max(max_error['mean_curves'],float(np.max(np.abs(mean-np.asarray(report['mean_curves'][metric]['values'])))))
        for threshold,value in zip(protocol['thresholds'][metric],mean):
            curve_rows.append({'run':run,'metric':metric,'threshold':threshold,'value':float(value)})
    for attr,members in reconstructed.items():
        ids=[names.index(name) for name in members]
        check(prefix+'_'+attr+'_count', report['attributes'][attr]['sequences']==len(ids))
        for metric,curves in independent_curves.items():
            value=float(curves[ids].mean() if metric in ('SR','MSR') else curves[ids].mean(0)[20])*100
            max_error['attributes']=max(max_error['attributes'],abs(value-report['attributes'][attr][metric]))
            attribute_rows.append({'run':run,'attribute':attr,'sequences':len(ids),'metric':metric,'percent':value,'reported_percent':report['attributes'][attr][metric]})
    lat=np.concatenate(all_latency)
    timing={'fps':len(lat)/float(lat.sum()),'latency_p50_ms':float(np.percentile(lat,50)*1000),'latency_p95_ms':float(np.percentile(lat,95)*1000)}
    check(prefix+'_latency_summary', all(abs(value-report['timing'][key])<1e-12 for key,value in timing.items()), timing)
    check(prefix+'_frame_totals', totals['frames']==report['frames']==sum(r['frames'] for r in receipts) and totals['latencies']==totals['frames']-n_expected, totals)
    check(prefix+'_all_arithmetic', all(max_error[key]<1e-12 for key in ['curves','reported_per_sequence','csv','attributes','mean_curves']) and max_error['CLE']<1e-4 and max_error['IoU']<1e-7 and max_error['latency_record_seconds']<1e-8, max_error)
    # Dataset-provided invalid GT stays in the native denominator. It is an
    # observed protocol condition, not a failure of raw prediction validity.
    check(prefix+'_finite_inputs_and_positive_predictions_latencies', all(totals[k]==0 for k in totals if 'nonfinite' in k or ('nonpositive' in k and 'gt_' not in k)))
    all_runs[run]={'metrics_percent':metric_values,'totals':totals,'maximum_absolute_errors':max_error,'timing':timing,'checkpoint':report['checkpoint'],'epoch':report['epoch'],'policy_family':receipts[0]['policy_family'],'raw_attribute_membership_verified':True,'sequence_order':names,'sequence_metric_values':{k:(v.mean(1)*100 if k in ('SR','MSR') else v[:,20]*100).tolist() for k,v in independent_curves.items()}}
    print(json.dumps({'run':run,'metrics':metric_values,'frames':totals['frames'],'max_errors':max_error,'timing':timing}))

# Fixed-checkpoint, paired-sequence bootstrap is justified: both metric scalars
# are linear means over the same234 per-sequence scalar metrics.
main,pure=all_runs['main/rgbt234'],all_runs['pure_rgbt234']
check('bootstrap_paired_sequence_order', main['sequence_order']==pure['sequence_order'])
seed=20261015
rng=np.random.default_rng(seed)
indices=rng.integers(0,234,size=(5000,234))
bootstrap={'replicates':5000,'seed':seed,'generator':'NumPy1.26.4 default_rng / PCG64','pairing':'Same named234 sequences, retain paired main and pure scalar metrics; uniformly resample234 sequence indices with replacement per replicate; both metrics use the same index draws. Main minus pure, percentage points.','interval':'Two-sided percentile95%, numpy.quantile([.025,.975], method=linear)','scope':'Fixed selected checkpoint/benchmark sequence uncertainty only; not training-seed stability, causal module attribution, latency uncertainty or independent datasets. Sequences treated as exchangeable units; related videos may violate independence.','draw_index_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),'metrics':{}}
bootstrap_rows=[]
replicates={}
for metric in ('MPR','MSR'):
    delta=np.asarray(main['sequence_metric_values'][metric])-np.asarray(pure['sequence_metric_values'][metric])
    reps=delta[indices].mean(1)
    replicates[metric]=reps
    bootstrap['metrics'][metric]={'point_difference_pp':float(delta.mean()),'percentile95_interval_pp':np.quantile(reps,[.025,.975],method='linear').tolist(),'replicate_mean_pp':float(reps.mean()),'replicate_std_pp':float(reps.std(ddof=1)),'fraction_replicates_above_zero':float(np.mean(reps>0)),'sequence_win_tie_loss':[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]}
for i in range(5000): bootstrap_rows.append({'replicate':i+1,'MPR_difference_pp':float(replicates['MPR'][i]),'MSR_difference_pp':float(replicates['MSR'][i])})
dump('bootstrap_rgbt234.json',bootstrap)
csvout('bootstrap_rgbt234_replicates.csv',bootstrap_rows)
# Per-sequence metrics differ by dataset; use a long-format output.
long_rows=[]
for row in seq_rows:
    for metric in metric_functions[row['dataset']]:
        long_rows.append({k:row[k] for k in ['run','dataset','sequence','frames']} | {'metric':metric,'percent':row[metric]})
csvout('recomputed_per_sequence.csv',long_rows)
csvout('recomputed_attributes.csv',attribute_rows)
csvout('recomputed_mean_curves.csv',curve_rows)
csvout('recomputed_sequence_curves.csv',sequence_curve_rows)
csvout('recomputed_latency_statistics.csv',latency_rows)
dump('deterministic_verification.json',{'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','python':sys.version,'numpy':np.__version__,'numpy_promotion_state':np._get_promotion_state(),'historical_numpy_documentation':'documentation/HANDOFF_snapshot.md:4945 records warm runtime NumPy2.2.6; this diagnostic changes only local process promotion semantics, not inputs or installed environments. It does not attest the remote runtime version.','script_sha256':sha(Path(__file__)),'toolkit_execution':'Unmodified toolkit utility module and exact AST-selected scoring classes/load functions under NumPy1.26.4 NEP50 weak promotion; no package initializer, plotting, project or neural imports. Independently vectorized float32 geometry and original loading/rounding verified per frame. Legacy-promotion mismatch is preserved in attempt01_numpy126_legacy.','runs':all_runs,'checks':checks,'check_count':len(checks),'failed_checks':[c for c in checks if not c['pass']],'bootstrap':bootstrap})
print(json.dumps({'checks':len(checks),'failed_checks':[c for c in checks if not c['pass']],'bootstrap':bootstrap}))
sys.exit(0 if all(c['pass'] for c in checks) else 1)
