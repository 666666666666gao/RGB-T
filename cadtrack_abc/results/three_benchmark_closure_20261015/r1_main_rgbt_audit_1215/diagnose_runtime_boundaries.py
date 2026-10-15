import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'main_rgbt_audit_inputs_1150'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
old=j(OUT/'attempt01_numpy126_legacy/deterministic_verification.json')
new=j(OUT/'deterministic_verification.json')
oldrows=list(csv.DictReader((OUT/'attempt01_numpy126_legacy/recomputed_sequence_curves.csv').open(newline='')))
newrows=list(csv.DictReader((OUT/'recomputed_sequence_curves.csv').open(newline='')))
frames={}
for row in csv.DictReader((OUT/'recomputed_per_sequence.csv').open(newline='')):
    frames[(row['run'],row['sequence'])]=int(row['frames'])
changes=[]
summary={run:{'changed_curve_cells':0,'affected_sequences':set(),'frame_threshold_success_changes':0,'by_threshold':{},'metric_difference_new_minus_legacy_pp':{m:v-old['runs'][run]['metrics_percent'][m] for m,v in item['metrics_percent'].items()}} for run,item in new['runs'].items()}
for a,b in zip(oldrows,newrows):
    assert {k:a[k] for k in ['run','sequence','metric','threshold']}=={k:b[k] for k in ['run','sequence','metric','threshold']}
    difference=float(b['value'])-float(a['value'])
    if difference:
        count=round(difference*frames[(b['run'],b['sequence'])])
        assert abs(difference*frames[(b['run'],b['sequence'])]-count)<1e-9
        changes.append(dict(run=b['run'],sequence=b['sequence'],metric=b['metric'],threshold=b['threshold'],legacy_value=float(a['value']),numpy226_value=float(b['value']),frame_success_count_change=count))
        s=summary[b['run']]
        s['changed_curve_cells']+=1;s['affected_sequences'].add(b['sequence']);s['frame_threshold_success_changes']+=count
        key=b['metric']+'@'+b['threshold']
        s['by_threshold'][key]=s['by_threshold'].get(key,0)+count
for item in summary.values():item['affected_sequences']=len(item['affected_sequences'])
with (OUT/'numpy_runtime_curve_differences.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
invalid=[]
gt_counts={}
for dataset in ['rgbt210','rgbt234']:
    counts={'files':0,'noninteger_coordinates':0,'nonfinite_coordinates':0,'nonpositive_xy_rows':0,'nonpositive_wh_rows':0}
    for path in sorted((ROOT/'ground_truth'/dataset).rglob('*.txt')):
        gt=np.loadtxt(path,delimiter=',',dtype=np.float32,ndmin=2)
        counts['files']+=1
        counts['noninteger_coordinates']+=int(np.sum(gt!=np.round(gt)))
        counts['nonfinite_coordinates']+=int(np.sum(~np.isfinite(gt)))
        counts['nonpositive_xy_rows']+=int(np.sum(np.any(gt[:,:2]<=0,axis=1)))
        mask=np.any(gt[:,2:]<=0,axis=1)
        counts['nonpositive_wh_rows']+=int(mask.sum())
        for idx in np.flatnonzero(mask):
            invalid.append({'input_file':path.relative_to(ROOT).as_posix(),'line':int(idx)+1,'classification':'nonpositive GT width and/or height; retained in denominator by native PR/SR; no replacement/filter added'})
    gt_counts[dataset]=counts
scalar=np.float32(.2)
threshold=np.float64(.2)
result={'numpy_version':np.__version__,'runtime_comparison':summary,'actual_numpy226_boundary_probe':{'float32_scalar_plus_python_int_dtype':str((np.float32(1)+1).dtype),'float32_one_fifth_as_float64':float(scalar),'threshold_float64':float(threshold),'strict_success_at_threshold':bool(scalar>threshold)},'cause':'Original source loads float32. NumPy1.26 legacy promotes many float32 scalar/Python scalar operations to float64, while NumPy2 retains float32 (NEP50); the float64 linspace success threshold then distinguishes rounded float32 IoU from exact threshold. Both full original-toolkit executions plus separately derived equations are preserved. The installed source inequalities and denominator were never changed.','source_evidence':['toolkit/rgbt/utils.py:31','toolkit/rgbt/utils.py:171','toolkit/rgbt/utils.py:185','toolkit/rgbt/metrics/metrics.py:57','toolkit/rgbt/metrics/metrics.py:75','toolkit/rgbt/metrics/metrics.py:228','documentation/HANDOFF_snapshot.md:4945'],'ground_truth_diagnostics':gt_counts,'invalid_gt_locations':invalid,'auditor_correction':'Initial verifier classified any invalid GT as an input failure. Corrected only that audit assertion: dataset-supplied invalid GT is an observed native protocol condition; raw predictions and latencies must still be positive/finite. All denominators and original source unchanged.','unexecuted_diagnostic':'verify_metrics_nep50.py was prepared but never executed; exact NumPy2.2.6 became available first.'}
(OUT/'runtime_boundary_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
