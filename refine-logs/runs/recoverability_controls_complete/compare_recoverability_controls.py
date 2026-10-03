import csv,json
from pathlib import Path
import numpy as np
root=Path('/data/gb/outputs/recoverability_control_comparison_20261003')
root.mkdir(exist_ok=True)
sources={'full':('/data/gb/outputs/recoverability_online_full_report_20261003/per_sequence.csv','recoverability_s42'),'no_search':('/data/gb/outputs/recoverability_no_search_s42_20261003/report/per_sequence.csv','no_search'),'unsafe_writes':('/data/gb/outputs/recoverability_unsafe_writes_s42_20261003/report/per_sequence.csv','unsafe_writes')}
values={}
for label,(source,variant) in sources.items():
    with Path(source).open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['variant']==variant]
    assert len(rows)==98
    values[label]={r['sequence']:float(r['mean_valid_iou']) for r in rows}
names=sorted(values['full'])
assert all(set(v)==set(names) for v in values.values())
indices=np.random.default_rng(42).integers(98,size=(5000,98))
report={'completed':True,'scope':'same locked checkpoint and original98 complete internal videos; fixed-weight5000 paired sequence uncertainty, not native metrics','sources':sources,'sequence_mean_iou':{k:float(np.mean(list(v.values()))) for k,v in values.items()},'paired':{}}
for second,first in [('no_search','full'),('full','unsafe_writes')]:
    delta=np.asarray([values[second][name]-values[first][name] for name in names])*100
    report['paired'][second+'_minus_'+first]={'mean_delta_percentage_points':float(delta.mean()),'paired95_ci_percentage_points':np.percentile(delta[indices].mean(1),(2.5,97.5)).tolist(),'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),'tied':int((delta==0).sum())}
(root/'paired_controls.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))