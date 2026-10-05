"""Closed TRAIN event analysis; group correlated query offsets before bootstrap."""
import argparse
import json
from pathlib import Path
import random
import statistics


def interval(values):
    rng=random.Random(42)
    boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
    return [boot[124],boot[4874]]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--folder',required=True); a=p.parse_args()
    f=Path(a.folder)
    data=json.loads((f/'matched_velocity_train88_old4/merged88_event_analysis.json').read_text())
    jobs=json.loads((f/'prepared_velocity_train88_jobs.json').read_text())['jobs']
    role={j['event_id']:j['cached_event_role'] for j in jobs}
    comparisons=data['equal_event_comparisons']
    rows={}
    for comparison in comparisons:
        for horizon in ['3','32']:
            for group in ['all','normal','persistent33','recovered32']:
                events=[e for e in data['event_rows'] if group=='all' or role[e['event_id']]==group]
                values=[e['paired'][comparison][horizon]['mean_future_iou_delta']*100 for e in events]
                rows[comparison+' / H'+horizon+' / '+group]={
                    'event_units':len(events),'mean_delta_pp':statistics.mean(values),
                    'paired_event_95_interval_pp':interval(values),
                    'events_up':sum(v>1e-6 for v in values),'events_down':sum(v < -1e-6 for v in values),
                    'events_equal':sum(abs(v)<=1e-6 for v in values)}
    geometry=[(e['sequence'],e['paired']['search_velocity_vs_raw']['32']['mean_future_iou_delta']*100) for e in data['event_rows']]
    ranked=sorted(geometry,key=lambda row:row[1])
    target=f/'actual_velocity_event_strata_CPU_analysis.json'
    assert not target.exists()
    result={'status':'ACTUAL_TRAIN24_EVENT_BOOTSTRAP_AND_STRATA_COMPLETE','queries':88,'events':24,
            'actual_write_pairs':data['actual_write_eligible_queries'],'comparisons':rows,
            'largest32_benefits':ranked[-3:],'largest32_harms':ranked[:3],
            'scope':'TRAIN development causal diagnostic, fixed old4, 5000 paired EVENT resamples seed42; not native accuracy or training-seed intervals',
            'selection_bias':'Inputs deliberately stratified from previously observed TRAIN failure/recovery; no natural population frequency claim',
            'interpretation':'Positive mean is concentrated in two large beneficial events; never claim uniform or stable geometry benefit.'}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))
    print(json.dumps({k:v for k,v in rows.items() if 'H32' in k and 'search_velocity_vs_raw' in k}))


if __name__=='__main__':
    main()
