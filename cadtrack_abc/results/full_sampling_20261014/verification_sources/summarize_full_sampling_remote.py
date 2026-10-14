from pathlib import Path
import json

project=Path('/home/gaob/CADTrack-ABC')
root=project/'outputs/native_future_sampling_20261014'
diagnosis=json.loads((root/'inventory_diagnosis.json').read_text())
assert all(not row['proc_exists'] for row in diagnosis['drivers'])
assert diagnosis['total_sequences']==979 and diagnosis['total_real_frames']==diagnosis['total_receipt_frames']==514081
assert all(not row['mismatches'] for row in diagnosis['rows'])
assert all(not row['missing'] and not row['unexpected'] and row['observed']==row['unique']==row['expected'] for row in diagnosis['coverage'].values())
results={}
for split in ['train','validation']:
    data=[json.loads((root/('queries_'+split)/f'sampling_shard{shard}.json').read_text()) for shard in [0,1]]
    records=[record for receipt in data for record in receipt['records']]
    results[split]=dict(sequences=len(records),frames=sum(r['frames'] for r in records),queries=sum(r['queries'] for r in records),
                       failures=sum(r['failure_frames'] for r in records),
                       event_counts={key:sum(r['event_counts'][key] for r in records) for key in records[0]['event_counts']},
                       search_gates={key:sum(r['search_gates'][key] for r in records) for key in records[0]['search_gates']},
                       cue_min=[min(r['search_cues_min'][k] for r in records) for k in range(3)],
                       cue_max=[max(r['search_cues_max'][k] for r in records) for k in range(3)],
                       worker_seconds=[r['seconds'] for r in data])
result=dict(status='FULL_881_TRAIN_98_DEV_NATIVE_SAMPLING_COMPLETE_REAL_FILES_VERIFIED',
            splits=results,teacher_sha256=diagnosis['rows'][0]['teacher_sha256'],
            actual_total_frames=514081,initial_forecast_total_frames=514083,
            forecast_frame_count_error=2,full_sampling_complete=True,all_original_drivers_exited=True,
            new_optimizer_updates=0,new_formal_metrics=None,baseline_rerun=False)
(root/'sampling_complete_verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
