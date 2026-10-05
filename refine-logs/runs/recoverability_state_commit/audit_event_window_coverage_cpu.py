"""Read closed TRAIN prefixes; inventory event-offset supervision, no models."""
import json
from pathlib import Path

import numpy as np

BASE = Path('/data/gb/outputs/recoverability_best_native_policy_merged_20261005/own/train')
EVENT = Path('/data/gb/GOLA/refine-logs/runs/recoverability_best_native_policy/actual_train_event_query_eligibility_proposal.json')
OUT = Path('/data/gb/outputs/recoverability_event_window_write_coverage_20261005.json')
PARENT = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'


def main():
    assert not OUT.exists()
    config = json.loads((BASE/'config.json').read_text())
    original = json.loads(EVENT.read_text())
    split = json.loads(Path(config['split']).read_text())
    assert config['partition']=='train' and config['prefix_model']==PARENT
    assert not set(split['train']) & set(split['validation'])
    jobs = config['jobs']
    assert len(jobs)==1762 and {j['sequence'] for j in jobs}==set(split['train'])
    queried = {(j['sequence'],j['query_frame']) for j in jobs+original['new_event_jobs']}
    states = {}
    with np.load(BASE/'samples.npz', allow_pickle=False) as data:
        valid = data['history_valid']; frames = data['history_frames']
        quality = data['history_iou']; evidence = data['history_evidence']; writes = data['history_write']
        assert valid.shape==frames.shape==quality.shape==writes.shape==(1762,1024)
        # Prefer each sequence's longest recorded causal prefix; no query boxes inserted.
        for index in sorted(range(len(jobs)), key=lambda i:jobs[i]['query_frame']):
            name = jobs[index]['sequence']
            timeline = states.setdefault(name,{})
            for slot in np.flatnonzero(valid[index]):
                frame = int(frames[index,slot])
                timeline[frame] = {'iou':float(quality[index,slot]),'raw_score':float(evidence[index,slot,0]),
                                   'template_written':bool(writes[index,slot])}
    events = []
    for name, timeline in sorted(states.items()):
        for frame, current in sorted(timeline.items()):
            # Match the closed inventory: the previous state must be a prediction,
            # not the ground-truth initialization at frame0; unknown GT stays unknown.
            if frame>1 and frame-1 in timeline and timeline[frame-1]['iou']>=.5 and 0<=current['iou']<.2:
                seen = [timeline.get(frame+offset) for offset in range(1,33)]
                recovered = any(all(r is not None and r['iou']>=.5 for r in seen[start:start+3])
                                for start in range(30))
                complete = all(r is not None and r['iou']>=0 for r in seen)
                role = 'recovered_within32' if recovered else 'not_recovered_within32' if complete else 'future_window_incomplete'
                events.append({'sequence':name,'transition_frame':frame,'event_id':f'{name}:transition:{frame}',
                               'role':role})
    assert len(events)==original['observed_correct_to_failed_transitions']==721
    records = []
    offsets = [-2,-1,0,1,3]
    for event in events:
        for offset in offsets:
            query = event['transition_frame']+offset
            if not 0<query<=1024:
                continue
            row = states[event['sequence']].get(query)
            records.append(event|{'query_frame':query,'offset':offset,
                'already_action_queried':(event['sequence'],query) in queried,
                'actual_prefix_state_observed':row is not None,
                'selected_raw_score':None if row is None else row['raw_score'],
                'selected_IoU':None if row is None else row['iou'],
                'actual_template_written':None if row is None else row['template_written'],
                'selected_raw_write_eligible':None if row is None else row['raw_score']>.84})
    by_offset = {}
    for offset in offsets:
        rows = [r for r in records if r['offset']==offset]
        new = [r for r in rows if not r['already_action_queried']]
        by_offset[str(offset)]={'event_links':len(rows),'already_queried_links':len(rows)-len(new),
            'not_queried_links':len(new),'new_observed_links':sum(r['actual_prefix_state_observed'] for r in new),
            'new_selected_raw_write_eligible_links':sum(r['selected_raw_write_eligible'] is True for r in new),
            'new_actual_template_written_links':sum(r['actual_template_written'] is True for r in new)}
    unqueried = [r for r in records if not r['already_action_queried']]
    unique = {(r['sequence'],r['query_frame']) for r in unqueried}
    write_episodes = []
    for name, timeline in sorted(states.items()):
        episode = []
        def finish_episode():
            written = [t for t in episode if timeline[t]['template_written']]
            if written:
                query = written[0]
                assert timeline[query]['raw_score']>.84
                write_episodes.append({'sequence':name,'episode_start':episode[0],'observed_episode_end':episode[-1],
                    'event_id':f'{name}:wrong_localization_write:{episode[0]}','query_frame':query,
                    'wrong_localization_writes_in_episode':len(written),'selected_IoU':timeline[query]['iou'],
                    'selected_raw_score':timeline[query]['raw_score'],
                    'already_action_queried':(name,query) in queried})
        for frame, row in sorted(timeline.items()):
            if frame>0 and 0<=row['iou']<.2:
                if episode and frame!=episode[-1]+1:
                    finish_episode();episode=[]
                episode.append(frame)
            else:
                finish_episode();episode=[]
        finish_episode()
    result={'status':'ACTUAL_CLOSED_TRAIN_EVENT_WINDOW_INVENTORY_COMPLETE','events':len(events),
        'prefix_frames':sum(t>0 and r['iou']>=0 for s in states.values() for t,r in s.items()),
        'stored_prefix_frames_including_init_and_unknown':sum(len(s) for s in states.values()),
        'unknown_noninitial_GT_labels':sum(t>0 and r['iou']<0 for s in states.values() for t,r in s.items()),
        'initialization_to_first_failed_observation_count':sum(1 in s and 0<=s[1]['iou']<.2 for s in states.values()),
        'TRAIN_sequences':len(states),'offsets':offsets,
        'by_offset':by_offset,'unique_unqueried_sequence_frame_pairs':len(unique),
        'roles':{role:sum(e['role']==role for e in events) for role in sorted({e['role'] for e in events})},
        'write_risk_inventory':{'known_wrong_localization_written_frames':sum(t>0 and 0<=r['iou']<.2 and r['template_written'] for s in states.values() for t,r in s.items()),
            'failure_episodes_with_actual_wrong_localization_writes':len(write_episodes),
            'episodes_first_wrong_write_already_action_queried':sum(e['already_action_queried'] for e in write_episodes),
            'episodes_first_wrong_write_not_action_queried':sum(not e['already_action_queried'] for e in write_episodes),
            'representative_rule':'First actual write within each contiguous known-IoU<.2 failure episode; unknown GT and missing history break the episode. One query per episode, no repeated adjacent-write count as independent samples.',
            'episodes':write_episodes,
            'limit':'Wrong localization is an IoU proxy, not proof of wrong semantic identity or causal damage from writing. New actual query/future GT eligibility and matched write-pause NN outcomes are still required.'},
        'records':records,'source':str(BASE),'teacher':PARENT,
        'scope':'Existing TRAIN histories only. Query annotations and future frames not read. IoU here is a stored offline TRAIN label, never NN input.',
        'limits':'Event links are correlated and may share the same query. Eligibility describes the previously selected candidate, not every current action. No new visual candidate recall, pause benefit, unobserved outcome or NN result is inferred. Collection jobs still require real GT validity before launch.',
        'future_label_boundary':'Not recovered within a complete32-frame observed window is not permanently unrecoverable; incomplete future windows remain unknown.',
        'neural_forwards':0,'optimizer_updates':0,'GPU_queries':0}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
