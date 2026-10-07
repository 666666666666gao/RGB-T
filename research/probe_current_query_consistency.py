"""Matched deployment query versus its batch16 own-policy training cache.

Read the first actual collector batch, replay its frozen policy prefixes, then
compare serial query extraction, grouped extraction, and reconstructed A/B
state. No test videos, optimizer or future images are used. This diagnostic
does not assume a numeric difference is a tracking-performance cause.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .collect_rollouts import observe_actions, iou
from .collect_temporal import paths
from .evaluate_online import read_pair
from .recoverability_modules import RecoverabilityModules, DECISION_FIELDS, select_actions
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


class RecordedQueryTracker(RecoverabilityTracker):
    @torch.inference_mode()
    def local_observation(self, image):
        observation = super().local_observation(image)
        self.query_original = observation
        return observation

    def original_inputs(self, observation, distribution, history):
        data = super().original_inputs(observation, distribution, history)
        self.query_data = {key: value.clone() for key, value in data.items()}
        self.query_anchor = self.identity_anchor.clone()
        self.query_memory = self.identity_memory.clone()
        return data


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return dict(exact=np.array_equal(a,b), max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))


def host(value):
    return value.detach().cpu().numpy()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--queries', type=int, default=16, choices=(1,16))
    args=p.parse_args()
    source=Path(args.source);config=json.loads((source/'config.json').read_text())
    done=json.loads((source/'completion.json').read_text())
    assert done['completed'] and config['partition']=='train' and config['batch_clips']==16
    assert config['prefix_search_value']=='gross' and config['prefix_write_verification']=='action'
    assert config['prefix_model_family']=='ABC_candidate_relations' and config['future_policy_mode']=='own'
    split=json.loads(Path(config['split']).read_text())
    jobs=config['jobs'][:args.queries]
    assert len(jobs)==args.queries and all(job['sequence'] in split['train'] for job in jobs)
    assert not set(split['train']) & set(split['validation'])
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    device=torch.device('cuda:0')
    c1=torch.load(config['head'],map_location='cpu',weights_only=False)
    checkpoint=torch.load(config['prefix_model'],map_location='cpu',weights_only=False)
    assert checkpoint['module']=='ABC_candidate_relations' and checkpoint['epoch']==5
    modules=RecoverabilityModules(c1,candidate_relations=True).to(device)
    modules.load_state_dict(checkpoint['model'],strict=True);modules.eval().requires_grad_(False)
    motion_root=Path(config['motion_run']);motion_config=json.loads((motion_root/'config.json').read_text())
    motion_checkpoint=torch.load(motion_root/'last.pth',map_location='cpu',weights_only=False)
    motion=TemporalModules(c1,motion_config['slots'],motion_config['motion_history'],motion_config['modes'],motion_checkpoint['horizon']).to(device)
    motion.load_state_dict(motion_checkpoint['model'],strict=True);motion.eval().requires_grad_(False)
    extractor=InstanceExtractor(config['pretrained'],5,.45,c1['args']['nms_iou']).to(device)
    dataset=MultiModalObjectTrackingDataset_MemoryMapped.load(config['root'],config['cache'])
    by_name={dataset[i].get_name():i for i in range(len(dataset))}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    contexts=[];started=time.perf_counter()
    for job in jobs:
        seq=dataset[by_name[job['sequence']]];query=job['query_frame']
        initial=seq[0].get_bounding_box().copy()
        tracker=RecordedQueryTracker(extractor,modules,motion,read_pair(*paths(seq,0),device),initial,
            config['prefix_threshold'],write_verification='action',search_value='gross')
        for frame in range(1,query):
            tracker.step(read_pair(*paths(seq,frame),device))
        assert tracker.frame==query-1
        contexts.append((job,seq,tracker,read_pair(*paths(seq,query),device)))
        print(json.dumps(dict(prefix_closed=job,elapsed_seconds=time.perf_counter()-started)),flush=True)
    extractor.proposal_policy='peaks'
    grouped=observe_actions([(tracker,tracker.branch,image) for _,_,tracker,image in contexts],extractor,modules.c1)
    with np.load(source/'samples.npz') as archive:
        keys=tuple(dict.fromkeys((*DECISION_FIELDS,'history_descriptors')))
        cached={key:archive[key][:args.queries].copy() for key in keys}
    rows=[]
    for index,((job,seq,tracker,image),batch) in enumerate(zip(contexts,grouped)):
        actual_box=tracker.step(image)
        serial=tracker.query_original
        data={key:torch.from_numpy(value[index:index+1]).to(device) for key,value in cached.items()}
        reconstructed_anchor,reconstructed_memory,_,_=modules.memory.history(data)
        motion_anchor,motion_memory,_=motion.history_memory(data)
        rebuilt_distribution=motion.motion(data['history_boxes'].float(),data['history_frames'],data['history_valid'],
            data['history_quality'].float(),motion_anchor,motion_memory)
        runtime=modules.decide(tracker.query_data,tracker.query_anchor,tracker.query_memory)
        replay=modules.decide(tracker.query_data,reconstructed_anchor,reconstructed_memory)
        runtime_action=select_actions(runtime,tracker.query_data,config['prefix_threshold'],'action','gross')
        replay_action=select_actions(replay,tracker.query_data,config['prefix_threshold'],'action','gross')
        comparisons={key:difference(host(serial[0][key][0]),host(batch[0][key][0])) for key in
            ('features','evidence','raw_score','boxes','instance_features','valid')}
        cache_comparisons={key:difference(host(batch[0][key][0]),cached[key][index,0]) for key in
            ('features','evidence','raw_score','boxes','instance_features','valid')}
        cache_comparisons.update(
            c1_quality=difference(host(batch[1]),cached['c1_quality'][index,0]),
            image_boxes=difference(batch[2].astype(np.float32),cached['image_boxes'][index,0]),
            anchor_features=difference(host(batch[0]['anchor_features'][0]),cached['anchor_features'][index]))
        serial_boxes,batch_boxes=serial[2],batch[2]
        # Labels read only after replay/extraction/decisions. No future decoded.
        gt=seq[job['query_frame']].get_bounding_box()
        assert np.isfinite(gt).all() and (gt[2:]>gt[:2]).all()
        rows.append(dict(sequence=job['sequence'],query_frame=job['query_frame'],
            serial_original_choice=serial[3],grouped_original_choice=batch[3],
            cached_original_choice=int(cached['original_choice'][index]),
            serial_vs_grouped=comparisons,grouped_vs_existing_cache=cache_comparisons,
            serial_vs_grouped_image_boxes=difference(serial_boxes,batch_boxes),
            actual_vs_reconstructed_identity_anchor=difference(host(tracker.query_anchor),host(reconstructed_anchor)),
            actual_vs_reconstructed_identity_memory=difference(host(tracker.query_memory),host(reconstructed_memory)),
            cached_motion_vs_actual=difference(cached['motion_means'][index],host(tracker.query_data['motion_means'][0])),
            cached_motion_log_weights_vs_actual=difference(cached['motion_log_weights'][index],host(tracker.query_data['motion_log_weights'][0])),
            rebuilt_motion_vs_actual=difference(host(rebuilt_distribution['means']),host(tracker.query_data['motion_means'])),
            rebuilt_motion_log_weights_vs_actual=difference(host(rebuilt_distribution['log_weights']),host(tracker.query_data['motion_log_weights'])),
            memory_reconstruction_changes_local_decision=(
                not torch.equal(runtime_action['flat_action'],replay_action['flat_action'])
                or not torch.equal(runtime_action['search_triggered'],replay_action['search_triggered'])
                or (bool(runtime_action['search_triggered'][0]) and
                    not torch.equal(runtime_action['searched_region'],replay_action['searched_region']))),
            memory_reconstruction_local_score_difference=difference(host(runtime['scores'][:,0]),host(replay['scores'][:,0])),
            serial_keep_current_iou=iou(serial_boxes[serial[3]],gt),grouped_keep_current_iou=iou(batch_boxes[batch[3]],gt),
            actual_ABC_current_iou=iou(actual_box,gt)))
    assert all(p.grad is None for group in (extractor,modules,motion) for p in group.parameters())
    report=dict(completed=True,source=str(source),queries=args.queries,partition='train',
        model=config['prefix_model'],scope='Matched numeric/input/state check, not native accuracy or proof of a unique cause',
        new_optimizer_updates=0,no_future_images_decoded=True,GT_only_first_init_and_post_decision_labels=True,
        serial_grouped_original_choice_disagreements=sum(row['serial_original_choice']!=row['grouped_original_choice'] for row in rows),
        grouped_existing_cache_choice_disagreements=sum(row['grouped_original_choice']!=row['cached_original_choice'] for row in rows),
        memory_reconstruction_decision_changes=sum(row['memory_reconstruction_changes_local_decision'] for row in rows),
        max_identity_memory_reconstruction_error=max(row['actual_vs_reconstructed_identity_memory']['max_abs'] for row in rows),
        elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'completion.json').write_text(json.dumps({key:value for key,value in report.items() if key!='rows'},indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!='rows'}),flush=True)


if __name__=='__main__':
    main()
