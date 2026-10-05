"""Fixed-head full core evaluation after actual four-fit/full98 acceptance.

Stages do not retrain or select on native data. Old4, C1 and full GOLA remain
completed references. Frame-balanced contiguous shards preserve sequence order.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path('/data/gb/GOLA')
sys.path.insert(0,str(ROOT))
FOLDER=ROOT/'refine-logs/runs/recoverability_state_commit'
FIT=Path('/data/gb/outputs/recoverability_geometry_commit_fit_20261005')
BASE=Path('/data/gb/outputs/recoverability_geometry_commit_native_20261005')
PARENT='/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
DATA={'lasher':'/data/wangwj/dataset/LasHeR/testingset','rgbt234':'/data/zhouy/DATASET/RGB-T234'}
EXPECTED={'lasher':(245,220703),'rgbt234':(234,116649)}
MODEL='geometry_commit_complete'
read=lambda path:json.loads(Path(path).read_text())


def reference(dataset,label):
    if label=='old4':
        return Path('/data/gb/outputs/recoverability_write_pair_native_'+dataset+'_own_b384_full_20261004')
    assert label in ['baseline','c1']
    return Path('/data/gb/outputs/'+('online_core' if label=='baseline' else 'online_core_v2'))/(dataset+'_'+label)


def review_gate():
    review=read(FOLDER/'geometry_commit_native_pipeline_source_review.json')
    assert review['status']=='PASS' and review['runtime_attested'] is False


def prepare():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    target=FOLDER/'prepared_geometry_native_frame_shards.json'; assert not target.exists()
    datasets={}
    for dataset,(count,frames) in EXPECTED.items():
        done=read(reference(dataset,'baseline')/'predictions/inference_completion.json')
        assert done['completed'] and not done['smoke_only'] and done['sequences']==count and done['frames']==frames
        lengths={r['sequence']:r['frames'] for r in done['records']}
        names=sorted(p.name for p in Path(DATA[dataset]).iterdir() if p.is_dir())
        assert len(names)==count and set(names)==set(lengths) and sum(lengths.values())==frames
        cumulative=np.concatenate(([0],np.cumsum([lengths[n] for n in names])))
        cuts=[0]
        for k in range(1,4):
            cuts.append(min(range(cuts[-1]+1,count-(4-k)+1),key=lambda i:abs(cumulative[i]-frames*k/4)))
        cuts.append(count)
        shards=[{'gpu':i,'offset':cuts[i],'count':cuts[i+1]-cuts[i],
                 'frames':int(cumulative[cuts[i+1]]-cumulative[cuts[i]]),
                 'sequences':names[cuts[i]:cuts[i+1]]} for i in range(4)]
        for label in ['baseline','c1','old4']:
            r=reference(dataset,label)
            accepted=read(r/'predictions/inference_completion.json')
            assert accepted['completed'] and not accepted['smoke_only']
            assert (accepted['sequences'],accepted['frames'])==(count,frames)
            if label=='old4':
                cfg=read(r/'predictions/inference_config.json')
                assert cfg['model']==PARENT and cfg['head_epoch']==4 and not cfg['disable_search'] and not cfg['unsafe_writes']
        datasets[dataset]={'root':DATA[dataset],'sequences':count,'frames':frames,'shards':shards}
    result={'status':'PREPARED_CPU_ONLY_NATIVE_PARTITION','datasets':datasets,'native_labels_read':False,
            'model_selected':False,'source':'Completed baseline frame counts and current directory-name union only; no native GT used for selection'}
    target.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


def select():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    review_gate(); records=[]
    for i in range(4):
        fit=FIT/('gpu'+str(i)); done=read(fit/'completion.json')
        score=read(fit/'continuous/full98_selection_score.json')
        cfg=read(fit/'continuous/predictions/inference_config.json')
        assert done['complete'] and done['optimizer_updates']==60 and done['frozen_parent_exact']
        assert score['completed'] and score['sequences']==98 and score['frames']==49418 and score['actual_TRAIN_GT_scored']
        assert cfg['model']==PARENT and cfg['commit_model']==str(fit/'best.pth')
        assert cfg['commit_head_epoch']==done['selected_epoch'] and cfg['max_frames']==cfg['limit_sequences']==cfg['sequence_offset']==0
        assert cfg['validation_split']=='/data/gb/outputs/c1_initial_seed42/split.json'
        records.append({'gpu':i,'parent':PARENT,'commit_model':cfg['commit_model'],
                        'commit_epoch':done['selected_epoch'],'sequence_mean_iou':score['sequence_mean_iou'],
                        'full98_score':str(fit/'continuous/full98_selection_score.json')})
    chosen=max(records,key=lambda r:r['sequence_mean_iou'])
    target=FOLDER/'selected_geometry_native_checkpoint.json'; assert not target.exists()
    result={'status':'SELECTED_ON_COMPLETE98_DEVELOPER_VIDEOS','parent':PARENT,'commit_model':chosen['commit_model'],
            'commit_epoch':chosen['commit_epoch'],'candidates':records,'all_four60fits_and98videos_complete':True,
            'selection_rule':'Maximum actual-GT sequence mean IoU, declared GPU order on ties; no native-test selection',
            'limitations':'Reused98 developer videos and diagnosed24 TRAIN events, not untouched confirmation; prototype commit head only',
            'selected_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    target.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


def launch(gpu):
    review_gate(); selection=read(FOLDER/'selected_geometry_native_checkpoint.json')
    assert selection['status']=='SELECTED_ON_COMPLETE98_DEVELOPER_VIDEOS' and selection['all_four60fits_and98videos_complete']
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    device=next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0])==gpu)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert float(device[2])>20000 and not any(line.split(', ')[0]==device[1] and 'python' in line.lower() for line in apps.splitlines())
    BASE.mkdir(exist_ok=True); receipt=BASE/('gpu'+str(gpu)+'_launch.json')
    assert not receipt.exists()
    command=['bash',str(FOLDER/'run_geometry_native.sh'),str(gpu)]
    with (BASE/('gpu'+str(gpu)+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result={'pid':child.pid,'gpu':gpu,'command':command,'parent':selection['parent'],'commit_model':selection['commit_model'],
            'commit_epoch':selection['commit_epoch'],'launched_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    receipt.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


def merge(dataset):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    review_gate(); selection=read(FOLDER/'selected_geometry_native_checkpoint.json')
    prepared=read(FOLDER/'prepared_geometry_native_frame_shards.json')['datasets'][dataset]
    count,frames=EXPECTED[dataset]; root=BASE/dataset; out=root/'predictions'; assert not out.exists()
    configs=[]; completions=[]; rows=[]; files=[]; times=[]
    for shard in prepared['shards']:
        source=root/'shards'/('gpu'+str(shard['gpu'])); p=source/'predictions'
        assert (source/'inference_completed.txt').is_file()
        cfg=read(p/'inference_config.json'); done=read(p/'inference_completion.json')
        assert done['completed'] and done['smoke_only'] and cfg['smoke_only']
        assert cfg['sequence_offset']==shard['offset'] and cfg['limit_sequences']==done['sequences']==shard['count']
        assert done['frames']==shard['frames'] and cfg['root']==DATA[dataset] and cfg['dataset']==dataset
        assert cfg['model']==selection['parent']==PARENT and cfg['commit_model']==selection['commit_model']
        assert cfg['head_epoch']==4 and cfg['commit_head_epoch']==selection['commit_epoch']
        assert cfg['max_frames']==0 and cfg['validation_split'] is None and cfg['write_verification']=='action'
        assert not any(cfg[k] for k in ['zero_init','parity_check','disable_search','unsafe_writes']) and cfg['policy']=='learned'
        assert [r['sequence'] for r in done['records']]==shard['sequences']
        assert sorted(path.stem for path in p.glob('*.txt'))==sorted(shard['sequences'])
        for r in done['records']:
            name=r['sequence']; prediction=np.loadtxt(p/(name+'.txt'),ndmin=2)
            latency=np.load(p/(name+'_latency.npy'),allow_pickle=False)
            visible=sum(v.is_file() for v in (Path(DATA[dataset])/name/'visible').iterdir())
            infrared=sum(v.is_file() for v in (Path(DATA[dataset])/name/'infrared').iterdir())
            assert prediction.shape==(r['frames'],4) and visible==infrared==r['frames']
            assert latency.shape==(r['frames']-1,) and np.isfinite(prediction).all() and np.isfinite(latency).all() and (latency>0).all()
            files.extend(p/(name+suffix) for suffix in ['.txt','_latency.npy','_recoverability_decisions.npz'])
            times.append(latency)
        configs.append(cfg); completions.append(done); rows.extend(done['records'])
    common=['model','commit_model','commit_head_epoch','head_epoch','pretrained_load','seed','amp_dtype','c1_head','motion_run','threshold','new_commit_parameters']
    assert all(all(c[k]==configs[0][k] for k in common) for c in configs)
    assert len(rows)==count and sum(r['frames'] for r in rows)==frames and len({r['sequence'] for r in rows})==count
    assert len({p.name for p in files})==len(files)==count*3
    out.mkdir()
    for p in files:assert p.is_file(); os.link(p,out/p.name)
    cfg=dict(configs[0],output=str(out),sequence_offset=0,limit_sequences=0,smoke_only=False,full_shard_union_verified=True)
    (out/'inference_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    latency=np.concatenate(times)
    for i,r in enumerate(rows,1):r['sequence_index']=i; r['sequences']=count
    done={'completed':True,'smoke_only':False,'sequences':count,'frames':frames,'records':rows,'gt_scoring_completed':False,
          'full_shard_union_verified':True,'fps_including_decode_crop_update':len(latency)/latency.sum(),
          'latency_p50_ms':float(np.percentile(latency,50)*1000),'latency_p95_ms':float(np.percentile(latency,95)*1000),
          'peak_cuda_allocated_mib':max(c['peak_cuda_allocated_mib'] for c in completions),
          'peak_cuda_reserved_mib':max(c['peak_cuda_reserved_mib'] for c in completions),
          'wall_seconds_including_initialization_and_diagnostic_serialization':max(c['wall_seconds_including_initialization_and_diagnostic_serialization'] for c in completions),
          'timing_note':'Instrumented concurrent four-GPU latency; no algorithm speedup claim.'}
    (out/'inference_completion.json').write_text(json.dumps(done,indent=2)+'\n')
    (root/'full_inference_merge_acceptance.json').write_text(json.dumps({'status':'PASS','dataset':dataset,'sequences':count,'frames':frames,'parent':selection['parent'],'commit_model':selection['commit_model'],'commit_epoch':selection['commit_epoch'],'native_metrics_computed':False},indent=2)+'\n')


def report(dataset):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    review_gate(); root=BASE/dataset; assert read(root/'full_inference_merge_acceptance.json')['status']=='PASS'
    variants=['baseline','c1','old4',MODEL]; runs=[str(reference(dataset,l)) for l in variants[:3]]+[str(root)]
    python='/data/gb/envs/gola/bin/python'
    def call(module,*args):subprocess.run([python,'-u','-m',module,*args],cwd=ROOT,check=True)
    call('research.collect_core_metrics','--dataset',dataset,'--data-root',DATA[dataset],'--variants',*variants,'--runs',*runs,'--output',str(root/'core_report'))
    for label,run in zip(variants[:3],runs[:3]):
        out=root/(label+'_paired_report')
        call('research.collect_core_metrics','--dataset',dataset,'--data-root',DATA[dataset],'--variants',label,MODEL,'--runs',run,str(root),'--output',str(out))
        call('research.paired_sequence_bootstrap','--reports',str(out/'full_report.json'),'--output',str(out/'paired_bootstrap.json'))
        call('research.plot_core_metrics','--report',str(out/'full_report.json'),'--output',str(out))
    call('research.collect_recoverability_metrics','--dataset',dataset,'--root',DATA[dataset],
         '--labels',MODEL,'--runs',str(root/'predictions'),'--reference-labels',*variants[:3],
         '--references',*[str(Path(run)/'predictions') for run in runs[:3]],'--output',str(root/'mechanism_report'))
    (root/'report_completed.txt').write_text('Native metrics, all attributes, curves, three paired reports and mechanisms complete\n')


def complete():
    from research.merge_abc_complete import native_report,uncertainty
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    review_gate(); selection=read(FOLDER/'selected_geometry_native_checkpoint.json')
    datasets={}; acceptance=[]; sequence_rows=[]; attribute_rows=[]
    variants=['baseline','c1','old4',MODEL]
    for dataset,(count,frames) in EXPECTED.items():
        root=BASE/dataset; assert (root/'report_completed.txt').is_file()
        audit=read(root/'independent_native_cpu_acceptance.json'); assert audit['status']=='PASS'
        native,seq=native_report(root/'core_report/full_report.json',dataset,variants)
        cfg=native['variants'][MODEL]['original_inference_config']
        assert cfg['model']==PARENT and cfg['commit_model']==selection['commit_model'] and cfg['commit_head_epoch']==selection['commit_epoch']
        paired={}
        for label in variants[:3]:
            folder=root/(label+'_paired_report'); pair,_=native_report(folder/'full_report.json',dataset,[label,MODEL])
            paired[label]=uncertainty(folder/'paired_bootstrap.json',dataset,[label,MODEL],pair)
            assert paired[label]['datasets'][dataset]['metrics']==audit['paired_5000_seed42_bootstrap_exact'][label]
            for name in ['official_curves.png','official_curves.pdf','attribute_deltas.png','attribute_deltas.pdf']:assert (folder/name).stat().st_size>0
        mechanism=read(root/'mechanism_report/full_recoverability_report.json')
        assert mechanism['completed'] and mechanism['sequences']==count and (root/'mechanism_report/recoverability_metrics_completed.txt').is_file()
        datasets[dataset]={'native':native,'paired':paired,'mechanism':mechanism,'actual_GT_acceptance':audit}
        sequence_rows.extend({'dataset':dataset,**row} for row in seq)
        for label in variants:
            for attribute,values in native['variants'][label]['attributes'].items():attribute_rows.append({'dataset':dataset,'variant':label,'attribute':attribute,**values})
        for metric,baseline in native['variants']['baseline']['overall_metrics_percent'].items():
            model=native['variants'][MODEL]['overall_metrics_percent'][metric]
            acceptance.append({'dataset':dataset,'metric':metric,'baseline_percent':baseline,
                'c1_percent':native['variants']['c1']['overall_metrics_percent'][metric],
                'old4_percent':native['variants']['old4']['overall_metrics_percent'][metric],
                'method_percent':model,'delta_vs_baseline_pp':model-baseline,'target_percent':baseline+2,'meets_plus_two':model-baseline>=2})
    assert len(acceptance)==5 and len(sequence_rows)==479*4 and len(attribute_rows)==31*4
    out=BASE/'complete_report'; assert not out.exists(); out.mkdir()
    result={'completed':True,'official_tracking_accuracy':True,'selected_checkpoint':selection,'datasets':datasets,
            'acceptance':{'metrics':acceptance,'all_five_overall_meet_plus_two':all(r['meets_plus_two'] for r in acceptance)},
            'training_scope':'Four60update commit-head prototypes,18TRAINevents66queries; frozen old4 A/B/C/C1. Not full881 or backbone retraining.',
            'limitations':['24event diagnostics and98developer videos reused; prior native tests informed development.','Fixed-checkpoint sequence bootstrap, no cross-training-seed stability claim.','Concurrent instrumented speed is not isolated efficiency.','Eight actual inference combinations, budget curves, second baseline not completed here.'],
            'created_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    (out/'complete_core_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rows in [('acceptance.csv',acceptance),('per_sequence.csv',sequence_rows),('attributes.csv',attribute_rows)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); writer.writeheader(); writer.writerows(rows)
    (out/'COMPLETE').write_text('Complete479/337352 native GT, all5/31attributes/curves/paired intervals\n')
    print(json.dumps(result['acceptance']))


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','select','launch','merge','report','complete'])
    p.add_argument('--gpu',type=int,choices=range(4)); p.add_argument('--dataset',choices=EXPECTED)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='select':select()
    elif a.stage=='launch':assert a.gpu is not None; launch(a.gpu)
    elif a.stage=='merge':assert a.dataset is not None; merge(a.dataset)
    elif a.stage=='report':assert a.dataset is not None; report(a.dataset)
    else:complete()


if __name__=='__main__':main()
