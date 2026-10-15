"""One-card complete R2 run:20 epochs, six full98 snapshots, three benchmarks.

Launch H on physical GPU2 and HA on GPU3 independently so the earlier fit can
begin its own complete video selection while the other card finishes training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--m0-root',required=True)
    parser.add_argument('--variant',choices=['H','HA'],required=True)
    parser.add_argument('--gpu',type=int,choices=[2,3],required=True)
    parser.add_argument('--baseline-234',required=True)
    parser.add_argument('--baseline-210',required=True)
    args=parser.parse_args()
    assert args.gpu=={'H':2,'HA':3}[args.variant]
    m0_root=Path(args.m0_root).resolve()
    m0=json.loads((m0_root/'state.json').read_text())
    assert m0['status']=='DENSE_M0_COMPLETE'
    assert json.loads((m0_root/'exit.json').read_text())['actual_controller_exit_code']==0
    batch=m0['chosen_common_batch']
    assert batch in (32,64,128)
    project=Path(__file__).resolve().parent.parent
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text())
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    fit=output/'fit'
    state=dict(status='DENSE_FULL_RUN_RUNNING',variant=args.variant,physical_gpu=args.gpu,
        controller_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        batch=batch,phases={},reports={},full_ABC=False,goal_achieved=False,
        M0_state_sha256=hashlib.sha256((m0_root/'state.json').read_bytes()).hexdigest())

    def save():
        (output/'queue_state.json').write_text(json.dumps(state,indent=2))

    def run(name,commands,device):
        phase=dict(status='RUNNING',commands=commands,physical_gpu=device,
            started_utc=datetime.now(timezone.utc).isoformat())
        state['phases'][name]=phase
        children=[]
        for index,command in enumerate(commands):
            with (output/f'{name}_{index}.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-m',*command],cwd=project,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='4',
                        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
                    stdout=stream,stderr=subprocess.STDOUT)
            children.append(child)
        phase['pids']=[child.pid for child in children]
        save()
        codes=[child.wait() for child in children]
        phase.update(status='COMPLETE' if all(code==0 for code in codes) else 'FAILED',
            actual_exit_codes=codes,completed_utc=datetime.now(timezone.utc).isoformat())
        save()
        assert all(code==0 for code in codes),(name,codes)

    save()
    train=['cadtrack_abc.train_dense','--config',str(config_path),'--output',str(fit),
        '--batch',str(batch),'--epochs','20','--samples','20000','--steps','8','--workers','4',
        '--head-lr','0.00001','--prompt-lr','0.0001']
    if args.variant=='HA':
        train.append('--memory')
    run('full_training',[train],args.gpu)
    terminal=json.loads((fit/'fit_complete.json').read_text())
    assert terminal['status']=='DENSE_TRAINING_COMPLETE' and not terminal['probe']
    assert terminal['epochs']==20 and terminal['raw_clips']==400000
    assert terminal['updates']==((20000+batch-1)//batch)*20
    assert terminal['query_frame_observations']==3200000
    assert terminal['strict_dense_reload_passed'] and all(terminal['changed_parameter_tensors'].values())
    assert terminal['frozen_backbone_before']==terminal['frozen_backbone_after']
    assert [c['epoch'] for c in terminal['candidates']]==[0,1,2,5,10,20]
    for candidate in terminal['candidates']:
        checkpoint=Path(candidate['checkpoint'])
        development=output/'development'/checkpoint.stem
        commands=[['cadtrack_abc.validate_dense','--checkpoint',str(checkpoint),
            '--manifest',str(fit/'manifest.json'),'--output',str(development),'--shard',str(shard)]
            for shard in (0,1)]
        run('full98_'+checkpoint.stem,commands,args.gpu)
    run('select_trained_checkpoint',[['cadtrack_abc.select_dense','--fit',str(fit),
        '--development',str(output/'development')]],'')
    selection=json.loads((fit/'selection.json').read_text())
    assert selection['full98_development_selection_complete'] and selection['strict_dense_reload']
    selected=Path(selection['trained_best_checkpoint'])
    assert selection['trained_selected']['epoch']>0
    digest=hashlib.sha256(selected.read_bytes()).hexdigest()
    assert digest==selection['trained_selected']['checkpoint_sha256']
    state['selected_trained_checkpoint']=str(selected)
    state['selected_trained_checkpoint_sha256']=digest
    state['selected_trained_epoch']=selection['trained_selected']['epoch']
    save()
    roots={'lasher':Path(config['test_root']),'rgbt234':project/'dataset/rgbt234',
        'rgbt210':project/'dataset/rgbt210/RGBT210'}
    baseline={'rgbt234':Path(args.baseline_234).resolve(),'rgbt210':Path(args.baseline_210).resolve()}
    for dataset,count,frames in [('lasher',245,220703),('rgbt234',234,116649),('rgbt210',210,104706)]:
        predictions=output/'formal'/f'{dataset}_predictions'
        commands=[['cadtrack_abc.infer_dense','--checkpoint',str(selected),'--dataset',dataset,
            '--root',str(roots[dataset]),'--output',str(predictions),'--shard',str(shard)] for shard in (0,1)]
        run('infer_'+dataset,commands,args.gpu)
        receipts=[json.loads((predictions/f'complete_shard{shard}.json').read_text()) for shard in (0,1)]
        assert all(r['completed'] and r['checkpoint_sha256']==digest for r in receipts)
        assert sum(r['sequences'] for r in receipts)==count and sum(r['frames'] for r in receipts)==frames
        command=['cadtrack_abc.score_dense','--dataset',dataset,'--root',str(roots[dataset]),
            '--predictions',str(predictions),'--output',str(output/'formal'/f'{dataset}_report')]
        if dataset!='lasher':
            command+=['--baseline-report',str(baseline[dataset])]
        run('score_'+dataset,[command],'')
        report_path=output/'formal'/f'{dataset}_report'/'qualified_full_report.json'
        report=json.loads(report_path.read_text())
        assert report['completed'] and report['checkpoint_sha256']==digest
        assert report['sequences']==count and report['frames']==frames
        state['reports'][dataset]=dict(path=str(report_path),metrics=report['overall_metrics_percent'],
            baseline=report['baseline_metrics_percent'],delta=report['delta_percentage_points'],
            plus_two_reached=report['plus_two_reached'])
        save()
    assert hashlib.sha256(selected.read_bytes()).hexdigest()==digest
    state.update(status='DENSE_FULL_TRAINING_AND_THREE_BENCHMARKS_COMPLETE',
        completed_utc=datetime.now(timezone.utc).isoformat(),full98_selection_complete=True,
        seven_metric_plus_two=all(all(r['plus_two_reached'].values()) for r in state['reports'].values()),
        not_full_ABC=True,goal_achieved=False,
        matched_clip_order='Final H-versus-HA acceptance also requires matching all20 actual history clip-order hashes.',
        checkpoint_cleanup='Deferred until both runs, full results and required audit close; no unique best/pretrain removed.')
    save()
    (output/'pipeline_complete.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
