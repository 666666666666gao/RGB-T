"""Honor the user's stop boundary without interrupting the current fit workers."""
import ctypes
import datetime
import json
import os
import select
import signal
import time
from pathlib import Path

ROOT=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix')
OWNER=2809287
REQUEST=ROOT/'user_stop_request.json'

def read(path):
    return json.loads(path.read_text())

def process(pid):
    path=Path('/proc',str(pid),'stat')
    if not path.exists():
        return dict(pid=pid,alive=False,state=None,exit_code=None)
    fields=path.read_text().split(') ',1)[1].split()
    return dict(pid=pid,alive=fields[0]!='Z',state=fields[0],exit_code=int(fields[49]) if fields[0]=='Z' else None)

def write(name,**fields):
    value=dict(at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**fields)
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
assert fd>=0,ctypes.get_errno()
assert libc.inotify_add_watch(fd,os.fsencode(ROOT),0x8)>=0,ctypes.get_errno()
paused=False
fit=None
write('user_stop_guard_started.json',guard_pid=os.getpid(),owner_pid=OWNER,request=read(REQUEST),CPU_only=True,new_NN_or_optimizer=False)
while True:
    request=read(REQUEST)
    if request['boundary']=='complete_current_pipeline':
        if paused:
            assert process(OWNER)['state']=='T'
            os.kill(OWNER,signal.SIGCONT)
        write('user_stop_guard_released_for_current_evaluation.json',owner_pid=OWNER,coordinator_resumed=paused,reason='User chose completion of the current training plus previously requested full evaluation; no next research iteration')
        break
    assert request['boundary']=='complete_current_training'
    current=read(ROOT/'progress.json')
    if not paused and current['stage']=='FIT_FULL':
        command=Path('/proc',str(OWNER),'cmdline').read_bytes().split(b'\0')
        assert b'scripts.run_current_event_rollout_training' in command and os.fsencode(ROOT) in command
        fit=current['children']
        assert len(fit)==4
        for item in fit:
            args=item['command']
            assert 'research.train_event_recoverability' in args
            assert args[args.index('--epochs')+1]=='24'
            assert Path(args[args.index('--output')+1]).parent==ROOT/'fit_full'
        os.kill(OWNER,signal.SIGSTOP)
        while process(OWNER)['state']!='T':
            time.sleep(.1)
        paused=True
        write('user_stop_coordinator_held_during_current_fit.json',owner_pid=OWNER,owner_state=process(OWNER),fit_children=fit,only_coordinator_stopped=True,current_four_fit_workers_continue=True,post_fit_evaluation_not_started=True)
    if paused:
        states=[process(item['pid']) for item in fit]
        if all(not row['alive'] for row in states):
            receipts=[]
            for item,state in zip(fit,states):
                assert state['exit_code'] in (0,None),state
                args=item['command']
                output=Path(args[args.index('--output')+1])
                done=read(output/'completion.json')
                assert done['completed'] and done['epochs']==24 and done['optimizer_steps']==12360
                assert done['modules_changed']=={'A':True,'B':True,'C':True}
                assert all(value>0 for value in done['max_module_gradient_norms'].values())
                assert done['frozen_C1_weights_exact'] and not done['GOLA_and_motion_new_gradients']
                assert all(value['strict_reload'] for value in done['checkpoints'].values())
                receipts.append(dict(gpu=item['gpu'],output=str(output),process=state,completion=done))
            assert process(OWNER)['state']=='T'
            write('user_stop_after_current_training_complete.json',completed=True,owner_pid=OWNER,owner_intentionally_held=True,all_four_fit_workers_closed=True,epochs_per_arm=24,updates_per_arm=12360,total_optimizer_updates=49440,receipts=receipts,new_full98_or_native_evaluation_started=False,weights_preserved=True,scope='Current new-module training finished; paused queue retains post-fit evaluation for explicit future continuation, not a claim of native +2 accuracy')
            break
    ready,_,_=select.select([fd],[],[],240)
    if ready:
        os.read(fd,65536)
os.close(fd)
