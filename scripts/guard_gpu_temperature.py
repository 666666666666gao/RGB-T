"""Throttle only the named RGB-T jobs; hardware power caps require an admin."""
import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_GPU = {
    '/data/gb/outputs/recoverability_native_lasher_s42_20261003': 0,
    '/data/gb/outputs/recoverability_paired_c1_prefix_20261003': 1,
    '/data/gb/outputs/recoverability_prefix_c1_s42_20261003': 1,
    '/data/gb/outputs/recoverability_prefix_own_s42_20261003': 2,
    '/data/gb/outputs/recoverability_prefix_own_s43_20261003': 3,
}
MODULES = {'research.collect_recoverability', 'research.train_recoverability',
           'research.evaluate_recoverability'}


def owned_jobs(process_rows):
    jobs = []
    for row in process_rows.splitlines():
        pid, state, command = row.split(None, 2)
        words = command.split()
        if words[:3] != ['/data/gb/envs/gola/bin/python', '-u', '-m']:
            continue
        if words[3] not in MODULES or '--output' not in words:
            continue
        output = words[words.index('--output') + 1].removesuffix('/predictions')
        if output in OUTPUT_GPU:
            jobs.append({'pid': int(pid), 'state': state, 'command': command,
                         'output': output, 'gpu': OUTPUT_GPU[output]})
    return jobs


def action_for(temperature, power, hardware_power_limit, stopped, pause_temp, resume_temp, pause_power, resume_power,
               allow_uncapped_resume=False):
    if not stopped and (temperature >= pause_temp or power >= pause_power):
        return 'STOP'
    if (stopped and (hardware_power_limit <= 250 or allow_uncapped_resume)
            and temperature <= resume_temp and power <= resume_power):
        return 'CONT'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pause-temp', type=float, default=74.)
    parser.add_argument('--resume-temp', type=float, default=70.)
    parser.add_argument('--pause-power', type=float, default=245.)
    parser.add_argument('--resume-power', type=float, default=220.)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--allow-uncapped-resume', action='store_true',
                        help='User-authorized continuation while administrator hardware-cap setup is deferred.')
    parser.add_argument('--log', required=True)
    args = parser.parse_args()
    assert args.resume_temp < args.pause_temp <= 75
    assert args.resume_power < args.pause_power <= 250 and args.interval > 0
    with Path(args.log).open('a', buffering=1) as log:
        while True:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw,power.limit',
                                           '--format=csv,noheader,nounits'], text=True)
            gpus = {int(i): {'temperature': float(t), 'power': float(p), 'hardware_power_limit': float(limit)}
                    for i, t, p, limit in csv.reader(raw.splitlines())}
            processes = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,stat=,args='], text=True)
            jobs = owned_jobs(processes)
            actions = []
            for job in jobs:
                gpu = gpus[job['gpu']]
                action = action_for(gpu['temperature'], gpu['power'], gpu['hardware_power_limit'], 'T' in job['state'],
                                    args.pause_temp, args.resume_temp, args.pause_power, args.resume_power,
                                    args.allow_uncapped_resume)
                if action:
                    # UID and exact command restrict signals to the observed experiment.
                    # pkill returns1 when a normal train/evaluation phase has already exited.
                    result = subprocess.run(['pkill', '-' + action, '-u', str(os.getuid()),
                                             '-f', '^' + re.escape(job['command']) + '$'],
                                            capture_output=True, text=True)
                    assert result.returncode in (0, 1), result.stderr
                    actions.append({'pid': job['pid'], 'gpu': job['gpu'], 'action': action,
                                    'signal_matched': result.returncode == 0})
            record = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'gpus': gpus,
                      'jobs': [{key: job[key] for key in ('pid', 'state', 'output', 'gpu')} for job in jobs],
                      'actions': actions, 'hardware_caps_configured_by_this_script': False,
                      'allow_uncapped_resume': args.allow_uncapped_resume}
            log.write(json.dumps(record, allow_nan=False) + '\n')
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
