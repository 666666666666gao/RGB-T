from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

private = Path(__file__).parent
repo = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
desktop = Path('C:/Users/gb/Desktop/document/RGBT_GOLA_20261002')
names = json.loads((private / 'requested_resource_1058_owned_files.json').read_bytes())['files']
assert len(names) == len(set(names)) == 4
allowed = ['resource_concurrency_20261015']
assert all(name in ['cadtrack_abc/plan.json', 'docs/HANDOFF_20261002.md', '.gitattributes']
    or any(name.startswith('cadtrack_abc/results/' + folder + '/') for folder in allowed) for name in names)


def git(*args):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True, check=True).stdout


ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
       '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028']
guards = ['baseline.py', 'data.py', 'model.py', 'collect_actions.py', 'future_actions.py',
    'pure_baseline_tracker.py', 'infer_baseline.py', 'baseline_pipeline.py', 'future_pipeline.py',
    'train_future_actions.py', 'r1_pair_pipeline.py', 'select_future_actions.py', 'infer.py', 'score.py']
expected = {'cadtrack_abc/' + name: hashlib.sha256((repo / 'cadtrack_abc' / name).read_bytes()).hexdigest() for name in guards}
guard = ('from pathlib import Path\nimport hashlib\nr=Path("/home/gaob/CADTrack-ABC")\ne=' + repr(expected) +
    '\nassert {n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in e}==e\n')
subprocess.run(ssh + ['python3 -'], input=guard.encode(), capture_output=True, check=True)
assert git('rev-parse', 'HEAD').decode().strip().startswith('c8f41a9')
assert not git('diff', '--cached', '--name-only').strip()
commit = git('rev-parse', 'HEAD').decode().strip()
remote_head = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=repo, capture_output=True, check=True).stdout.decode().split()[0]
assert remote_head == commit
payload = io.BytesIO()
with tarfile.open(fileobj=payload, mode='w:gz') as archive:
    for name in names:
        assert git('show', 'HEAD:' + name) == (repo / name).read_bytes()
        archive.add(repo / name, arcname='HANDOFF_20261002.md' if name == 'docs/HANDOFF_20261002.md' else name)
subprocess.run(ssh + ['tar -xzf - -C /home/gaob/CADTrack-ABC'], input=payload.getvalue(), check=True)
subprocess.run(ssh + ['python3 -'], input=guard.encode(), capture_output=True, check=True)
remote_paths = ['HANDOFF_20261002.md' if name == 'docs/HANDOFF_20261002.md' else name for name in names]
code = ('from pathlib import Path\nimport hashlib,json\nr=Path("/home/gaob/CADTrack-ABC")\nn=' + repr(remote_paths) +
    '\nprint(json.dumps({p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in n}))\n')
actual = json.loads(subprocess.run(ssh + ['python3 -'], input=code.encode(), capture_output=True, check=True).stdout)
for name, remote in zip(names, remote_paths):
    assert actual[remote] == hashlib.sha256((repo / name).read_bytes()).hexdigest()
    if name.startswith('cadtrack_abc/results/'):
        assert (desktop / 'results' / Path(name).relative_to('cadtrack_abc/results')).read_bytes() == (repo / name).read_bytes()
handoff = (repo / 'docs/HANDOFF_20261002.md').read_bytes()
assert handoff == (desktop / 'HANDOFF_20261002.md').read_bytes()
receipt = dict(commit=commit, files=names, handoff_sha256=hashlib.sha256(handoff).hexdigest(),
    desktop_repo_git_server_same_bytes=True, running_sources_before_after_exact=True,
    new_neural_jobs_started=False, existing_jobs_restarted=False,
    decision='Keep two inference workers per physical GPU2/3; four existing jobs total.',
    resource_sample_utc='2026-10-15T02:57:43.558912+00:00/2026-10-15T02:58:13.644468+00:00',
    next_original_observers_bjt='Original 11:10 observations completed; next nodes to be recorded separately.', original_publisher_exit_code=1, original_failure='SSH banner timeout during server tar sync after Git commit/push; no neural job restart', resumed_action='Exact committed four-file server sync and byte verification only', goal_achieved=False)
(private / 'requested_resource_1058_recovered_publication.json').write_text(json.dumps(receipt, indent=2))
with Path('C:/Users/gb/memory/2026-10-15.md').open('a', encoding='utf-8') as stream:
    stream.write(f"CAD user-resource request 10:58 completed: CPU11.3083/iowait1.1446%, GPU2/3 free22901/22897MiB meanutil81.4333/35.0333%, actual4NN bothbeforeafter. Keep2/card basedexisting equalwork1.573856578x, no3/card test/no training concurrency claim/no newNN/noTP. Git{commit}/4ownedfiles solehandoffSHA{receipt['handoff_sha256']} exactDesktopGitServer/14running sources unchanged. MainLasraw result audit remainsPENDING; otherformal3 pending/R2R3unimplemented/goalACTIVE_UNMET. Originalmain69524/control13255 consumedtrue0; main11:10 moved210 with234report89.159547/65.933886, controlstillLas. Original80292 failedSSH255aftercommit/push; exact committedfourfile sync recovered only, no recommit orNNrestart.\n")
print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, indent=2), flush=True)
