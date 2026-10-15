from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

private = Path(__file__).parent
repo = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
desktop = Path('C:/Users/gb/Desktop/document/RGBT_GOLA_20261002')
names = json.loads((private / 'control_full98_closed_owned_files.json').read_bytes())['files']
assert len(names) == len(set(names)) == 22
allowed = ['r1_control_selected_and_formal_20261015']
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
assert not git('diff', '--cached', '--name-only').strip()
git('fetch', '--no-tags', 'origin', 'main')
assert git('rev-list', '--count', 'HEAD..origin/main').strip() == b'0'
git('add', '--', *names)
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(names)
git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check')
for name in names:
    assert git('show', ':' + name) == (repo / name).read_bytes()
print(git('commit', '-m', 'Close all six matched-control candidates and admit selected epoch20 formal evaluation').decode(), flush=True)
git('push', 'origin', 'main')
commit = git('rev-parse', 'HEAD').decode().strip()
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
    full_h3_labels_complete=True, actual_r1_training_complete=True, full_new_results_complete=False,
    goal_achieved=False)
(private / 'control_full98_closed_publication.json').write_text(json.dumps(receipt, indent=2))
with Path('C:/Users/gb/memory/2026-10-15.md').open('a', encoding='utf-8') as stream:
    stream.write(f"CAD R1 control full98 PROGRESS Git{commit}/22ownedfiles, solehandoffSHA{receipt['handoff_sha256']} exactDesktopGitServer;14running NN sources beforeafter exact. All6control candidates0/1/2/5/10/20 real98/52200/52120 and12child0/588rawrecordsrecomputed. Actualselector0 chosepositive20 IoU.6646335296, best2,125,728B SHAcffca0302460b60f505c2a09a5b651b10f1d1e2211178f68307a4d708080bf62. Existing68351/PID85361 admittedfullLas09:52/NN260113+260114 physical3 actualpslive09:55, samebest/config then234/210. Original13592closed0, solecontrol13255 first11:10; mainoriginal28531 andsole1709 first10:40 unchanged/notqueriedearly. R1newformal/pure210/R2R3/independent3contributions stillpending/goalACTIVE_UNMET. PreviousgoalturnPROGRESS; thisturncompletecontrolselectionPROGRESS.\n")

print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, indent=2), flush=True)
