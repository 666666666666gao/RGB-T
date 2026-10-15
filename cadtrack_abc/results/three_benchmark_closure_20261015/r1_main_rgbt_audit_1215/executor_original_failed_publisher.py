from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

private = Path(__file__).parent
repo = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
desktop = Path('C:/Users/gb/Desktop/document/RGBT_GOLA_20261002')
names = json.loads((private / 'main_rgbt_audit_owned_files.json').read_bytes())['files']
assert len(names) == len(set(names)) == 111
prefixes = ['cadtrack_abc/results/three_benchmark_closure_20261015/r1_main_rgbt_audit_1215/',
    'cadtrack_abc/results/three_benchmark_closure_20261015/r1_main_three_benchmarks_1150/']
assert all(name == 'docs/HANDOFF_20261002.md' or any(name.startswith(prefix) for prefix in prefixes) for name in names)

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
print(git('commit', '-m', 'Archive exact RGB result audit, runtime boundary diagnosis and qualified paired comparison').decode(), flush=True)
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
    desktop_repo_git_server_same_bytes=True, subset14_running_sources_before_after_exact=True,
    audit_verdict='WARN', blocking_issues=0, checks=1437, exact_numpy_runtime='2.2.6',
    original_failure_attempts_preserved=True, actual_reviewer_routing='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional',
    raw_gt_prediction_latency_arrays_or_weight_bytes_uploaded=False,
    r1_pair_closed=False, r2_or_r3_started=False, goal_achieved=False)
(private / 'main_rgbt_audit_publication.json').write_text(json.dumps(receipt, indent=2))
with Path('C:/Users/gb/memory/2026-10-15.md').open('a', encoding='utf-8') as stream:
    stream.write(f"CAD goal PROGRESS RGBfresh audit terminal WARN0/1437allpass actualexistingnp2.2.6, all2234inputsunchanged. Preserve1.26.4 firstmismatch/invalidGTassert and auditor09lineref failure;05/10/11 actual0, NEP50preparedNOTEXECUTED/noinstall/noNN. RGBT234 paired5000CI MPR[-2.269728897,.817623852] MSR[-1.448652461,.705736097], bothnegativepoint/no+2; rawscoreunchanged/qualifiedbaseline_source fixed. Git{commit}/111ownedfiles solehandoffSHA{receipt['handoff_sha256']} exactDesktopGitServer/14sourceguardunchanged. ActualroutingUNATTESTEDsamefamilyprovisional; fullruntimehistoricalgraph/publisherorigin/causalmodulesnotcertified. Pure210original52993 remains, sole90417 at12:25 after69537actual12:20live191seq/97169frames; control68351/51230at12:40 unchanged. R1pair/R2R3/fullgoalACTIVE_UNMET.\n")
print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, indent=2), flush=True)
