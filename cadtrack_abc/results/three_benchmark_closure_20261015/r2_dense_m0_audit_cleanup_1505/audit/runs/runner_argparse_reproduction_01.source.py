"""Execute only this audit's CPU checkers and retain exact output and exit codes."""
import argparse, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--legacy-encoding',action='store_true')
p.add_argument('tag')
p.add_argument('script')
p.add_argument('arguments',nargs='*')
a=p.parse_args()
script=(root/a.script).resolve()
assert script.parent==root and script.suffix=='.py'
prefix=root/'runs'/a.tag
prefix.parent.mkdir(exist_ok=True)
assert not prefix.with_suffix('.exit.json').exists()
command=[sys.executable,'-B',str(script),*a.arguments]
prefix.with_suffix('.source.py').write_bytes(script.read_bytes())
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
if not a.legacy_encoding:
    env['PYTHONIOENCODING']='utf-8'
else:
    env['PYTHONIOENCODING']='gbk'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(command,capture_output=True,env=env,cwd=root)
prefix.with_suffix('.stdout.txt').write_bytes(r.stdout)
prefix.with_suffix('.stderr.txt').write_bytes(r.stderr)
record=dict(command=command,started_utc=started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actual_exit_code=r.returncode,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),
    stderr_sha256=hashlib.sha256(r.stderr).hexdigest(),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    legacy_encoding=a.legacy_encoding)
prefix.with_suffix('.exit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
print(r.stdout.decode('utf-8',errors='replace'))
print(r.stderr.decode('utf-8',errors='replace'))
sys.exit(r.returncode)
