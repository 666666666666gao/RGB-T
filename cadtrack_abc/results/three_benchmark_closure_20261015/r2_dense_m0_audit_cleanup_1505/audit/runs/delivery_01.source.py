"""Final deliverable checks only; no experimental scope expansion."""
import ast, hashlib, json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'dense_m0_fresh_audit_inputs_1435'/'project'
checks=[]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def check(n,c):
    checks.append(dict(name=n,passed=bool(c)))
    assert c,n
r=read(OUT/'EXPERIMENT_AUDIT.json')
check('WARN verdict zero blockers',r['verdict']=='WARN' and r['blocking_findings_count']==0 and not r['blocking_findings'])
check('six checklist statuses',{k:v['status'] for k,v in r['checks'].items()}==dict(A='WARN',B='PASS',C='WARN',D='WARN',E='WARN',F='PASS'))
check('honest requested actual attribution',r['requested_reviewer_model']=='gpt-6-astra' and r['requested_reviewer_reasoning_effort']=='max' and r['actual_reviewer_model']==r['actual_reviewer_routing']==r['actual_reviewer_effort']=='UNATTESTED')
check('same-family provisional',r['review_independence']=='same-family' and r['acceptance_status']=='provisional')
check('848 counted substantive checks',r['deterministic_verification']['counted_assertions']==848)
for letter,checklist in r['checks'].items():
    for cite in checklist['evidence']:
        f,line=cite.rsplit(':',1)
        check('valid citation '+letter+' '+cite,(ROOT/f).is_file() and 1<=int(line)<=len((ROOT/f).read_text(encoding='utf-8-sig').splitlines()))
before=read(OUT/'input_seal_before.json'); after=read(OUT/'input_seal_after.json')
check('before after identical input rows',before['files']==after['files'])
check('before after manifest and index unchanged',before['manifest_sha256']==after['manifest_sha256'] and before['index_sha256']==after['index_sha256'])
for f in before['files']:
    check('final live local seal '+f['relative'],hashlib.sha256((ROOT/f['relative']).read_bytes()).hexdigest()==f['sha256'])
for receipt_path in sorted((OUT/'runs').glob('*.exit.json')):
    x=read(receipt_path); prefix=receipt_path.name.removesuffix('.exit.json')
    for suffix,key in [('.stdout.txt','stdout_sha256'),('.stderr.txt','stderr_sha256')]:
        check('exact captured bytes '+prefix+suffix,hashlib.sha256((receipt_path.parent/(prefix+suffix)).read_bytes()).hexdigest()==x[key])
    copy=receipt_path.parent/(prefix+'.source.py')
    if not copy.exists(): copy=OUT/Path(x['command'][2]).name
    check('executed source preserved '+prefix,hashlib.sha256(copy.read_bytes()).hexdigest()==x['script_sha256'])
check('NumPy failed attempt preserved',read(OUT/'runs/source_cpu_01.exit.json')['actual_exit_code']==1)
check('GBK failed reproduction preserved',read(OUT/'runs/read_gbk_reproduction_01.exit.json')['actual_exit_code']==1)
check('argparse failed reproduction preserved',read(OUT/'runs/runner_argparse_reproduction_01.exit.json')['actual_exit_code']==2)
trace=OUT/r['trace_path']
check('private full substantive response matches report',(trace/'001-dense-m0.response.md').read_bytes()==(OUT/'EXPERIMENT_AUDIT.md').read_bytes())
check('private request preserved',(trace/'001-dense-m0.request.json').is_file())
for path in OUT.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8-sig'))
result=dict(status='PASS',checks_count=len(checks),checks=checks,scope='Deliverable/citations/retained execution bytes/input seals; separate from848 substantive assertions.')
(OUT/'delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',delivery_checks=len(checks),substantive_checks=848),indent=2))
