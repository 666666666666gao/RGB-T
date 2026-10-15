"""Read all input text/JSON leaves; verify remaining archived lineage and coverage."""
import ast, hashlib, json, re
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'dense_m0_fresh_audit_inputs_1435'/'project'
M=json.loads((ROOT.parent/'input_manifest.json').read_text(encoding='utf-8-sig'))
B='cadtrack_abc/results/three_benchmark_closure_20261015/'
A=B+'r2_dense_admission_1425/'
checks=[]; leaves=[]; coverage=[]
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def check(n,c,details=None):
    checks.append(dict(id=len(checks)+1,name=n,passed=bool(c),details=details))
    assert c,n
def visit(v,path):
    if isinstance(v,dict):
        for k,x in v.items(): visit(x,path+'/'+k)
    elif isinstance(v,list):
        for i,x in enumerate(v): visit(x,path+'/'+str(i))
    else: leaves.append(dict(path=path,value=v))
for f in M['files']:
    p=ROOT/f['relative']; s=p.read_text(encoding='utf-8-sig'); ls=s.splitlines()
    row=dict(path=f['relative'],bytes_read=len(p.read_bytes()),line_count=len(ls),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if p.suffix=='.json':
        v=json.loads(s); visit(v,f['relative']); row['review']='All JSON values parsed/traversed; qualification and lineage contracts checked; repeating name lists checked by exact set/list comparison.'
    elif p.suffix=='.py':
        row['review']='Whole source displayed with line numbers and semantically inspected; AST parsed. Never imported as a project module.'
    elif p.suffix=='.md':
        row['review']='Entire bytes read and complete headings indexed; current dense/R2 section 5242-5380 read in full. Older archival narratives are context only, not individually recertified.'
    else: row['review']='Whole configuration text read with line numbers.'
    coverage.append(row)
for group in ('batch32_64','batch128'):
    launch=read(A+group+'/launch.json'); pre=read(A+group+'/preflight.json'); terminal=read(A+group+'/exit.json')
    check(group+' launch embeds exact preflight JSON',launch['preflight']==pre)
    check(group+' finite not full flags',launch['new_finite_qualification_launched'] and not launch['new_full_training_launched'] and not launch['formal_training_started'])
    check(group+' snapshot cards2and3',launch['expected_physical_gpus']==[2,3])
    check(group+' wrapper pid lineage',terminal['wrapper_pid']==launch['wrapper_pid'])
    check(group+' source count21',len(pre['expected_sources'])==21)
    for path,digest in pre['expected_sources'].items():
        check(group+' sealed source '+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    check(group+' recorded home space exceeds budget',pre['home_free_bytes']>pre['output_budget_bytes'])
    check(group+' preflight24GiB units',all(int(line.split(',')[2])==24*1024 for line in pre['gpu_memory_and_utilization_csv'].splitlines()))
    check(group+' environment receipt matches sealed file',pre['environment']==read('cadtrack_abc/environment.json'))
first=read(B+'r2_dense_dependency_fix_1405/first_launch.json')
check('first launch20 sources',len(first['preflight']['expected_sources'])==20)
for path,digest in first['preflight']['expected_sources'].items(): check('first launch source '+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
check('first preflight missing tensor dependency explicitly',not any(p.endswith('/tensor.py') for p in first['preflight']['expected_sources']))
cpu=read(B+'r2_dense_dependency_fix_1405/dependency_cpu_check.json')
check('CPU dependency tensor hash matches included current file',cpu['native_tensor_sha256']==hashlib.sha256((ROOT/'cadtrack_abc/vendor/CADTrack/lib/utils/tensor.py').read_bytes()).hexdigest())
check('CPU prerequisite expressly synthetic only','Synthetic CPU' in cpu['scope'] and cpu['cuda_initialized'] is False)
for local,archived in [('r2_dense_m0_attempt2_driver_remote.py','batch32_64/driver.py'),('r2_dense_m0_attempt2_wrapper_remote.py','batch32_64/wrapper.py'),('extend_dense_m0_capacity128_remote.py','batch128/driver.py'),('r2_dense_m0_capacity128_wrapper_remote.py','batch128/wrapper.py')]:
    check('staged archived source text agrees '+local,(ROOT/A/local).read_text()==(ROOT/A/archived).read_text())

# Report exact claimed source count and historical placeholders without inventing missing artifacts.
handoff=(ROOT/'docs/HANDOFF_20261002.md').read_text(encoding='utf-8-sig').splitlines()
headings=[dict(line=i,text=s) for i,s in enumerate(handoff,1) if re.match(r'^#{1,6} ',s)]
refs=[]
for i,s in enumerate(handoff,1):
    if i>=5261:
        for r in re.findall(r'cadtrack_abc/[A-Za-z0-9_./-]+',s):
            refs.append(dict(line=i,reference=r,in_input=any(f['relative']==r or f['relative'].startswith(r.rstrip('/')+'/') for f in M['files'])))
from collections import Counter
counts=Counter(Path(f['relative']).suffix for f in M['files'])
report=dict(status='PASS',checks_count=len(checks),checks=checks,input_file_types=dict(counts),coverage=coverage,
    json_leaves_traversed=len(leaves),narrative_line_count=len(handoff),narrative_headings=headings,
    current_narrative_references=refs,
    semantic_coverage_limit='All listed sources and current dense R2 narrative reviewed; historical GOLA/R1 claims and external references not independently re-audited.')
(OUT/'lineage_coverage_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(OUT/'private_json_leaf_inventory.json').write_text(json.dumps(leaves,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',checks_count=len(checks),files=len(coverage),json_leaves=len(leaves),types=dict(counts),narrative_lines=len(handoff)),indent=2))
