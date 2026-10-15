"""Create a separate stdlib checker after NumPy absence; preserve failed source."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'check_source_cpu.py').read_text(encoding='utf-8')
s=s.replace('AST contracts and NumPy-only unit witnesses','AST contracts and stdlib-only witnesses').replace('import numpy as np','import math')
a=s.index('# Execute exact sampler method')
b=s.index('# Execute source max-selection')
s=s[:a]+'''# Execute exact independent RNG/start-selection expressions on synthetic metadata.
get=method(tree('data.py'),'VideoClips','__getitem__')
selected=[n for n in get.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('rng','sequence_index','first')]
def sample(index):
    env=dict(random=random,self=SimpleNamespace(seed=42,epoch=3,samples=20000,sequences=[None]*881),index=index,starts=list(range(39)))
    exec(compile(ast.Module(body=copy.deepcopy(selected),type_ignores=[]),'<sealed RNG statements>','exec'),env)
    return env['sequence_index'],env['first']
a=[sample(i) for i in range(100)]
random.seed(999)
for _ in range(1000): random.random()
b=[sample(i) for i in range(100)]
check('exact sampler RNG statements independent of global draws',a==b)
check('sampler reads initial plus query images not future labels',"ids[:self.steps + 1]" in ast.unparse(get))
check('sampler explicit continuous stride1','stride = 1' in ast.unparse(get))

# Execute actual stdlib diagnostic functions; NumPy IoU execution remains unavailable.
ms=tree('mechanism_summary.py')
env={}
execute_functions([method(ms,None,n) for n in ('recovery_events','ratio')],env)
events=env['recovery_events']([.1,.1,.7,.1,.8],[True,False,True,True,False])
check('actual recovery excludes invalid end and counts frame delay',events==[dict(start_frame=1,recovery_frame=3,delay_frames=2,right_censored=False),dict(start_frame=4,recovery_frame=None,delay_frames=None,observed_until_frame=5,right_censored=True)],events)
check('actual ratio uses explicit event denominator',env['ratio'](2,5)==.4)
check('zero denominator represented null',env['ratio'](0,0) is None)

'''+s[b:]
a=s.index('# Reproduce exact candidate normalized geometry')
b=s.index('# Function-call inventory')
s=s[:a]+'''# Independent scalar Center-head convention oracle; no neural execution.
rng=random.Random(42)
for i in range(7):
    index=rng.randrange(256); w,h=rng.uniform(.05,.8),rng.uniform(.05,.8)
    dx,dy=rng.uniform(-.2,.9),rng.uniform(-.2,.9)
    cx,cy=((index%16)+dx)/16,((index//16)+dy)/16
    edges=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
    check(f'Center scalar convention case{i}',math.isclose(edges[2]-edges[0],w) and math.isclose(edges[3]-edges[1],h))

'''+s[b:]
s=s.replace("execution_scope='NumPy/std library only; extracted non-neural functions and AST. Synthetic witnesses are not dataset/neural validation.'", "execution_scope='Stdlib only; extracted non-neural expressions/functions and AST. Synthetic witnesses are not dataset/neural validation.'")
s=s.replace("(OUT/'source_cpu_checks.json')", "(OUT/'source_stdlib_checks.json')")
s=s.replace("numpy_version=np.__version__,", "")
(root/'check_source_stdlib.py').write_text(s,encoding='utf-8')
