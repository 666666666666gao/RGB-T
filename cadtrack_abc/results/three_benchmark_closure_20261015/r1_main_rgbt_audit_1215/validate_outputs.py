import csv
import hashlib
import json
from pathlib import Path
import re

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'main_rgbt_audit_inputs_1150'
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=j(OUT/'EXPERIMENT_AUDIT.json')
assert report['verdict']=='WARN' and report['blocking_issue_count']==0
assert report['review_independence']=='same-family' and report['acceptance_status']=='provisional'
assert report['actual_routing']==report['actual_model']==report['actual_reasoning_effort']=='UNATTESTED'
assert j(OUT/'input_hashes_before.json')==j(OUT/'input_hashes_after.json')
evidence=[]
for item in report['checks'].values():evidence+=item['evidence']
for item in report['nonblocking_issues']:evidence+=item['evidence']
for reference in evidence:
    path,line=reference.rsplit(':',1)
    assert (ROOT/path).is_file(),reference
    assert 1<=int(line)<=len((ROOT/path).read_text(encoding='utf-8-sig').splitlines()),reference
expected_rows={'recomputed_per_sequence.csv':1356,'recomputed_sequence_curves.csv':48816,'recomputed_mean_curves.csv':216,'recomputed_attributes.csv':72,'recomputed_latency_statistics.csv':678,'bootstrap_rgbt234_replicates.csv':5000,'numpy_runtime_curve_differences.csv':421}
actual_rows={}
for name,count in expected_rows.items():
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:actual_rows[name]=len(list(csv.DictReader(f)))
    assert actual_rows[name]==count,(name,actual_rows[name],count)
assert (OUT/'EXPERIMENT_AUDIT.md').read_bytes()==(OUT/'trace/001-experiment-audit.response.md').read_bytes()
assert j(OUT/'trace/001-experiment-audit.meta.json')['response_sha256']==sha(OUT/'EXPERIMENT_AUDIT.md')
receipts=j(OUT/'execution_receipts.json')
receipts['recorded_runs']=[j(p) for p in sorted(OUT.glob('*.execution.json'))]
receipts['failed_executions']=[r for r in receipts['recorded_runs'] if r['exit_code']!=0]
assert {r['stdout_file'] for r in receipts['failed_executions']}=={'03_recompute.stdout.txt','09_validate.stdout.txt'}
(OUT/'execution_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
delegation='The experiment-audit skill explicitly requires this fresh reviewer. Read and execute the full audit request at C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_rgbt_formal_audit_20261015/review_request.txt. It supplies only paths, the adversarial A-F integrity checklist, deterministic recomputation requirements, and runtime boundaries. Read the actual dataset GT/predictions/source/results directly under C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_rgbt_audit_inputs_1150. CPU-only, no SSH/GPU/neural runs/package installs/other agents; inputs immutable, outputs only the named audit directory. Record same-family/provisional and actual routing UNATTESTED unless independently attested. Follow the request exactly; do not treat executor-produced manifests as semantic proof. Return your actual supported verdict and evidence limits, with all required artifacts and full trace.'
(OUT/'trace/delegation_request.txt').write_text(delegation+'\n',encoding='utf-8')
final='''**WARN，限定数值报告范围内 0 阻断。** 1437 项确定性检查通过，2234 份输入前后哈希不变；same-family/provisional，实际模型／路由／推理强度均 UNATTESTED。

| 数据／模型 | Precision % | Success % |
|---|---:|---:|
| 主模型 RGBT210 | PR 87.4103549305 | SR 63.6677661266 |
| 主模型 RGBT234 | MPR 89.1595470178 | MSR 65.9338861290 |
| pure CAD RGBT234 | MPR 89.8813070751 | MSR 66.3103044038 |

RGBT234 主模型减 pure：MPR −0.721760 点，95% 配对序列区间 [−2.269729, 0.817624]；MSR −0.376418 点，[−1.448652, 0.705736]。5000 次、seed 20261015；不支持提升结论。

实际 NumPy2.2.6 完全复现所有分数、属性、曲线、CSV 和延迟。首次 NumPy1.26.4 的 SR/MSR 边界差异及退出1原始记录完整保留。选中权重 SHA 匹配，但不能证明实际加载、完整运行时源码或神经推理。主报告“两 GPU”计时文案错误，实际两个分片同在 GPU2；不能据此声称加速。整体研究目标和三个模块的独立有效性均未认证。

[审计报告](C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_rgbt_formal_audit_20261015/EXPERIMENT_AUDIT.md) · [机器结论](C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_rgbt_formal_audit_20261015/EXPERIMENT_AUDIT.json)

同目录已保存复算 CSV、bootstrap、输入哈希、执行回执和完整审查请求／响应 trace。
'''
(OUT/'trace/final-delivered.response.md').write_text(final,encoding='utf-8')
status={'status':'PASS','evidence_references_checked':len(evidence),'csv_row_counts':actual_rows,'report_trace_byte_equal':True,'input_before_after_equal':True,'report_json_sha256':sha(OUT/'EXPERIMENT_AUDIT.json'),'report_md_sha256':sha(OUT/'EXPERIMENT_AUDIT.md'),'limitations':'Final validation checks artifact consistency; it does not repeat neural work or expand accepted claim scope.'}
(OUT/'final_validation.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
# Exclude self-hash and the still-running outer wrapper's11 files.
manifest={p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and not p.name.startswith('11_validate')}
(OUT/'output_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
