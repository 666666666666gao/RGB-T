"""Publish the terminal original experiment and its complete native metrics."""
import datetime
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
repo = parent / 'GOLA-source'
base = repo / 'refine-logs/runs/post_search_relation_control'
closed = base / 'complete/training'
archive = parent / 'post_search_complete_publication_20261007.tar.gz'
proof_path = parent / 'post_search_complete_publication_proof_20261007.json'
assert not archive.exists() and not proof_path.exists()
receipt = json.loads((base / 'actual_completed_artifact_intake.json').read_text())
assert receipt['completed_stage'] == 'COMPLETE_POST_SEARCH_RELATION_CONTROL_AND_NATIVE'
assert receipt['no_neural_training_inference_or_metric_report_replay']
for item in receipt['manifest']:
    raw = (base / 'complete' / item['path']).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
native = receipt['native_complete']
assert native['completed'] and len(native['five_metrics']) == 5
assert not native['all_five_plus_two']
selection = receipt['selected_model']
assert selection['selected_epoch'] == 0 and selection['selected_candidate'] == 'one_way_lr5_best'
cleanup = json.loads((closed / 'unused_own_weight_cleanup.json').read_text())
assert len(cleanup['removed']) == 19 and cleanup['kept'] == selection['parent_model']
stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
goal_path = repo / 'refine-logs/current_goal.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_cst'] = stamp
control = goal['current_post_search_relation_control']
control.update(updated_cst=stamp, full_training_complete=True, full98_complete=True, native_complete=True,
    actual_observation=json.loads((base / 'sole_observer_complete.json').read_text())['final_observation'],
    next_check_cst=None, complete_receipt='refine-logs/runs/post_search_relation_control/actual_completed_artifact_intake.json',
    native_five_metrics=native['five_metrics'], all_five_plus_two=native['all_five_plus_two'],
    selected_epoch=selection['selected_epoch'], selected_candidate=selection['selected_candidate'],
    same_model_both_datasets=native['same_ABC_model_both_datasets'],
    unused_own_weights_removed=19, all_consumers_closed_before_cleanup=True,
    new_learning_gain_established=False, goal_status='ACTIVE_UNMET')
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
rows = []
for row in native['five_metrics']:
    paired = json.loads((closed / 'native' / row['dataset'] / 'baseline_paired_report/paired_bootstrap.json').read_text())
    ci = paired['datasets'][row['dataset']]['metrics'][row['metric']]['percentile_95_interval_percentage_points']
    rows.append(f"| {row['dataset']} {row['metric']} | {row['baseline_percent']:.4f} | {row['percent']:.4f} | {row['delta_vs_baseline_pp']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['target_percent'] - row['percent']:.4f} |")
table = '\n'.join(rows)
doc = repo / 'docs/HANDOFF_20261002.md'
old = doc.read_text(encoding='utf-8')
prefix = '# RGB-T / GOLA 研究交接文档\n\n'
assert old.startswith(prefix)
section = f'''## {stamp} 本轮完整训练、九候选全视频、两个正式数据集及权重清理全部闭合

原controller1311071和原sole observer35412已结束；原完成状态`COMPLETE_POST_SEARCH_RELATION_CONTROL_AND_NATIVE`。complete-only接收器一次性核验并复制原结果，没有重新训练、推理或计算GT指标。四组完整64 epochs、每组576次优化更新及严格重载验收已完成；九个候选全部98视频评测及原actual-GT选模已完成。正式评测在TEST之前锁定`one_way_lr5_best` epoch0，两个数据集同一权重，完整GOLA-B保留在线模板。

| 正式指标（%） | 本地完整GOLA-B | 本轮选定模型 | 增量（百分点） | 固定模型95%配对区间 | 距baseline+2还差 |
|---|---:|---:|---:|---|---:|
{table}

LasHeR完整245条/220703帧对，RGBT234完整234条/116649帧对；31项属性设置、完整曲线、逐序列结果、配对5000次bootstrap、候选补回/实际接纳/损害/写入与失败恢复诊断均保留在原报告中。固定权重的配对区间不是训练seed波动，也不消除开发集和测试集历史重复使用的影响。并发推理FPS只描述本次运行，不用来证明相对不同磁盘负载的baseline加速。

本轮选到epoch0，不能把以上分数记成这次双向关联训练产生的新收益。双向两个最佳内部IoU均低于单向父模型，四个训练末轮全部退步；九候选原报告和各轮损失曲线保留，负结果不覆盖。五项全部+2仍未达到，目标ACTIVE_UNMET。主方法三个模块的独立必要性、完整八组合和隔离负载的效率对照仍未完成。

全部消费者结束后，原队列已删除本轮19个自有未选中权重，仅保留本轮选定best；当前父模型、old4、完整GOLA、C1、运动依赖及预训练基础权重受保护。原删除清单及字节统计见`complete/training/unused_own_weight_cleanup.json`，不能提前删除仍被推理或报告引用的文件。

下一步依据已完成控制推进：优先做同视觉/候选预算下的实例身份表征对照，再决定是否增加跨帧被拒候选关联；风险校准与几何/RGB/TIR选择性提交作为后续受控验证。当前source-only审计确认A描述来自Transformer前的patch投影，未证明它导致退步。不会把本次失败的双向结构继续加长训练当作已验证方案，也不改测试序列规则或事后重写输出。

完整原始证据入口：`refine-logs/runs/post_search_relation_control/actual_completed_artifact_intake.json`；原完整训练及正式报告目录`refine-logs/runs/post_search_relation_control/complete/training/`。所有复制文件逐项SHA核验，ONE交接文档在仓库/GitHub/Desktop/服务器同字节。仍只做LasHeR、RGBT234，一个预先选定最佳模型即可，不加三个seed同时超过的条件。

---

'''
doc.write_text(prefix + section + old[len(prefix):], encoding='utf-8', newline='\n')
desktop = Path('C:/Users/gb/Desktop/document/RGBT_GOLA_20261002/HANDOFF_20261002.md')
desktop.write_bytes(doc.read_bytes())
(base / 'publish_post_search_complete_20261007.py').write_bytes(Path(__file__).read_bytes())
files = ['docs/HANDOFF_20261002.md', 'refine-logs/current_goal.json']
files += [(base / name).relative_to(repo).as_posix() for name in ['actual_completed_artifact_intake.json',
    'collect_post_search_relation_control_complete_20261007.py', 'complete_intake_source_verification.json',
    'publish_post_search_complete_20261007.py']]
files += [p.relative_to(repo).as_posix() for p in sorted((base / 'complete').rglob('*')) if p.is_file()]
with tarfile.open(archive, 'w:gz') as stream:
    for name in files:
        stream.add(repo / name, arcname=name, recursive=False)
subprocess.run(['scp', str(archive), '2027:/data/gb/setup/' + archive.name], capture_output=True, check=True)
payload = "import hashlib,tarfile;from pathlib import Path\nwith tarfile.open('/data/gb/setup/post_search_complete_publication_20261007.tar.gz') as stream:stream.extractall('/data/gb/GOLA')\nprint(hashlib.sha256(Path('/data/gb/GOLA/docs/HANDOFF_20261002.md').read_bytes()).hexdigest())\n"
remote = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=payload,
    text=True, encoding='utf-8', capture_output=True, check=True)
def git(*arguments, input=None):
    return subprocess.run(['git', *arguments], cwd=repo, input=input, capture_output=True, check=True).stdout
assert not git('diff', '--cached', '--name-only')
git('-c', 'core.autocrlf=false', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul', input=('\0'.join(files) + '\0').encode())
assert set(git('diff', '--cached', '--name-only', '-z').decode().split('\0')[:-1]) <= set(files)
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
git('commit', '-m', 'Close matched relation control with full native metrics and retire unused checkpoints')
git('push', 'origin', 'main')
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert doc.read_bytes() == desktop.read_bytes() == git('show', 'HEAD:docs/HANDOFF_20261002.md')
digest = hashlib.sha256(doc.read_bytes()).hexdigest()
assert remote.stdout.strip() == digest
proof = dict(main=head, handoff_sha256=digest, one_handoff_exact_local_Git_Desktop_server=True,
    native_complete=True, five_metrics=native['five_metrics'], all_five_plus_two=False,
    actual_full_training_updates=2304, selected_epoch=0, unused_own_weights_removed=19,
    published_files=len(files), goal_status='ACTIVE_UNMET')
proof_path.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof), flush=True)
