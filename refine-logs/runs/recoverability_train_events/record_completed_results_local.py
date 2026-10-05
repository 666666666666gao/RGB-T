"""Put the actually collected complete420 results first in the single human handoff."""
import datetime
import hashlib
import json
from pathlib import Path


def native_tables(report):
    metrics = report['acceptance']['metrics']
    assert [(row['dataset'], row['metric']) for row in metrics] == [
        ('lasher', 'PR'), ('lasher', 'NPR'), ('lasher', 'SR'), ('rgbt234', 'MPR'), ('rgbt234', 'MSR')]
    lines = ['| 数据集／指标，% | 完整GOLA-B | C1 | 本轮ABC | ABC−GOLA，百分点 | 95%配对区间 |',
             '|---|---:|---:|---:|---:|---|']
    variants = ['baseline', 'c1', 'best_native_policy_complete']
    for row in metrics:
        actual = report['datasets'][row['dataset']]['native']['variants']
        assert [actual[name]['overall_metrics_percent'][row['metric']] for name in variants] == [
            row['baseline_percent'], row['c1_percent'], row['ABC_percent']]
        label = 'LasHeR' if row['dataset'] == 'lasher' else 'RGBT234'
        lines.append(f"| {label} {row['metric']} | {row['baseline_percent']:.4f} | {row['c1_percent']:.4f} | "
                     f"{row['ABC_percent']:.4f} | {row['delta_vs_baseline_percentage_points']:+.4f} | "
                     f"[{row['paired_95_lower_percentage_points']:.4f}, {row['paired_95_upper_percentage_points']:.4f}] |")
    lines.extend(['', '以下保留全部属性，不筛选上涨项。各单元格指标顺序由表头标明，单位为%。'])
    for dataset, count, columns in [('lasher', 19, ['PR', 'NPR', 'SR']), ('rgbt234', 12, ['MPR', 'MSR'])]:
        label = 'LasHeR' if dataset == 'lasher' else 'RGBT234'
        variant = report['datasets'][dataset]['native']['variants']
        names = list(variant['baseline']['attributes'])
        assert len(names) == count and all(set(variant[name]['attributes']) == set(names) for name in variants)
        order = '／'.join(columns)
        lines.extend(['', f'**{label}全部{count}项属性：{order}**', '',
                      f'| 属性 | 序列数 | 完整GOLA-B {order} | C1 {order} | 本轮ABC {order} |',
                      '|---|---:|---|---|---|'])
        for attribute in names:
            values = [variant[name]['attributes'][attribute] for name in variants]
            assert len({item['sequences'] for item in values}) == 1
            cells = ['／'.join(f'{item[column]:.4f}' for column in columns) for item in values]
            lines.append(f"| {attribute} | {values[0]['sequences']} | " + ' | '.join(cells) + ' |')
    lines.extend(['', '**运行效率：并发评测实测，不能据此宣称算法加速。**', '',
                  '| 数据集／方法 | FPS | P50，ms | P95，ms | P99，ms |', '|---|---:|---:|---:|---:|'])
    for dataset, item in report['datasets'].items():
        label = 'LasHeR' if dataset == 'lasher' else 'RGBT234'
        for name, title in zip(variants, ['GOLA-B', 'C1', '本轮ABC']):
            values = item['native']['variants'][name]['efficiency']
            lines.append(f"| {label} {title} | {values['tracking_fps']:.4f} | {values['latency_p50_ms']:.4f} | "
                         f"{values['latency_p95_ms']:.4f} | {values['latency_p99_ms']:.4f} |")
    lines.extend(['', '**本轮ABC机制诊断：这些计数来自实际各自轨迹，不能视为跨方法同状态因果对照。**', '',
                  '| 指标 | LasHeR | RGBT234 |', '|---|---:|---:|'])
    counter_names = [('实际额外视觉前向', 'extra_visual_forwards'),
                     ('原区域正确候选缺失帧', 'local_missing_frames'),
                     ('补回正确候选帧', 'missing_correct_candidates_reintroduced'),
                     ('补回后实际选对帧', 'reintroduced_candidates_selected_correctly'),
                     ('同状态相对C1挽救', 'same_state_rescues'), ('同状态相对C1损害', 'same_state_harms'),
                     ('定位错误模板写入计数', 'wrong_template_updates_localization_proxy'),
                     ('可核对模板写入分母', 'known_template_updates'),
                     ('暂停查询写入', 'paused_query_writes')]
    for title, key in counter_names:
        values = [report['datasets'][dataset]['mechanism']['variants']['best_native_policy_complete']['counters'][key]
                  for dataset in ['lasher', 'rgbt234']]
        lines.append(f'| {title} | {values[0]} | {values[1]} |')
    for title, key in [('失败事件', 'failure_events'), ('已恢复事件', 'recovered_events'),
                       ('失败帧，IoU<0.2', 'failed_tracking_frames'), ('恢复延迟中位数，帧', 'median_recovery_delay_frames')]:
        values = [report['datasets'][dataset]['native']['variants']['best_native_policy_complete']['localization_diagnostics'][key]
                  for dataset in ['lasher', 'rgbt234']]
        lines.append(f'| {title} | {values[0]} | {values[1]} |')
    return '\n'.join(lines)


def main():
    root = Path(r'C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
    folder = root / 'refine-logs/runs/recoverability_train_events'
    evidence = folder / 'native_complete'
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    intake = read(folder / 'actual_complete_report_local_intake.json')
    assert intake['status'] == 'PASS'
    manifest = read(evidence / 'export_manifest.json')
    assert manifest['status'] == 'PASS' and manifest['actual_two_native_GT_audits_passed']
    for row in manifest['files']:
        path = evidence / row['path']
        assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    report_path = evidence / 'complete_report/complete_core_report.json'
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == intake['full_report_sha256']
    report = read(report_path)
    assert report['completed'] and report['official_tracking_accuracy']
    assert report['acceptance']['metrics'] == intake['five_metrics']
    training = report['training_CPU_acceptance']
    assert training['status'] == 'PASS' and training['all_four_ABC_fits_passed']
    assert training['validation_queries'] == 196
    assert {name: (row['epochs'], row['optimizer_steps'], row['train_queries']) for name, row in training['arms'].items()} == {
        'base_pairwise': (84, 420, 1762), 'aug_pairwise': (60, 420, 2478),
        'base_budgeted': (84, 420, 1762), 'aug_budgeted': (60, 420, 2478)}
    selection = report['selected_checkpoint']
    assert selection['both_datasets_same_fixed_checkpoint'] and selection['selected_best_epoch'] > 0
    for dataset, sequences, frames in [('lasher', 245, 220703), ('rgbt234', 234, 116649)]:
        audit = report['datasets'][dataset]['independent_native_CPU']
        assert audit['status'] == 'PASS' and audit['sequences'] == sequences and audit['frames'] == frames
    passed = report['acceptance']['all_five_overall_meet_plus_two']
    verdict = ('五项均达到完整GOLA-B＋2，仍需根代理逐项完成审计后才能标记goal完成。' if passed
               else '完整训练与正式评测已结束；五项全部＋2未达成，goal仍为ACTIVE_UNMET。')
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    heading = '## ' + report['created_at_cst'] + '：本轮完整420次训练与两数据集全量正式结果'
    lines = [heading, '', verdict, '',
             '保留完整预训练GOLA-B及原在线模板更新，冻结GOLA和C1，训练全部A／B／C；batch384、lr1e-4、seed42。'
             '原样本两组84 epochs、扩充样本两组60 epochs，每组均420次实际优化更新。'
             '样本覆盖原881条TRAIN，加入716个失败转折查询后扩充组共2478个状态；VAL仍为98条视频／196个状态。'
             '这是缓存时序状态上的模块训练，不是遍历每个原始视频帧的端到端训练。', '',
             '| 训练组 | epochs | 优化更新 | TRAIN查询数 | 峰值CUDA allocated，GiB | 训练耗时，秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in training['arms'].items():
        lines.append(f"| {name} | {row['epochs']} | {row['optimizer_steps']} | {row['train_queries']} | "
                     f"{row['peak_cuda_mib'] / 1024:.4f} | {row['elapsed_seconds']:.2f} |")
    lines.extend(['', f"完整98条连续内部视频共评测{len(selection['candidates'])}个正epoch候选权重，锁定`{selection['checkpoint']}`，"
                  f"epoch{selection['selected_best_epoch']}，SHA256 `{report['selected_checkpoint_sha256_at_report_export']}`。"
                  '同一个权重用于LasHeR245条／220703帧对和RGBT234234条／116649帧对，合计479条／337352帧对。'
                  '实际GT、逐序列指标、全部属性、官方曲线及5000次配对bootstrap已独立CPU核对；不把缓存utility当正式SR。', '',
                  native_tables(report), '',
                  '完整机器结果：`refine-logs/runs/recoverability_train_events/native_complete/complete_report/`；'
                  '包含五项验收CSV、1437行逐序列CSV、93行属性CSV；另有四组训练历史、完整98条选模证据、'
                  '两数据集官方曲线与属性差值PNG／PDF、原始机制统计和GT验收。'
                  f"归档{manifest['file_count']}个文件／{manifest['total_bytes']}字节。"
                  '本记录没有删除权重，也没有标记system goal完成。', '',
                  '**实验边界：**固定checkpoint的序列bootstrap不代表跨训练seed波动；不要求三个seed稳定超过。'
                  '98条内部视频已多次用于开发与选模，正式测试历史也已影响研究方向，应披露。'
                  '四卡并发效率不是隔离负载测速；八组合消融、预算曲线、FLOPs及第二基线不由本次结果代替。', '',
                  '原训练日志命名冲突及接续修复、上一轮300次训练负结果保留在后文，不能混入本轮指标。', ''])
    doc = root / 'docs/HANDOFF_20261002.md'
    text = doc.read_text(encoding='utf-8')
    assert heading not in text
    rows = text.splitlines()
    rows[2] = '更新：' + stamp + '，北京时间。仅维护这一份人工交接文档，历史证据全部保留在下方。'
    rows[4] = '**本轮四组ABC各完成420次优化；同一选定权重的两数据集全量正式指标已完成。' + verdict + '**'
    rows[6] = '**当前完成状态：**本轮训练、完整98条选模、479条全量评测及实际GT CPU核对已结束，最新完整结果列于下方。'
    text = '\n'.join(rows) + '\n'
    position = text.index('\n## ')
    doc.write_text(text[:position] + '\n' + '\n'.join(lines) + text[position:], encoding='utf-8', newline='\n')
    goal_path = root / 'refine-logs/current_goal.json'
    goal = read(goal_path)
    current = goal['current_train_event_data_iteration']
    current.update(status='COMPLETE_FOUR420_ABC_FITS_FULL98_AND_BOTH_NATIVE_GT_ACCEPTED',
                   observed_at_cst=report['created_at_cst'], full_native_metrics_completed_this_new_iteration=True,
                   fullfit_CPU_accepted=True, actual_native_five_metrics=report['acceptance']['metrics'],
                   actual_complete_native_report='refine-logs/runs/recoverability_train_events/native_complete/complete_report/complete_core_report.json',
                   actual_full98_selection=selection, goal_status=report['goal_result'],
                   all_ABC_trained_at_selected_positive_epoch=True,
                   next_required_work='Audit actual best-weight retention and full objective; report all five even when negative. No new training launched by this delivery.')
    current['completed_result_delivery_preparation']['actual_delivery_executed'] = True
    goal['updated_cst'] = stamp
    goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'created_at_cst': stamp, 'five_metrics': report['acceptance']['metrics'],
                      'goal_result': report['goal_result'], 'weights_deleted': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
