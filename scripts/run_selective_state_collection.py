"""Reviewed four-card sanity then full TRAIN/developer-state supervision collection."""
import argparse
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path('/data/gb/GOLA')
PYTHON = '/data/gb/envs/gola/bin/python'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', required=True)
    p.add_argument('--review', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    review = json.loads(Path(args.review).read_text())
    assert review['status'] == 'PASS' and review['scope'] == 'STATE_HEAD_COLLECTOR_PLANNER_CONTROLLER_SOURCE'
    root = Path(args.output)
    assert not root.exists()
    assert shutil.disk_usage('/data/gb').free > 2 * 1024**3
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    cards = [tuple(int(x.strip()) for x in row.split(',')) for row in inventory.splitlines()]
    assert [r[0] for r in cards] == [0, 1, 2, 3] and all(r[1] < 1024 and r[2] == 0 for r in cards), cards
    plan = json.loads((Path(args.queries) / 'plan.json').read_text())
    assert plan['status'] == 'CPU_H32_QUERIES_PREPARED' and plan['train_sequences'] == 881
    groups = [json.loads((Path(args.queries) / f'gpu{g}_jobs.json').read_text())['jobs'] for g in range(4)]
    root.mkdir(parents=True)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    for stage in ('sanity', 'full'):
        children = []
        for gpu, group in enumerate(groups):
            selected = [min(group, key=lambda j: j['query_frame'])] if stage == 'sanity' else group
            jobfile = root / f'{stage}_gpu{gpu}_jobs.json'
            jobfile.write_text(json.dumps({'jobs': selected}, indent=2))
            out = root / stage / f'gpu{gpu}'
            log = (root / f'{stage}_gpu{gpu}.log').open('w')
            command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'collect_selective_state_commit',
                       '--jobs-file', str(jobfile), '--output', str(out), '--horizon', '32']
            if stage == 'sanity':
                command += ['--parity']
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, gpu, out, len(selected)))
        record(stage.upper(), children=[{'gpu': gpu, 'pid': child.pid} for child, _, gpu, _, _ in children])
        exits = []
        for child, log, gpu, out, count in children:
            code = child.wait()
            log.close()
            exits.append((gpu, code))
            record(stage.upper() + '_WORKER_ENDED', gpu=gpu, exit_code=code)
        assert all(code == 0 for _, code in exits), (stage, exits, 'Original logs retained; no automatic retry')
        for _, _, gpu, out, count in children:
            done = json.loads((out / 'summary.json').read_text())
            assert (out / 'COMPLETE').is_file() and done['queries'] == count
            assert done['status'] == 'COMPLETE_MATCHED_STATE_COMMIT_LABELS'
            if stage == 'sanity':
                assert done['M0_parity_checked'] and all(r['M0_parity_pass'] for r in done['results'])
        record(stage.upper() + '_PASS')
    results = [r for g in range(4) for r in json.loads((root / f'full/gpu{g}/summary.json').read_text())['results']]
    assert len(results) == plan['queries']
    assert len({(r['sequence'], r['query_frame']) for r in results}) == len(results)
    assert len({r['sequence'] for r in results if r['partition'] == 'train'}) == 881
    assert len({r['sequence'] for r in results if r['partition'] == 'validation'}) == 98
    partitions = {}
    for part in ('train', 'validation'):
        rows = [r for r in results if r['partition'] == part]
        events = {}
        for row in rows:
            events.setdefault(row['actual_event_id'], set()).add(row['actual_event_role'])
        partitions[part] = {'queries': len(rows), 'distinct_actual_events': len(events),
                            'query_roles': dict(Counter(r['actual_event_role'] for r in rows)),
                            'event_roles': dict(Counter(role for roles in events.values() for role in roles)),
                            'search_executed_queries': sum(r['extra_visual_forward_at_query'] for r in rows),
                            'alternative_candidate_queries': sum(r['alternative_candidates'] > 0 for r in rows),
                            'legal_write_pair_queries': sum(r['legal_write_pairs'] > 0 for r in rows)}
    complete = {'complete': True, 'queries': len(results), 'partitions': partitions,
                'no_validation_in_optimizer': 'Enforced by the separate trainer partition filter',
                'GT_scope': 'TRAIN root only; actual matched consequences, no TEST selection', 'results': results}
    (root / 'collection_complete.json').write_text(json.dumps(complete, indent=2))
    record('COMPLETE_TRAIN881_VALIDATION98_MATCHED_STATE_LABELS', queries=len(results), partitions=partitions,
           optimizer_updates=0, formal_metric_claim=False)


if __name__ == '__main__':
    main()
