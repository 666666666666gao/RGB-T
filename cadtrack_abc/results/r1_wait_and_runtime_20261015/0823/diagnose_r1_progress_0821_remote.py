from datetime import datetime, timezone
from pathlib import Path
import json

project = Path('/home/gaob/CADTrack-ABC')
main_root = project / 'outputs/r1_main_reports_20261015'
state = json.loads((main_root / 'benchmark_state.json').read_text())
phase = state['phases']['infer_lasher']
elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(phase['started_utc'])).total_seconds()
sequences = sorted(path for path in Path('/data/wangwj/dataset/LasHeR/testingset').iterdir() if path.is_dir())
progress = []
for shard in (0, 1):
    text = (main_root / f'infer_lasher_shard{shard}.log').read_text()
    records = [json.loads(line) for line in text.rsplit('\n', 1)[0].splitlines() if line.startswith('{')]
    records = [row for row in records if row.get('event') == 'sequence_complete']
    expected_names = sequences[shard::2]
    assert [row['sequence'] for row in records] == [path.name for path in expected_names[:len(records)]]
    total_frames = sum(sum(1 for _ in (path / 'visible').iterdir()) for path in expected_names)
    completed_frames = sum(row['frames'] for row in records)
    progress.append(dict(shard=shard, completed_sequences=len(records), total_sequences=len(expected_names),
        completed_frames=completed_frames, total_frames=total_frames,
        completed_sequence_extra_forwards=sum(row['extra_forwards'] for row in records),
        last_completed=records[-1],
        completion_only_remaining_seconds_estimate=elapsed * (total_frames - completed_frames) / completed_frames))
control = json.loads((project / 'outputs/r1_control_reports_20261015/benchmark_state.json').read_text())
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), elapsed_main_lasher_seconds=elapsed,
    main_lasher=progress, control_status=control['status'],
    control_phases={name: dict(status=phase['status'], started_utc=phase.get('started_utc'),
        completed_utc=phase.get('completed_utc'), evidence=phase.get('evidence'))
        for name, phase in control['phases'].items()},
    estimate_scope='Conservative sequence-completion-only wall estimate; in-progress sequence not counted.',
    new_neural_jobs_started=False)
(main_root / 'progress_diagnostic_0821.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
