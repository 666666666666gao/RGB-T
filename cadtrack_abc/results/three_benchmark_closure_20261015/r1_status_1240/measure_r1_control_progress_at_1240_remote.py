from datetime import datetime, timezone
from pathlib import Path
import json

project = Path('/home/gaob/CADTrack-ABC')
root = project / 'outputs/r1_control_reports_20261015'
state = json.loads((root / 'benchmark_state.json').read_text())
phase = state['phases']['infer_selected_lasher']
started = datetime.fromisoformat(phase['started_utc'])
observed = datetime.now(timezone.utc)
rows = []
for shard in (0, 1):
    records = [json.loads(line) for line in (root / f'infer_selected_lasher_shard{shard}.log').read_text().splitlines() if line.startswith('{')]
    records = [row for row in records if row.get('event') == 'sequence_complete']
    expected = json.loads((project / f'outputs/r1_main_reports_20261015/selected/lasher_predictions/complete_shard{shard}.json').read_text())
    completed_frames = sum(row['frames'] for row in records)
    assert completed_frames > 0
    elapsed = (observed - started).total_seconds()
    rows.append(dict(shard=shard, completed_sequences=len(records), total_sequences=expected['sequences'],
        completed_frame_lower_bound=completed_frames, total_frame_pairs=expected['frames'],
        last_completed_sequence=records[-1]['sequence'],
        current_sequence_partial_frames_not_counted=True,
        linear_remaining_seconds=elapsed * (expected['frames'] - completed_frames) / completed_frames))
result = dict(observed_utc=observed.isoformat(), phase_status=phase['status'],
    purpose='One necessary throughput/progress forecast at the existing12:40 diagnostic node, not a new monitor loop.',
    elapsed_seconds=(observed - started).total_seconds(), shards=rows,
    forecast_limitation='Linear forecast from completed-sequence frame lower bounds; current partial sequences, latency variation and subsequent datasets are not measured here.',
    new_neural_work=False, gpu_or_temperature_power_queries=False)
(root / 'progress_forecast_1240.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
