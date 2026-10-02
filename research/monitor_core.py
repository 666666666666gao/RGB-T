"""Read the four existing online jobs once; no restarts or parameter changes."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/data/gb/outputs/online_core')
RUNS = (
    ('lasher_baseline', 'online_lasher_baseline_20261002', 245, 220703),
    ('lasher_c1', 'online_lasher_c1_20261002', 245, 220703),
    ('rgbt234_baseline', 'online_rgbt234_baseline_20261002', 234, 116649),
    ('rgbt234_c1', 'online_rgbt234_c1_20261002', 234, 116649),
)


def main():
    now = datetime.now(timezone.utc)
    report = {'checked_utc': now.isoformat(), 'runs': {}}
    for name, session, expected_sequences, expected_frames in RUNS:
        folder = ROOT / name
        live = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0
        predictions = folder / 'predictions'
        records = [json.loads(line) for line in (predictions / 'progress.jsonl').read_text().splitlines()]
        frames = sum(r['frames'] for r in records)
        elapsed = sum(r['elapsed_seconds'] for r in records)
        fps = (frames - len(records)) / elapsed if elapsed else None
        inference_receipt_path = predictions / 'inference_completion.json'
        inference_receipt = json.loads(inference_receipt_path.read_text()) if inference_receipt_path.exists() else None
        metrics_path = folder / 'official_metrics.log'
        metrics_text = metrics_path.read_text() if metrics_path.exists() else None
        rgbt_gt_verified = ('Ground truth: /data/zhouy/DATASET/RGB-T234' in (metrics_text or '')) if name.startswith('rgbt234') else None
        completed = (folder / 'job_completed.txt').exists()
        if completed:
            assert inference_receipt['completed'] and not inference_receipt['smoke_only']
            assert inference_receipt['sequences'] == expected_sequences
            assert inference_receipt['frames'] == expected_frames
            assert (folder / 'official_metrics.log').is_file()
            phase = 'needs_actual_gt_scoring' if rgbt_gt_verified is False else 'completed'
        elif live:
            phase = 'official_scoring' if inference_receipt else 'inference'
        else:
            phase = 'stopped_without_completion'
        report['runs'][name] = {
            'tmux_session': session, 'session_live': live, 'phase': phase,
            'completed_sequences': len(records), 'expected_sequences': expected_sequences,
            'completed_sequence_frames': frames, 'expected_frames': expected_frames,
            'completed_sequence_tracking_fps': fps,
            'remaining_inference_minutes_estimate': (expected_frames - frames) / fps / 60 if fps else None,
            'eta_basis': 'completed sequence tracking time; includes unfinished sequence conservatively; excludes first-frame init',
            'last_sequence': records[-1]['sequence'] if records else None,
            'official_metrics_text': metrics_text,
            'rgbt234_actual_ground_truth_verified': rgbt_gt_verified,
        }
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True)
    report['gpu_memory_mib_and_utilization_percent'] = gpu.stdout.splitlines()
    filename = ROOT / f'monitor_{now:%Y%m%d_%H%M%S_UTC}.json'
    filename.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    print(f'MONITOR_RECEIPT {filename}')


if __name__ == '__main__':
    main()
