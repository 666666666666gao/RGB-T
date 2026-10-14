from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

private = Path(__file__).parent
plan = json.loads((private / 'rgbt210_verified_transport_resume_plan.json').read_text())
rows = json.loads((private / 'rgbt210_http500_remaining_inventory.json').read_text())['inventory']
assert plan['server_start'] == 3827263607 and plan['remaining_transfer_bytes'] == 3030058071
assert plan['exact4096_boundary_bytes_equal'] and len(rows) == 49
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
    '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028']
started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
written = 0
stdout_path = private / 'rgbt210_remaining_transfer.stdout.txt'
stderr_path = private / 'rgbt210_remaining_transfer.stderr.txt'
with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
    transfer = subprocess.Popen(ssh + ['python3 /home/gaob/CADTrack-ABC/outputs/rgbt210_transport_resume_20261015/receive.py'],
        stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
    for row in rows[26:]:
        path = private / 'rgbt210_resume_parts_20261015' / f"part{row['index']:03d}.bin"
        assert path.stat().st_size == row['expected_bytes']
        offset = max(plan['server_start'] - row['offset'], 0)
        remaining = row['expected_bytes'] - offset
        with path.open('rb') as stream:
            stream.seek(offset)
            while remaining:
                count = min(4 * 1024**2, remaining)
                data = stream.read(count)
                assert len(data) == count
                assert transfer.stdin.write(data) == count
                remaining -= count
                written += count
    assert written == plan['remaining_transfer_bytes']
    transfer.stdin.close()
    code = transfer.wait()
assert code == 0, stderr_path.read_text()
receipt = json.loads(stdout_path.read_text())
assert receipt['status'] == 'OFFICIAL_RGBT210_DOWNLOAD_COMPLETE'
assert receipt['expected_bytes'] == plan['expected_bytes']
receipt.update(client_actual_child_exit_code=code, transferred_bytes=written,
    started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
    seconds=time.perf_counter()-started, local_parts_retained=True)
(private / 'rgbt210_remaining_transfer_complete.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)
