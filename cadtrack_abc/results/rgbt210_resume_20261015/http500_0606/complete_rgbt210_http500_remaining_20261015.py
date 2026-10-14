from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import shlex
import shutil
import subprocess
import time
import urllib.request
from urllib.error import HTTPError

private = Path(__file__).parent
inventory = json.loads((private / 'rgbt210_http500_remaining_inventory.json').read_text())['inventory']
previous = json.loads((private / 'rgbt210_download_progress.json').read_text())
start, expected = 301830263, 6857321678
assert len(inventory) == 49 and sum(row['expected_bytes'] for row in inventory) == expected - start
missing = sum(row['expected_bytes'] - row['actual_bytes'] for row in inventory)
assert missing == 1086023792 and shutil.disk_usage(private).free > missing + 1024**3
parts = private / 'rgbt210_resume_parts_20261015'
for row in inventory:
    path = parts / f"part{row['index']:03d}.bin"
    assert (path.stat().st_size if path.exists() else 0) == row['actual_bytes']
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
       '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028']
archive_path = '/home/gaob/CADTrack-ABC/dataset/RGBT210.rar'
check = ('from pathlib import Path; p=Path(' + repr(archive_path) + '); '
    'assert p.stat().st_size==' + str(start) + '; '
    'assert not p.with_name("rgbt210_download_complete.json").exists()')
subprocess.run(ssh + ['python3 -'], input=check.encode(), check=True)
started = time.perf_counter()
started_utc = datetime.now(timezone.utc).isoformat()


def finish(row):
    path = parts / f"part{row['index']:03d}.bin"
    if not path.exists():
        path.touch(exist_ok=False)
    size = row['actual_bytes']
    requests, short_reads, http500 = 0, 0, 0
    while size < row['expected_bytes']:
        low = row['offset'] + size
        high = min(low + 8 * 1024**2, row['offset'] + row['expected_bytes']) - 1
        wanted = high - low + 1
        for attempt in range(3):
            request = urllib.request.Request(previous['url'], headers={
                'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={low}-{high}'})
            requests += 1
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {low}-{high}/{expected}'
                    assert int(response.headers['Content-Length']) == wanted
                    body = response.read(wanted)
            except HTTPError as error:
                if error.code != 500 or attempt == 2:
                    raise
                http500 += 1
                time.sleep(2)
                continue
            if len(body) == wanted:
                break
            short_reads += 1
        assert len(body) == wanted, (row['index'], low, wanted, len(body), requests)
        with path.open('ab') as stream:
            stream.write(body)
        size += len(body)
    assert path.stat().st_size == row['expected_bytes']
    return dict(index=row['index'], file=str(path), bytes=size,
        reused_bytes=row['actual_bytes'], new_requests=requests, short_read_attempts=short_reads, http500_retries=http500)


completed = []
with ThreadPoolExecutor(max_workers=2) as executor:
    jobs = [executor.submit(finish, row) for row in inventory]
    for job in as_completed(jobs):
        completed.append(job.result())
        progress = dict(status='FINISHING_ACTUAL_RGBT210_PARTIAL_GAPS',
            started_utc=started_utc, observed_utc=datetime.now(timezone.utc).isoformat(),
            completed_parts=completed, total_parts=49,
            newly_downloaded_bytes=sum(row['bytes'] - row['reused_bytes'] for row in completed),
            reused_local_bytes=5469467623, missing_bytes_at_start=missing,
            elapsed_seconds=time.perf_counter() - started, concurrent_connections=2,
            request_limit_bytes=8 * 1024**2, maximum_total_attempts_per_range=3, retry_http_status=500)
        (private / 'rgbt210_http500_finish_progress.json').write_text(json.dumps(progress, indent=2))
assert len(completed) == 49 and sum(row['bytes'] for row in completed) == expected - start
code = ('from pathlib import Path; import sys,shutil,json,hashlib; '
    'p=Path(' + repr(archive_path) + '); assert p.stat().st_size==' + str(start) + '; '
    'f=p.open("ab"); shutil.copyfileobj(sys.stdin.buffer,f); f.close(); '
    'assert p.stat().st_size==' + str(expected) + '; '
    'h=hashlib.file_digest(p.open("rb"),"sha256").hexdigest(); '
    'r=dict(status="OFFICIAL_RGBT210_DOWNLOAD_COMPLETE",expected_bytes=' + str(expected) +
    ',sha256=h,url=' + repr(previous['url']) + ',resumed_from_bytes=' + str(start) +
    ',extraction_complete=False,inventory_verified=False); '
    'p.with_name("rgbt210_download_complete.json").write_text(json.dumps(r,indent=2)); print(json.dumps(r),flush=True)')
transfer = subprocess.Popen(ssh + ['python3 -c ' + shlex.quote(code)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for row in sorted(completed, key=lambda item: item['index']):
    with Path(row['file']).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            transfer.stdin.write(data)
transfer.stdin.close()
transfer.stdin = None
stdout, stderr = transfer.communicate()
(private / 'rgbt210_http500_finish_transfer.stdout.txt').write_bytes(stdout)
(private / 'rgbt210_http500_finish_transfer.stderr.txt').write_bytes(stderr)
assert transfer.returncode == 0, stderr.decode()
receipt = json.loads(stdout)
receipt.update(observed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
    original_three_failures_preserved=True, local_parts_retained=True)
(private / 'rgbt210_http500_finish_complete.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)
