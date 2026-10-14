from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import shlex
import shutil
import subprocess
import time
import urllib.request

private = Path(__file__).parent
previous = json.loads((private / 'rgbt210_download_progress.json').read_text())
start, expected = 301830263, 6857321678
assert previous['bytes_transferred'] == start and previous['expected_bytes'] == expected
assert shutil.disk_usage(private).free > expected - start + 1024**3
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
       '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028']
archive_path = '/home/gaob/CADTrack-ABC/dataset/RGBT210.rar'
check = ('from pathlib import Path; p=Path(' + repr(archive_path) + '); '
         'assert p.stat().st_size==' + str(start) + '; '
         'assert not p.with_name("rgbt210_download_complete.json").exists()')
subprocess.run(ssh + ['python3 -'], input=check.encode(), check=True)
parts = private / 'rgbt210_resume_parts_20261015'
parts.mkdir(exist_ok=False)
chunk = 128 * 1024**2
ranges = [(offset, min(offset + chunk, expected) - 1) for offset in range(start, expected, chunk)]
started = time.perf_counter()
started_utc = datetime.now(timezone.utc).isoformat()


def download(index, bounds):
    low, high = bounds
    request = urllib.request.Request(previous['url'], headers={
        'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={low}-{high}'})
    path = parts / f'part{index:03d}.bin'
    count = 0
    with urllib.request.urlopen(request, timeout=60) as response, path.open('xb') as stream:
        assert response.status == 206
        assert response.headers['Content-Range'] == f'bytes {low}-{high}/{expected}'
        assert int(response.headers['Content-Length']) == high - low + 1
        for data in iter(lambda: response.read(4 * 1024**2), b''):
            stream.write(data)
            count += len(data)
    assert count == high - low + 1, (index, count, high - low + 1)
    return dict(index=index, low=low, high=high, bytes=count, file=str(path))


completed = []
with ThreadPoolExecutor(max_workers=4) as executor:
    jobs = [executor.submit(download, index, bounds) for index, bounds in enumerate(ranges)]
    for job in as_completed(jobs):
        completed.append(job.result())
        progress = dict(status='RESUMING_OFFICIAL_RGBT210_BOUNDED_RANGES',
            started_utc=started_utc, observed_utc=datetime.now(timezone.utc).isoformat(),
            completed_parts=completed, total_parts=len(ranges),
            newly_downloaded_bytes=sum(row['bytes'] for row in completed),
            elapsed_seconds=time.perf_counter() - started, expected_bytes=expected,
            original_server_prefix_bytes=start, concurrent_connections=4)
        (private / 'rgbt210_resume_progress.json').write_text(json.dumps(progress, indent=2))
assert len(completed) == len(ranges)
code = ('from pathlib import Path; import sys,shutil,json,hashlib; '
        'p=Path(' + repr(archive_path) + '); assert p.stat().st_size==' + str(start) + '; '
        'f=p.open("ab"); shutil.copyfileobj(sys.stdin.buffer,f); f.close(); '
        'assert p.stat().st_size==' + str(expected) + '; '
        'h=hashlib.file_digest(p.open("rb"),"sha256").hexdigest(); '
        'r=dict(status="OFFICIAL_RGBT210_DOWNLOAD_COMPLETE",expected_bytes=' + str(expected) +
        ',sha256=h,url=' + repr(previous['url']) +
        ',resumed_from_bytes=' + str(start) + ',extraction_complete=False,inventory_verified=False); '
        'p.with_name("rgbt210_download_complete.json").write_text(json.dumps(r,indent=2)); '
        'print(json.dumps(r),flush=True)')
transfer = subprocess.Popen(ssh + ['python3 -c ' + shlex.quote(code)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for row in sorted(completed, key=lambda item: item['index']):
    with Path(row['file']).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            transfer.stdin.write(data)
transfer.stdin.close()
transfer.stdin = None
stdout, stderr = transfer.communicate()
(private / 'rgbt210_resume_transfer.stdout.txt').write_bytes(stdout)
(private / 'rgbt210_resume_transfer.stderr.txt').write_bytes(stderr)
assert transfer.returncode == 0, stderr.decode()
receipt = json.loads(stdout)
assert receipt['status'] == 'OFFICIAL_RGBT210_DOWNLOAD_COMPLETE'
receipt.update(observed_utc=datetime.now(timezone.utc).isoformat(),
               elapsed_seconds=time.perf_counter() - started, completed_parts=len(completed),
               original_failed_download_preserved=True, local_parts_retained=True)
(private / 'rgbt210_resume_complete.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)
