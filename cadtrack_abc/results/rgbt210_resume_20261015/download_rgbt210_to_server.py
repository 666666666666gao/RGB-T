from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time
import urllib.request

private = Path(__file__).parent
url = 'https://drive.usercontent.google.com/download?id=0B3i2rdXLNbdUTkhsLVRwcTBTMlU&export=download&confirm=t'
expected = 6857321678
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
       '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028']
check = 'from pathlib import Path; import shutil; p=Path("/home/gaob/CADTrack-ABC/dataset"); assert p.is_dir(); assert shutil.disk_usage(p).free>40*1024**3; assert not (p/"RGBT210.rar").exists()'
subprocess.run(ssh + ['python3 -'], input=check.encode(), check=True)
request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
started = time.perf_counter()
started_utc = datetime.now(timezone.utc).isoformat()
digest = hashlib.sha256()
count = 0
with urllib.request.urlopen(request, timeout=60) as response:
    assert int(response.headers['Content-Length']) == expected
    assert response.headers['Content-Type'] == 'application/octet-stream'
    assert 'RGBT210.rar' in response.headers['Content-Disposition']
    process = subprocess.Popen(ssh + ['cat > /home/gaob/CADTrack-ABC/dataset/RGBT210.rar'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for chunk in iter(lambda: response.read(4 * 1024**2), b''):
        if count == 0:
            assert chunk.startswith(b'Rar!\x1a\x07\x00')
        process.stdin.write(chunk)
        digest.update(chunk)
        count += len(chunk)
        state = dict(status='DOWNLOADING_OFFICIAL_RGBT210_TO_SERVER', started_utc=started_utc,
                     observed_utc=datetime.now(timezone.utc).isoformat(), bytes_transferred=count,
                     expected_bytes=expected, elapsed_seconds=time.perf_counter()-started,
                     url=url, destination='/home/gaob/CADTrack-ABC/dataset/RGBT210.rar')
        (private / 'rgbt210_download_progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    process.stdin.close()
    process.stdin = None
    stdout, stderr = process.communicate()
    assert process.returncode == 0, stderr.decode()
assert count == expected
state.update(status='OFFICIAL_RGBT210_DOWNLOAD_COMPLETE', sha256=digest.hexdigest(),
             observed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
             extraction_complete=False, inventory_verified=False)
(private / 'rgbt210_download_complete.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
code = ('from pathlib import Path; import json,hashlib; p=Path("/home/gaob/CADTrack-ABC/dataset/RGBT210.rar"); '
        'assert p.stat().st_size==' + str(expected) + '; h=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(); '
        'assert h==' + repr(digest.hexdigest()) + '; Path("/home/gaob/CADTrack-ABC/dataset/rgbt210_download_complete.json").write_text(' + repr(json.dumps(state, indent=2)) + ')')
subprocess.run(ssh + ['python3 -'], input=code.encode(), check=True)
print(json.dumps(state, indent=2), flush=True)
