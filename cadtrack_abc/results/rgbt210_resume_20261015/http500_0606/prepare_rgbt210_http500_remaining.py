from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

private = Path(__file__).parent
original = json.loads((private / 'rgbt210_partial_parts_inventory.json').read_text())
rows = []
for row in original['inventory']:
    path = private / 'rgbt210_resume_parts_20261015' / f"part{row['index']:03d}.bin"
    size = path.stat().st_size
    assert 0 <= size <= row['expected_bytes']
    rows.append(dict(row, actual_bytes=size))
reused = sum(row['actual_bytes'] for row in rows)
missing = sum(row['expected_bytes'] - row['actual_bytes'] for row in rows)
assert reused == 5469467623 and missing == 1086023792
assert shutil.disk_usage(private).free > missing + 1024**3
inventory = dict(observed_utc=datetime.now(timezone.utc).isoformat(), inventory=rows,
    reused_bytes=reused, missing_bytes=missing, free_bytes=shutil.disk_usage(private).free,
    prior_original_session=53052, prior_actual_exit_code=1,
    prior_primary_error='urllib.error.HTTPError: HTTP Error 500: Internal Server Error at urlopen',
    prior_source_preserved='finish_rgbt210_partial_parts_20261015.py')
(private / 'rgbt210_http500_remaining_inventory.json').write_text(json.dumps(inventory, indent=2))
source = (private / 'finish_rgbt210_partial_parts_20261015.py').read_text()
source = source.replace('import urllib.request', 'import urllib.request\nfrom urllib.error import HTTPError')
source = source.replace('rgbt210_partial_parts_inventory.json', 'rgbt210_http500_remaining_inventory.json')
source = source.replace('missing == 5579932557', 'missing == 1086023792')
source = source.replace('requests, short_reads = 0, 0', 'requests, short_reads, http500 = 0, 0, 0')
old = '''            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {low}-{high}/{expected}'
                assert int(response.headers['Content-Length']) == wanted
                body = response.read(wanted)
            requests += 1'''
new = '''            requests += 1
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
                continue'''
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace('new_requests=requests, short_read_attempts=short_reads)',
    'new_requests=requests, short_read_attempts=short_reads, http500_retries=http500)')
source = source.replace('reused_local_bytes=975558858', 'reused_local_bytes=5469467623')
source = source.replace('maximum_short_read_attempts_per_range=3)',
    'maximum_total_attempts_per_range=3, retry_http_status=500)')
source = source.replace('rgbt210_finish_', 'rgbt210_http500_finish_')
source = source.replace('original_two_failures_preserved=True', 'original_three_failures_preserved=True')
(private / 'complete_rgbt210_http500_remaining_20261015.py').write_text(source, encoding='utf-8')
print(json.dumps({key: value for key, value in inventory.items() if key != 'inventory'}, indent=2))
