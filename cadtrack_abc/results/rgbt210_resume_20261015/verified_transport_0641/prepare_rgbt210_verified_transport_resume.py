from pathlib import Path
import json

private = Path(__file__).parent
boundary = json.loads((private / 'inspect_rgbt210_resume_boundary_remote.stdout.txt').read_text())
inventory = json.loads((private / 'rgbt210_http500_remaining_inventory.json').read_text())['inventory']
assert boundary['bytes'] == 3827263607
row = inventory[26]
assert row['index'] == 26 and row['offset'] <= boundary['boundary_offset']
assert boundary['bytes'] <= row['offset'] + row['expected_bytes']
with (private / 'rgbt210_resume_parts_20261015/part026.bin').open('rb') as stream:
    stream.seek(boundary['boundary_offset'] - row['offset'])
    assert stream.read(4096).hex() == boundary['tail_hex']
assert all((private / 'rgbt210_resume_parts_20261015' / f"part{r['index']:03d}.bin").stat().st_size == r['expected_bytes'] for r in inventory)
receipt = dict(server_start=3827263607, expected_bytes=6857321678,
    remaining_transfer_bytes=3030058071, first_local_part=26,
    exact4096_boundary_bytes_equal=True, original_local_parts=49,
    actual_prior_error='BrokenPipeError at transfer.stdin.write(data); original30429 trueexit1',
    read_only_quote_probe_exit_code=0, primary_stderr_of_original_pipe_unavailable=True,
    original_failure_cause_not_established=True, no_google_redownload=True)
(private / 'rgbt210_verified_transport_resume_plan.json').write_text(json.dumps(receipt, indent=2))
url = json.loads((private / 'rgbt210_download_progress.json').read_text())['url']
receiver = '''from pathlib import Path
import hashlib
import json
import shutil
import sys

archive = Path('/home/gaob/CADTrack-ABC/dataset/RGBT210.rar')
assert archive.stat().st_size == 3827263607
marker = archive.with_name('rgbt210_download_complete.json')
assert not marker.exists()
with archive.open('rb') as stream:
    stream.seek(3827263607 - 4096)
    assert stream.read(4096).hex() == BOUNDARY
with archive.open('ab') as stream:
    shutil.copyfileobj(sys.stdin.buffer, stream)
assert archive.stat().st_size == 6857321678
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
result = dict(status='OFFICIAL_RGBT210_DOWNLOAD_COMPLETE', expected_bytes=6857321678,
    sha256=digest, url=URL, resumed_from_bytes=3827263607,
    original_initial_prefix_bytes=301830263, extraction_complete=False,
    inventory_verified=False, prior_failed_transfers_preserved=True)
marker.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
'''.replace('BOUNDARY', repr(boundary['tail_hex'])).replace('URL', repr(url))
(private / 'receive_rgbt210_verified_remaining.py').write_text(receiver)
print(json.dumps(receipt, indent=2))
