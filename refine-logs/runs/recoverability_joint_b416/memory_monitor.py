"""Sample real allocated-device occupancy; never read power or temperature."""
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

stage = sys.argv[1]
assert stage in ('m0', 'full')
setup = Path('/data/gb/setup')
launch = json.loads((setup / ('joint_b416_' + stage + '_launch_20261004.json')).read_text())
rows = launch['runs']
out = setup / ('joint_b416_' + stage + '_memory_20261004.csv')
with out.open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(('observed_at_cst', 'gpu', 'memory_used_mib', 'memory_total_mib', 'gpu_utilization_percent'))
    while True:
        stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        for values in csv.reader(result.stdout.splitlines()):
            writer.writerow((stamp, *[int(value.strip()) for value in values]))
        stream.flush()
        if all((Path(row['root']) / 'job_completed.txt').is_file() for row in rows):
            break
        states = [subprocess.run(['ps', '-o', 'stat=', '-p', str(row['runner_pid'])],
                                 capture_output=True, text=True).stdout.strip() for row in rows]
        if any((not state or state.startswith('Z')) and not (Path(row['root']) / 'job_completed.txt').is_file()
               for row, state in zip(rows, states)):
            break
        time.sleep(20)
print(str(out))
