"""Fetch the missing core dataset into this project's own home filesystem."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    project = Path('/home/gaob/CADTrack-ABC')
    destination = project / 'dataset'
    destination.mkdir(exist_ok=True)
    assert shutil.disk_usage(destination).free > 40 * 1024**3
    archive = destination / 'rgbt234.tar.gz'
    url = 'https://hf-mirror.com/datasets/xche32/rgbt234/resolve/main/rgbt234.tar.gz'
    subprocess.run(['curl', '--location', '--fail', '--connect-timeout', '30', '--max-time', '21600',
                    '--output', str(archive), url], check=True)
    assert archive.stat().st_size == 7665568241
    checksum = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    assert digest == '137f45fa9860e2ce7c5f882142703c6176ffce604cfcc1cebced3364b9cbc1ee'
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert member.isfile() or member.isdir()
        stream.extractall(destination, members=members)
    directories = [destination] + [p for p in destination.iterdir() if p.is_dir()]
    roots = []
    for root in directories:
        sequences = [p for p in root.iterdir() if p.is_dir()]
        if len(sequences) == 234 and all((p/'visible').is_dir() and (p/'infrared').is_dir() and (p/'visible.txt').is_file() and (p/'infrared.txt').is_file() for p in sequences):
            roots.append(root)
    assert len(roots) == 1, roots
    import numpy as np
    frames = 0
    for sequence in roots[0].iterdir():
        if not sequence.is_dir(): continue
        rgb, tir = sorted((sequence/'visible').iterdir()), sorted((sequence/'infrared').iterdir())
        visible = np.loadtxt(sequence/'visible.txt', delimiter=',', ndmin=2)
        infrared = np.loadtxt(sequence/'infrared.txt', delimiter=',', ndmin=2)
        assert len(rgb) == len(tir) == len(visible) == len(infrared), sequence.name
        frames += len(rgb)
    config_path = project/'cadtrack_abc/config.json'
    config = json.loads(config_path.read_text())
    config['rgbt234_root'] = str(roots[0])
    config['rgbt234_verified_frames'] = frames
    config_path.write_text(json.dumps(config, indent=2))
    (destination/'rgbt234_complete.json').write_text(json.dumps({'completed':True,'root':str(roots[0]),'sequences':234,'frames':frames,'archive_bytes':archive.stat().st_size,'sha256':digest,'url':url}, indent=2))
    print(json.dumps({'completed':True,'root':str(roots[0]),'sequences':234,'frames':frames}), flush=True)


if __name__ == '__main__':
    main()
