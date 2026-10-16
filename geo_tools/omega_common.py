"""Small, CPU-safe provenance helpers for the isolated Omega study."""
import datetime
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/data1/liuyeqiang/WWW_omega_runtime_20261016')
NATIVE = Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1')
DATA = Path('/data/DATASETS/TRACKING/JDE/VisionTrack')
BASE = 'c6999a7ab1d22549816b5219734f53c8dcb74178'
FIT = (4, 5, 6, 7, 8, 9, 13, 17, 20, 21, 24)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def provenance():
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                source_hashes={str(p.relative_to(ROOT)): sha256(p)
                               for pattern in ['geo_tools/omega_*.py', 'gtr/geometry/omega_*.py']
                               for p in ROOT.glob(pattern)})


def verified_native(scene):
    directory = NATIVE / f'scene{scene}_full_v1'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not manifest['full_video'] or not manifest['continuous_from_first_frame']:
        raise ValueError('native source is not a continuous full-video audit')
    if manifest['seed'] != 20261014 or manifest['video_id'] != scene:
        raise ValueError('native source identity mismatch')
    for name, expected in manifest['artifacts'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'changed sealed native artifact: {directory / name}')
    return directory, dict(path=str(directory / 'manifest.json'), sha256=sha256(directory / 'manifest.json'),
                           artifacts=manifest['artifacts'])
