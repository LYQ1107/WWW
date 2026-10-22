"""Compact source and immutable-cache provenance for the registered experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_stamp():
    paths = sorted((ROOT / 'worldmot').rglob('*.py'))
    paths += [ROOT / 'scripts' / name for name in (
        'train_world_query.py', 'select_query_model.py', 'run_fullvideo.py',
        'run_gmt_test_baseline.py', 'evaluate_fullvideo.py', 'official_worldmot_metrics.m',
        'experiment_provenance.py', 'preserve_inference_attempts.py')]
    files = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    encoded = json.dumps(files, sort_keys=True).encode()
    return {'files': files, 'sha256': hashlib.sha256(encoded).hexdigest()}


def cache_stamp(roots, scenes):
    manifests = {}
    for root in roots:
        for scene in sorted(scenes):
            p = Path(root) / f'scene_{scene:02d}' / 'manifest.json'
            if p.is_file():
                data = json.loads(p.read_text())
                if data.get('status') != 'COMPLETE':
                    raise RuntimeError('training cache is incomplete: ' + str(p))
                manifests[str(scene)] = {'path': str(p), 'sha256': digest(p),
                    'schema': data['schema'], 'timestamps': data['timestamps']}
    return manifests
