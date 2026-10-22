#!/usr/bin/env python3
"""Read-only coordinate diagnostics; transform changes are not GT drift errors."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from cache_fulltrain import atomic_json, sha


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return dict(zip(('min', 'p50', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .5, .95, 1]))))


def coordinate_summary(rows, positions):
    scales = [r['reference']['scale'] for r in rows
              if r['alignment_success'] and r.get('reference')]
    normalization = [r['normalization']['scale'] for r in rows
                     if r.get('normalization')]
    reference = [r['reference'] for r in rows
                 if r['alignment_success'] and r.get('reference')]
    rotation_errors = [np.linalg.norm(np.asarray(r['rotation']).T @
                                     np.asarray(r['rotation']) - np.eye(3))
                       for r in reference]
    radius = np.linalg.norm(np.concatenate(positions), axis=1) if positions else []
    return {
        'timestamps': len(rows),
        'alignment_success_timestamps': sum(r['alignment_success'] for r in rows),
        'alignment_reasons': dict(collections.Counter(r['alignment_reason'] for r in rows)),
        'accepted_reference_scale': quantiles(scales),
        'accepted_reference_scale_first_last': [float(scales[0]), float(scales[-1])] if scales else [],
        'rotation_orthogonality_error': quantiles(rotation_errors),
        'normalization_scale': quantiles(normalization),
        'normalization_changes': sum(a != b for a, b in zip(normalization, normalization[1:])),
        'valid_normalized_observation_radius': quantiles(radius),
        'is_metric_reconstruction_accuracy': False,
        'is_independent_static_background_drift_measurement': False,
        'interpretation': 'Sim(3) scale changes may reflect reconstruction scale, scene motion or alignment drift; no metric GT or fixed-background anchor is available in these cache statistics.'
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    scenes = []
    for directory in sorted(Path(a.cache).glob('scene_*')):
        manifest_path = directory / 'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'COMPLETE':
            continue
        frames, positions, fingerprints = [], [], []
        for entry in manifest['shards']:
            path = directory / entry['file']
            if sha(path) != entry['sha256']:
                raise RuntimeError('cache checksum mismatch: ' + str(path))
            frames.extend(json.loads(path.with_suffix('.json').read_text())['per_frame'])
            with np.load(path, allow_pickle=False) as d:
                positions.append(d['geometry'][d['geometry_valid'], :3])
            fingerprints.append(entry['sha256'])
        if len(frames) != manifest['timestamps']:
            raise RuntimeError('coordinate audit timeline mismatch')
        scenes.append({'scene': manifest['scene'], 'manifest_sha256': sha(manifest_path),
                       'shard_sha256': fingerprints, **coordinate_summary(frames, positions)})
    report = {'status': 'COMPLETE_SCENES_COORDINATE_DIAGNOSTICS', 'split': 'train',
              'scene_count': len(scenes), 'timestamps': sum(s['timestamps'] for s in scenes),
              'full_TRAIN_coverage': len(scenes) == 24 and sum(s['timestamps'] for s in scenes) == 28754, 'TEST_read': False,
              'audit_source_sha256': sha(__file__),
              'model_or_threshold_changes': False, 'scenes': scenes}
    atomic_json(a.output, report)
    print(json.dumps(report, indent=2))
