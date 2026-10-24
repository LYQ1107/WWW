"""Read-only registry and validation for the frozen GMT/point TEST routes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

NAS = Path('/data1/liuyeqiang/nas/A100/WorldMOT')
LEGAL = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 18, 19)
BLOCKED = (5, 12, 13, 14, 17, 20, 21, 22)
FIELDS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'CVIDF1', 'MOTA', 'IDSW', 'Frag', 'CVMA')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


@dataclass(frozen=True)
class Route:
    key: str
    name: str
    association: str
    predictions: Path
    metrics: Path
    gate: str


def new_routes(nas=NAS):
    det = nas / 'gmt_detector_comparison_20261023'
    point = nas / 'vggt_point_association_20261023'
    control = nas / 'final_comparison_20261023/2d_controls_v2'
    routes = [Route('iou_motion', 'GMTDet + 2D IoU/Motion', 'IoU/Motion',
                    control / 'test/iou_motion', control / 'test_evaluation/iou_motion/metrics.json',
                    'gmt_2d_iou_motion_test_gate_20261024.json'),
              Route('appearance_motion', 'GMTDet + 2D Appearance/Motion', 'ROI/Motion',
                    control / 'test/appearance_motion', control / 'test_evaluation/appearance_motion/metrics.json',
                    'gmt_2d_appearance_motion_test_gate_20261024.json'),
              Route('gmt_v1', 'GMTDet + V1', 'Fixed V1 rule/Hungarian',
                    det / 'test_gmt_v1_v2', det / 'evaluation_gmt_v1_v2/metrics.json',
                    'gmt_v1_test_gate_complete_20261023.json')]
    for key in ('joint_seed20261022', 'joint_seed20261023', 'no_geometry_seed20261022'):
        association = 'PWQT_NO_GEOMETRY' if key.startswith('no_geometry') else 'PWQT'
        routes.append(Route(key, 'GMTDet + V2 ' + key.replace('_seed', ' seed').replace('no_geometry', 'no-geometry'),
                            association, det / 'test_gmt_v2' / key,
                            det / 'test_gmt_v2_evaluation' / key / 'metrics.json',
                            f'gmt_v2_{key}_test_gate_20261024.json'))
    for key in ('temporal', 'joint'):
        routes.append(Route('point_' + key, 'GMTDet + VGGT ' + key.title(),
                            'Single-view points' if key == 'temporal' else 'Multi-view points',
                            point / 'test_v1' / key, point / 'test_evaluation_v1' / key / 'metrics.json',
                            f'gmt_vggt_{key}_test_gate_20261024.json'))
    return routes


def validate_route(route, results):
    """Validate raw evaluator, frozen Gate hash and full inference manifests."""
    metrics = read_json(route.metrics)
    if (metrics.get('status') != 'COMPLETE_VALID_SCOPE' or metrics.get('split') != 'test'
            or metrics.get('raw_GT_modified') is not False
            or metrics.get('custom_metrics_used') is not False
            or metrics.get('box_source') != 'tracks_2d_by_camera'):
        raise ValueError(f'Invalid official protocol: {route.metrics}')
    gate = read_json(Path(results) / route.gate)
    expected = gate.get('evaluation', {}).get('metrics_sha256')
    if expected is None:
        expected = gate.get('provenance', {}).get('raw_metrics_sha256')
    if not expected or sha(route.metrics) != expected:
        raise ValueError(f'Official metrics changed after Gate: {route.key}')
    if gate.get('TEST_used_for_selection', gate.get('test_used_for_selection')) is not False:
        raise ValueError(f'Missing TRAIN-only selection evidence: {route.key}')
    coverage = {int(row['scene']): row for row in metrics['coverage']}
    if set(coverage) != set(range(1, 23)) or sum(x['frames'] for x in coverage.values()) != 29019:
        raise ValueError(f'Incomplete official inference coverage: {route.key}')
    if len(metrics['evaluated_scenes']) != 14 or {r['scene'] for r in metrics['blocked']} != set(BLOCKED):
        raise ValueError(f'Different legal GT scope: {route.key}')
    manifests = {}
    for sid in range(1, 23):
        path = route.predictions / f'scene_{sid:02d}/manifest.json'
        data = read_json(path)
        if (data.get('status') != 'COMPLETE' or data.get('full_video') is not True
                or data.get('GT_in_inference') is not False
                or int(data['scene']) != sid or data['frames'] != coverage[sid]['frames']
                or data['prediction_sha256'] != coverage[sid]['predictions_sha256']):
            raise ValueError(f'Prediction manifest/evaluator mismatch: {path}')
        manifests[sid] = data
    return metrics, manifests
