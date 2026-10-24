"""Reporting must reject partial or changed evidence instead of inventing scores."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_gmt_point_test import match_boxes
from build_gmt_point_comparison import throughput
from gmt_point_report_common import BLOCKED, Route, new_routes, sha, validate_route


def test_primary_throughput_uses_wall_time_not_arithmetic_scene_average():
    rows = [{'frames': 10, 'real_e2e_seconds': 1, 'real_e2e_fps': 10},
            {'frames': 90, 'real_e2e_seconds': 90, 'real_e2e_fps': 1}]
    result = throughput(rows)
    assert result['frame_weighted_fps'] == pytest.approx(100 / 91)
    assert result['mean_scene_fps'] == 5.5


def test_cancelled_seed24_cannot_enter_new_completed_route_table():
    routes = new_routes(Path('/unused'))
    assert len(routes) == 8
    assert all('seed20261024' not in r.key for r in routes)
    assert all('2d_controls_v2' in str(r.predictions) for r in routes[:2])


def test_box_level_diagnostic_is_one_to_one_and_does_not_change_boxes():
    detections = [{'bbox_xywh': [0, 0, 10, 20]}, {'bbox_xywh': [0, 0, 10, 20]},
                  {'bbox_xywh': [100, 100, 10, 20]}]
    gt = [{'bbox': [0, 0, 10, 20]}, {'bbox': [200, 200, 10, 20]}]
    assert match_boxes(detections, gt) == {1: 0}
    assert match_boxes([], gt) == {}
    assert detections[0]['bbox_xywh'] == [0, 0, 10, 20]


def make_route(tmp_path):
    pred = tmp_path / 'predictions'
    path = tmp_path / 'metrics.json'
    route = Route('test', 'test', 'test', pred, path, 'gate.json')
    coverage = []
    for sid in range(1, 23):
        frames = 28998 if sid == 1 else 1
        manifest = dict(status='COMPLETE', full_video=True, GT_in_inference=False,
                        scene=sid, frames=frames, prediction_sha256=f'hash{sid}')
        p = pred / f'scene_{sid:02d}/manifest.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(manifest))
        coverage.append(dict(scene=sid, frames=frames, predictions_sha256=f'hash{sid}'))
    metrics = dict(status='COMPLETE_VALID_SCOPE', split='test', raw_GT_modified=False,
                   custom_metrics_used=False, box_source='tracks_2d_by_camera', coverage=coverage,
                   evaluated_scenes=list(range(14)), blocked=[dict(scene=sid) for sid in BLOCKED])
    path.write_text(json.dumps(metrics))
    (tmp_path / 'gate.json').write_text(json.dumps(dict(TEST_used_for_selection=False,
                                                      evaluation=dict(metrics_sha256=sha(path)))))
    return route


def test_report_accepts_complete_evidence_then_rejects_changed_metrics(tmp_path):
    route = make_route(tmp_path)
    _, manifests = validate_route(route, tmp_path)
    assert len(manifests) == 22
    route.metrics.write_text(route.metrics.read_text() + '\n')
    with pytest.raises(ValueError, match='metrics changed'):
        validate_route(route, tmp_path)


def test_report_rejects_manifest_from_another_prediction_attempt(tmp_path):
    route = make_route(tmp_path)
    path = route.predictions / 'scene_01/manifest.json'
    data = json.loads(path.read_text())
    data['prediction_sha256'] = 'other_attempt'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='manifest/evaluator mismatch'):
        validate_route(route, tmp_path)
