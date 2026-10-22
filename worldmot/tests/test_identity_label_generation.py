from types import SimpleNamespace

from worldmot.training.identity_targets import assign_relative_identity_classes, match_detections_to_gt


def test_identity_targets_are_scene_qualified_and_clip_classes_are_relative():
    detections = {1: [SimpleNamespace(bbox_xywh=[0, 0, 10, 10])]}
    ground_truth = {1: [{"bbox": [0, 0, 10, 10], "instance_id": 23}]}
    target = match_detections_to_gt(detections, ground_truth, scene_id=8)[(1, 0)]
    assert target.identity_key == "scene8:id23"
    mapped = assign_relative_identity_classes([target], seed=11)[0]
    assert mapped.identity_class == 0

