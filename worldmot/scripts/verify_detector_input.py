#!/usr/bin/env python3
"""Real RGB adapter versus official MMDetection filename inference."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from mmdet.apis import inference_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dataset import Compose
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.person_detector import PersonDetector
from cache_fulltrain import DEFAULT_DETECTOR, DEFAULT_WEIGHTS, atomic_json


def selected(result, detector):
    pred = result.pred_instances
    scores = pred.scores.detach().cpu().numpy()
    indices = [int(i) for i in np.argsort(-scores)[:detector.max_detections]
               if int(pred.labels[i]) == 0 and scores[i] >= detector.score_threshold]
    boxes = pred.bboxes.detach().cpu().numpy()[indices].copy()
    boxes[:, 2:] -= boxes[:, :2]
    masks = pred.masks.detach().cpu().numpy()[indices]
    return boxes, scores[indices], masks


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.set_num_threads(2)
    detector = PersonDetector(DEFAULT_DETECTOR, DEFAULT_WEIGHTS, a.device)
    if detector.status != 'READY':raise RuntimeError(detector.manifest())
    # This MMDetection version mutates its shallow-copied ndarray loader config.
    # Preserve the official filename pipeline before the first ndarray call.
    file_pipeline = Compose(copy.deepcopy(get_test_pipeline_cfg(detector.model.cfg)))
    ds = VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack', annotation=a.metadata,
                     split='train', load_annotations=False)
    checks = []
    for sid in [8, 9, 24]:
        pair = next(iter(ds.synchronized_pairs(sid)))
        for view, rec in sorted(pair.images.items()):
            path = ds.image_path(rec)
            rgb = np.asarray(Image.open(path).convert('RGB'))
            actual = detector.infer({view: rgb})[view]
            boxes, scores, masks = selected(inference_detector(detector.model, str(path), test_pipeline=file_pipeline), detector)
            actual_boxes = np.asarray([d.bbox_xywh for d in actual]).reshape(-1, 4)
            actual_scores = np.asarray([d.score for d in actual])
            if actual_boxes.shape != boxes.shape:raise RuntimeError('RGB adapter and official file input detection counts differ')
            box_error = float(np.max(np.abs(actual_boxes-boxes))) if len(boxes) else 0.
            score_error = float(np.max(np.abs(actual_scores-scores))) if len(scores) else 0.
            mask_equal = all(np.array_equal(d.mask, mask) for d, mask in zip(actual, masks))
            if box_error > 1e-3 or score_error > 1e-5 or not mask_equal:
                raise RuntimeError('RGB adapter does not reproduce official file inference')
            wrong_boxes, wrong_scores, _ = selected(inference_detector(detector.model, rgb), detector)
            wrong_same = wrong_boxes.shape == boxes.shape and bool(np.allclose(wrong_boxes, boxes, atol=1e-3))
            checks.append({'scene': sid, 'view': view, 'timestamp': pair.frame_id,
                'image': str(path), 'detections': len(actual),
                'official_file_max_box_error_px': box_error,
                'official_file_max_score_error': score_error,
                'official_file_masks_identical': mask_equal,
                'old_wrong_channel_detections': len(wrong_scores),
                'old_wrong_channel_boxes_match': wrong_same})
    source = Path(inspect.getfile(inference_detector))
    report = {'status': 'PASS_OFFICIAL_FILENAME_EQUIVALENCE', 'checks': checks,
        'detector': detector.manifest(), 'official_inference_source': str(source),
        'official_inference_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'GT_used': False, 'old_channel_cache_valid_for_training': False}
    atomic_json(a.output, report)
    print(json.dumps(report, indent=2))
