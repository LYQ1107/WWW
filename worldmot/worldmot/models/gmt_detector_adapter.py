"""Frozen current-image GMT candidates, before GTA, memory and identity IDs.

The native environment is Python 3.10 with its own Detectron2 extensions.
``GMTDetectorAdapter`` runs directly there or uses a persistent subprocess
from other environments, without changing either environment's dependencies.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from .person_detector import PersonDetection, sha256_file

GMT_SOURCE = '/data1/liuyeqiang/WWW_cir_gmt_runtime_20261016/sources/p2_full_controls_v2'
GMT_PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
GMT_CHECKPOINT_SHA256 = 'cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8'


def bbox_xyxy_to_float32_xywh(box):
    """Use the immutable NPZ schema's FP32 representation at the boundary."""
    x0,y0,x1,y1=map(float,box)
    return tuple(map(float,np.asarray([x0,y0,x1-x0,y1-y0],dtype=np.float32)))


class GMTDetectorAdapter:
    """PersonDetector-compatible pre-GTA adapter; caller images are uint8 RGB."""

    def __init__(self, source=GMT_SOURCE, device='cuda:0', python=None,
                 score_threshold=None, expose_intermediate_features=True):
        self.source = Path(source).resolve()
        self.device = device
        self.score_threshold = score_threshold
        self.expose_intermediate_features = expose_intermediate_features
        self.last_timings = {}
        self.last_feature_maps = {}
        self.last_roi_features = {}
        self.last_diagnostics = {}
        self.status = 'INITIALIZING'
        self._worker = None
        self._temporary = None
        if python is not None:
            self._start_worker(python)
        else:
            self._init_direct()
        self.status = 'READY'

    def _start_worker(self, python):
        entry = Path(__file__).resolve().parents[2] / 'scripts/gmt_detector_worker.py'
        self._temporary = tempfile.TemporaryDirectory(prefix='worldmot_gmtdet_', dir='/dev/shm')
        self._worker = subprocess.Popen(
            [str(python), '-u', str(entry), '--source', str(self.source), '--device', self.device,
             *([] if self.score_threshold is None else ['--score-threshold', str(self.score_threshold)])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, 'PYTHONPATH': str(entry.parents[1]),
                 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'})
        hello = self._receive()
        if hello.get('status') != 'READY':
            raise RuntimeError('GMT worker failed: ' + repr(hello))
        self._manifest = hello['manifest']

    def _receive(self):
        line = self._worker.stdout.readline()
        if not line:
            raise RuntimeError(f'GMT worker exited with {self._worker.poll()}')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    def _init_direct(self):
        sys.path[:0] = [str(self.source / 'geo_tools'), str(self.source),
                        str(self.source / 'third_party/CenterNet2')]
        import torch
        from common import load_gmt, CHECKPOINT, SOURCE_COMMIT, SOURCE_DIFF_SHA256, SOURCE_HASHES
        from gtr.data.custom_build_augmentation import build_custom_augmentation
        started = time.perf_counter()
        if sha256_file(CHECKPOINT) != GMT_CHECKPOINT_SHA256:
            raise RuntimeError('Frozen GMT checkpoint SHA mismatch')
        with contextlib.redirect_stdout(sys.stderr):
            self.cfg, self.model = load_gmt(20261014, {'ENABLED': False})
        self.model.to(self.device).eval().requires_grad_(False)
        if self.cfg.INPUT.FORMAT != 'RGB' or self.model.roi_heads.delay_cls:
            raise RuntimeError('Expected RGB/no delayed classification in frozen GMT')
        if not self.model.roi_heads.no_box_head:
            raise RuntimeError('This audited adapter requires native NO_BOX_HEAD configuration')
        self.augmentations = build_custom_augmentation(self.cfg, False)
        self.threshold = float(self.model.roi_heads.asso_thresh_test if self.score_threshold is None
                               else self.score_threshold)
        self.forbidden_calls = {}
        # Any accidental tracking invocation becomes a visible engineering failure.
        for name in ('forward', 'sliding_inference_GMT', 'run_global_tracker_plus',
                     'run_first_tracker_plus', 'run_memory_tracker', 'get_asso'):
            if hasattr(self.model, name):
                def reject(*args, _name=name, **kwargs):
                    self.forbidden_calls[_name] = self.forbidden_calls.get(_name, 0) + 1
                    raise RuntimeError('Forbidden GMT tracking call: ' + _name)
                setattr(self.model, name, reject)
        self._manifest = {
            'source': 'gmt_pre_gta_detector', 'checkpoint': str(CHECKPOINT),
            'checkpoint_sha256': GMT_CHECKPOINT_SHA256,
            'config_sha256': hashlib.sha256(self.cfg.dump().encode()).hexdigest(),
            'source_commit': SOURCE_COMMIT, 'source_diff_sha256': SOURCE_DIFF_SHA256,
            'source_file_sha256': SOURCE_HASHES, 'input_color': 'RGB',
            'threshold': self.threshold, 'candidate_rule': 'CenterNet proposals: objectness_logits > threshold',
            'mask_available': False, 'GT_in_inference': False, 'history_or_future_images': False,
            'output_bbox_dtype': 'float32 xywh, identical to immutable cache storage',
            'GTA_called': False, 'roi_feature_source': 'frozen asso_pooler + asso_head, unjittered current boxes',
            'roi_feature_dim': int(self.model.roi_heads.feature_dim),
            'total_backbone_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'torch': torch.__version__, 'precision': 'float32',
            'initialization_seconds': time.perf_counter() - started,
        }

    def manifest(self):
        return {**self._manifest, 'status': self.status, 'device': self.device,
                'worker_process': self._worker.pid if self._worker else None}

    def infer(self, images_by_camera):
        started = time.perf_counter()
        images = {int(v): np.asarray(x) for v, x in images_by_camera.items()}
        if any(x.ndim != 3 or x.shape[2] != 3 or x.dtype != np.uint8 for x in images.values()):
            raise ValueError('GMT detector expects uint8 HxWx3 RGB arrays')
        if self._worker is not None:
            files = {}
            for view, image in sorted(images.items()):
                path = Path(self._temporary.name) / f'camera_{view}.npy'
                np.save(path, image, allow_pickle=False)
                files[str(view)] = str(path)
            self._worker.stdin.write(json.dumps({'images': files}) + '\n')
            self._worker.stdin.flush()
            reply = self._receive()
            result = {}
            self.last_roi_features = {}
            for view, rows in reply['detections'].items():
                result[int(view)] = []
                for row in rows:
                    det = PersonDetection(int(view), int(row['detection_id']), tuple(row['bbox_xywh']),
                                          float(row['score']), source='gmt_pre_gta_detector')
                    det.detector_roi_feature = np.asarray(row['roi_feature'], np.float32)
                    self.last_roi_features[int(view), det.detection_id] = det.detector_roi_feature
                    result[int(view)].append(det)
            self.last_diagnostics = reply['diagnostics']
            self.last_timings = {**reply['timings'], 'ipc_infer_wall_s': time.perf_counter() - started}
            return result
        return self._infer_direct(images, started)

    def _infer_direct(self, images, started):
        import torch
        from detectron2.data import transforms as T
        from detectron2.structures import Boxes, Instances
        from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
        self.last_timings = {}
        self.last_diagnostics = {}
        self.last_roi_features = {}
        result = {}
        for view, original in sorted(images.items()):
            aug = T.StandardAugInput(original)
            transforms = aug.apply_augmentations(self.augmentations)
            resized = aug.image
            h, w = original.shape[:2]
            rh, rw = resized.shape[:2]
            if any(getattr(t, 'offset_x', 0) or getattr(t, 'offset_y', 0) for t in transforms.transforms):
                raise RuntimeError('Unexpected test crop; audited full-view mapping invalid')
            inputs = [{'image': torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1))),
                       'height': h, 'width': w}]
            if self.device.startswith('cuda'):
                torch.cuda.synchronize(self.device)
            begin = time.perf_counter()
            with torch.inference_mode():
                tensors = self.model.preprocess_image(inputs)
                features = self.model.backbone(tensors.tensor)
                proposals, _ = self.model.proposal_generator(tensors, features, None)
                p = proposals[0]
                if p.has('track_ids'):
                    raise RuntimeError('Track ID reached the raw detector')
                p = p[p.objectness_logits > self.threshold]
                boxes = p.proposal_boxes.tensor.clone()
                # Independent frozen image descriptor. No spatial/camera token,
                # transformer, track bank or random ROI jitter is involved.
                head = self.model.roi_heads
                maps = [features[f] for f in head.asso_in_features]
                roi = head.asso_head(head.asso_pooler(maps, [p.proposal_boxes]))
                inst = Instances((rh, rw))
                inst.pred_boxes = Boxes(boxes)
                inst.scores = p.objectness_logits
                inst.pred_classes = torch.zeros(len(p), dtype=torch.long, device=boxes.device)
                # Preserve native descriptor indices if postprocessing removes
                # a zero-area box. This is an index, never an identity ID.
                inst.raw_candidate_index = torch.arange(len(p), device=boxes.device)
                out = CustomRCNN._postprocess([inst], inputs, [(rh, rw)],
                                              not_clamp_box=self.model.not_clamp_box)[0]['instances']
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - begin
                xyxy = out.pred_boxes.tensor.cpu().numpy()
                scores = out.scores.cpu().numpy()
                indices = out.raw_candidate_index.cpu().numpy()
                rois = roi.cpu().numpy()
            result[view] = []
            for index, (box, score, raw_index) in enumerate(zip(xyxy, scores, indices)):
                det = PersonDetection(view, index, bbox_xyxy_to_float32_xywh(box), float(score),
                                      source='gmt_pre_gta_detector')
                det.detector_roi_feature = np.asarray(rois[int(raw_index)], np.float32)
                self.last_roi_features[view, index] = det.detector_roi_feature
                result[view].append(det)
            self.last_timings[f'camera_{view}_detector_s'] = elapsed
            self.last_diagnostics[str(view)] = {
                'candidate_count': len(p), 'output_count': len(out),
                'roi_shape': list(rois.shape), 'original_hw': [h, w],
                'network_hw': [rh, rw], 'padded_hw': list(tensors.tensor.shape[-2:]),
                'native_postprocess_scale_xy': [w/rw, h/rh],
                'class_ids': [0], 'GT_in_inference': False,
            }
        self.last_timings['infer_wall_s'] = time.perf_counter() - started
        self.last_diagnostics['forbidden_calls'] = dict(self.forbidden_calls)
        return result

    def close(self):
        if self._worker is not None and self._worker.poll() is None:
            self._worker.stdin.write(json.dumps({'close': True}) + '\n')
            self._worker.stdin.flush()
            self._worker.wait(timeout=30)
        if self._temporary:
            self._temporary.cleanup()


def detection_rows(detections):
    return {str(view): [{**d.to_dict(), 'roi_feature': d.detector_roi_feature.tolist()}
                       for d in ds] for view, ds in sorted(detections.items())}
