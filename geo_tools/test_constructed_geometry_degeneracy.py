"""Constructed CPU stress inputs; no claim about natural UAV failure causes."""
import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT, save_json, provenance
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from gtr.config import add_gtr_config
from gtr.geometry.runtime import GeometryRuntime
from gtr.geometry.update_signals import CausalBackgroundMotion
from gtr.geometry.temporal_geometry import TemporalGeometryConsumer
from gtr.geometry.sparse_scheduler import SparseScheduler


def tiny_detection():
    instance = Instances((180, 320))
    instance.pred_boxes = Boxes(torch.tensor([[100., 80., 101., 81.]]))
    instance.reid_features = torch.ones(1, 8)
    instance.scores = torch.ones(1)
    return instance


class Tests(unittest.TestCase):
    observations = []
    def test_textureless_and_unrelated_textures_leave_actual_runtime_logits_exact(self):
        rng = np.random.default_rng(20261014)
        blank = np.zeros((180, 320), np.uint8)
        backgrounds = [cv2.GaussianBlur(rng.integers(0, 256, blank.shape, dtype=np.uint8), (5, 5), 0) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f'view{view}.png' for view in (1, 2)]
            records = [dict(view_id=view, frame_id=1, width=320, height=180, file_name=str(path)) for view, path in zip((1, 2), paths)]
            for case, images in [('textureless', [blank, blank]), ('unrelated_procedural_textures', backgrounds)]:
                for path, image in zip(paths, images):
                    self.assertTrue(cv2.imwrite(str(path), image))
                for method in ('orb', 'sift'):
                    with self.subTest(case=case, method=method):
                        cfg = get_cfg()
                        add_gtr_config(cfg)
                        geo = cfg.MODEL.GEO
                        geo.ENABLED = True
                        geo.FEATURE_ENABLED = False
                        geo.GEOMETRY_ENABLED = True
                        geo.FEATURE_WEIGHT = 0.
                        geo.GEOMETRY_WEIGHT = .5
                        geo.GEOMETRY_METHOD = method
                        runtime = GeometryRuntime(geo)
                        refs, query = tiny_detection(), tiny_detection()
                        runtime.observe([refs, query], records, 0)
                        self.assertEqual(runtime.failed, 0)
                        self.assertEqual(runtime.vggt_calls + runtime.dinov2_calls, 0)
                        self.assertFalse(runtime.geometry[0]['valid'])
                        original = [torch.tensor([[.375]])]
                        self.assertIs(runtime.adapt(original, [refs, query], 1), original)
                        observation = dict(case=case, method=method, valid=runtime.geometry[0]['valid'],
                            matches=runtime.geometry[0]['matches'], inliers=runtime.geometry[0]['inliers'],
                            exact_original_logits=True, encoder_calls=0)
                        self.observations.append(observation)
                        print(json.dumps(observation), flush=True)

    def test_large_constructed_rotation_cannot_make_unvalidated_past_geometry_usable(self):
        rng = np.random.default_rng(20261014)
        original = cv2.GaussianBlur(rng.integers(0, 256, (180, 320), dtype=np.uint8), (5, 5), 0)
        rotated = cv2.warpAffine(original, cv2.getRotationMatrix2D((160, 90), 60., 1.), (320, 180))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frame.png'
            motion = CausalBackgroundMotion()
            records = [dict(view_id=view, file_name=str(path)) for view in (1, 2)]
            boxes = [torch.empty(0, 4), torch.empty(0, 4)]
            self.assertTrue(cv2.imwrite(str(path), original))
            motion.observe(records, boxes)
            self.assertTrue(cv2.imwrite(str(path), rotated))
            value, details = motion.observe(records, boxes)
            consumer = TemporalGeometryConsumer(5)
            F = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            self.assertTrue(consumer.receive(dict(source_frame=0, completed_at=1., result=dict(F=F, valid=True, camera_order=(1, 2))), 0, now=2.))
            self.assertIsNotNone(consumer.candidate(0, now=2.))
            consumer.add_motion(1, details)
            candidate = consumer.candidate(1, now=2.)
            self.assertTrue(candidate is None or not candidate['valid'])
            if not all(row['valid'] for row in details):
                self.assertIsNone(candidate)
            motion_trigger = SparseScheduler(policy='motion').should_update(1, motion=value)
            observation = dict(case='constructed_60_degree_in_plane_rotation', motion=value,
                motion_valid=[row['valid'] for row in details], old_geometry_usable_without_current_validation=False,
                temporal_candidate_returned=candidate is not None, temporal_consumer_stats=consumer.stats,
                motion_only_update_requested=motion_trigger,
                scope='CPU optical flow and temporal consumer; no VGGT or physical UAV pose claim')
            self.observations.append(observation)
            print(json.dumps(observation), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report-output', type=Path)
    args, remaining = parser.parse_known_args()
    if args.report_output and args.report_output.exists():
        raise FileExistsError(args.report_output)
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.report_output:
        save_json(args.report_output, dict(status='PASS_CONSTRUCTED_CPU_GEOMETRY_STRESS', tests_run=result.testsRun,
            observations=Tests.observations, seed=20261014, encoder_calls=0, model_strict_load_performed=False,
            natural_dataset_failure_causes_identified=False, full_tracker_inference_performed=False, speed_eligible=False,
            scope='actual CPU OpenCV/background geometry, GeometryRuntime score fallback and temporal consumer on fixed constructed inputs; detector instances and initial past F are synthetic',
            limitations='independent procedural textures are not a calibrated physical camera pair; in-plane rotation is not real UAV pose/rapid-rotation evidence; invalid motion returns0 and can miss the motion-only trigger, not proof of no motion; no VGGT encoder, full GMT or natural-data efficacy claim', provenance=provenance()))
