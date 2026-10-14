"""Causal exclusion, bounded clips and stream isolation contracts."""
import random
import unittest
import numpy as np
import torch
from common import ROOT
from gtr.geometry.retrieval_index import CausalClipIndex
from gtr.geometry.isolated_stream import IsolatedCausalStream
from gtr.geometry.runtime import GeometryRuntime
from gtr.geometry.memory_audit import retained_payload_bytes
from detectron2.config import get_cfg
from gtr.config import add_gtr_config
from test_empty_tracking import model_stub, detections


class Tests(unittest.TestCase):
    def test_memory_audit_deduplicates_aliases_and_views(self):
        tensor = torch.zeros(100, dtype=torch.float32)
        array = np.zeros(100, dtype=np.float32)
        result = retained_payload_bytes([tensor, tensor[:3], array, array[4:]])
        self.assertEqual(result['cpu_tensor_bytes'], 400)
        self.assertEqual(result['numpy_bytes'], 400)
        self.assertEqual(result['total_bytes'], 800)

    def test_other_camera_scene_namespace_and_no_future_backfill(self):
        index = CausalClipIndex(clip_groups=3, ttl_groups=4)
        a = index.publish(1, 1, 1, 0, 1, 1, [0, 0, 10, 10], [1, 0])
        b = index.publish(1, 2, 1, 0, 2, 1, [0, 0, 10, 10], [1, 0])
        c = index.publish(2, 1, 1, 0, 3, 1, [0, 0, 10, 10], [0, 1])
        short = index.query(1, 1, 0, [1, 0])
        self.assertEqual([r['clip_id'] for r in short['results']], [b, c])
        snapshot = index.snapshot()
        index.publish(1, 2, 1, 1, 4, 2, [0, 0, 10, 10], [0, 1])
        self.assertEqual(len(snapshot[1]['observations']), 1)
        self.assertEqual(short['results'][0]['score'], 1.)
        with self.assertRaises(ValueError):
            index.query(1, 1, 0, [1, 0])
        self.assertEqual(index.query(1, 1, 6, [1, 0])['candidates'], 0)

    def test_missing_stale_extra_neutral_and_memory_eviction(self):
        index = CausalClipIndex(feature_age=1, max_bytes=1000)
        index.publish(1, 2, 1, 0, 1, 1, [0, 0, 10, 10], [1, 0], [0, 1])
        neutral = index.query(1, 1, 0, [1, 0], feature_weight=2)
        fresh = index.query(1, 1, 0, [1, 0], [0, 1], feature_weight=2)
        self.assertGreater(fresh['results'][0]['score'], neutral['results'][0]['score'])
        stale = index.query(1, 1, 2, [1, 0], [0, 1], feature_weight=2)
        self.assertEqual(stale['results'][0]['score'], neutral['results'][0]['score'])
        for i in range(20):
            index.publish(2, 1, i, 2, i+2, 3, [0, 0, 10, 10], [1, 0])
        self.assertLessEqual(index.bytes, index.max_bytes)
        self.assertGreater(index.evictions, 0)

    def test_interleaving_preserves_each_stream_rng_and_track_output(self):
        def run(interleaved):
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            model = model_stub()
            cfg = get_cfg();add_gtr_config(cfg)
            model.geo_runtime = GeometryRuntime(cfg.MODEL.GEO)
            def inference(*args, **kwargs):
                instance = detections(2)
                # Exercise every RNG used by GMT, independently per stream.
                instance.reid_features = torch.rand(2, 8)+random.random()+np.random.rand()
                return [instance]
            model.inference = inference
            def transformer(instances, reid, *args):
                k = args[1]
                refs = torch.cat([x.reid_features for i, x in enumerate(instances) if i != k])
                logits = instances[k].reid_features @ refs.T
                boxes, _ = model.roi_heads._get_boxes_time(instances)
                return [logits], boxes, None, None, None, None
            model.roi_heads._forward_transformer = transformer
            streams = [IsolatedCausalStream(model) for _ in range(2)]
            output = [[], []]
            order = [(s, t) for t in range(1, 5) for s in range(2)] if interleaved else [(s, t) for s in range(2) for t in range(1, 5)]
            for s, t in order:
                records = [dict(frame_id=t, view_id=v, height=100, width=100) for v in [1, 2]]
                rows = streams[s].step(records)
                output[s].append([(r['instances'].track_ids.tolist(), r['instances'].reid_features.tolist()) for r in rows])
            return output
        self.assertEqual(run(False), run(True))


if __name__ == '__main__':
    unittest.main()
