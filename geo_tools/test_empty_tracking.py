"""Actual GMT GTA and causal runner contracts across empty detector groups."""
import types, unittest, torch
from common import ROOT
from detectron2.structures import Instances, Boxes
from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN
from gtr.modeling.roi_heads.gtr_roi_heads import GTRROIHeads
from gtr.geometry.online_runner import CausalGMT

def detections(count):
    inst = Instances((20, 30))
    inst.pred_boxes = Boxes(torch.tensor([[3., 2., 9., 8.]]).repeat(count, 1))
    inst.reid_features = torch.ones(count, 8)
    inst.scores = torch.ones(count)
    inst.pred_classes = torch.zeros(count, dtype=torch.long)
    return inst

def model_stub():
    def forbidden(*args, **kwargs):
        raise AssertionError('empty attention must be bypassed')
    roi = types.SimpleNamespace(delay_cls=False, _forward_transformer=forbidden)
    roi._get_boxes_time = types.MethodType(GTRROIHeads._get_boxes_time, roi)
    roi._activate_asso = types.MethodType(GTRROIHeads._activate_asso, roi)
    model = types.SimpleNamespace(roi_heads=roi, test_len=2, with_bank=False,
                                 overlap_thresh=.5, not_mult_thresh=False, not_clamp_box=False,
                                 geo_runtime=types.SimpleNamespace(reset=lambda: None, observe=lambda *args: None))
    for name in ['get_asso', 'run_first_tracker_plus', 'run_global_tracker_plus']:
        setattr(model, name, types.MethodType(getattr(GTRRCNN, name), model))
    return model

class Tests(unittest.TestCase):
    def test_empty_shapes_boxes_and_rng(self):
        model = model_stub()
        for history_count, query_count in [(0, 0), (0, 2), (2, 0)]:
            history, query = detections(history_count), detections(query_count)
            history.track_ids = torch.arange(history_count)
            before = torch.get_rng_state().clone()
            outputs, boxes, counts, npast, indices = model.get_asso([history, query], 1)
            self.assertEqual(outputs[0].shape, (query_count, history_count))
            self.assertEqual(boxes.shape, (history_count + query_count, 4))
            self.assertEqual(counts, [history_count, query_count])
            self.assertEqual(npast, history_count)
            self.assertEqual(indices, list(range(history_count, history_count + query_count)))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            if len(boxes):
                self.assertTrue(torch.allclose(boxes[0], torch.tensor([.1, .1, .3, .4])))

    def test_causal_empty_first_groups_and_reappearance(self):
        model = model_stub()
        model.inference = lambda batch, *args, **kwargs: [detections(batch[0]['count'])]
        stream = CausalGMT(model)
        for timestamp, counts in enumerate([(0, 0), (0, 2), (0, 0), (0, 0), (0, 1)], 1):
            records = [dict(frame_id=timestamp, count=n, height=20, width=30) for n in counts]
            output = stream.step(records)
            self.assertEqual([len(x['instances']) for x in output], list(counts))
            if timestamp == 2:
                self.assertEqual(output[1]['instances'].track_ids.tolist(), [1., 2.])
            if timestamp == 5:
                self.assertEqual(output[1]['instances'].track_ids.tolist(), [3.])
        self.assertEqual(stream.id_count, 3)

if __name__ == '__main__':
    unittest.main()
