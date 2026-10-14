import unittest, torch
from common import ROOT
from test_empty_tracking import detections, model_stub
from gtr.geometry.association_trace import association_effect, AssociationTrace

class Tests(unittest.TestCase):
    def test_original_per_frame_activation_and_assignment(self):
        model = model_stub()
        refs = [detections(1), detections(1)]
        refs[0].track_ids = torch.tensor([10.])
        refs[1].track_ids = torch.tensor([20.])
        instances = refs + [detections(1)]
        before, after = torch.tensor([[3., -3.]]), torch.tensor([[-3., 3.]])
        result = association_effect(before, after, instances, 2, model.roi_heads, .5, False)
        self.assertEqual(result['changed_logit_pairs'], 2)
        self.assertEqual(result['counterfactual_assignment_changes'], 1)
        self.assertAlmostEqual(result['max_abs_gta_delta'], float(torch.sigmoid(torch.tensor(3.)) - torch.sigmoid(torch.tensor(-3.))), places=6)

    def test_read_only_and_restore(self):
        model = model_stub()
        model.geo_runtime.adapt = lambda outputs, *args: outputs
        original = model.geo_runtime.adapt
        trace = AssociationTrace(model)
        trace.install()
        history, query = detections(1), detections(1)
        history.track_ids = torch.tensor([1.])
        outputs = [torch.tensor([[2.]])]
        before = outputs[0].clone();rng = torch.get_rng_state().clone()
        self.assertIs(model.geo_runtime.adapt(outputs, [history, query], 1), outputs)
        self.assertTrue(torch.equal(before, outputs[0]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(trace.records[0]['changed_logit_pairs'], 0)
        trace.close()
        self.assertIs(model.geo_runtime.adapt, original)

if __name__ == '__main__':
    unittest.main()
