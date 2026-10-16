"""Metric formatting and deployment namespace contracts."""
from pathlib import Path
import tempfile
import unittest
from cir_evaluation import flatten_official,validate_rows,read_mot
from test_cir_native_commit import model


class Contracts(unittest.TestCase):
    def test_duplicate_gt_is_rejected_not_relabelled(self):
        with self.assertRaises(ValueError):validate_rows([[1,5,0,0,10,10],[1,5,2,2,10,10]],'raw')
    def test_official_view_mapping(self):
        gt={'scene_View1':[[1,2,10,20,30,40,-1,-1,-1,-1]],
            'scene_View2':[[2,2,20,30,40,50,-1,-1,-1,-1]]}
        with tempfile.TemporaryDirectory() as tmp:
            jobs=flatten_official({'scene':list(gt)},gt,gt,Path(tmp))
            seq=read_mot(jobs[0]['prediction']);inter=read_mot(jobs[1]['prediction'])
            self.assertEqual([r[0] for r in seq],[1,4])
            self.assertEqual([r[0] for r in inter],[2,5])
            self.assertEqual(seq[0][2:6],[10,20,30,40])
    def test_cpu_contract_does_not_initialize_cuda(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        model(True)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
