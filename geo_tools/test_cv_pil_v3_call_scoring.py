"""Require bitwise score parity; this is an engineering optimization only."""
import unittest
import torch

from test_cv_pil_v3_replay import fixture
from cv_pil_v2_model import CandidatePredictor,tensorize,fused_scores
from cv_pil_v3_call_scoring import call_scores


class TestCallScoring(unittest.TestCase):
    def test_bitwise_parity_on_homogeneous_and_mixed_calls(self):
        torch.manual_seed(7001)
        models={arm:CandidatePredictor(use_camera=arm!='C4').eval() for arm in ('C2','C3','C4')}
        with torch.no_grad():
            for m in models.values():m.up.weight.normal_(0,.03);m.up.bias.normal_(0,.02)
            for homogeneous in (True,False):
                a,c=fixture()
                if homogeneous:
                    old=a.observations.pop((201,1));old.update(source=[200,1],image_id=200,frame=10,camera=2)
                    a.observations[(200,1)]=old;c['queries'][1]=[200,1]
                    c['query_feature_keys'][1]['source']=[200,1]
                    a.query_index={(12,200,0):(0,0),(12,200,1):(0,1)}
                for o in a.observations.values():o['feature']=torch.randn(1152)
                fast=call_scores(a,c,models)
                for row,source in enumerate(c['queries']):
                    b=tensorize(a.actor((12,*source)))
                    slow=dict(C0=b.native_scores,C1=fused_scores(b,b.histories))
                    slow.update({arm:m(b)[0] for arm,m in models.items()})
                    for arm,value in slow.items():
                        with self.subTest(homogeneous=homogeneous,row=row,arm=arm):
                            self.assertTrue(torch.equal(value,fast[arm][row]))


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
