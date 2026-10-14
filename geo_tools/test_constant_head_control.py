import unittest
from unittest.mock import patch
import torch
import run_online
import run_constant_head_control as control
from gtr.geometry.lightweight_head import LowRankIdentityHead


class Tests(unittest.TestCase):
    def test_validation_preserves_weights_rng_and_restores_methods_on_failure(self):
        model=LowRankIdentityHead(2);state={key:value.clone() for key,value in model.state_dict().items()}
        original=LowRankIdentityHead.evidence;rng=torch.get_rng_state().clone()
        def interrupted():
            evidence=model.evidence(torch.tensor([[1.,0.]]),torch.tensor([[1.,0.],[0.,1.]]))
            self.assertTrue(torch.equal(evidence,torch.ones(1,2)))
            self.assertEqual(run_online.provenance()['validation_constant_head_evidence'],1.)
            self.assertIn('--trace-association',control.sys.argv)
            raise RuntimeError('fixture interruption')
        with patch.object(run_online,'main',interrupted),patch.object(run_online,'provenance',lambda:{}),patch.object(control.sys,'argv',['control','--head-path','fixture']):
            with self.assertRaisesRegex(RuntimeError,'fixture interruption'):control.main()
        self.assertIs(LowRankIdentityHead.evidence,original)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,state[key]))


if __name__=='__main__':unittest.main()
