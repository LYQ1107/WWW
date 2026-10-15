"""Research boundary tests: partial-fit refusal, UNKNOWN and global competition."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from cv_pil_v2_model import CandidatePredictor
from pirga_indexed_store import sha
from run_pirga_phase1_replay import load_epoch_heads,certified_outcomes,primary_solve


class ReplayContractTests(unittest.TestCase):
    def test_epoch_checkpoint_provenance_and_finite_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol={'admission':{'sha256':'a'*64}}
            pins={'epoch':1,'training_protocol':{'sha256':'p'*64},'heads':{}}
            states={}
            for arm in ('C2','C3','C4'):
                path=Path(directory)/(arm+'.pth')
                states[arm]=dict(arm=arm,epoch=1,step=113415,seed=20261015,protocol_sha256='p'*64,
                    admission_sha256='a'*64,state_dict=CandidatePredictor().state_dict())
                torch.save(states[arm],path);pins['heads'][arm]={'path':str(path),'sha256':sha(path)}
            heads=load_epoch_heads(pins,protocol)
            self.assertFalse(heads['C4'].use_camera)
            for field,value in [('epoch',None),('step',300),('arm','C3'),('seed',1),
                                ('protocol_sha256','x'),('admission_sha256','x')]:
                with self.subTest(field=field):
                    state=copy.deepcopy(states['C2']);state[field]=value
                    path=Path(pins['heads']['C2']['path']);torch.save(state,path)
                    pins['heads']['C2']['sha256']=sha(path)
                    with self.assertRaisesRegex(ValueError,'source/budget'):load_epoch_heads(pins,protocol)
            path=Path(pins['heads']['C2']['path']);state=copy.deepcopy(states['C2'])
            state['state_dict']['up.bias'][0]=float('nan');torch.save(state,path)
            pins['heads']['C2']['sha256']=sha(path)
            with self.assertRaisesRegex(ValueError,'parameter'):load_epoch_heads(pins,protocol)
            pins['heads']['C2']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):load_epoch_heads(pins,protocol)

    def test_unknown_never_certified_and_identity_aliases_are_multi_positive(self):
        ids=[101,102,103];scores=torch.tensor([.8,.7,.6])
        unknown=dict(GT_person=None,certified_target_indices=[],candidate_known_mask=[True,True,False])
        for accepted in (101,103,None):
            self.assertEqual(certified_outcomes(scores,ids,accepted,unknown),(101,None,None,None))
        known=dict(GT_person=7,certified_target_indices=[0,1],candidate_known_mask=[True,True,False])
        self.assertIs(certified_outcomes(scores,ids,102,known)[3],True)
        self.assertIsNone(certified_outcomes(scores,ids,103,known)[3])
        self.assertIs(certified_outcomes(scores,ids,None,known)[3],False)

    def test_complete_joint_assignment_and_strict_native_threshold(self):
        scores=torch.tensor([[.8,.7],[.79,.1]])
        self.assertEqual(primary_solve(scores,[101,102],[.1,.1]),[102,101])
        self.assertEqual(primary_solve(scores,[101,102],[scores[1,0],scores[0,1]]),[None,None])
        self.assertEqual(primary_solve(torch.empty(2,0),[],[]),[None,None])

if __name__=='__main__':unittest.main()
