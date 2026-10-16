import unittest
import torch
import common
from gtr.modeling.identity_revision.native_evidence import native_conflicts
from gtr.modeling.identity_revision.conflict_detector import FixedConflictDetector
from gtr.modeling.identity_revision.types import ConflictState


class NativeEvidenceContracts(unittest.TestCase):
    def inputs(self):
        q=torch.zeros(1,1152);q[:,0]=1
        h=torch.zeros(6,1152);h[:3,0]=1;h[3:,1]=1
        return [q,torch.tensor([[0.,0,10,10]]),4,1,h,
            torch.tensor([[0.,0,10,10]]*3+[[20.,20,30,30]]*3),
            torch.tensor([2,2,2,1,1,1]),torch.tensor([1,2,3,1,2,3]),torch.tensor([1,1,1,2,2,2]),
            torch.tensor([1,2]),torch.tensor([[2.5,2.49]]),torch.tensor([1])]
    def test_extract_joint_local_conflict(self):
        e,x=native_conflicts(*self.inputs())
        self.assertEqual(e[0].local_id,2);self.assertEqual(e[0].local_consecutive_frames,3)
        self.assertEqual(FixedConflictDetector().classify(e[0])[0],ConflictState.SUSPECT)
        self.assertEqual(tuple(x.shape),(1,12));self.assertTrue(torch.isfinite(x).all())
    def test_descriptor_version_does_not_certify_raw_local_chain(self):
        physical=torch.tensor([False,True,True,True,True,True])
        e,_=native_conflicts(*self.inputs(),history_is_physical=physical)
        self.assertEqual(e[0].local_consecutive_frames,2)
        self.assertEqual(FixedConflictDetector().classify(e[0])[0],ConflictState.UNKNOWN)
    def test_future_rejected(self):
        args=self.inputs();args[7][0]=5
        with self.assertRaises(ValueError):native_conflicts(*args)
    def test_full1152_required(self):
        args=self.inputs();args[0]=args[0][:,:1024]
        with self.assertRaises(ValueError):native_conflicts(*args)
    def test_empty_query_contract(self):
        args=self.inputs();args[0]=args[0][:0];args[1]=args[1][:0];args[10]=args[10][:0];args[11]=args[11][:0]
        e,x=native_conflicts(*args);self.assertEqual(e,[]);self.assertEqual(tuple(x.shape),(0,12))
    def test_rejected_proposal_is_unknown(self):
        args=self.inputs();args[-1][0]=-1
        e,_=native_conflicts(*args)
        self.assertEqual(FixedConflictDetector().classify(e[0])[0],ConflictState.UNKNOWN)


if __name__=='__main__':unittest.main()
