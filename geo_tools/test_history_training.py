import unittest
import numpy as np
import torch
from train_history_heads import history_clauses,ranking_loss


class Tests(unittest.TestCase):
    def test_only_known_past_other_camera_frames_with_positive_and_negative(self):
        observations=[dict(frame=f,camera=c) for f,c in [(20,1),(5,2),(5,2),(20,2),(21,2),(-1,2),(5,1),(5,2)]]
        labels=np.array([1,1,2,1,1,1,1,-1]);event=dict(query=[0,-1],reference=list(range(1,8)),logits=torch.arange(14.).reshape(2,7))
        clauses=history_clauses(observations,labels,[event]);self.assertEqual(len(clauses),1)
        self.assertEqual(clauses[0]['reference'],[1,2]);self.assertEqual(clauses[0]['base'].tolist(),[0.,1.])
        self.assertEqual(clauses[0]['positive'].tolist(),[True,False])
        self.assertFalse(history_clauses(observations,np.ones(8,dtype=int),[event]))

    def test_uniform_confidence_bonus_cancels_but_relative_evidence_changes_loss(self):
        logits=torch.tensor([.1,.4,-.2],requires_grad=True);positive=torch.tensor([True,False,False])
        loss=ranking_loss(logits,positive);shifted=ranking_loss(logits+2,positive)
        self.assertTrue(torch.allclose(loss,shifted,atol=1e-6))
        loss.backward();self.assertLess(float(logits.grad[0]),0);self.assertGreater(float(logits.grad[1]),0)
        self.assertLess(float(ranking_loss(logits.detach()+torch.tensor([.5,0.,0.]),positive)),float(loss))


if __name__=='__main__':unittest.main()
