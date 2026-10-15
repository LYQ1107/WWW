import unittest
import torch
from dev12_native_schema import CommitDictionary, activate, replay_call, validate_past, tensor_digest, digest_json

class Tests(unittest.TestCase):
    def test_joint_solver_competition_and_strict_threshold(self):
        call=dict(queries=[[1,0],[1,1]],track_ids=[4,9],track_scores=torch.tensor([[.9,.8],[.85,.1]]),thresholds=torch.tensor([.4,.8]))
        pairs,accepted=replay_call(call)
        self.assertEqual([(p['row'],p['track']) for p in pairs],[(0,9),(1,4)])
        self.assertEqual(accepted,[-1,4])
    def test_null_softmax_per_actual_block(self):
        values=activate(torch.zeros((2,3)),[1,2])
        self.assertTrue(torch.equal(values[:,0],torch.full((2,),.5)))
        self.assertTrue(torch.allclose(values[:,1:],torch.full((2,2),1/3)))
        with self.assertRaises(ValueError):activate(torch.zeros((2,3)),[2])
    def test_strict_causal_processed_camera_only(self):
        q=dict(frame=3);r=dict(frame=3,source=[42,0])
        with self.assertRaises(ValueError):validate_past(q,r,set())
        validate_past(q,r,{42})
        with self.assertRaises(ValueError):validate_past(q,dict(frame=4,source=[1,0]),{1})
    def test_passive_commit_preserves_object_and_fingerprint(self):
        seen=[];d=CommitDictionary(lambda k,v:seen.append((k,v)));obj=object();d[1]=obj
        self.assertIs(d[1],obj);self.assertIs(seen[0][1],obj)
        self.assertEqual(digest_json({'a':1,'b':2}),digest_json({'b':2,'a':1}))
    def test_precision_and_nonfinite_rejected(self):
        x=torch.tensor([1.0001]);self.assertNotEqual(tensor_digest(x),tensor_digest(x.half()))
        with self.assertRaises(ValueError):tensor_digest(torch.tensor([float('nan')]))
    def test_empty_gallery_still_retains_every_query(self):
        c=dict(queries=[[1,0]],track_ids=[],track_scores=torch.empty(1,0),thresholds=torch.empty(0))
        self.assertEqual(replay_call(c),([],[-1]))

if __name__=='__main__':unittest.main()
