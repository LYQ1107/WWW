import unittest,time,numpy as np,torch
from common import ROOT
from gtr.geometry.pair_score_adapter import PairScoreAdapter
from gtr.geometry.sparse_scheduler import SparseScheduler
from gtr.geometry.geometry_state import GeometryState
from gtr.geometry.stream_worker import GeometryWorker

class Tests(unittest.TestCase):
    def test_off_zero_and_invalid(self):
        l=torch.randn(4,7);r=torch.randn(4,7)
        self.assertIs(PairScoreAdapter().apply(l,r,r),l)
        self.assertIs(PairScoreAdapter(True).apply(l,r,r),l)
        x=PairScoreAdapter(True,1,0).apply(l,torch.full_like(l,float('nan')))
        self.assertTrue(torch.equal(l,x))
        mask=torch.zeros_like(l,dtype=torch.bool);mask[0,1]=True
        result=PairScoreAdapter(True,4,0,.5).apply(l,r,None,mask)
        self.assertTrue(torch.equal(result[~mask],l[~mask]));self.assertLessEqual(float((result-l).abs().max()),.500001)
    def test_symmetric_feature_order(self):
        g=torch.Generator().manual_seed(42)
        a,b,c,d=[torch.nn.functional.normalize(torch.randn(6,8,generator=g),dim=-1) for _ in range(4)]
        first=torch.cat([a,c],dim=-1)/2**.5;second=torch.cat([b,d],dim=-1)/2**.5
        score=second@first.T
        reversed_restored=(torch.cat([d,b],dim=-1)/2**.5)@(torch.cat([c,a],dim=-1)/2**.5).T
        self.assertTrue(torch.allclose(score,reversed_restored,atol=1e-6))

    def test_prefix_and_budget(self):
        def run(n):
            s=SparseScheduler(budget=3,window=20)
            return [s.should_update(t,motion=(.1 if t%7==0 else 0),margin=(.03 if t%5==0 else 1)) for t in range(n)]
        a,b=run(100),run(150);self.assertEqual(a,b[:100])
        for t in range(100):self.assertLessEqual(sum(a[max(0,t-19):t+1]),3)
    def test_stale_future_nonfinite(self):
        s=GeometryState(10,100.,np.eye(3),True,1.)
        self.assertFalse(s.usable(9,101));self.assertFalse(s.usable(11,99));self.assertFalse(s.usable(31,101));self.assertTrue(s.usable(11,101))
        s.F=np.full((3,3),np.inf);self.assertFalse(s.usable(11,101))
    def test_bounded_worker(self):
        def compute(x):time.sleep(.03);return dict(F=np.eye(3),valid=True,confidence=1.)
        w=GeometryWorker(compute)
        try:
            self.assertTrue(w.submit(0,None));self.assertFalse(w.submit(1,None));self.assertIsNone(w.poll(0));time.sleep(.05)
            state=w.poll(1);self.assertIsNotNone(state);self.assertTrue(state.usable(1,time.monotonic()))
        finally:w.close()
        def fail(x):raise RuntimeError('simulated OOM')
        w=GeometryWorker(fail)
        try:w.submit(0,None);time.sleep(.03);self.assertIsNone(w.poll(1));self.assertEqual(w.stats['failed'],1)
        finally:w.close()
if __name__=='__main__':unittest.main()
