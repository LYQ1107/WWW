"""Shared actions, solver externalities, complete clones and temporal error tests."""
import json,math,unittest
from collections import Counter
import torch
from jev_phase7_common import *

class Contracts(unittest.TestCase):
    def test_actual_solver_can_move_an_accept(self):
        from gtr.modeling.jev_assignment import constrained_hungarian
        scores=torch.tensor([[.9,.8],[.85,.1]])
        # Reject row0->col1 from the original optimum; row1's accepted col0 moves.
        original=dict(constrained_hungarian(scores,set()))
        self.assertEqual(original,{0:1,1:0})
        changed=dict(constrained_hungarian(scores,{(0,1)}))
        self.assertEqual(changed,{0:0,1:1});self.assertNotEqual(original[1],changed[1])
    def test_same_legal_support_and_probability_invariants(self):
        from jev_phase7_policies import MatchActor,A,R,N
        from run_jev_phase7_normalized import NormalizedActor
        capture=torch.load(next((OUT/'paired_snapshots/video09').glob('*.pth')),map_location='cpu')
        record=next(r for r in capture['mechanism']['first_decisions']if r['action']==R)
        feature=torch.tensor(record['feature_vector'])
        for actor in [MatchActor(n)for n in ['FIXED','DYNAMIC','MLP','BINARY','B2']]+[NormalizedActor('MLP')]:
            for legal in ([A,R,N],[A,N]):
                d=actor.decide(feature,'MATCH_DECISION',legal,off_action=A,context=record['context'])
                self.assertIn(d.committed_action,legal);self.assertEqual(set(d.probabilities),set(legal))
                self.assertTrue(all(math.isfinite(p)and 0<=p<=1 for p in d.probabilities.values()))
                self.assertAlmostEqual(sum(d.probabilities.values()),1,places=5)
    def test_complete_clone_independence_and_digest_fields(self):
        from jev_phase7_state import CompleteStateDigest
        capture=torch.load(next((OUT/'paired_snapshots/video09').glob('*.pth')),map_location='cpu')
        state=capture['state'];digest=CompleteStateDigest();start=digest(state,capture['metadata']);clone=state.clone()
        self.assertEqual(start,digest(clone,capture['metadata']))
        clone.counters['probe']=1;self.assertNotEqual(start,digest(clone,capture['metadata']))
        self.assertEqual(start,digest(state,capture['metadata']))
        clone=state.clone();track=next(iter(clone.memory));clone.memory[track][0].add_(.01)
        self.assertNotEqual(start,digest(clone,capture['metadata']));self.assertEqual(start,digest(state,capture['metadata']))
        clone=state.clone();clone.assignments['probe']=999
        self.assertNotEqual(start,digest(clone,capture['metadata']))
    def test_true_temporal_error_and_gap_censoring(self):
        from jev_phase7_offline import IdentityEvaluator
        evaluator=IdentityEvaluator.__new__(IdentityEvaluator)
        evaluator.denominator=Counter({1:2,2:5});evaluator.gt_views={1:{0},2:{0}}
        rows={(0,0,0):{'id':1,'gt':1},(1,0,0):{'id':1,'gt':1},
              (2,0,0):{'id':1,'gt':2},(3,0,0):{'id':1,'gt':2},
              (7,0,0):{'id':1,'gt':2},(8,0,0):{'id':2,'gt':2},(9,0,0):{'id':2,'gt':2},
              (10,0,0):{'id':3,'gt':None}}
        summary=evaluator.summarize(rows)
        self.assertEqual(summary['wrong_ID_duration_total_camera_frames'],3)
        self.assertEqual([e['duration_frames']for e in summary['wrong_ID_episodes']],[2,1])
        self.assertEqual([e['right_censored']for e in summary['wrong_ID_episodes']],[True,False])
        self.assertEqual(summary['unknown_GT_observations'],1)

if __name__=='__main__':
    result=unittest.main(exit=False)
    save(REPORTS/'CONTRACT_TESTS.json',{'status':'PASS'if result.result.wasSuccessful()else 'FAIL',
        'tests':result.result.testsRun,'failures':len(result.result.failures),'errors':len(result.result.errors),'binding':binding()})
    if not result.result.wasSuccessful():raise SystemExit(1)
