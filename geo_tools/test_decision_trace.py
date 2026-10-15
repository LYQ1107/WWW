import unittest
import common
import gzip
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from scipy.optimize import linear_sum_assignment
from gtr.geometry.decision_trace import solver_record,DecisionTrace
from audit_decision_trace import decision_state


class Tests(unittest.TestCase):
    def test_actual_solver_assignment_and_track_aggregation_threshold(self):
        context=dict(stage='global',queries=[[2,0],[2,1]],references=[dict(source=[1,0],track=4),dict(source=[1,1],track=4),dict(source=[1,2],track=9)])
        cost=-torch.tensor([[.8,.1],[.2,.6]])
        assignment=linear_sum_assignment(cost)
        record=solver_record(cost,assignment,context,.4,False)
        self.assertEqual(record['accepted_track_ids'],[-1,9]) # .8 equals 2*.4: strict rejection
        self.assertEqual(record['hungarian_pairs'][0]['track'],4)
        self.assertEqual(solver_record(cost,assignment,context,.4,True)['accepted_track_ids'],[4,9])

    def test_empty_context_and_bad_matrix_fail_closed(self):
        context=dict(stage='global',queries=[[2,0]],references=[])
        cost=torch.empty((1,0));record=solver_record(cost,linear_sum_assignment(cost),context,.4,False)
        self.assertEqual(record['accepted_track_ids'],[-1])
        with self.assertRaises(ValueError):solver_record(torch.zeros((1,1)),([],[]),context,.4,False)

    def test_wrapper_retains_actual_solver_objects_and_model_return(self):
        from detectron2.structures import Instances
        history=Instances((10,10));history.track_ids=torch.tensor([4]);history.decision_trace_source=torch.tensor([[1,0]])
        query=Instances((10,10));query.decision_trace_source=torch.tensor([[2,0]])
        module=SimpleNamespace(linear_sum_assignment=linear_sum_assignment);calls=[]
        def solver(cost):
            result=linear_sum_assignment(cost);calls.append(result);return result
        module.linear_sum_assignment=solver
        model=SimpleNamespace(overlap_thresh=.4,not_mult_thresh=True,thred_bank=.8,inference=lambda:None)
        result=([history,query],4,{},{})
        def first(instances,outputs,boxes,k,*args):
            assignment=module.linear_sum_assignment(-torch.tensor([[.9]]))
            self.assertIs(assignment,calls[-1]);instances[k].track_ids=torch.tensor([4]);return result
        model.run_first_tracker_plus=first;model.run_global_tracker_plus=lambda:None;model.run_memory_tracker=lambda:None
        state=torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            trace=DecisionTrace(model,Path(directory),module);trace.install()
            try:self.assertIs(model.run_first_tracker_plus([history,query],None,None,1),result)
            finally:trace.close()
            with gzip.open(Path(directory)/'actual_decisions.jsonl.gz','rt') as stream:record=json.loads(stream.readline())
            self.assertEqual(record['accepted_track_ids'],[4]);self.assertEqual(record['final_track_ids'],[4])
            self.assertIs(module.linear_sum_assignment,solver);self.assertIs(model.run_first_tracker_plus,first)
        self.assertEqual(len(calls),1);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_unknown_mixed_and_candidate_conditional_truth(self):
        self.assertEqual(decision_state(3,{4:[3,3]},4),'correct_link')
        self.assertEqual(decision_state(3,{4:[5]},4),'wrong_link')
        self.assertEqual(decision_state(3,{4:[3,5]},4),'ambiguous_reference')
        self.assertEqual(decision_state(3,{4:[3,None]},4),'ambiguous_reference')
        self.assertEqual(decision_state(3,{4:[3]},-1),'missed_known_positive')
        self.assertEqual(decision_state(3,{4:[5]},-1),'correct_reject_within_recorded_candidates')
        self.assertEqual(decision_state(3,{4:[None]},-1),'unknown_candidate_coverage')
        self.assertEqual(decision_state(None,{4:[3]},4),'unknown_query')


if __name__=='__main__':unittest.main()
