import unittest
import common
import gzip
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
from scipy.optimize import linear_sum_assignment
from gtr.geometry.decision_trace import solver_record,DecisionTrace
from audit_decision_trace import decision_state,verify_detector_cohort,verify_query_truth


class Tests(unittest.TestCase):
    def test_equal_proposal_counts_do_not_prove_same_physical_queries(self):
        rows=[dict(source=[1,0],detection=dict(bbox=[0,0,1,1])),dict(source=[1,1],detection=dict(bbox=[4,0,1,1]))]
        self.assertEqual(verify_detector_cohort(rows,list(reversed(rows))),2)
        changed=[dict(source=rows[0]['source'],detection=rows[1]['detection']),dict(source=rows[1]['source'],detection=rows[0]['detection'])]
        with self.assertRaises(ValueError):verify_detector_cohort(rows,changed)
        with self.assertRaises(ValueError):verify_detector_cohort(rows,rows+[rows[0]])

    def test_identical_correctness_categories_do_not_prove_same_query_gt(self):
        left={(1,0):dict(state='correct_link',gt_identity=3)}
        right={(1,0):dict(state='correct_link',gt_identity=4)}
        with self.assertRaises(ValueError):verify_query_truth(left,right,left.keys())
        verify_query_truth(left,left,left.keys())

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

    def test_original_first_global_and_memory_methods_with_real_instances(self):
        from detectron2.structures import Instances
        from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN
        activate=lambda values:[torch.softmax(torch.cat([value,value.new_zeros((len(value),1))],dim=1),dim=1)[:,:-1] for value in values]
        for stage in ('run_first_tracker_plus','run_global_tracker_plus','run_memory_tracker'):
            history=Instances((10,10));history.track_ids=torch.tensor([4,9]);history.reid_features=torch.zeros((2,2));history.decision_trace_source=torch.tensor([[1,0],[1,1]])
            query=Instances((10,10));query.reid_features=torch.zeros((2,2));query.decision_trace_source=torch.tensor([[2,0],[2,1]])
            model=SimpleNamespace(overlap_thresh=.4,not_mult_thresh=True,thred_bank=.8,with_bank=False,bank_size=100,inference=lambda:None,roi_heads=SimpleNamespace(_activate_asso=activate))
            for name in ('run_first_tracker_plus','run_global_tracker_plus','run_memory_tracker'):
                setattr(model,name,MethodType(getattr(GTRRCNN,name),model))
            contexts=[history,query];logits=[torch.tensor([[5.,-5.],[-5.,5.]])];counts={4:1,9:1};bank={4:history[0],9:history[1]}
            with tempfile.TemporaryDirectory() as directory:
                trace=DecisionTrace(model,Path(directory));trace.install()
                try:
                    if stage=='run_first_tracker_plus':model.run_first_tracker_plus(contexts,logits,None,1,9,counts,bank)
                    elif stage=='run_global_tracker_plus':model.run_global_tracker_plus(2,contexts,logits,None,1,9,counts,bank,[],1)
                    else:model.run_memory_tracker(contexts,[-torch.ones((2,2))*5],None,1,counts)
                finally:trace.close()
                with gzip.open(Path(directory)/'actual_decisions.jsonl.gz','rt') as stream:record=json.loads(stream.readline())
                self.assertEqual(record['accepted_track_ids'],[-1,-1] if stage=='run_memory_tracker' else [4,9])
                self.assertEqual(record['stage'],stage)

    def test_detector_source_fields_preserve_public_boxes_and_live_coordinates(self):
        from detectron2.structures import Instances,Boxes
        from gtr.geometry.online_runner import clone_instances
        from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
        from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
        instance=Instances((10,10));instance.pred_boxes=Boxes(torch.tensor([[1.,2.,3.,4.]]));instance.scores=torch.tensor([.9]);instance.pred_classes=torch.tensor([0])
        record=dict(image_id=7,height=20,width=20)
        expected=custom_instances_to_coco_json(CustomRCNN._postprocess([clone_instances(instance)],[record],[(0,0)],not_clamp_box=True)[0]['instances'],7)
        original=instance.pred_boxes.tensor.clone();model=SimpleNamespace(inference=lambda *args,**kwargs:[instance],not_clamp_box=True)
        with tempfile.TemporaryDirectory() as directory:
            trace=DecisionTrace(model,Path(directory));trace.originals['inference']=model.inference
            try:result=trace.inference([record],2,[1,1,1],0,do_postprocess=False)
            finally:trace.decisions.close();trace.detectors.close()
            with gzip.open(Path(directory)/'detector_provenance.jsonl.gz','rt') as stream:actual=json.loads(stream.readline())
            self.assertIs(result[0],instance);self.assertEqual(actual['detection'],expected[0]);self.assertEqual(actual['source'],[7,0])
        self.assertTrue(torch.equal(original,instance.pred_boxes.tensor))


if __name__=='__main__':unittest.main()
