import unittest
import numpy as np
import torch
from prepare_trackvggt_v3_cohort import cycle_round_robin
from extract_trackvggt_v3_tokens import target_requests
from trackvggt_data import TrackEventData
from evaluate_teacher_pairs import predicted_state


class V3DataTests(unittest.TestCase):
    def test_existing_V2_targets_are_reused_and_auxiliary_is_explicit(self):
        cohort=dict(primary_new_target_keys=[[9,101,0]],reused_V2_target_keys=[[9,100,0]],
            fit=[dict(key=[9,100,0],scene=9,event_file='old',event_index=1),dict(key=[9,101,0],scene=9,event_file='new',event_index=2)],
            auxiliary_target_index=[dict(key=[9,102,0],scene=9,event_file='new',source=[102,0],parents=[[9,101,0]])])
        result=target_requests(cohort,9)
        self.assertEqual([r['key'] for r in result],[[9,101,0],[9,102,0]])
        self.assertEqual(result[1]['event_index'],None)
        self.assertEqual(result[1]['roles'],['cycle_B_auxiliary'])

    def test_cycle_subset_is_scene_balanced_without_GT_fields(self):
        events=[dict(key=[scene,i,0],cycle={} if i%2 else None) for scene in [4,9,24] for i in range(20)]
        chosen=cycle_round_robin(events,12)
        self.assertEqual(len(chosen),12)
        self.assertEqual([sum(e['key'][0]==s for e in chosen) for s in [4,9,24]],[4,4,4])
        for e in events:e['query_gt_identity']=123 if e['key'][1]%3 else None
        self.assertEqual([e['key'] for e in chosen],[e['key'] for e in cycle_round_robin(events,12)])

    def test_empty_gallery_loader_preserves_real_NULL_without_history(self):
        data=TrackEventData.__new__(TrackEventData)
        item=dict(key=[9,1,0],scope='fit',scene=9,native_top1_state='no_candidate')
        query=dict(feature=torch.ones(3),frame=1,camera=1)
        event=dict(query_source=[1,0],call_index=0,native_track_scores=[],candidate_tracks=[],positive_tracks=[],ambiguous_tracks=[],known_unmatched_within_all_recorded_history=True,
            loss_eligible=True,query_gt_identity=10,all_history_positive=False,top8_positive=False,actual_primary_state='correct_reject_within_recorded_candidates')
        payload=dict(observations={(1,0):query},calls={0:dict(histories={})})
        data.event=lambda index:(item,payload,event);data.rows={(9,1,0):dict(patch_bin='0')}
        arrays={f'layer_{k}':np.zeros((0,8),np.float16) for k in [4,11,17,23]}
        arrays.update(attention=np.empty(0,np.float32),xy=np.empty((0,2),np.float32),inside=np.empty(0,bool),roi=np.ones((4,8),np.float16))
        data.arrays=lambda key:arrays
        features,target=data.get(0)
        self.assertEqual(features['histories'],[]);self.assertEqual(tuple(features['pair'].shape),(0,8))
        self.assertEqual(features['native_scores'].numel(),0);self.assertEqual(target['positive'].tolist(),[True])
        self.assertEqual(predicted_state([0],target),'correct_null_within_recorded_history')
        self.assertEqual(data.native_prior_scores(0).tolist(),[0.])


if __name__=='__main__':unittest.main()
