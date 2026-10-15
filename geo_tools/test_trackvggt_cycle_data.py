import copy
import unittest
import torch
from trackvggt_cycle_data import observable_cycle,label_cycle


def fixture():
    observations={}
    for source,frame,camera in [(1,2,1),(2,2,2),(3,3,1),(4,2,2),(6,3,1),(7,3,2)]:observations[source,0]=dict(frame=frame,camera=camera)
    call=dict(references=[dict(source=[s,0],track=t) for s,t in [(1,10),(2,10),(3,10),(4,20),(6,20)]],
        histories={10:dict(sources=[[1,0],[2,0],[3,0]],observed_length=6),20:dict(sources=[[4,0],[6,0]],observed_length=7)})
    event=dict(query_source=[7,0],candidate_tracks=[10,20],native_track_scores=[.9,.3])
    evidence={(3,0):dict(track_ids=[10,20],scores=torch.tensor([.8,.2]),counts={10:5,20:6}),(6,0):dict(track_ids=[10,20],scores=torch.tensor([.1,.9]),counts={10:5,20:6})}
    labels={(s,0):(100 if s in [1,2,3,7] else 200) for s in [1,2,3,4,6,7]}
    metadata={s:dict(frame_id=v['frame']) for (s,_),v in observations.items()}
    return call,event,observations,evidence,labels,metadata


class CycleDataTests(unittest.TestCase):
    def test_observation_only_indices_and_before_B_counts(self):
        call,event,obs,evidence,_,_=fixture();graph,needed=observable_cycle(call,event,obs,evidence)
        self.assertEqual([r['source'] for r in graph['A_previous']],[[2,0],[4,0]])
        self.assertEqual([r['source'] for r in graph['B_current']],[[3,0],[6,0]])
        self.assertEqual(graph['A_previous'][0]['history_sources'],[[1,0],[2,0]])
        self.assertEqual(graph['A_previous'][0]['observed_length'],5)
        self.assertEqual(graph['B_current'][0]['observed_length'],6)
        self.assertEqual(graph['A_current_sources'],[[7,0]])
        self.assertEqual(needed,{(1,0),(2,0),(3,0),(4,0),(6,0)})
        self.assertLess(graph['previous_time'],graph['current_time'])

    def test_no_after_current_or_future_camera_bridge(self):
        call,event,obs,evidence,_,_=fixture()
        self.assertEqual(observable_cycle(call,event,obs,{}),(None,set()))
        obs[3,0]['frame']=4;obs[6,0]['frame']=4
        self.assertEqual(observable_cycle(call,event,obs,evidence),(None,set()))

    def test_visible_rows_share_actual_identity_space(self):
        call,event,obs,evidence,labels,metadata=fixture();graph,_=observable_cycle(call,event,obs,evidence)
        target=label_cycle(graph,call,100,labels,metadata)
        self.assertEqual(target['visible_A_rows'],[True,False]);self.assertEqual(target['known_B_columns'],[True,True])
        self.assertEqual(target['legal_AB_pairs'],[[True,True],[True,True]])
        # Labels change masks only, not the feature/history graph.
        before=copy.deepcopy(graph);labels[3,0]=None;target=label_cycle(graph,call,100,labels,metadata)
        self.assertEqual(target['visible_A_rows'],[False,False]);self.assertEqual(target['known_B_columns'],[False,True]);self.assertEqual(graph,before)

    def test_unknown_past_reference_cannot_certify_pure_anchor(self):
        call,event,obs,evidence,labels,metadata=fixture();graph,_=observable_cycle(call,event,obs,evidence)
        labels[1,0]=None;target=label_cycle(graph,call,100,labels,metadata)
        self.assertEqual(target['A_past_GT'],[None,200]);self.assertFalse(any(target['visible_A_rows']))


if __name__=='__main__':unittest.main()
