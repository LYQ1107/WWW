import unittest
from audit_cir_identity_propagation import owner_states,episodes,certify_queries


class PropagationContracts(unittest.TestCase):
    def test_owner_uses_strictly_past_frames_and_stays_fixed(self):
        metadata={f:dict(frame_id=f,view_id=1) for f in range(1,7)}
        rows=[dict(image_id=f,track_id=1) for f in metadata]
        labels={(f,0):(1 if f<=3 else 2) for f in metadata}
        states,summary=owner_states(rows,metadata,labels)
        self.assertEqual([states[(f,0)]['state'] for f in range(1,7)],['UNANCHORED']*3+['WRONG']*3)
        self.assertEqual(summary['mixed_native_IDs'],1)
        e=episodes(4,states);self.assertEqual(len(e),1);self.assertEqual(e[0]['observed_wrong_camera_frames'],3)
        self.assertEqual(e[0]['end_reason'],'VIDEO_OR_OBSERVATION_END_CENSORED')
    def test_other_current_camera_cannot_certify_owner(self):
        metadata={1:dict(frame_id=1,view_id=1),2:dict(frame_id=1,view_id=2),3:dict(frame_id=2,view_id=1),4:dict(frame_id=2,view_id=2)}
        rows=[dict(image_id=f,track_id=1) for f in metadata]
        states,_=owner_states(rows,metadata,{(1,0):1,(2,0):1,(3,0):1,(4,0):2})
        self.assertTrue(all(r['state']=='UNANCHORED' for r in states.values()))
    def test_duplicate_gt_event_is_unknown_not_removed_from_prediction(self):
        rows=[dict(image_id=1,bbox=[0,0,10,10],track_id=1)]
        gt={'s_View1':[[1,5,0,0,10,10,1,1,1],[1,5,20,20,10,10,1,1,1]]}
        labels,grouped=certify_queries(rows,{1:dict(frame_id=1,view_id=1)},gt,'s')
        self.assertIsNone(labels[(1,0)]);self.assertEqual(len(grouped[1]),1)
    def test_gaps_are_censored_not_interpolated(self):
        states={(1,0):dict(frame=1,camera=1,person=2,identity=1,owner=1,state='WRONG'),
                (3,0):dict(frame=3,camera=1,person=2,identity=1,owner=1,state='WRONG')}
        e=episodes(4,states);self.assertEqual(len(e),2)
        self.assertEqual([r['observed_wrong_camera_frames'] for r in e],[1,1]);self.assertEqual(e[0]['end_reason'],'GAP_CENSORED')
    def test_unknown_never_establishes_owner(self):
        metadata={f:dict(frame_id=f,view_id=1) for f in range(1,5)}
        rows=[dict(image_id=f,track_id=f) for f in metadata]
        states,summary=owner_states(rows,metadata,{(f,0):None for f in metadata})
        self.assertEqual(summary['fixed_causal_owners'],0);self.assertTrue(all(r['state']=='GT_UNKNOWN' for r in states.values()))


if __name__=='__main__':unittest.main()
