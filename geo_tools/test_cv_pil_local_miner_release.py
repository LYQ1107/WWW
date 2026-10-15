import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_cv_pil_native_handoff import fixture
from audit_cv_pil_local_miner_release import joined_row


def source_event():
    return dict(key=[4,200,0],frame=10,camera=2,top8_tracks=[10,20],positive_tracks=[20],
        accepted_primary_track=10,final_output_track=10,actual_primary_state='wrong_link',
        known_wrong=True,query_gt_identity=7,ranking_state='wrong_link')


class TestLocalJoin(unittest.TestCase):
    def test_only_real_candidate_and_actual_commit_match_is_accepted(self):
        asset=dict(scene=4,image_id=200,row_id=0,call_index=0,query_row=0)
        r=joined_row(fixture(),asset,source_event(),(10,10))
        self.assertTrue(r['reachable_primary_error']);self.assertTrue(r['certified_accepted_final_wrong'])
        event=source_event();event['top8_tracks']=[20,10]
        with self.assertRaisesRegex(ValueError,'ranking drift'):joined_row(fixture(),asset,event,(10,10))
        with self.assertRaisesRegex(ValueError,'commit drift'):joined_row(fixture(),asset,source_event(),(20,20))

    def test_missed_primary_is_not_certified_as_a_final_new_id_error(self):
        asset=dict(scene=4,image_id=200,row_id=0,call_index=0,query_row=0)
        event=source_event();event.update(actual_primary_state='missed_known_positive',accepted_primary_track=-1,final_output_track=50)
        r=joined_row(fixture(),asset,event,(-1,50))
        self.assertTrue(r['reachable_primary_error']);self.assertFalse(r['certified_accepted_final_wrong'])

    def test_unknown_is_not_a_known_native_error(self):
        asset=dict(scene=4,image_id=200,row_id=0,call_index=0,query_row=0)
        event=source_event();event['query_gt_identity']=None
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):joined_row(fixture(),asset,event,(10,10))


if __name__=='__main__':unittest.main()
