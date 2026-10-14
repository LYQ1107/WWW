import unittest
from audit_tracking_anchor_errors import anchor_states,recent_output_links


class Tests(unittest.TestCase):
    def test_simultaneous_cameras_are_not_past_anchors(self):
        rows=[dict(frame=1,camera=1,gt=9,base=4),dict(frame=1,camera=2,gt=9,base=4),dict(frame=2,camera=1,gt=9,base=4)]
        states=anchor_states(rows,'base',True)
        self.assertEqual([r['state'] for r in states],['no_past_other_camera_anchor','no_past_other_camera_anchor','consistent_past_gt'])

    def test_id_renaming_neutrality_and_mixed_anchor_are_explicit(self):
        rows=[dict(frame=1,camera=2,gt=1,base=4),dict(frame=2,camera=1,gt=2,base=4),dict(frame=3,camera=1,gt=1,base=4)]
        expected=anchor_states(rows,'base')
        self.assertEqual([r['state'] for r in expected],['unanchored_id_gt_first_matched','conflicting_past_gt','mixed_past_gt_anchor'])
        for row in rows:row['base']+=1000
        self.assertEqual(anchor_states(rows,'base'),expected)

    def test_actual_output_links_count_missing_repairs_and_new_harms_once_per_query(self):
        rows=[dict(frame=1,camera=2,gt=1,base=10,new=100,image_id=1),
              dict(frame=2,camera=1,gt=1,base=20,new=100,image_id=2),
              dict(frame=3,camera=1,gt=1,base=10,new=300,image_id=3)]
        result=recent_output_links(rows)
        self.assertEqual(result['counts']['baseline_wrong_method_correct'],1)
        self.assertEqual(result['counts']['baseline_correct_method_wrong'],1)
        before=result['cohort_transition_counts']
        for row in rows:row['base']+=999;row['new']+=555
        self.assertEqual(recent_output_links(rows)['cohort_transition_counts'],before)

    def test_same_group_and_expired_truth_do_not_create_positive_opportunities(self):
        rows=[dict(frame=1,camera=1,gt=1,base=10,new=100,image_id=1),
              dict(frame=1,camera=2,gt=1,base=20,new=200,image_id=2),
              dict(frame=22,camera=1,gt=1,base=30,new=300,image_id=3)]
        result=recent_output_links(rows)
        self.assertEqual(result['cohort_transition_counts']['positive_opportunity'],{})
        self.assertEqual(result['counts']['baseline_wrong_method_correct'],0)


if __name__=='__main__':unittest.main()
