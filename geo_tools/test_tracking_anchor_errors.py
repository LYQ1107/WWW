import unittest
from audit_tracking_anchor_errors import anchor_states


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


if __name__=='__main__':unittest.main()
