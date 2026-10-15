import unittest
from collect_track_hard_events import choose_events,pure_reference_identity


class Tests(unittest.TestCase):
    def test_validation_sampling_needs_no_label_fields(self):
        events=[dict(key=[12,5,0],relative_margin=.1),dict(key=[12,2,0],relative_margin=.1),dict(key=[12,9,0],relative_margin=.5)]
        selected=choose_events(events,2,False)
        self.assertEqual([row['key'] for row in selected],[[12,2,0],[12,5,0]])
        labeled=[dict(row,top8_positive=i%2==0,native_top1_correct=i%2==1) for i,row in enumerate(events)]
        self.assertEqual([row['key'] for row in choose_events(labeled,2,False)],[row['key'] for row in selected])

    def test_fit_native_errors_with_reachable_correct_candidate_first(self):
        rows=[dict(key=[5,i,0],relative_margin=margin,top8_positive=positive,native_top1_correct=correct,native_top1_state='correct_link' if correct else 'wrong_link',actual_primary_state='correct_link' if correct else 'wrong_link')
              for i,margin,positive,correct in [(1,.01,True,True),(2,.8,True,False),(3,.9,False,False)]]
        self.assertEqual(choose_events(rows,1,True)[0]['key'],[5,2,0])

    def test_mixed_or_unmatched_track_never_gets_pure_identity(self):
        self.assertEqual(pure_reference_identity([17,17]),17)
        for values in ([17,None],[17,18],[None],[]):self.assertIsNone(pure_reference_identity(values))


if __name__=='__main__':unittest.main()
