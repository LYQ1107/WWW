import itertools
import unittest
from audit_tracking_identity_pairs import pair_outcomes


class Tests(unittest.TestCase):
    def test_contingency_counts_equal_brute_force_for_both_scopes(self):
        rows=[dict(gt=i%3,base=i%4,new=(i+1)%5,camera=i%2) for i in range(19)]
        for cross in [False,True]:
            result=pair_outcomes(rows,cross);counts=dict(fixes_false_negative_pairs=0,fixes_false_positive_pairs=0,harms_true_positive_pairs=0,harms_true_negative_pairs=0)
            for left,right in itertools.combinations(rows,2):
                if cross and left['camera']==right['camera']:continue
                truth=left['gt']==right['gt'];old=left['base']==right['base'];new=left['new']==right['new']
                counts['fixes_false_negative_pairs']+=truth and not old and new
                counts['fixes_false_positive_pairs']+=not truth and old and not new
                counts['harms_true_positive_pairs']+=truth and old and not new
                counts['harms_true_negative_pairs']+=not truth and not old and new
            self.assertEqual({key:result[key] for key in counts},counts)

    def test_track_id_renaming_is_not_an_identity_change(self):
        rows=[dict(gt=i%3,base=i%3,new=100+i%3,camera=i%2) for i in range(12)]
        for cross in [False,True]:
            result=pair_outcomes(rows,cross)
            self.assertTrue(all(result[key]==0 for key in result if key.startswith(('fixes_','harms_'))))


if __name__=='__main__':unittest.main()
