"""UNKNOWN transitions and repeated-person fixes cannot inflate certification."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_cv_pil_v2 import paired,metrics


def row(key,outcome,person=7):
    return dict(key=[12,key,0],scene=12,person=person,outcome=outcome,
        prediction=10 if outcome else 20,positive_covered=True,known_query=person is not None,
        cross_view_positive=True,strict_retrieval_correct=outcome is True,rr=1. if outcome else .5,
        any_unknown_candidate=outcome is None,alias_positive_count=2,primary_state='correct_link')


class TestMetrics(unittest.TestCase):
    def test_unknown_transitions_are_neither_certified_fixes_nor_harms(self):
        before=[row(1,None),row(2,True),row(3,False)]
        after=[row(1,True),row(2,None),row(3,True)]
        p=paired(before,after)
        self.assertEqual(p['certified_pair_rank_fixes'],1)
        self.assertEqual(p['certified_pair_rank_harms'],0)
        self.assertEqual(p['unknown_to_correct'],1);self.assertEqual(p['correct_to_unknown'],1)
        self.assertIsNone(p['native_committed_corrections'])

    def test_repeated_queries_of_one_person_are_one_dependence_group(self):
        p=paired([row(1,False),row(2,False)],[row(1,True),row(2,True)])
        self.assertEqual(p['certified_pair_rank_fixes'],2);self.assertEqual(p['fix_person_groups'],1)
        m=metrics([row(1,True),row(2,None)])
        self.assertEqual(m['top1'],.5);self.assertEqual(m['prediction_unknown_ratio'],.5)
        self.assertIsNone(m['certified_native_committed_fixes'])


if __name__=='__main__':unittest.main()
