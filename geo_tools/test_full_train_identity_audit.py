import unittest
from audit_full_train_identity_events import annotate_query, BoundedExamples
from census_trackvggt_train import patch_count


class FullAuditTests(unittest.TestCase):
    def test_known_wrong_is_separate_from_impure(self):
        r=annotate_query(10,{1:10,2:20,3:None},{1:[10],2:[20],3:[10,None]},[2,1,3],2)
        self.assertEqual(r['types'],['A','B','E'])
        self.assertTrue(r['correctable_known_wrong'])
        impure=annotate_query(10,{1:None},{1:[10,20]},[1],1)
        self.assertEqual(impure['types'],['D','E','F'])
        self.assertFalse(impure['known_wrong'])
        self.assertFalse(impure['genuine_NULL_within_legal_candidates'])

    def test_ranking_failure_can_be_solver_corrected(self):
        r=annotate_query(10,{1:10,2:20},{1:[10],2:[20]},[2,1],1)
        self.assertEqual(r['types'],['A'])
        self.assertFalse(r['known_wrong'])

    def test_positive_available_but_impure_selected_is_unknown_not_typeB(self):
        r=annotate_query(10,{1:10,2:None},{1:[10],2:[10,20]},[2,1],2)
        self.assertEqual(r['types'],['E','F'])
        self.assertFalse(r['known_wrong']);self.assertFalse(r['correctable_known_wrong'])

    def test_reject_pure_positive_is_assignment_failure(self):
        r=annotate_query(10,{1:10,2:20},{1:[10],2:[20]},[1,2],-1)
        self.assertEqual(r['types'],['B'])
        self.assertEqual(r['actual_primary_state'],'missed_known_positive')

    def test_top8_pruning_not_unmatched(self):
        pure={i:100+i for i in range(1,10)}; refs={i:[v] for i,v in pure.items()}
        r=annotate_query(109,pure,refs,list(range(1,10)),1)
        self.assertIn('C',r['types'])
        self.assertFalse(r['genuine_NULL_within_legal_candidates'])
        self.assertEqual(r['positive_tracks'],[9])

    def test_unknown_and_local_NULL_have_different_semantics(self):
        r=annotate_query(None,{1:10},{1:[10]},[1],1)
        self.assertEqual(r['types'],['F']);self.assertFalse(r['known_wrong'])
        r=annotate_query(20,{1:10},{1:[10]},[1],-1)
        self.assertTrue(r['genuine_NULL_within_legal_candidates']);self.assertFalse(r['known_wrong'])
        self.assertEqual(r['types'],['D'])

    def test_known_wrong_without_positive_not_correctable(self):
        r=annotate_query(20,{1:10},{1:[10]},[1],1)
        self.assertTrue(r['known_wrong']);self.assertFalse(r['correctable_known_wrong'])

    def test_bounded_examples_not_all_history(self):
        examples=BoundedExamples(100)
        for i in range(10000):
            examples.add(dict(key=[9,i,0],correctable_known_wrong=True,types=['A','B'],relative_margin=.1))
        self.assertEqual(len(examples.rows()),40)
        self.assertEqual(len({tuple(r['key']) for r in examples.rows()}),40)

    def test_patch_centers_content_padding_and_half_open(self):
        self.assertEqual(patch_count([0,0,518,518],518,518),1369)
        self.assertEqual(patch_count([0,0,518,259],518,259),37*18)
        self.assertEqual(patch_count([0,0,7,7],518,518),0)
        self.assertEqual(patch_count([7,7,1,1],518,518),1)


if __name__=='__main__':unittest.main()
