import unittest
from audit_retrieval_outcomes import compare,shared_rank1_fixes


def rows(values):
    return [dict(query=dict(image_id=index,bbox=[0,0,1,1]),gt_identity=[18,index],matched=True,
                 eligible=eligible,first_positive_rank=rank,positive_gallery_clips=int(eligible),published_track_id=index)
            for index,(eligible,rank) in enumerate(values)]


class Tests(unittest.TestCase):
    def test_one_net_hit_can_include_multiple_corrections_and_harms(self):
        result=compare(rows([(True,2),(True,2),(True,1)]),rows([(True,1),(True,1),(True,2)]))
        self.assertEqual(result['common_eligible_rank1_fixes'],2);self.assertEqual(result['common_eligible_rank1_harms'],1)
        self.assertEqual(result['all_request_rank1_hit_delta'],1);self.assertTrue(result['identical_eligible_query_set'])

    def test_equal_eligible_counts_do_not_imply_equal_denominators(self):
        result=compare(rows([(True,1),(False,None)]),rows([(False,None),(True,1)]))
        self.assertFalse(result['identical_eligible_query_set']);self.assertEqual(result['eligible_gained'],1);self.assertEqual(result['eligible_lost'],1)
        self.assertEqual(result['common_eligible'],0);self.assertIsNone(result['common_eligible_rank1_delta'])

    def test_changed_query_or_gt_matching_is_not_a_paired_comparison(self):
        baseline=rows([(True,1)]);changed=rows([(True,1)]);changed[0]['query']['bbox']=[0,0,2,2]
        with self.assertRaises(ValueError):compare(baseline,changed)
        changed=rows([(True,1)]);changed[0]['gt_identity']=[18,99]
        with self.assertRaises(ValueError):compare(baseline,changed)

    def test_same_aggregate_gain_is_not_a_shared_query_correction(self):
        baseline=rows([(True,2),(True,2)]);vggt=rows([(True,1),(True,2)]);control=rows([(True,2),(True,1)])
        # Even duplicate physical query dictionaries retain distinct ordered
        # occurrences; a set of serialized boxes would incorrectly merge them.
        for values in (baseline,vggt,control):values[1]['query']=dict(values[0]['query'])
        result=shared_rank1_fixes(baseline,vggt,control)
        self.assertEqual(result['vggt_rank1_fixes'],1);self.assertEqual(result['control_rank1_fixes'],1)
        self.assertEqual(result['shared_rank1_fixes'],0);self.assertEqual(result['vggt_fixes_not_reproduced_by_this_control'],1)

    def test_same_query_fix_is_retained_even_when_other_rank_changes_differ(self):
        result=shared_rank1_fixes(rows([(True,2),(True,4)]),rows([(True,1),(True,3)]),rows([(True,1),(True,2)]))
        self.assertEqual(result['shared_rank1_fixes'],1);self.assertEqual(result['vggt_fixes_not_reproduced_by_this_control'],0)
        self.assertEqual(result['shared_fix_examples'][0]['query_ordinal'],0)


if __name__=='__main__':unittest.main()
