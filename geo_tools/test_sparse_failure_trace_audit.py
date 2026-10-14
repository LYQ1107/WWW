import unittest
from audit_sparse_failure_trace import first_partition_change,summarize_trace


class Tests(unittest.TestCase):
    def test_arbitrary_id_renaming_is_neutral_and_split_merge_get_witnesses(self):
        rows=[dict(frame_id=i,baseline_id=a,method_id=b) for i,(a,b) in enumerate([(1,90),(2,80),(1,90)])]
        self.assertIsNone(first_partition_change(rows))
        rows[-1]['method_id']=70
        change=first_partition_change(rows)
        self.assertEqual(change['kind'],'split_baseline_identity')
        self.assertEqual(change['previous']['frame_id'],0)
        self.assertEqual(change['current']['frame_id'],2)
        rows[-1].update(baseline_id=3,method_id=80)
        self.assertEqual(first_partition_change(rows)['kind'],'merge_baseline_identities')

    def test_rejects_future_stale_and_invalid_age_and_preserves_zero(self):
        row=dict(query_feature_valid=[True,False],feature_ages=[0,None],query_count=2,valid_query_features=1,
                 changed_logit_pairs=1,counterfactual_assignment_changes=0,max_abs_logit_delta=.5,max_abs_gta_delta=.1)
        self.assertEqual(summarize_trace([row],20)['valid_source_age_histogram'],{0:1})
        for age in [-1,21,None]:
            row['feature_ages'][0]=age
            with self.assertRaises(ValueError):summarize_trace([row],20)
        row['feature_ages']=[0,0]
        with self.assertRaises(ValueError):summarize_trace([row],20)


if __name__=='__main__':unittest.main()
