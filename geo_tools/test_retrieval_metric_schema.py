import copy
import unittest
from evaluate_live_retrieval import verify_retained_metrics


class Tests(unittest.TestCase):
    def fixture(self):
        old=dict(requests=3,mrr=.5,per_scene={'18':dict(requests=3,matched=2,eligible=2,hits1=1)})
        new={**copy.deepcopy(old),'query_scene':None,'scoring_diagnostics':{},'scoring_diagnostic_denominator':'full gallery'}
        new['per_scene']['18'].update(hits5=2,mrr=.5,reciprocal=1.)
        return old,new

    def test_explicit_known_enrichment_preserves_all_legacy_values(self):
        old,new=self.fixture()
        with self.assertRaises(ValueError):verify_retained_metrics(old,new)
        self.assertEqual(verify_retained_metrics(old,new,True),'verified_legacy_counter_enrichment')
        self.assertEqual(verify_retained_metrics(new,new),'exact')

    def test_changed_numeric_or_unknown_legacy_fields_rejected(self):
        for change in ('top_metric','counter','unknown_field','filtered_scene'):
            old,new=self.fixture()
            if change=='top_metric':new['mrr']=.6
            elif change=='counter':new['per_scene']['18']['hits1']=2
            elif change=='unknown_field':old['per_scene']['18']['unrecognized']=0
            else:new['query_scene']=18
            with self.assertRaises(ValueError):verify_retained_metrics(old,new,True)


if __name__=='__main__':unittest.main()
