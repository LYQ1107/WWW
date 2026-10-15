import copy
import unittest
from finalize_research import require_complete_factors,require_closed_resources


class Tests(unittest.TestCase):
    def test_seed_flag_cannot_hide_missing_trial_or_detector_drift(self):
        groups=[dict(method=method,factor=factor,complete=True,completed_trials=3,pending=[],rows=[dict(exact_detector_membership=True) for _ in range(3)])
                for method,factor in [('B5','training_head_seed'),('B7','training_head_seed'),('B7','inference_rng_seed')]]
        require_complete_factors(dict(groups=groups))
        missing=copy.deepcopy(groups);missing[2]['rows'].pop()
        with self.assertRaises(ValueError):require_complete_factors(dict(groups=missing))
        drift=copy.deepcopy(groups);drift[0]['rows'][0]['exact_detector_membership']=False
        with self.assertRaises(ValueError):require_complete_factors(dict(groups=drift))

    def test_first_probe_completion_does_not_finalize_live_replacement_probe(self):
        ready=dict(status='FINISHED_PROBE_SAMPLED_RESOURCES',sample_groups=3)
        require_closed_resources([ready,ready])
        with self.assertRaises(ValueError):require_closed_resources([ready])
        with self.assertRaises(ValueError):require_closed_resources([ready,dict(ready,status='SAMPLING')])


if __name__=='__main__':unittest.main()
