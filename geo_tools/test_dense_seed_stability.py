import copy
import unittest
from unittest.mock import patch
from audit_dense_seed_stability import check_trial


class Tests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(seed=20261014, head_checkpoint_sha256='head14', full_video=True,
            causal_online=True, geo_failures=0, total_vggt_calls=10, images=10,
            geo_options=dict(HEAD_PATH='head14.pt', GEOMETRY_WEIGHT=.5, FEATURE_WEIGHT=2))
        self.head = dict(seed=20261014, checkpoint_sha256='head14')

    def verify(self, run, seed):
        def settings(value):
            return dict(seed=value['seed'], head_checkpoint_sha256=value['head_checkpoint_sha256']), True
        def options(value):
            return {key: val for key, val in value['geo_options'].items() if key != 'GEOMETRY_WEIGHT'}
        with patch('audit_dense_seed_stability.sha256', return_value='head14'), \
             patch('audit_dense_seed_stability.signature', side_effect=settings), \
             patch('audit_dense_seed_stability.shared_options', side_effect=options):
            check_trial(run, self.reference, self.head, seed)

    def test_fixed_head_allows_explicit_inference_seed_change(self):
        run = copy.deepcopy(self.reference)
        run.update(seed=20261015, explicit_inference_seed=20261015)
        self.verify(run, 20261015)

    def test_seed_or_head_name_does_not_prove_actual_inference_rng(self):
        with self.assertRaisesRegex(ValueError, 'actual inference RNG'):
            self.verify(self.reference, 20261015)
        run = copy.deepcopy(self.reference)
        run['seed'] = 20261015
        with self.assertRaisesRegex(ValueError, 'wrapper evidence'):
            self.verify(run, 20261015)

    def test_reject_wrong_head_in_fixed_head_trial(self):
        run = copy.deepcopy(self.reference)
        run['head_checkpoint_sha256'] = 'head15'
        with self.assertRaisesRegex(ValueError, 'wrong TRAIN head'):
            self.verify(run, 20261014)

    def test_reject_other_fusion_change(self):
        for key in ('FEATURE_WEIGHT', 'GEOMETRY_WEIGHT'):
            run = copy.deepcopy(self.reference)
            run['geo_options'][key] = .7
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                self.verify(run, 20261014)


if __name__ == '__main__':
    unittest.main()
