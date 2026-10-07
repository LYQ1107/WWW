"""Validate the scientific input restriction and binary-target intervention."""
import copy
import unittest

from jev_phase6_common import ROOT
import torch
from gtr.modeling.jev_lifecycle_gates import MatchThresholdGate
from run_jev_phase6_gating import binary_record


class Phase6GateContract(unittest.TestCase):
    def test_scalar_cannot_read_other_scores_and_strict_tie_is_new(self):
        gate = MatchThresholdGate()
        state = torch.zeros(64)
        state[56] = gate.base_threshold.detach()
        changed = torch.arange(64).float() * 1000
        changed[56] = state[56]
        legal = ['ACCEPT_CURRENT', 'START_NEW']
        first = gate(state, ['MATCH_DECISION'], [legal])
        second = gate(changed, ['MATCH_DECISION'], [legal])
        self.assertTrue(torch.equal(first['probs'], second['probs']))
        self.assertEqual(gate.decide_strict(changed, legal), 'START_NEW')
        changed[56] += 0.001
        self.assertEqual(gate.decide_strict(changed, legal), 'ACCEPT_CURRENT')
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 1)

    def test_dynamic_is_parameter_matched_and_rejects_third_action(self):
        gate = MatchThresholdGate(hidden_dim=154)
        self.assertLess(abs(sum(p.numel() for p in gate.parameters()) / 34080 - 1), 0.01)
        with self.assertRaises(ValueError):
            gate(torch.zeros(64), ['MATCH_DECISION'], [['ACCEPT_CURRENT', 'REASSOCIATE', 'START_NEW']])

    def test_binary_target_is_conditioned_utility_not_folded_label(self):
        record = {'legal_actions': ['ACCEPT_CURRENT', 'REASSOCIATE', 'START_NEW'],
                  'action_outcomes': {a: {'utility': u} for a, u in
                     [('ACCEPT_CURRENT', 0), ('REASSOCIATE', 100), ('START_NEW', 1)]},
                  'state': {'features': [1, 2], 'digest': 'preserve'},
                  'labeling': {'temperature': 1}, 'sample_weight': 0.75}
        before = copy.deepcopy(record)
        derived = binary_record(record)
        self.assertEqual(record, before)
        self.assertEqual(derived['state'], record['state'])
        self.assertEqual(derived['sample_weight'], 0.75)
        self.assertEqual(derived['best_actions'], ['START_NEW'])
        self.assertAlmostEqual(derived['target_probs'][0], 0.2689414214)
        self.assertNotIn('REASSOCIATE', derived['action_outcomes'])


if __name__ == '__main__':
    unittest.main()
