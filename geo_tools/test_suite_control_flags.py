import unittest
from evaluate_suite import validate_control_flags


class Tests(unittest.TestCase):
    def test_homogeneous_legacy_and_explicit_control_are_preserved(self):
        self.assertIsNone(validate_control_flags([dict(causal_online=True)]*5)['validation_constant_head_evidence'])
        result=validate_control_flags([dict(causal_online=True,validation_constant_head_evidence=1.)]*5)
        self.assertEqual(result['validation_constant_head_evidence'],1.)

    def test_each_mixed_protocol_or_counterfactual_variant_is_rejected(self):
        for flag,value in [('causal_online',False),('validation_corrected_gmt_camera_index',True),
                           ('validation_reversed_vggt_order',True),('validation_constant_head_evidence',1.),('compute_neutral',True)]:
            rows=[dict(causal_online=True) for _ in range(5)];rows[-1][flag]=value
            with self.assertRaisesRegex(ValueError,flag):validate_control_flags(rows)


if __name__=='__main__':unittest.main()
