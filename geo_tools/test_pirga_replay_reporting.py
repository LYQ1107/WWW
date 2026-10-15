import unittest
from pirga_replay_reporting import mechanism_audit,uncertainty_report

ARMS=('C0','C1','C3','A2_rank','AC_contrastive','A4_uncertainty')


def row(key,person,state,covered=True):
    r=dict(key=[12,key,0],person=person,source_primary_state=state,positive_covered=covered)
    for arm in ARMS:r.update({arm+'_primary':False if arm=='C0' else True,arm+'_accepted_id':None if arm=='C0' else 9})
    return r


class ReportingTests(unittest.TestCase):
    def test_c1_explained_repairs_are_not_prediction_specific_gain(self):
        rows=[row(1,1,'missed_known_positive'),row(2,2,'wrong_link')]
        result=mechanism_audit(rows,ARMS)
        self.assertEqual(result['status'],'NO_PREDICTION_SPECIFIC_GAIN')
        self.assertEqual(result['methods']['A4_uncertainty']['original_error_repairs'],2)
        self.assertFalse(result['online_admitted'])
        self.assertFalse(result['phase2_mechanism_gate_pass'])

    def test_suppression_and_unknown_conversions_do_not_become_identity_repairs(self):
        a=row(1,1,'wrong_link',False);a['A4_uncertainty_accepted_id']=None
        b=row(2,2,'unknown');b['C0_primary']=None
        result=mechanism_audit([a,b],ARMS)
        self.assertEqual(result['methods']['A4_uncertainty']['original_error_repairs'],0)
        self.assertEqual(result['methods']['A4_uncertainty']['original_error_rejected_without_legal_positive'],1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):mechanism_audit([a,a],ARMS)

    def test_original_unknown_to_wrong_is_visible_outside_known_harms(self):
        a=row(1,1,'unknown');a['C0_primary']=None;a['A4_uncertainty_primary']=False
        result=mechanism_audit([a],ARMS)
        self.assertEqual(result['methods']['A4_uncertainty']['original_unknown_to_known_wrong'],1)
        self.assertEqual(result['methods']['A4_uncertainty']['known_harms_to_original_correct'],0)

    def test_constant_uncertainty_is_detected_without_false_correlation(self):
        rows=[dict(key=[12,i,0],person=i,positive_variance=.1,positive_cosine_error=i/10,primary_correct=i==1) for i in [1,2]]
        report=uncertainty_report(rows)
        self.assertTrue(report['variance_effectively_constant'])
        self.assertIsNone(report['error_variance_pearson'])
        self.assertEqual(report['physical_queries'],2)

if __name__=='__main__':unittest.main()
