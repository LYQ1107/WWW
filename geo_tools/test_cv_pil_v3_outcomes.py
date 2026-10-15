import unittest

from audit_cv_pil_v3_outcomes import targeted_audit


def row(index,person,state='wrong_link',covered=True):
    r=dict(key=[12,index,0],person=person,source_primary_state=state,positive_covered=covered)
    for arm in ('C0','C1','C2','C3','C4'):
        r[arm+'_primary']=state=='correct_link';r[arm+'_accepted_id']=10 if state=='correct_link' else None
    return r


class TestOriginalErrorAttribution(unittest.TestCase):
    def test_c2_harm_avoidance_does_not_become_original_error_repair(self):
        r=row(0,1,'correct_link');r['C2_primary']=False
        other=row(1,2)
        a=targeted_audit([r,other]);p=a['comparisons']['C3_vs_C2']
        self.assertEqual(p['net_original_error_repairs'],0)
        self.assertEqual(p['avoided_ordinary_model_harms'],1)
        self.assertFalse(a['algorithm_gate_pass'])

    def test_repeated_original_repairs_of_one_person_fail_two_people_gate(self):
        rs=[row(i,7) for i in range(5)]
        for r in rs:r['C3_primary']=True;r['C3_accepted_id']=20
        a=targeted_audit(rs)
        self.assertEqual(a['methods']['C3']['certified_reachable_original_primary_errors_repaired'],5)
        self.assertFalse(a['algorithm_gate_conditions']['C3_net_original_repairs_in_at_least2people'])

    def test_no_candidate_rejection_and_unknown_are_not_identity_reuse_fixes(self):
        r=row(1,3,covered=False);r['C3_primary']=True;r['C3_accepted_id']=None
        s=row(2,4);s['C3_primary']=None
        a=targeted_audit([r,s]);self.assertEqual(a['methods']['C3']['certified_reachable_original_primary_errors_repaired'],0)
        self.assertEqual(a['methods']['C3']['rejected_without_legal_positive'],1)

    def test_algorithm_gate_does_not_self_admit_online_or_scientific_success(self):
        rs=[row(i,i) for i in (1,2)]
        for r in rs:r['C3_primary']=True;r['C3_accepted_id']=20
        a=targeted_audit(rs);self.assertTrue(a['algorithm_gate_pass'])
        self.assertFalse(a['native_online_run_admitted']);self.assertFalse(a['scientific_success'])


if __name__=='__main__':unittest.main()
