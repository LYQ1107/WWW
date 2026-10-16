"""Certified causal effect durations must exclude unrelated errors and gaps."""
import unittest
from audit_cir_identity_propagation import paired_effect_episodes


def states(pairs):
    before={};after={}
    for frame,camera,person,b,a in pairs:
        key=(frame,camera);meta=dict(frame=frame,camera=camera,person=person)
        before[key]=dict(**meta,state=b);after[key]=dict(**meta,state=a)
    return before,after


class PairedEffectTests(unittest.TestCase):
    def test_recovery_and_unrelated_wrong_are_distinguished(self):
        b,a=states([(10,1,2,'CORRECT','WRONG'),(11,1,2,'CORRECT','WRONG'),(12,1,2,'CORRECT','CORRECT'),(10,2,4,'WRONG','WRONG')])
        e=paired_effect_episodes(b,a,10,12,'HARM')
        self.assertEqual(len(e),1);self.assertEqual(e[0]['observed_affected_camera_frames'],2)
        self.assertEqual(e[0]['end_reason'],'NEXT_CERTIFIED_NON_HARM')
    def test_missing_observation_censors_instead_of_inventing_duration(self):
        b,a=states([(10,1,2,'CORRECT','WRONG'),(12,1,2,'CORRECT','WRONG')])
        e=paired_effect_episodes(b,a,10,12,'HARM')
        self.assertEqual([r['observed_affected_camera_frames'] for r in e],[1,1]);self.assertEqual(e[0]['end_reason'],'GAP_CENSORED')
    def test_unknown_is_not_carried_as_known_person(self):
        b,a=states([(10,1,2,'CORRECT','WRONG'),(11,1,None,'GT_UNKNOWN','GT_UNKNOWN'),(12,1,2,'CORRECT','WRONG')])
        self.assertEqual(len(paired_effect_episodes(b,a,10,12,'HARM')),2)
    def test_cameras_are_separate_durations(self):
        b,a=states([(10,1,2,'CORRECT','WRONG'),(10,2,2,'CORRECT','WRONG')])
        e=paired_effect_episodes(b,a,10,10,'HARM');self.assertEqual(len(e),2)
        self.assertEqual([r['observed_affected_camera_frames'] for r in e],[1,1])
    def test_unanchored_to_correct_is_not_certified_repair(self):
        b,a=states([(10,1,2,'UNANCHORED','CORRECT'),(11,1,2,'WRONG','CORRECT')])
        e=paired_effect_episodes(b,a,10,11,'REPAIR');self.assertEqual(len(e),1)
        self.assertEqual(e[0]['start_frame'],11);self.assertEqual(e[0]['first_effect_delay_groups'],1)


if __name__=='__main__':unittest.main()
