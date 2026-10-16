"""Identity certification refuses unknown anchors and self-labeling of new tracks."""
import unittest
from omega_512_identity_adjudicate import classify


def decision(accepted,final,refs=None,query=(10,0)):
    return dict(stage='run_global_tracker_plus',queries=[list(query)],references=refs or [],
                accepted_track_ids=[accepted],final_track_ids=[final],hungarian_pairs=[])

class Tests(unittest.TestCase):
    def test_unknown_anchor_is_not_wrong_even_for_pure_legal_positive(self):
        d=decision(1,1,[dict(source=[9,0],track=1)])
        r=classify([d],{(10,0):4,(9,0):4},{1:{4,None}})[10,0]
        self.assertEqual(r['primary'],'correct')
        self.assertEqual(r['final_state'],'unknown_or_mixed_anchor')

    def test_new_id_needs_prior_pure_identity_and_legal_positive_for_fragmentation(self):
        d=decision(-1,3,[dict(source=[9,0],track=1)])
        labels={(10,0):4,(9,0):4}
        self.assertEqual(classify([d],labels,{1:{4}})[10,0]['final_state'],'certified_fragmentation')
        self.assertEqual(classify([d],labels,{1:{4,None}})[10,0]['final_state'],'new_track_unscored')
        self.assertEqual(classify([decision(-1,3)],labels,{1:{4}})[10,0]['final_state'],'new_track_unscored')

    def test_samegroup_published_camera_anchor_can_support_later_camera(self):
        a=decision(-1,3,query=(10,0));b=decision(3,3,[dict(source=[10,0],track=3)],query=(11,0))
        r=classify([a,b],{(10,0):4,(11,0):4},{})
        self.assertEqual(r[10,0]['final_state'],'new_track_unscored')
        self.assertEqual(r[11,0]['final_state'],'correct')
        self.assertEqual(classify([b,a],{(10,0):4,(11,0):4},{})[11,0]['final_state'],'new_track_unscored')

if __name__=='__main__':unittest.main()
