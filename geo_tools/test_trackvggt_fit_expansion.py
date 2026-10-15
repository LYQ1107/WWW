import collections
import unittest
from decide_trackvggt_fit_expansion import category, scene_rotating_select, cohort_statistics


def event(scene,frame,identity=10,**kwargs):
    return dict(key=[scene,scene*10000+frame,0],camera=1,frame=frame,query_gt_identity=identity,patch_bin='1-3',
        **dict(dict(pure_positive_top8=True,ranking_state='correct_link',known_wrong=False,relative_margin=.9),**kwargs))


class ExpansionTests(unittest.TestCase):
    def test_pruned_and_impure_never_become_positive_or_error(self):
        e=event(9,1,pure_positive_top8=False,ranking_state='ambiguous_reference')
        self.assertEqual(category(e),'missing_or_unknown')
        self.assertEqual(category(event(9,2,ranking_state='ambiguous_reference')),'ambiguous')

    def test_scene_rotation_unique_and_temporally_spaced(self):
        pool={s:[event(s,f) for f in range(1,51)] for s in [9,13]}
        used=set();frames=collections.defaultdict(set);selected=scene_rotating_select(pool,20,used,frames)
        self.assertEqual(len(selected),20);self.assertEqual(len(used),20)
        self.assertLessEqual(abs(sum(e['key'][0]==9 for e in selected)-sum(e['key'][0]==13 for e in selected)),1)
        for values in frames.values():
            ordered=sorted(values)
            self.assertTrue(all(b-a>=3 for a,b in zip(ordered,ordered[1:])))
        more=scene_rotating_select(pool,100,used,frames)
        self.assertFalse({tuple(e['key']) for e in selected}&{tuple(e['key']) for e in more})
        self.assertEqual(len(used),len(selected)+len(more))
        self.assertEqual(scene_rotating_select(pool,100,used,frames),[])

    def test_locked_V2_exclusion_and_neighborhood(self):
        e=event(9,10);used={tuple(e['key'])};frames=collections.defaultdict(set);frames[9,1,10].add(10)
        selected=scene_rotating_select({9:[event(9,f) for f in [9,10,11,12,13]]},10,used,frames)
        self.assertEqual([e['frame'] for e in selected],[13])

    def test_unknowns_are_not_a_single_GT_identity(self):
        stats=cohort_statistics([event(9,1,None),event(9,2,None),event(9,3,10)])
        self.assertEqual(stats['unique_known_scene_identities'],1)
        self.assertEqual(stats['adjacent_group_pairs_same_scene_camera_identity'],0)


if __name__=='__main__':unittest.main()
