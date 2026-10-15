"""CV-PIL consumer tests for actual shard semantics and one-step replay."""
import copy
import unittest
import torch

from cv_pil_dev12_shards import NativeShardAssets, tensor_digest
from run_cv_pil_v3_old_heads import primary_solve, classify, paired


def fixture():
    a = NativeShardAssets.__new__(NativeShardAssets); a.variants = {}; a.max_aggregation_error = 0.
    ids = list(range(10, 19)); refs = [[100+i, 0] for i in range(9)]; q = [[200, 0], [201, 1]]
    a.observations = {tuple(s): dict(source=s, frame=i+1, camera=1, image_id=s[0], feature=torch.ones(1152))
                      for i, s in enumerate(refs)}
    a.observations.update({tuple(s): dict(source=s, frame=10+i, camera=2-i, image_id=s[0], feature=torch.ones(1152))
                           for i, s in enumerate(q)})
    raw = torch.zeros(2, 9); score = torch.full((2, 9), .1)
    c = dict(call_id=0, stage='run_global_tracker_plus', queries=q,
             references=[dict(source=s, track=t) for s, t in zip(refs, ids)], track_ids=ids,
             track_scores=score, thresholds=torch.full((9,), .05), reference_block_sizes=[9],
             query_feature_keys=[dict(source=s, variant=None) for s in q],
             reference_feature_keys=[dict(source=s, variant=None) for s in refs],
             observed_track_lengths={str(t): 3 for t in ids}, actual_candidate_support_lengths=[1]*9,
             raw_logits=raw, activation_block_sizes=[9], single_reference_probabilities=score.clone(),
             completed_camera_images_before_call=[],
             decision=dict(accepted_track_ids=[10, 11], final_track_ids=[10, 11],
                hungarian_pairs=[dict(row=i, track=10+i, score=float(score[i,i]),
                                      threshold=float(torch.tensor(.05)), accepted=True) for i in range(2)]))
    a.calls = {0:c}; a.query_index = {(12,*s):(0,i) for i,s in enumerate(q)}
    return a,c


class TestV3Replay(unittest.TestCase):
    def test_full_native_shard_call_and_all_candidates(self):
        a,c=fixture();a._validate_call(c)
        actor=a.actor((12,200,0));self.assertEqual(len(actor['native_candidate_ids']),9)
        self.assertEqual(len(actor['past_histories']),9)

    def test_top8_and_misaligned_columns_rejected(self):
        for kind in ('matrix','ids'):
            a,c=fixture()
            if kind=='matrix':c['track_scores']=c['track_scores'][:,:8]
            else:c['track_ids'].reverse()
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'Full native'):a._validate_call(c)

    def test_actual_descriptor_variant_preserved_not_replaced_by_raw_roi(self):
        a,c=fixture();spec=c['reference_feature_keys'][0];spec['variant']='actual-native-version'
        value=torch.full((1152,),2.)
        a.variants[spec['variant']]=dict(source=spec['source'],feature=value,feature_sha256=tensor_digest(value))
        a._validate_call(c)
        self.assertEqual(a.actor((12,200,0))['past_histories'][0][0]['feature'][0].item(),2.)
        self.assertEqual(a.observations[(100,0)]['feature'][0].item(),1.)

    def test_future_and_unprocessed_same_frame_native_reference_rejected(self):
        for frame in (10,11):
            a,c=fixture();a.observations[(100,0)]['frame']=frame
            with self.subTest(frame=frame),self.assertRaisesRegex(ValueError,'Noncausal'):a._validate_call(c)

    def test_joint_solver_coupling_and_strict_original_threshold(self):
        s=torch.tensor([[.9,.8],[.85,.1]])
        self.assertEqual(primary_solve(s,[10,11],torch.tensor([.1,.1])),[11,10])
        self.assertEqual(primary_solve(torch.tensor([[.5]]),[10],torch.tensor([.5])),[None])

    def test_unknown_is_not_known_error_and_aliases_are_positive(self):
        label=dict(GT_person=7,certified_target_indices=(0,1),candidate_known_mask=(True,True,False))
        self.assertTrue(classify(10,label,[10,11,12]));self.assertTrue(classify(11,label,[10,11,12]))
        self.assertIsNone(classify(12,label,[10,11,12]));self.assertFalse(classify(None,label,[10,11,12]))

    def test_repeated_fixes_are_one_person_and_unknown_changes_separate(self):
        r=[dict(key=[12,i,0],person=7,C2_primary=False,C3_primary=True) for i in range(5)]
        r.append(dict(key=[12,9,0],person=8,C2_primary=None,C3_primary=True))
        p=paired(r,'C2','C3','primary')
        self.assertEqual(p['certified_fixes'],5);self.assertEqual(p['unique_fix_person_groups'],1)
        self.assertEqual(p['unknown_to_correct'],1)


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
