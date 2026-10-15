"""Synthetic adversarial contract tests, never a real-video performance claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from cv_pil_dev12_assets import (Dev12Assets, SCHEMA, SCORE_CONTRACT, digest_json,
                                off_passthrough, sha256, FROZEN_STAGE2_SHA256,
                                FROZEN_CONFIG_SHA256)
from cv_pil_v2_model import tensorize
from inspect_cv_pil_dev12_assets import coverage


def fixture():
    ids = list(range(10, 19)); queries = [[12, 200, 0], [12, 201, 1]]
    refs = [[12, 100 + i, 0] for i in range(9)]
    observations = [dict(key=k, frame=i + 1, camera=1, causal_index=i,
                         feature_ref=str(i)) for i, k in enumerate(refs)]
    observations += [dict(key=k, frame=10+i, camera=2-i, causal_index=100+i,
                          feature_ref='q'+str(i)) for i, k in enumerate(queries)]
    features = {o['feature_ref']: torch.arange(1152).float() + 1 for o in observations}
    logits = torch.zeros(2, 9)
    scores = torch.cat((logits, torch.zeros(2, 1)), 1).softmax(1)[:, :-1].tolist()
    call = dict(call_id='original-call-0', query_keys=queries, candidate_ids=ids,
                track_scores=scores, solver_matrix=copy.deepcopy(scores), reference_keys=refs,
                reference_candidate_ids=ids, reference_block_sizes=[9], logits=logits.tolist(),
                histories=[dict(global_id=g, sources=[s], observed_length=3, support_count=1)
                           for g, s in zip(ids, refs)], score_contract=SCORE_CONTRACT)
    action = dict(pre_state_sha256='a'*64, post_state_sha256='b'*64,
                  hungarian=[[0, 0], [1, 1]], thresholds=[.05]*9,
                  accepted_ids=[10, 11], bank_ids=[None, None], birth_ids=[None, None],
                  final_ids=[10, 11], gallery_update_sha256='c'*64, bank_update_sha256='d'*64)
    context = {'original-call-0': dict(query_keys=copy.deepcopy(queries), candidate_ids=ids.copy(),
                                     solver_matrix_sha256=digest_json(scores),
                                     observation_metadata_sha256=digest_json(observations),
                                     action_sha256=digest_json(action))}
    supervision = [dict(key=k, GT_person=7, candidate_persons=[7, 7, 8, None, 9, 10, 11, 12, 13],
                        primary_state='correct_link', unknown_reason=None, impure_candidate_ids=[],
                        reachable_positive_ids=[10, 11]) for k in queries]
    provenance = dict(checkpoint_sha256=FROZEN_STAGE2_SHA256, config_sha256=FROZEN_CONFIG_SHA256,
                      source_commit='a'*40, video_sha256='b'*64, camera_map_sha256='c'*64,
                      gt_sha256='d'*64)
    scope = dict(video=12, selection_uses_gt=False, all_primary_queries=True,
                 synchronized_groups=1200, frozen_before_export=True,
                 preregistration_sha256='f'*64, query_count=2,
                 query_keys_sha256=digest_json(queries))
    return dict(inputs=dict(observations=observations, calls=[call]), features=features,
                actions={'original-call-0': action}, native_context=context,
                offline_supervision=supervision, provenance=provenance, scope=scope)


def write_release(folder, f):
    # Deliberately arbitrary filenames, independent of semantic role names.
    artifacts = {}
    for i, role in enumerate(('inputs', 'features', 'actions', 'native_context', 'offline_supervision')):
        file = folder / ('piece_' + str(i) + ('.pt' if role == 'features' else '.json'))
        if role == 'features': torch.save(f[role], file)
        else: file.write_text(json.dumps(f[role]))
        artifacts[role] = dict(path=file.name, sha256=sha256(file))
    m = dict(schema=SCHEMA, status='FROZEN_COMPLETE', provenance=f['provenance'], scope=f['scope'],
             artifacts=artifacts)
    p = folder/'arbitrary-release-name.json'; p.write_text(json.dumps(m))
    return p


def fingerprint(actor):
    b = tensorize(actor)
    return (actor['event_key'], actor['native_candidate_ids'], b.query.numpy().tobytes(),
            b.histories.numpy().tobytes(), b.owners.numpy().tobytes(), b.age.numpy().tobytes(),
            b.cross_view.numpy().tobytes(), b.native_scores.numpy().tobytes(), b.destination_camera)


class TestFullNativeAssets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name); self.f = fixture()

    def load(self):
        p = write_release(self.folder, self.f)
        return Dev12Assets(p, sha256(p), self.f['provenance'], self.folder, self.f['scope'])

    def test_full_1152_including_tail_is_preserved(self):
        a = self.load().actor((12, 200, 0))
        self.assertEqual(a['query_feature'].shape, (1152,))
        self.assertEqual(a['query_feature'][-1].item(), 1152)
        self.f['features']['q0'] = torch.ones(1024)
        with self.assertRaisesRegex(ValueError, '1152'): self.load()

    def test_strict_causal_clock_and_future_frames(self):
        for field, value in [('causal_index', 100), ('frame', 11)]:
            self.f = fixture(); self.f['inputs']['observations'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'causal|future'): self.load()

    def test_physical_camera_not_candidate_index(self):
        a = self.load().actor((12, 200, 0))
        self.assertEqual(a['camera'], 2)
        self.assertTrue(all(h[0]['camera'] == 1 and h[0]['cross_view'] for h in a['past_histories']))
        self.f['inputs']['observations'][0]['camera'] = 0
        with self.assertRaisesRegex(ValueError, 'physical camera'): self.load()

    def test_candidate_score_column_alignment(self):
        self.f['inputs']['calls'][0]['candidate_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'full Hungarian'): self.load()

    def test_gt_cannot_add_candidate_oracle(self):
        self.f['offline_supervision'][0]['reachable_positive_ids'].append(999)
        a = self.load()
        self.assertNotIn(999, a.actor((12, 200, 0))['native_candidate_ids'])
        with self.assertRaisesRegex(ValueError, 'GT cannot add'): a.supervision()

    def test_unknown_is_not_a_negative(self):
        s = self.load().supervision()[(12, 200, 0)]
        self.assertFalse(s['candidate_known_mask'][3])
        self.f['offline_supervision'][0].update(GT_person=None, reachable_positive_ids=[],
                                              unknown_reason='unmatched_GT', primary_state='unknown')
        s = self.load().supervision()[(12, 200, 0)]
        self.assertFalse(any(s['candidate_known_mask']))
        self.assertEqual(s['certified_target_indices'], ())

    def test_multiple_native_aliases_are_all_positives(self):
        s = self.load().supervision()[(12, 200, 0)]
        self.assertEqual(s['certified_target_indices'], (0, 1))
        self.assertTrue(s['candidate_known_mask'][0] and s['candidate_known_mask'][1])

    def test_missing_history_feature_fails_no_interpolation(self):
        del self.f['features']['0']
        with self.assertRaisesRegex(ValueError, 'Missing native history'): self.load()

    def test_sha_tamper_fails_for_manifest_and_payload(self):
        p = write_release(self.folder, self.f)
        with self.assertRaisesRegex(ValueError, 'Manifest SHA'):
            Dev12Assets(p, '0'*64, self.f['provenance'], self.folder, self.f['scope'])
        (self.folder/'piece_0.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Artifact SHA'):
            Dev12Assets(p, sha256(p), self.f['provenance'], self.folder, self.f['scope'])

    def test_gt_modification_cannot_change_actor_generation(self):
        a = self.load(); before = fingerprint(a.actor((12, 200, 0)))
        for s in self.f['offline_supervision']:
            s.update(GT_person=999, candidate_persons=[999]*9, reachable_positive_ids=list(range(10, 19)))
        b = self.load()
        self.assertEqual(before, fingerprint(b.actor((12, 200, 0))))
        # Poison labels *after* verification: actor still never opens that file.
        b.paths['offline_supervision'].write_text('not JSON')
        self.assertEqual(before, fingerprint(b.actor((12, 200, 0))))
        with self.assertRaisesRegex(ValueError, 'Offline SHA'): b.supervision()

    def test_top8_local_matrix_cannot_impersonate_full_context(self):
        c = self.f['inputs']['calls'][0]
        c['candidate_ids'] = c['candidate_ids'][:8]
        c['track_scores'] = [r[:8] for r in c['track_scores']]
        c['solver_matrix'] = [r[:8] for r in c['solver_matrix']]
        with self.assertRaisesRegex(ValueError, 'Top8/local'): self.load()

    def test_missing_full_query_row_also_rejected(self):
        self.f['inputs']['calls'][0]['query_keys'].pop()
        with self.assertRaisesRegex(ValueError, 'Top8/local'): self.load()

    def test_off_boundary_returns_original_predictions_and_state_exactly(self):
        predictions = b'original GMT prediction bytes\x00\xff'
        state = {'Gallery': [10], 'Bank': [11], 'counter': 42}; calls = []
        def original(images, *, token):
            calls.append((images, token)); state['counter'] += 1
            return predictions, state
        inputs = object(); out = off_passthrough(original, inputs, token='unchanged')
        self.assertIs(out[0], predictions); self.assertIs(out[1], state)
        self.assertEqual(state, {'Gallery': [10], 'Bank': [11], 'counter': 43})
        self.assertEqual(calls, [(inputs, 'unchanged')])
        # This certifies the OFF call boundary only, not real full GMT parity.

    def test_gt_is_rejected_inside_native_input_and_features(self):
        self.f['inputs']['observations'][0]['GT_person'] = 7
        with self.assertRaisesRegex(ValueError, 'Illegal fields'): self.load()
        self.f = fixture(); self.f['features']['GT_person'] = torch.ones(1152)
        with self.assertRaisesRegex(ValueError, 'Unexpected data'): self.load()

    def test_missing_or_truncated_history_is_not_silently_filled(self):
        self.f['inputs']['calls'][0]['histories'][0]['sources'] = []
        with self.assertRaisesRegex(ValueError, 'truncated native history'): self.load()

    def test_native_score_aggregation_verified(self):
        c = self.f['inputs']['calls'][0]
        c['logits'][0][0] = 3.
        with self.assertRaisesRegex(ValueError, 'normalization/aggregation'): self.load()

    def test_native_bank_birth_final_semantics_checked(self):
        self.f['actions']['original-call-0']['final_ids'][0] = 999
        self.f['native_context']['original-call-0']['action_sha256'] = digest_json(self.f['actions']['original-call-0'])
        with self.assertRaisesRegex(ValueError, 'commit semantics'): self.load()

    def test_unfrozen_scope_or_different_gmt_is_rejected(self):
        self.f['scope']['selection_uses_gt'] = True
        with self.assertRaisesRegex(ValueError, 'DEV scope'): self.load()
        self.f = fixture(); self.f['provenance']['checkpoint_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Original GMT'): self.load()

    def test_same_frame_native_reference_excluded_from_old_head_only(self):
        self.f['inputs']['observations'][0]['frame'] = 10
        self.f['native_context']['original-call-0']['observation_metadata_sha256'] = digest_json(self.f['inputs']['observations'])
        a = self.load()
        self.assertEqual(a.actor((12, 200, 0))['past_histories'][0], [])
        self.assertEqual(len(a.calls['original-call-0']['reference_keys']), 9)

    def test_input_mutation_does_not_change_verified_source(self):
        a = self.load(); actor = a.actor((12, 200, 0)); actor['query_feature'].zero_()
        self.assertGreater(a.actor((12, 200, 0))['query_feature'].sum().item(), 0)

    def test_empty_candidate_call_retains_native_birth_semantics(self):
        c = self.f['inputs']['calls'][0]
        c.update(candidate_ids=[], track_scores=[[], []], solver_matrix=[[], []],
                 reference_keys=[], reference_candidate_ids=[], reference_block_sizes=[],
                 logits=[[], []], histories=[])
        a = self.f['actions']['original-call-0']
        a.update(hungarian=[], thresholds=[], accepted_ids=[None, None],
                 birth_ids=[20, 21], final_ids=[20, 21])
        self.f['native_context']['original-call-0'].update(candidate_ids=[],
            solver_matrix_sha256=digest_json([[], []]), action_sha256=digest_json(a),
            observation_metadata_sha256=digest_json(self.f['inputs']['observations'][-2:]))
        for s in self.f['offline_supervision']:
            s.update(candidate_persons=[], reachable_positive_ids=[], primary_state='candidate_missing')
        assets = self.load()
        self.assertEqual(assets.actor((12, 200, 0))['native_candidate_ids'], ())
        self.assertEqual(assets.actions['original-call-0']['final_ids'], [20, 21])
        self.assertTrue(assets.supervision()[(12, 200, 0)]['no_legal_correct_candidate'])

    def test_independent_scope_pin_cannot_be_replaced_by_header(self):
        p = write_release(self.folder, self.f)
        scope = dict(self.f['scope'], query_count=14557)
        with self.assertRaisesRegex(ValueError, 'Independent frozen scope'):
            Dev12Assets(p, sha256(p), self.f['provenance'], self.folder, scope)

    def test_bank_recovery_is_distinct_from_birth(self):
        a = self.f['actions']['original-call-0']
        a.update(thresholds=[1.]*9, accepted_ids=[None, None], bank_ids=[50, None],
                 birth_ids=[None, 51], final_ids=[50, 51])
        self.f['native_context']['original-call-0']['action_sha256'] = digest_json(a)
        for s in self.f['offline_supervision']: s['primary_state'] = 'missed_known_positive'
        assets = self.load()
        self.assertEqual(assets.actions['original-call-0']['bank_ids'], [50, None])
        self.assertEqual(assets.actions['original-call-0']['birth_ids'], [None, 51])

    def test_other_valid_camera_id_still_must_match_original_source(self):
        self.f['inputs']['observations'][0]['camera'] = 2
        with self.assertRaisesRegex(ValueError, 'Original physical camera'): self.load()

    def test_coverage_counts_people_not_repeated_candidate_edges(self):
        report = coverage(self.load())
        self.assertEqual(report['total_queries'], 2)
        self.assertEqual(report['unique_video_person_groups'], 1)
        self.assertEqual(report['unique_video_person_camera_groups'], 2)
        self.assertIsNone(report['actual_counterfactual_committed_fixes'])
        self.assertEqual(report['certified_primary_errors'], 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
