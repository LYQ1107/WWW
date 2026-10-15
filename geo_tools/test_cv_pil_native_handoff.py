"""Adversarial source, identity and causal contracts for CV-PIL's own adapter."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv_pil_native_handoff as h


def fixture():
    def obs(key,frame,camera):
        return dict(source=list(key),image_id=key[0],frame=frame,camera=camera,
                    feature=torch.ones(1152,dtype=torch.float16))
    call=dict(queries=[[200,0]],track_ids=[10,20],track_scores=torch.tensor([[1.,0.]]),
              references=[dict(source=[100,0],track=10),dict(source=[101,0],track=20)],
              histories={10:dict(sources=[[100,0]]),20:dict(sources=[[101,0]])})
    event=dict(key=[4,200,0],query_source=[200,0],query_row=0,call_index=0,
               candidate_tracks=[10,20],candidate_track_indices=[0,1],native_track_scores=[1.,0.],
               positive_tracks=[20],ambiguous_tracks=[],query_gt_identity=7,top8_positive=True,
               native_top1_state='wrong_link',actual_primary_state='wrong_link',
               actual_solver_scope='primary before memory; own frozen OFF history')
    body=dict(feature_dim=1152,events=[event],calls={0:call},
              observations={(200,0):obs((200,0),10,2),(100,0):obs((100,0),8,1),(101,0):obs((101,0),9,1)})
    manifest=dict(status='PASS_READONLY_EVENTS_POST_PARITY_LABELS',prefix40_raw_prediction_exact=True,
                  gt_query_features=False,future_query_features=False,vggt_calls=0,dinov2_calls=0,
                  scope='fit',video_id=4,selection_uses_gt=True,native_reid_dim=1152,selected_events=1,
                  provenance=dict(checkpoint_sha256=h.FROZEN_STAGE2_SHA256,config_sha256=h.FROZEN_CONFIG_SHA256,
                                  annotation_sha256='a'*64,commit='b'*40,
                                  source_hashes={'gtr/modeling/roi_heads/gtr_roi_heads.py':'c'*64}))
    return dict(manifest=manifest,payload=body)


class TestNativeHandoff(unittest.TestCase):
    def test_source_manifest_and_payload_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);b=fixture();torch.save(b['payload'],folder/'events.pt')
            b['manifest']['artifacts']={'events.pt':h.sha256(folder/'events.pt')}
            (folder/'manifest.json').write_text(json.dumps(b['manifest']))
            expected=h.sha256(folder/'manifest.json')
            loaded=h.load_trackvggt_bundle(folder,expected_scope='fit',expected_manifest_sha256=expected)
            self.assertEqual(loaded['feature_contract'],h.FEATURE_CONTRACT)
            with self.assertRaises(ValueError):h.load_trackvggt_bundle(folder)
            with self.assertRaises(ValueError):h.load_trackvggt_bundle(folder,expected_manifest_sha256='d'*64)
            (folder/'events.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):h.load_trackvggt_bundle(folder,expected_manifest_sha256=expected)

    def test_checkpoint_config_and_split_contract(self):
        for field,value in [('checkpoint_sha256','0'*64),('config_sha256','0'*64),('annotation_sha256','bad')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                b=fixture();b['manifest']['provenance'][field]=value;folder=Path(d)
                torch.save(b['payload'],folder/'events.pt');b['manifest']['artifacts']={'events.pt':h.sha256(folder/'events.pt')}
                (folder/'manifest.json').write_text(json.dumps(b['manifest']))
                with self.assertRaises(ValueError):h.load_trackvggt_bundle(folder,expected_manifest_sha256=h.sha256(folder/'manifest.json'))

    def test_full_1152_contract_finite_nonzero_and_physical_source(self):
        for change in ('dim','nan','zero','source','camera'):
            b=fixture();obs=b['payload']['observations'][(101,0)]
            if change=='dim':obs['feature']=torch.ones(1024)
            if change=='nan':obs['feature'][0]=float('nan')
            if change=='zero':obs['feature'].zero_()
            if change=='source':obs['source']=[999,0]
            if change=='camera':obs['camera']=3
            with self.subTest(change=change),self.assertRaises(ValueError):h.normalize_event(b,0)

    def test_candidate_native_id_index_score_and_top8_legality(self):
        for change in ('id','index','score','not_top','hist_owner','duplicate'):
            b=fixture();e=b['payload']['events'][0];call=b['payload']['calls'][0]
            if change=='id':e['candidate_tracks'][0]=99
            if change=='index':e['candidate_track_indices'][0]=8
            if change=='score':e['native_track_scores'][0]=1.1
            if change=='not_top':
                e['candidate_tracks'].reverse();e['candidate_track_indices'].reverse();e['native_track_scores'].reverse()
            if change=='hist_owner':call['histories'][10]['sources']=[[101,0]]
            if change=='duplicate':e['candidate_tracks']=[10,10]
            with self.subTest(change=change),self.assertRaises(ValueError):h.normalize_event(b,0)

    def test_future_rejected_same_time_excluded_and_empty_history_allowed(self):
        b=fixture();b['payload']['observations'][(101,0)]['frame']=11
        with self.assertRaisesRegex(ValueError,'future'):h.normalize_event(b,0)
        b['payload']['observations'][(101,0)]['frame']=10
        r=h.normalize_event(b,0)
        self.assertEqual(r['INPUT_ONLY']['past_histories'][1],[])
        self.assertEqual(r['OFFLINE_ONLY']['same_time_sources_excluded'],1)
        self.assertFalse(r['OFFLINE_ONLY']['correctable_primary_error'])

    def test_aliases_are_multi_positive_not_false_negative_identities(self):
        b=fixture();e=b['payload']['events'][0];e['positive_tracks']=[10,20];e['native_top1_state']='correct_link'
        r=h.normalize_event(b,0)['OFFLINE_ONLY']
        self.assertEqual(r['certified_target_indices'],(0,1))
        self.assertIsNone(r['certified_target_index'])
        self.assertEqual(r['alias_positive_count'],2)
        b['payload']['events'][0]['ambiguous_tracks']=[10]
        with self.assertRaisesRegex(ValueError,'Conflicting'):h.normalize_event(b,0)

    def test_unknown_absent_candidates_and_primary_are_not_certified_final_fixes(self):
        for state in ['unknown_query','unknown_candidate_coverage','correct_reject_within_recorded_candidates']:
            b=fixture();e=b['payload']['events'][0];e.update(query_gt_identity=None,positive_tracks=[],top8_positive=False,actual_primary_state=state,native_top1_state='unknown_query')
            r=h.normalize_event(b,0)['OFFLINE_ONLY']
            self.assertEqual(r['certified_target_indices'],())
            self.assertFalse(r['correctable_primary_error']);self.assertFalse(r['final_native_commit_certified'])
        b=fixture();b['payload']['events'][0]['ambiguous_tracks']=[10];b['payload']['events'][0]['native_top1_state']='ambiguous_reference'
        self.assertEqual(h.normalize_event(b,0)['OFFLINE_ONLY']['candidate_known_mask'],(False,True))

    def test_legacy_missing_top1_label_is_derived_and_conflicts_rejected(self):
        b=fixture();del b['payload']['events'][0]['native_top1_state']
        self.assertEqual(h.normalize_event(b,0)['OFFLINE_ONLY']['native_top1_state'],'wrong_link')
        b['payload']['events'][0]['native_top1_state']='correct_link'
        with self.assertRaisesRegex(ValueError,'conflicts'):h.normalize_event(b,0)

    def test_duplicate_dev_query_and_cohort_sha_fail_before_payload_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cohort.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA'):h.load_frozen_cohort(Path(d),p,'0'*64)
            c=dict(status='LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT',native_feature_dim=1152,
                used_teacher_results_for_selection=False,fit_scene_ids=[4,5,6,7,8],validation_scene_ids=[12],
                fit=[dict(key=[4,i,0]) for i in range(1000)],validation=[dict(key=[12,0,0]) for _ in range(300)])
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'twice'):h.load_frozen_cohort(Path(d),p,h.sha256(p))

    def test_gt_is_offline_and_cannot_change_or_enter_actor_inputs(self):
        b=fixture();r=h.normalize_event(b,0);actor=h.inference_inputs(r)
        self.assertEqual(set(actor),h.INPUT_KEYS)
        old=actor['query_feature'].clone();r['OFFLINE_ONLY']['GT_person']=999
        self.assertTrue(torch.equal(h.inference_inputs(r)['query_feature'],old))
        r['INPUT_ONLY']['GT_person']=999
        with self.assertRaisesRegex(ValueError,'offline'):h.inference_inputs(r)
        r=h.normalize_event(b,0);r['INPUT_ONLY']['past_histories'][0][0]['GT_person']=7
        with self.assertRaisesRegex(ValueError,'Offline'):h.inference_inputs(r)
        actor=h.normalize_event(b,0)['INPUT_ONLY'];actor['query_feature'].zero_()
        self.assertGreater(float(b['payload']['observations'][(200,0)]['feature'].sum()),0)

    def test_physical_keys_and_indices_fail_on_bool_or_negative(self):
        for key in [[True,0],[-1,0],[1.1,0]]:
            with self.assertRaises(ValueError):h.source_key(key)
        with self.assertRaises(ValueError):h.normalize_event(fixture(),-1)

    def test_release_header_requires_sha_authorized_split_and_bound_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);artifact=folder/'source.json';artifact.write_text('{}')
            m=dict(schema='cvpil_native_release/v1',status='FROZEN_READY_FOR_CONSUMER',
                feature_contract=h.FEATURE_CONTRACT,gt_in_actor_inputs=False,
                checkpoint_sha256=h.FROZEN_STAGE2_SHA256,config_sha256=h.FROZEN_CONFIG_SHA256,
                fit_scene_ids=[4],development_scene_ids=[12],development_selection_uses_gt=False,
                official_test_used=False,source_artifacts={str(artifact):h.sha256(artifact)})
            p=folder/'release.json';p.write_text(json.dumps(m))
            h.verify_mined_release(p,h.sha256(p),folder,{4},{12})
            with self.assertRaises(ValueError):h.verify_mined_release(p,h.sha256(p),folder,{4},{13})
            artifact.write_text('{"changed":true}')
            with self.assertRaises(ValueError):h.verify_mined_release(p,h.sha256(p),folder,{4},{12})


if __name__=='__main__':unittest.main()
