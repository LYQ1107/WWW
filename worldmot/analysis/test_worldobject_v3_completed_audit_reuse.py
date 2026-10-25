from copy import deepcopy
from audit_worldobject_v3_identities import completed_posthoc_matches


def complete():
    return {'source_analysis_sha256':'code','GT_annotations_sha256':'original_GT',
        'prediction_proofs':{'Full':[{'tracks_sha256':'immutable'}]},
        'fixed_candidate_boxes_verified':True,'fixed_candidate_scores_verified':True}


def test_only_identical_completed_scores_boxes_predictions_GT_and_code_can_be_reused():
    report=complete();proofs=report['prediction_proofs']
    assert completed_posthoc_matches(report,deepcopy(report),proofs,'code','original_GT')
    assert not completed_posthoc_matches(report,deepcopy(report),proofs,'new_code','original_GT')
    assert not completed_posthoc_matches(report,deepcopy(report),proofs,'code','changed_GT')
    changed=deepcopy(proofs);changed['Full'][0]['tracks_sha256']='changed'
    assert not completed_posthoc_matches(report,deepcopy(report),changed,'code','original_GT')


def test_partial_or_no_score_verification_never_skips_the_audit():
    report=complete();partial=deepcopy(report);del partial['fixed_candidate_scores_verified']
    assert not completed_posthoc_matches(report,partial,report['prediction_proofs'],'code','original_GT')
    assert not completed_posthoc_matches({},report,report['prediction_proofs'],'code','original_GT')
