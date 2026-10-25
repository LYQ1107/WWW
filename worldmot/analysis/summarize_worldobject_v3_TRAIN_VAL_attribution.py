#!/usr/bin/env python3
"""Publish actual retained-checkpoint TRAIN VAL evidence without reselection."""
import argparse
import json
from pathlib import Path
from summarize_worldobject_v3 import read,sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    rows=[];metrics=[];methods=('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3')
    refinement_path=a.output/'worldobject_v3_recovered_view_frequency_refinement.json'
    refinement=read(refinement_path) if refinement_path.is_file() else {'replays':[]}
    refined={(x['method'],x['selection_epoch']):x for x in refinement['replays']}
    for method in methods:
        for epoch in range(1,5):
            folder=a.run_root/'diagnostics/retained_TRAIN_VAL'/method/f'epoch_{epoch:02d}'
            path=folder/'attribution.json'
            if not path.is_file():continue
            x=read(path)
            if x['status']!='COMPLETE' or x['source_sha256']!='6bcdcdb1a6b3683c4e406b0791416a77e5e9e64f165fbad4a61667698b64645f':raise ValueError('INCOMPLETE_OR_DIFFERENT_FROZEN_REPLAY')
            rows.append({'method':method,'selection_epoch':epoch,'checkpoint_sha256':x['checkpoint_sha256'],
                'timestamps':x['timestamps'],'scenes':x['scenes'],'exact_original_diagnostics':x['original_epoch_prediction_diagnostics_exactly_reproduced'],
                'original_diagnostic_mismatches':x['original_epoch_diagnostic_mismatches'],
                'counts':x['counts'],'birth_funnel':x['birth_funnel'],'lifecycle_calibration':x['lifecycle_calibration'],
                'per_scene':{sid:{'timestamps':s['timestamps'],'counts':s['analysis']['counts'],
                    'error_strata':s['analysis']['error_strata'],
                    'recovery_examples':s['analysis']['examples'].get('RECOVERY_NEXT_FRAME_ID_BREAK',[])} for sid,s in x['per_scene'].items()},
                'recovered_view_refinement':refined.get((method,epoch)),
                'raw_report':str(path),'raw_report_sha256':sha(path),'analysis_sha256':x['analysis_sha256']})
            metric=folder/'TRAIN_VAL_metrics.json'
            if metric.is_file():metrics.append(read(metric))
    common={'schema':'worldobject_v3_retained_TRAIN_VAL_attribution_v1',
        'status':'COMPLETE_AVAILABLE_RETAINED_SELECTION_REPLAYS' if len(rows)==8 else 'PARTIAL_REAL_TRAIN_VAL_REPLAYS',
        'completed_replays':len(rows),'expected_replays':8,'R2_modified':False,'optimizer_used':False,'TEST_read':False,
        'scope':'Six complete fixed TRAIN VAL scenes 4/8/12/16/20/24, clean current GT matching only',
        'official_TRAIN_VAL_scope':'Original raw-GT-legal scenes 4/8/12 only; 16/20/24 excluded, GT unchanged',
        'official_TEST_result':False,'new_checkpoint_or_budget_selection':False,
        'missing_historical_evidence':{'FULL_NO_GEOMETRY_V3_selection_epochs':[1,2,3,4],
            'reason':'Original selection weights overwritten before archival. No retraining, final-refit weights or pre-R2 substitute.'},
        'source_sha256':'6bcdcdb1a6b3683c4e406b0791416a77e5e9e64f165fbad4a61667698b64645f',
        'summarizer_sha256':sha(__file__)}
    write_json(a.output/'worldobject_v3_retained_TRAIN_VAL_summary.json',{**common,'replays':rows,'official_TRAIN_VAL_metrics':metrics})
    identity_keys=('temporal_ID_changes','temporal_comparisons','previous_correct_two_view_fusion_next_frame_opportunities',
        'previous_correct_two_view_fusion_next_frame_ID_break','new_id_despite_unique_clean_active_history',
        'unique_clean_history_NULL_despite_current_detection','historical_id_reused_for_previously_different_clean_identity',
        'cumulative_identity_contamination_onsets','clean_observations_reusing_cumulatively_mixed_id')
    metadata=lambda x:{k:x[k] for k in ('method','selection_epoch','checkpoint_sha256','timestamps','exact_original_diagnostics','raw_report_sha256')}
    write_json(a.output/'worldobject_v3_birth_decision_attribution.json',{**common,'replays':[
        {**metadata(x),'birth_funnel':x['birth_funnel'],'first_clean_detection_pairs':x['counts'].get('true_first_clean_detection_same_person_pairs',0),
         'first_clean_detection_splits':x['counts'].get('true_first_clean_detection_splits',0),
         'false_Birth_outcome_observations':x['counts'].get('new_id_despite_unique_clean_active_history',0)} for x in rows],
        'definition':'NEW_SAME uses absence from clean active predicted history; first clean video detection is counted separately. Raw DIFFERENT high-score count is not a selected wrong merge count.'})
    write_json(a.output/'worldobject_v3_temporal_identity_comparison.json',{**common,'replays':[
        {**metadata(x),'counts':{k:x['counts'].get(k,0) for k in identity_keys},
         'per_scene_counts':{k:v['counts'] for k,v in x['per_scene'].items()}} for x in rows],
        'definition':'All counts are causal TRAIN VAL diagnostics, not official TEST IDSW or CVIDF1.'})
    write_json(a.output/'worldobject_v3_lifecycle_calibration.json',{**common,'replays':[
        {**metadata(x),'calibration':x['lifecycle_calibration'],'actual_NULL_despite_unique_current_person':x['counts'].get('unique_clean_history_NULL_despite_current_detection',0),
         'raw_NULL_logit_nonnegative':x['counts'].get('raw_NULL_logit_nonnegative',0),
         'trained_NULL_margin_probability_ge_0_5':x['counts'].get('trained_NULL_margin_probability_ge_0_5',0)} for x in rows],
        'NULL_semantics':'sigmoid(NULL logit minus max current assignment logit), exactly the trained loss; max zero when no current candidates',
        'existence_semantics':'Target includes matching current observation OR missed+1<=30; high probabilities must be compared to this retention target.'})
    spatial=[{**metadata(x),'recovered_view_counts_primary':None if x['recovered_view_refinement'] is None else x['recovered_view_refinement']['counts'],
        'raw_any_view_counts_v1':{k:v for k,v in x['counts'].items() if 'recovery' in k or 'recoveries' in k or 'GEOMETRY_GATE' in k},
        'per_scene':{sid:{'timestamps':s['timestamps'],'counts':{k:v for k,v in s['counts'].items() if 'recovery' in k or 'recoveries' in k or 'GEOMETRY_GATE' in k},
            'raw_v1_examples_any_view_only':s['recovery_examples']} for sid,s in x['per_scene'].items()},
        'refinement_source':'worldobject_v3_recovered_view_frequency_refinement.json'} for x in rows]
    write_json(a.output/'worldobject_v3_spatial_temporal_attribution.json',{**common,'replays':spatial,
        'mechanism':'Accepted correct visual identity recovery leaves old center valid and clears missed to zero. Next-frame hard spatial gating can reject that identity; actual ID reuse and new Birth are counted separately.',
        'definition_correction':'Primary counts follow only the camera(s) of the accepted recovery. Raw v1 any-view ID-change counts can include a pre-existing split in another camera and are not the primary recovery-failure frequency.',
        'single_vs_two_view_recovery':'All observed accepted recoveries in these eight replays have one assigned camera; do not describe them as already successful shared-ID two-view fusion.',
        'R3':'Separate branch research/worldmot-v3-r3-state-update-fix-20261025; CPU-verified spatial abstention, one TRAIN-only replay registered; no long training.'})
    write_json(a.output/'worldobject_v3_geometry_identity_attribution.json',{**common,'replays':[
        {**metadata(x),'counts':{k:v for k,v in x['counts'].items() if 'clean_correct_history_' in k or 'spatial' in k or 'recovery' in k},
         'recovered_view_counts_primary':None if x['recovered_view_refinement'] is None else x['recovered_view_refinement']['counts'],
         'birth_funnel':x['birth_funnel']} for x in rows],
        'causal_limit':'Actual gate rejection is observed directly. Model quality and geometry net benefit still require the registered Full-vs-NoGeometry TEST contrast; these per-epoch TRAIN rows do not replace it.'})
    inventory=read(a.run_root/'audit_snapshots/epoch_archive_index.json')
    write_json(a.output/'worldobject_v3_epoch_snapshot_inventory.json',{'status':'AVAILABLE_SNAPSHOTS_VERIFIED_BY_OBSERVER',
        'source_sha256':inventory['source_sha256'],'records':inventory['records'],
        'missing_NoGeometry_selection_epochs':[1,2,3,4]})
    print(json.dumps({'completed_replays':len(rows),'scored_TRAIN_VAL_replays':len(metrics),'official_TEST_claim':False}))


if __name__=='__main__':main()
