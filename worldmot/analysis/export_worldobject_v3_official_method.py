#!/usr/bin/env python3
"""Bundle a complete all22/legal14 official result and measured live status."""
import argparse
from pathlib import Path
from summarize_worldobject_v3 import read,sha,write_json,official,live_timing,REFERENCES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--method',required=True)
    p.add_argument('--candidate-proof',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    root=a.root;method=a.method;path=root/'evaluation'/method/'metrics.json'
    report=read(path)
    if report['status']!='COMPLETE_VALID_SCOPE' or not report.get('VisionTrack_MATLAB',{}).get('combined'):
        raise ValueError('NATIVE_TRACK_EVAL_AND_MATLAB_MUST_BE_COMPLETE')
    proof=read(a.candidate_proof)
    if proof['status']!='ALL22_ACTUAL_CURRENT_TEST_CANDIDATES_VERIFIED' or proof['method']!=method or proof['timestamps']!=29019:
        raise ValueError('ALL22_CANONICAL_CANDIDATE_VERIFICATION_REQUIRED')
    scenes=[]
    for verified in proof['scenes']:
        sid=verified['scene'];folder=root/'predictions'/method/f'scene_{sid:02d}'
        manifest=read(folder/'manifest.json')
        if manifest['status']!='COMPLETE' or manifest['tracks_sha256']!=verified['tracks_sha256'] or manifest['GT_in_inference'] is not False:
            raise ValueError('PREDICTION_CHANGED_AFTER_CANONICAL_VERIFICATION')
        scenes.append({'scene':sid,'timestamps':manifest['timestamps'],'tracks_sha256':manifest['tracks_sha256'],
            'manifest_sha256':sha(folder/'manifest.json')})
    gatepath=root/'train'/method/'train_gate.json';gate=read(gatepath)
    if sha(gatepath.parent/gate['checkpoint'])!=gate['checkpoint_sha256']:raise ValueError('WEIGHTS_CHANGED')
    metrics=official(path);references={name:official(root.parent/relative) for name,relative in REFERENCES.items()}
    if method!='FULL_NO_GEOMETRY_V3':references['FULL_NO_GEOMETRY_V3']=official(root/'evaluation/FULL_NO_GEOMETRY_V3/metrics.json')
    differences={name:{key:metrics[key]-values[key] for key in
        ('HOTA','AssA','IDF1','CVIDF1','MOTA','IDSW','TP','FP','FN') if key in metrics and key in values}
        for name,values in references.items()}
    speed,stages=live_timing(root,method)
    write_json(a.output,{'schema':'worldobject_v3_complete_official_method_bundle_v1','status':'COMPLETE_OFFICIAL_TEST',
        'method':method,'inference_scenes':22,'inference_paired_timestamps':29019,'official_scope':'14 original-GT-legal full TEST videos',
        'blocked8_excluded':True,'original_GT_modified':False,'TEST_parameter_selection':False,
        'checkpoint_sha256':gate['checkpoint_sha256'],'train_gate_sha256':sha(gatepath),'source_sha256':gate['source_sha256'],
        'seed':gate['seed'],'training_budget':gate['training_budget'],'all24_final_training_timestamps_per_epoch':gate['final_timestamps'],
        'same_current_candidates_bbox_score_verified':True,'candidate_proof_sha256':sha(a.candidate_proof),
        'current_TEST_detections':proof['current_detections'],'scenes':scenes,'official_metrics':metrics,
        'references':references,'differences_current_minus_reference':differences,
        'live_status':'COMPLETE_REGISTERED_300_PAIR_SAMPLE' if speed else 'PENDING_REGISTERED_LIVE_MEASUREMENT',
        'live_speed':speed,'live_stages':stages,'TEST_complete_is_not_goal_complete':True,
        'training_budget_limit':'Each model follows the same registered TRAIN-only selection rule; selected final epochs can differ. This is the registered end-to-end model comparison, not a same-weights geometry toggle.',
        'timing_limit':'Cache replay is not end-to-end FPS. Registered live uses 300 pairs; historical GMT/V1/V2 uses complete-video wall timing and must remain a separate scope.'})
    print({'method':method,'metrics':{k:metrics[k] for k in ('HOTA','AssA','IDF1','CVIDF1')},'live_status':bool(speed)})


if __name__=='__main__':main()
