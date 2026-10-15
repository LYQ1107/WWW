"""Read-only mapping of already selected V3 TRAIN; no mining or teachers."""
from pathlib import Path
import json

from cv_pil_dev12_assets import require
from cv_pil_native_handoff import (sha256, safe_torch_load, normalize_event,
                                   load_trackvggt_bundle, FROZEN_STAGE2_SHA256,
                                   FROZEN_CONFIG_SHA256)

FIT_SCENES = {4,5,6,7,8,9,13,17,20,21,24}


def load_fit(cohort_path, expected_sha, runtime, producer_repo):
    """Use original frozen Top8/latest4 TRAIN inputs for all three arms.

    This selected training interface is explicitly not full-Hungarian context
    and cannot be used as the DEV12 native evaluation input.
    """
    cohort_path=Path(cohort_path); runtime=Path(runtime).resolve(); producer_repo=Path(producer_repo).resolve()
    require(sha256(cohort_path)==expected_sha, 'V3 TRAIN cohort SHA differs')
    c=json.loads(cohort_path.read_text())
    require(c['cohort_version']==3 and c['native_feature_dim']==1152 and
            c['official_test_used'] is False and c['used_teacher_results_for_selection'] is False and
            set(c['fit_scene_ids'])==FIT_SCENES and c['validation_scene_ids']==[12], 'Unauthorized TRAIN split/source')
    fit=c['fit'];require(len(fit)==3000 and len({tuple(r['key']) for r in fit})==3000 and
                         all(r['scene'] in FIT_SCENES and r['key'][0]==r['scene'] for r in fit),
                         'V3 frozen TRAIN scope differs')
    artifacts=c['source_artifacts']
    for name,digest in artifacts.items():
        p=Path(name).resolve()
        require(p.is_relative_to(runtime) or p.is_relative_to(producer_repo), 'TRAIN artifact outside authorized roots')
        require(sha256(p)==digest, 'TRAIN artifact SHA differs: '+name)
    bundles={};result=[]
    for r in fit:
        p=Path(r['event_file']).resolve()
        require(p.is_relative_to(runtime) and str(p) in artifacts, 'Unbound TRAIN payload')
        if str(p) not in bundles:
            mp=p.parent/'manifest.json';require(str(mp) in artifacts, 'Unbound TRAIN manifest')
            m=json.loads(mp.read_text())
            if m['status']=='PASS_READONLY_EVENTS_POST_PARITY_LABELS':
                b=load_trackvggt_bundle(p.parent,expected_scope='fit',expected_manifest_sha256=artifacts[str(mp)])
            else:
                require(m['status']=='COMPLETE_LOCKED_V3_SELECTED_NATIVE_FEATURES' and
                        m['full_prediction_file_bytes_exact'] is True and m['selected_actual_decisions_exact'] is True and
                        m['gt_labels_after_prediction_and_solver_parity'] is True and
                        m['future_history'] is False and m['GT_query_features'] is False and m['teacher_calls']==0 and
                        m['video_id'] in FIT_SCENES, 'Incomplete/leaking V3 TRAIN source')
                require(m['provenance']['checkpoint_sha256']==FROZEN_STAGE2_SHA256 and
                        m['provenance']['config_sha256']==FROZEN_CONFIG_SHA256, 'TRAIN GMT source differs')
                payload=safe_torch_load(p);require(payload['feature_dim']==1152, 'Full1152 TRAIN feature required')
                b=dict(manifest=dict(m,scope='fit'),payload=payload)
            bundles[str(p)]=b
        event=normalize_event(bundles[str(p)],r['event_index'])
        require(tuple(event['INPUT_ONLY']['event_key'])==tuple(r['key']) and
                event['OFFLINE_ONLY']['split']=='fit', 'TRAIN physical key mismatch')
        result.append(event)
    return result,dict(cohort_sha256=expected_sha,fit_queries=3000,scenes=sorted(FIT_SCENES),
                       native_feature_dim=1152,teacher_inputs=False,
                       representation='frozen selected native Top8/latest4; not full native solver context',
                       unique_known_person_groups=len({(e['INPUT_ONLY']['video'],e['OFFLINE_ONLY']['GT_person'])
                           for e in result if e['OFFLINE_ONLY']['GT_person'] is not None}))
