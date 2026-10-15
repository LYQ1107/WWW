"""Paired inference-RNG trials for the fixed head-free control, no new fitting."""
import numpy as np
from common import *
from audit_camera_control import verify_pure_gmt


def main():
    seeds=[20261014,20261015,20261016];rows=[];pending=[];artifacts={}
    strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    for seed in seeds:
        baseline='causal_B0_strict_suite_v1' if seed==20261014 else f'causal_B0_seed{seed}_strict_suite_v1'
        prior=f'native_prior_seed{seed}_strict_suite_v1'
        paths=[ROOT/'results'/f'{tag}_metrics.json' for tag in (baseline,prior)]
        pair=ROOT/'results'/('native_prior_full_dev_identity_pair_audit_v1.json' if seed==20261014 else f'native_prior_seed{seed}_full_dev_identity_pair_audit_v1.json')
        if any(not path.exists() for path in [*paths,pair]):pending.append(seed);continue
        suites=[json.loads(path.read_text()) for path in paths];audit=json.loads(pair.read_text())
        if any(suite['status']!='PASS' or not suite['full_video'] or not suite['causal_online'] or suite['video_ids']!=strict for suite in suites):
            raise ValueError('incomplete causal suites')
        if audit['status']!='PASS' or not audit['full_strict_suite'] or sorted(row['video_id'] for row in audit['rows'])!=strict:
            raise ValueError('incomplete paired detection/identity audit')
        scenes=[]
        for scene in strict:
            captured=[suite['run_manifests'][str(scene)] for suite in suites]
            runs=[]
            for capture in captured:
                if sha256(capture['path'])!=capture['sha256']:raise ValueError('changed inference manifest')
                runs.append(json.loads(Path(capture['path']).read_text()))
            original,control=runs
            verify_pure_gmt(original)
            for key in ('seed','checkpoint_sha256','config_sha256','annotation_sha256','video_id','images','synchronized_groups'):
                if original[key]!=control[key]:raise ValueError('unpaired RNG/model/data inputs: '+key)
            if original['seed']!=seed or original.get('validation_native_output_prior',False) or not control.get('validation_native_output_prior',False):
                raise ValueError('wrong paired seed/control role')
            for key in ('validation_corrected_gmt_camera_index','validation_reversed_vggt_order','validation_constant_head_evidence'):
                default=None if key=='validation_constant_head_evidence' else False
                if original.get(key,default)!=control.get(key,default):raise ValueError('other control confounder')
            stats=control['native_prior_observation_stats']
            if control['native_prior_extra_encoder_or_head_loaded'] or control['native_prior_extra_learned_parameters'] or control['total_vggt_calls'] or control['dinov2_calls'] or stats['maximum_extra_vector_columns'] or stats['observed_groups']!=control['synchronized_groups']:
                raise ValueError('fixed zero-head/encoder/vector scope violated')
            pair_scene=next(row for row in audit['rows'] if row['video_id']==scene)
            predictions=[Path(pair_scene[key]) for key in ('baseline_prediction','method_prediction')]
            for index,prediction in enumerate(predictions):
                digest=sha256(prediction)
                if digest!=pair_scene['prediction_sha256'][index] or digest!=runs[index]['predictions_sha256'] or digest!=suites[index]['prediction_sha256'][str(prediction)]:
                    raise ValueError('prediction/evaluation scope mismatch')
            if pair_scene['added_detections'] or pair_scene['removed_detections']:raise ValueError('detector outputs differ')
            scenes.append(dict(video_id=scene,baseline_manifest=captured[0],prior_manifest=captured[1],
                inference_seed=seed,exact_detector_membership=True,additional_learned_parameters=0,extra_vector_columns=0,
                extra_encoder_calls=0,baseline_seconds=original['seconds'],prior_seconds=control['seconds'],
                baseline_images=original['images'],prior_images=control['images'],
                both_speed_eligible=original.get('speed_eligible',True) and control.get('speed_eligible',True)))
        delta={key:suites[1]['metrics'][key]-suites[0]['metrics'][key] for key in suites[0]['metrics']}
        rows.append(dict(inference_seed=seed,baseline_metrics=suites[0]['metrics'],prior_metrics=suites[1]['metrics'],paired_metric_delta=delta,
            baseline_cumulative_images_per_second=sum(row['baseline_images'] for row in scenes)/sum(row['baseline_seconds'] for row in scenes),
            prior_cumulative_images_per_second=sum(row['prior_images'] for row in scenes)/sum(row['prior_seconds'] for row in scenes),scenes=scenes))
        artifacts.update({str(path):sha256(path) for path in [*paths,pair]})
    deltas=[row['paired_metric_delta']['CVIDF1'] for row in rows]
    save_json(ROOT/'results/native_prior_inference_seed_stability_v1.json',dict(status='THREE_PAIRED_TRIALS_AUDITED' if len(rows)==3 else 'PARTIAL_PAIRED_TRIALS',
        declared_inference_seeds=seeds,completed_inference_seeds=[row['inference_seed'] for row in rows],pending_inference_seeds=pending,rows=rows,
        paired_cvidf1_delta_range=[min(deltas),max(deltas)] if deltas else None,
        descriptive_paired_cvidf1_sample_std_ddof1=float(np.std(deltas,ddof=1)) if len(deltas)>=2 else None,
        source_artifacts=artifacts,
        scope='same frozen checkpoint and head-free constant prior, each paired to a newly inferred baseline at the same actual inference RNG seed; all five strict scenes, exact detector keys',
        limitations='inference-RNG stability, not three independently learned models, confidence interval, architectural attribution or VGGT benefit; timing is single descriptive full replay per trial with shared-device/system effects, setup excluded; partial trials do not establish complete three-seed stability',provenance=provenance()))
    print(json.dumps(dict(completed=[row['inference_seed'] for row in rows],pending=pending,paired_cvidf1_delta_range=[min(deltas),max(deltas)] if deltas else None)))


if __name__=='__main__':main()
