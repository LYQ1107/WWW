"""Complete scene1 bounded-reuse backbone controls from actual run artifacts."""
from common import *


def main():
    baseline=RUNTIME/'online/online_B0_full_v1_source44_v1'
    base=json.loads((baseline/'manifest.json').read_text())
    base_predictions=json.loads((baseline/'predictions.json').read_text())
    base_metrics=json.loads((baseline/'predictions.metrics.json').read_text())['metrics']
    if sha256(baseline/'predictions.json')!=base['predictions_sha256']:raise ValueError('baseline changed')
    specs=[('vggt','cache_vggt_fixed20_total80_scene1_full_v1'),
           ('gmt','cache_gmt_fixed20_scene1_full_v1'),('dinov2','cache_dino_fixed20_total40_scene1_full_v1')]
    rows=[];chosen_frames=[]
    for backbone,tag in specs:
        path=RUNTIME/'online'/tag;manifest=path/'manifest.json';run=json.loads(manifest.read_text())
        if not run['full_video'] or not run['causal_online'] or not run['speed_eligible']:raise ValueError('complete clean causal control required')
        for field in ('video_id','checkpoint_sha256','config_sha256','annotation_sha256','seed','images'):
            if run[field]!=base[field]:raise ValueError('unpaired control: '+field)
        if run['geo_options']['FEATURE_BACKBONE']!=backbone or not run['geo_options']['REUSE_TARGET_FEATURES'] or run['geo_options']['GEOMETRY_WEIGHT']:
            raise ValueError('wrong control configuration')
        if sha256(path/'predictions.json')!=run['predictions_sha256'] or sha256(path/'schedule_trace.json')!=run['schedule_trace_sha256']:
            raise ValueError('changed runtime artifact')
        trace=json.loads((path/'schedule_trace.json').read_text());frames=[row['frame'] for row in trace if row['chosen']]
        chosen_frames.append(frames)
        metrics=json.loads((path/'predictions.metrics.json').read_text())['metrics']
        rows.append(dict(backbone=backbone,tag=tag,commit=run['commit'],manifest=str(manifest),manifest_sha256=sha256(manifest),
            metric_sha256=sha256(path/'predictions.metrics.json'),prediction_sha256=sha256(path/'predictions.json'),
            feature_observation_updates=run['geo_calls'],chosen_updates=len(frames),
            actual_vggt_calls=run['total_vggt_calls'],actual_dinov2_calls=run['dinov2_calls'],failures=run['geo_failures'],
            exact_baseline_predictions=json.loads((path/'predictions.json').read_text())==base_predictions,
            metrics=metrics,metric_delta={key:metrics[key]-base_metrics[key] for key in metrics},
            camera_image_fps=run['camera_image_fps'],input_to_identity_p50=run['input_to_identity_p50'],input_to_identity_p95=run['input_to_identity_p95'],
            target_cache=run['target_feature_cache'],head_checkpoint_sha256=run['head_checkpoint_sha256']))
    if any(row['feature_observation_updates']!=40 or row['chosen_updates']!=40 or row['failures'] for row in rows):
        raise ValueError('40 successful observed updates per backbone required')
    if any(frames!=chosen_frames[0] for frames in chosen_frames[1:]):raise ValueError('feature-observation times differ')
    save_json(ROOT/'results/target_cache_three_backbone_scene1_controls_v1.json',dict(status='COMPLETE_SCENE1_CONTROL_NOT_FULL_DEV',
        baseline=str(baseline),baseline_manifest_sha256=sha256(baseline/'manifest.json'),rows=rows,
        all_exact_baseline_predictions=all(row['exact_baseline_predictions'] for row in rows),same_selected_observation_groups=True,
        scope='three complete causal scene1 feature-only bounded-reuse controls, fixed TRAIN seed14 and same40 observed update groups; actual encoder costs are VGGT80 / DINO40 / GMT0, not equal compute',
        timing_scope='single descriptive full-pipeline V100 runs, static GMT initialization excluded, cold extra encoder load charged; no repeated timing superiority claim',
        conclusion='All three identity outputs equal baseline if all_exact_baseline_predictions is true; active reuse and bounded memory alone do not establish retained dense gains or independent foundation/geometry/scheduler benefit.',provenance=provenance()))
    print(json.dumps([dict(backbone=row['backbone'],exact=row['exact_baseline_predictions'],fps=row['camera_image_fps']) for row in rows]))


if __name__=='__main__':main()
