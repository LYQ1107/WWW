"""Explicit accuracy comparison for the validation-only camera input repair."""
import argparse
import collections
from common import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    runs=[args.baseline,args.control]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in runs]
    if [m.get('validation_corrected_gmt_camera_index',False) for m in manifests]!=[False,True]:
        raise ValueError('original baseline followed by explicit corrected-camera control required')
    for manifest in manifests:
        if not manifest['full_video'] or not manifest['causal_online']:
            raise ValueError('complete causal runs required')
        if manifest['feature_weight'] or manifest['geo_options']['GEOMETRY_WEIGHT']:
            raise ValueError('pure GMT control required')
    for field in ('checkpoint_sha256','config_sha256','annotation_sha256','video_id','seed','images','synchronized_groups'):
        if manifests[0][field]!=manifests[1][field]:
            raise ValueError('unpaired camera control: '+field)
    predictions=[];metrics=[]
    for path,manifest in zip(runs,manifests):
        if sha256(path/'predictions.json')!=manifest['predictions_sha256']:
            raise ValueError('changed prediction artifact')
        predictions.append(json.loads((path/'predictions.json').read_text()))
        metrics.append(json.loads((path/'predictions.metrics.json').read_text())['metrics'])
    def keys(rows):
        return collections.Counter((row['image_id'],row['category_id'],row['score'],*row['bbox']) for row in rows)
    detector_keys=[keys(rows) for rows in predictions]
    report=dict(status='PASS',video_id=manifests[0]['video_id'],full_strict_suite=False,
                scope='one complete causal scene; validation-only frozen original spatial/camera MLP input repair; locked baseline retained',
                exact_serialized_predictions=predictions[0]==predictions[1],
                identical_detector_membership=detector_keys[0]==detector_keys[1],
                baseline_metrics=metrics[0],control_metrics=metrics[1],
                metric_delta={key:metrics[1][key]-metrics[0][key] for key in metrics[0]},
                source_runs=[str(p) for p in runs],source_commits=[m['commit'] for m in manifests],
                artifact_sha256={str(p/name):sha256(p/name) for p in runs for name in ('manifest.json','predictions.json','predictions.metrics.json')},
                interpretation=('No observed effect on this scene; does not disprove the source camera-input issue or establish invariance on other scenes.' if predictions[0]==predictions[1] else 'Camera repair changes identity outputs; its effect is an explicit baseline control and cannot be credited to VGGT.'),
                speed_eligible=False,provenance=provenance())
    save_json(ROOT/'results'/f'{args.tag}.json',report)
    print(json.dumps({key:report[key] for key in ('exact_serialized_predictions','identical_detector_membership','metric_delta')},indent=2))


if __name__=='__main__':main()
