"""Explicit accuracy comparison for the validation-only camera input repair."""
import argparse
import collections
from common import *


def verify_pure_gmt(manifest):
    if manifest['feature_weight']:
        raise ValueError('pure GMT control required')
    if 'geo_options' in manifest:
        if manifest['geo_options']['GEOMETRY_WEIGHT']:
            raise ValueError('pure GMT control required')
        return 'recorded_options'
    # First scene18 OFF manifest predates recording geo_options. Its exact
    # captured driver sets ENABLED=(feature_weight != 0), with no geometry CLI.
    # Whitelist this inspected driver, never assume absent settings are zero.
    expected='38b783425133ba41875c4596015e7ae520c22ed88282cc068ea1879f7e0f086e'
    code=subprocess.check_output(['git','show',manifest['commit']+':geo_tools/run_online.py'],cwd=ROOT)
    if hashlib.sha256(code).hexdigest()!=expected or manifest['source_hashes']['geo_tools/run_online.py']!=expected:
        raise ValueError('unverified legacy OFF driver')
    if manifest['geo_calls'] or manifest['vggt_calls']:
        raise ValueError('legacy OFF unexpectedly computed enhancement')
    return 'captured_legacy_driver_hash_and_zero_calls'


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
    evidence=[]
    for manifest in manifests:
        if not manifest['full_video'] or not manifest['causal_online']:
            raise ValueError('complete causal runs required')
        evidence.append(verify_pure_gmt(manifest))
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
    detector_rows_equal=[{k:v for k,v in row.items() if k!='track_id'} for row in predictions[0]]==[{k:v for k,v in row.items() if k!='track_id'} for row in predictions[1]]
    forward=collections.defaultdict(set);reverse=collections.defaultdict(set)
    changed_ids=None;partition_equal=None
    if detector_rows_equal:
        changed_ids=0
        for original,corrected in zip(*predictions):
            a,b=original['track_id'],corrected['track_id'];forward[a].add(b);reverse[b].add(a);changed_ids+=a!=b
        partition_equal=all(len(values)==1 for values in [*forward.values(),*reverse.values()])
    report=dict(status='PASS',video_id=manifests[0]['video_id'],full_strict_suite=False,
                scope='one complete causal scene; validation-only frozen original spatial/camera MLP input repair; locked baseline retained',
                exact_serialized_predictions=predictions[0]==predictions[1],
                identical_detector_membership=detector_keys[0]==detector_keys[1],
                exact_detector_row_order=detector_rows_equal,changed_track_id_rows=changed_ids,
                same_identity_partition_up_to_global_bijection=partition_equal,
                baseline_metrics=metrics[0],control_metrics=metrics[1],
                metric_delta={key:metrics[1][key]-metrics[0][key] for key in metrics[0]},
                source_runs=[str(p) for p in runs],source_commits=[m['commit'] for m in manifests],pure_gmt_evidence=evidence,
                artifact_sha256={str(p/name):sha256(p/name) for p in runs for name in ('manifest.json','predictions.json','predictions.metrics.json')},
                interpretation=('No observed effect on this scene; does not disprove the source camera-input issue or establish invariance on other scenes.' if predictions[0]==predictions[1] else 'Camera repair changes identity outputs; its effect is an explicit baseline control and cannot be credited to VGGT.'),
                speed_eligible=False,provenance=provenance())
    save_json(ROOT/'results'/f'{args.tag}.json',report)
    print(json.dumps({key:report[key] for key in ('exact_serialized_predictions','identical_detector_membership','metric_delta')},indent=2))


if __name__=='__main__':main()
