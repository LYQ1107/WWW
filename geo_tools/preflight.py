"""Read-only verification of locked inputs; reports never replace the lock.

The historical initializer is retained in Git. Reproduction uses the committed
scene split and baseline lock. Metadata verification alone does not claim a new
strict model load or video inference. Model construction is explicit.
"""
import argparse
from common import *


def validate_locked_split(lock,split,video_ids):
    if lock['status']!='LOCKED_FULL_STRICT_DEVELOPMENT_BASELINE' or lock['split']!=split:
        raise ValueError('baseline lock and split disagree')
    groups={}
    for field in ('association_train_video_ids','full_development_video_ids','strict_development_video_ids','raw_gt_diagnostic_development_video_ids'):
        values=split[field]
        if any(not isinstance(value,int) or isinstance(value,bool) for value in values) or len(values)!=len(set(values)):
            raise ValueError('invalid/duplicate scene IDs: '+field)
        groups[field]=set(values)
    fit,dev,strict,diagnostic=[groups[key] for key in groups]
    if not fit or not dev or not strict or fit&dev or fit|dev!=set(video_ids):raise ValueError('TRAIN/development partition changed')
    if strict&diagnostic or strict|diagnostic!=dev:raise ValueError('strict/diagnostic partition changed')
    if not set(split['minimal_development_video_ids'])<=dev:raise ValueError('minimal scenes outside development')
    if set(map(int,lock['strict_video_runs']))!=strict:raise ValueError('locked prediction scene set changed')
    if split['official_test_used_for_this_project'] is not False or lock['test_used_for_current_selection'] is not False:
        raise ValueError('this project must not select on TEST')
    if split['historical_test_exposure'] is not True or lock['historical_test_exposure'] is not True:
        raise ValueError('historical TEST exposure disclosure changed')
    return sorted(strict)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-model',action='store_true',help='also construct and strictly load GMT on the allocated CUDA device; no video inference')
    parser.add_argument('--report-output',type=Path)
    args=parser.parse_args()
    output=args.report_output or RUNTIME/'preflight'/('verified_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    if output.exists():raise FileExistsError(output)
    split_path=ROOT/'manifests/geo_split_v1.json';lock_path=ROOT/'results/baseline_manifest.json'
    original_hashes={str(path):sha256(path) for path in (split_path,lock_path)}
    split=json.loads(split_path.read_text());lock=json.loads(lock_path.read_text())
    strict=validate_locked_split(lock,split,[video['id'] for video in train_data()['videos']])
    inputs={'checkpoint_sha256':CHECKPOINT,'config_sha256':ROOT/'configs/VISION_test.yaml','annotation_sha256':DATA/'annotations/train.json'}
    for field,path in inputs.items():
        if sha256(path)!=lock[field]:raise ValueError('locked input changed: '+field)
    suite_path=Path(lock['strict_suite_file'])
    if sha256(suite_path)!=lock['strict_suite_sha256']:raise ValueError('locked native suite changed')
    suite=json.loads(suite_path.read_text())
    if suite['status']!='PASS' or suite['causal_online'] or not suite['full_video'] or suite['video_ids']!=strict:
        raise ValueError('locked original offline suite scope changed')
    for scene,record in lock['strict_video_runs'].items():
        if sha256(record['run_manifest'])!=record['manifest_sha256'] or sha256(record['prediction'])!=record['prediction_sha256']:
            raise ValueError('locked inference artifact changed: '+scene)
        if suite['prediction_sha256'][record['prediction']]!=record['prediction_sha256']:
            raise ValueError('locked evaluation/inference differ')
    loaded_keys=None
    if args.build_model:
        cfg,model=load_gmt();loaded_keys=len(model.state_dict())
        if loaded_keys!=lock['gmt_state_keys']:raise ValueError('loaded model key count changed')
    if any(sha256(path)!=digest for path,digest in original_hashes.items()):raise ValueError('lock changed during verification')
    report=dict(status='LOCKED_INPUTS_AND_STRICT_MODEL_LOAD_VERIFIED' if args.build_model else 'LOCKED_INPUTS_VERIFIED_MODEL_LOAD_NOT_RUN',
        locked_files_unchanged=True,locked_file_sha256=original_hashes,strict_video_ids=strict,
        locked_suite_sha256=sha256(suite_path),model_strict_load_performed=args.build_model,loaded_state_keys=loaded_keys,
        new_video_inference_performed=False,source_checkpoint_sha256=lock['checkpoint_sha256'],
        scope='current model/config/annotation/split and retained original offline baseline artifacts only; preserves all lock bytes; does not replace causal baseline, inference regressions or final completion audit',provenance=provenance())
    save_json(output,report);print(json.dumps(dict(status=report['status'],report=str(output),locked_files_unchanged=True)))


if __name__=='__main__':main()
