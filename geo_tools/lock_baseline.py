"""Lock complete strict development baseline from verified immutable artifacts."""
from common import *

def main():
    source=json.loads((ROOT/'manifests/baseline_strict_dev_sources.json').read_text());suite=ROOT/'results/B0_strict_suite_v1_metrics.json';report=json.loads(suite.read_text());split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if report['status']!='PASS' or report['video_ids']!=split['strict_development_video_ids']:raise ValueError('incomplete baseline suite')
    if report['causal_online'] or not report['full_video']:raise ValueError('wrong baseline protocol')
    snapshots={}
    for video,path in source['video_predictions'].items():
        path=Path(path);manifest=path.with_name(path.stem+'_manifest.json');run=json.loads(manifest.read_text())
        if run.get('geo_options',{}).get('ENABLED',False):raise ValueError('baseline enabled GEO')
        if sha256(path)!=run['prediction_sha256'] or report['prediction_sha256'][str(path)]!=sha256(path):raise ValueError('prediction hash mismatch')
        if run['checkpoint_sha256']!=sha256(CHECKPOINT) or run['config_sha256']!=sha256(ROOT/'configs/VISION_test.yaml'):raise ValueError('model/config drift')
        snapshots[video]=dict(run_manifest=str(manifest),manifest_sha256=sha256(manifest),prediction=str(path),prediction_sha256=sha256(path),commit=run['commit'],seed=run['seed'])
    old=json.loads((ROOT/'results/baseline_manifest.json').read_text())
    old.update(status='LOCKED_FULL_STRICT_DEVELOPMENT_BASELINE',split=split,strict_suite_metrics=report['metrics'],strict_suite_file=str(suite),strict_suite_sha256=sha256(suite),
        strict_video_runs=snapshots,scope='all five structurally valid complete TRAIN-development scenes; remaining declared scenes require separate raw-GT diagnostic protocol',
        original_full_development_complete=False,causal_protocol_has_separate_baseline=True,lock_provenance=provenance(),historical_test_exposure=True,test_used_for_current_selection=False)
    save_json(ROOT/'results/baseline_manifest.json',old)
    print(json.dumps(dict(status=old['status'],video_ids=report['video_ids'],metrics=report['metrics']),indent=2))
if __name__=='__main__':main()
