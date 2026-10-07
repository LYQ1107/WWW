"""Publish a reviewable Phase V evidence bundle; exclude large model/data caches."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007')
DEST=ROOT/'reports/WWW_JEV_PHASE5_20261007'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if DEST.exists():raise RuntimeError('refusing to overwrite evidence bundle')
    assert json.loads((OUT/'MINIMAL_RETRAINING.json').read_text())['status']=='COMPLETE'
    DEST.mkdir(parents=True);entries=[]
    def copy(p,relative=None,compress=False):
        p=Path(p);relative=Path(relative) if relative else p.relative_to(OUT);target=DEST/relative
        if compress:target=target.with_name(target.name+'.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with p.open('rb') as source,target.open('wb') as output:
                with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0,compresslevel=6) as handle:shutil.copyfileobj(source,handle)
        else:shutil.copyfile(p,target)
        entries.append({'archived':str(target.relative_to(DEST)),'archive_sha256':sha(target),'source':str(p),'source_sha256':sha(p),'source_bytes':p.stat().st_size,'archived_bytes':target.stat().st_size,'compression':'gzip,mtime=0' if compress else None})
    for p in sorted(OUT.glob('*.json')):copy(p)
    for name in ['regression_tests.log','rng_tests.log','B1_training.log','B2_training.log','B1_tracking.log','B2_tracking.log']:
        copy(OUT/name)
    copy(OUT/'paired_decisions.jsonl',compress=True)
    for kind,variants in [('ablations',['A0','A1','A2','A3','A4']),('minimal_tracking',['B1','B2'])]:
        for variant in variants:
            root=OUT/kind/variant;method='gmt_off' if variant=='A0' else 'jev'
            for name in ['result.json','runtime_status.json']:copy(root/name)
            for directory,filename in [('tracking_predictions',method+'.json'),('tracking_decisions',method+'.json')]:copy(root/directory/filename,compress=True)
            copy(root/'online_decisions.jsonl',compress=True)
            evaluation=root/'tracking_eval_runtime_state'/method/'evaluation'
            copy(evaluation/'metrics.json')
            for name in ['pedestrian_summary.txt','pedestrian_detailed.csv']:copy(evaluation/'GMT'/name)
            copy(root/'tracking_eval_runtime_state'/method/'prepared'/'manifest.json')
    for task in ['memory','match_relabel']:
        root=OUT/task;copy(root/'plan.json');copy(root/'records.jsonl',compress=True)
        for p in sorted(root.glob('chunks/*/manifest.json')):copy(p)
        for p in sorted(root.glob('worker_*.log')):copy(p)
        if (root/'launch.json').exists():copy(root/'launch.json')
        for p in sorted(root.glob('*error.json')):copy(p)
    for name,contract in [('match_training','aligned'),('match_training_legacy','legacy')]:
        root=OUT/name
        for video in [6,7]:copy(root/f'video{video:02d}_{contract}_match.jsonl',compress=True)
        copy(root/'compact/manifest.json');copy(root/'policy_split.json')
    for condition in ['B1','B2']:
        root=OUT/'minimal_training'/condition
        for name in ['model.pth','metrics.json','binding.json','runtime_status.json','calibration/model_calibrated.pth','calibration/calibration_val_only.json']:copy(root/name)
    frozen=json.loads((OUT/'PHASE5_BASELINE_LOCK.json').read_text())['input_and_policy_hashes']['jev']['calibrated_checkpoint'];copy(frozen,'frozen_MATCH_checkpoint/model_calibrated.pth')
    manifest={'status':'COMPLETE','source_commit_at_archival':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'runtime':str(OUT),'official_test_read':False,'full24_authorized':False,'entries':entries,'excluded':'Large GMT checkpoint, full perception cache, raw dataset/GT images, generated eval_dataset symlinks and duplicate NumPy arrays; their immutable hashes/manifests remain available.'}
    (DEST/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    files=sorted(p for p in DEST.rglob('*') if p.is_file())
    (DEST/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(DEST))+'\n' for p in files))
    print(json.dumps({'status':'COMPLETE','files':len(files)+1,'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'destination':str(DEST)}))
if __name__=='__main__':main()
