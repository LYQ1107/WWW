"""Read-only final preservation check against pre-PIRGA byte-level baselines."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
from pirga_indexed_store import sha

BASE='560b43d5567ac8d4a73075450157766bc63d261d'


def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/PIRGA_FULLTRAIN';start=time.monotonic();issues=[];cache={}
    def check(path,expected,size=None):
        path=Path(path)
        if not path.is_file():issues.append(dict(path=str(path),reason='MISSING'));return
        if str(path) not in cache:cache[str(path)]=(path.stat().st_size,sha(path))
        actual_size,actual=cache[str(path)]
        if actual!=expected or (size is not None and size!=actual_size):issues.append(dict(path=str(path),reason='CONTENT_CHANGED',expected_sha256=expected,actual_sha256=actual))
    prior=repo/'results/cv_pil_v3_native_validation/FROZEN_PRIOR_EVIDENCE.json'
    pins=root/'SOURCE_SHA_MANIFEST.json'
    expected={prior:'a377dd24b5b72b6dbdecdf45641d9262e477421cdea9a858aca9f0b5a2600663',pins:'179f356ce3b359627680193847ef3b1cc10e61786a1102f4f1024cd404c6aa4c'}
    for path,digest in expected.items():
        if sha(path)!=digest:raise ValueError('Pre-training preservation manifest changed '+str(path))
    d=json.loads(prior.read_text());records=d['checkpoints']+[a for r in d['repositories'] for a in r['artifacts']]
    if len(records)!=2976:raise ValueError('Old preservation scope changed')
    for item in records:check(item['path'],item['sha256'],item['bytes'])
    inputs=json.loads(pins.read_text())['artifacts']
    for path,digest in inputs.items():check(path,digest)
    # The current reference also contains V3 artifacts created after the older2976 seal.
    entries=subprocess.check_output(['git','ls-tree','-r','-z',BASE],cwd=repo).split(b'\0');git_count=0
    for entry in filter(None,entries):
        meta,name=entry.split(b'\t',1);mode,kind,digest=meta.split();path=repo/os.fsdecode(name)
        if kind!=b'blob':raise ValueError('Unhandled reference object')
        if not path.exists() and not path.is_symlink():issues.append(dict(path=str(path),reason='BASE_FILE_MISSING'));continue
        content=os.fsencode(os.readlink(path)) if mode==b'120000' else path.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if actual!=digest.decode():issues.append(dict(path=str(path),reason='REFERENCE_GIT_BLOB_CHANGED',expected_git_blob=digest.decode(),actual_git_blob=actual))
        git_count+=1
    baseline=json.loads((root/'FROZEN_BASELINE_BINDINGS.json').read_text())
    for path,digest in baseline['old_artifacts'].items():check(repo/path,digest)
    for path,digest in baseline['producer_read_only_reports'].items():check(path,digest)
    check(baseline['checkpoint']['path'],baseline['checkpoint']['sha256'],baseline['checkpoint']['bytes'])
    oldfit=repo/'results/cv_pil_v3_native_validation/P3_MATCHED_TRAINING/RESULTS.json';fit=json.loads(oldfit.read_text())
    if set(fit['training'])!={'C2','C3','C4'}:raise ValueError('Old matched heads missing')
    for arm,spec in fit['training'].items():check(spec['local_checkpoint'],spec['checkpoint_sha256'])
    raw=fit['full_raw_training_result'];check(raw['path'],raw['sha256'])
    result=dict(status='PASS_PRE_PIRGA_PRESERVATION' if not issues else 'FAIL_RETAIN_EVIDENCE',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prior_frozen_record_count=len(records),prior_frozen_unique_paths=len({r['path'] for r in records}),prior_frozen_checkpoint_records=len(d['checkpoints']),
        current_reference_commit=BASE,current_reference_Git_blobs_checked=git_count,initial_input_source_pins_checked=len(inputs),
        V3_old_matched_checkpoint_count=3,distinct_SHA256_paths_checked=len(cache),distinct_SHA256_bytes=sum(v[0] for v in cache.values()),
        manifest_sha256={str(p.relative_to(repo)):v for p,v in expected.items()},issues=issues,wall_seconds=time.monotonic()-start,GPU_wall_hours=0,
        scientific_success=False,qualification='Preservation only. Counts from overlapping manifests are separate scopes, not independent experiments. Original records untouched; excludes newly created files and unsealed intermediate rolling resumes. No dataset GT rematching or new native inference.')
    (root/'OLD_EVIDENCE_PRESERVATION_RECHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if issues:raise ValueError('Preserved source mismatch; inspect without overwriting old files')


if __name__=='__main__':main()
