"""Read-only verification of the bounded Phase XV delivery and linked artifacts."""
import argparse
from jev_phase15_common import *


def main(all_phase15_refs=False):
    protect()
    delivery=read(REPORTS/'DELIVERY_AUDIT.json')
    assert len(delivery['required_results'])==20
    assert len(delivery['required_documents'])==6
    references=[]
    def collect(value):
        if isinstance(value,dict):
            path=value.get('path');digest=value.get('SHA256')
            if isinstance(path,str) and isinstance(digest,str):
                if path.startswith(str(ROOT)+'/') or path.startswith(str(OUT)+'/'):
                    references.append((path,digest))
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    collect(delivery)
    if all_phase15_refs:
        for path in REPORTS.glob('*.json'):collect(read(path))
    actual={};failures=[]
    for path,digest in sorted(set(references)):
        if path not in actual:actual[path]=sha(path) if Path(path).is_file() else None
        if actual[path]!=digest:failures.append(dict(path=path,expected=digest,actual=actual[path]))
    assert not failures,failures
    pilots=read(REPORTS/'PILOT_RESULTS.json')
    assert len(pilots['versions'])==3 and not pilots['formal_qualified']
    assert all(not p['full_pilot_qualified'] and not p['pending_native_videos'] for p in pilots['versions'])
    gate=read(REPORTS/'FINAL_GO_NO_GO.json')
    assert gate['status']=='SCIENTIFIC_NO_GO'
    assert all(gate[k] is False for k in ['GO_TRACKING','GO_JEV_INDEPENDENT_VALUE','GO_DEPLOYMENT'])
    for name in ['FORMAL_TRAINING_RESULTS','FAIR_BASELINE_RESULTS']:
        result=read(REPORTS/(name+'.json'))
        assert result['actual_updates']==0 and result['metrics'] is None
    online=read(REPORTS/'ONLINE_VALIDATION.json')
    assert sorted(c['video'] for c in online['cases'])==DEV
    assert sorted(c['frames'] for c in online['cases'])==[1029,1052,1200]
    assert online['full_video_live_frontend_and_native'] and online['full_FPS'] is None
    official=read(REPORTS/'OFFICIAL_MATLAB_RESULTS.json')
    assert official['status']=='COMPLETE' and official['unchanged_official_metrics']
    assert official['official_toolkit_full_hash_matches_prior']
    counts=official['sequential_identity_counts'];errors=official['interleaved_CLEAR_counts']
    assert abs(official['CVIDF1']-200*counts['IDTP']/(counts['n_gt']+counts['n_tr']))<1e-10
    assert abs(official['CVMA']-100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']))<1e-10
    efficiency=read(REPORTS/'EFFICIENCY.json')
    assert efficiency['status']=='COMPLETE' and len(efficiency['trials'])==3
    assert not efficiency['runtime_thresholds_pass']
    previous=read(REPORTS/'FINAL_PRIOR_INTEGRITY.json')
    assert previous['all_protected_bytes_unchanged'] and previous['checkpoint_count']==332
    # Verification prints its own result and never rewrites scientific reports.
    print(json.dumps(dict(status='PASS',unique_local_artifacts_hashed=len(actual),
        all_phase15_refs=all_phase15_refs,required_JSON=20,required_documents=6,
        scientific_status=gate['status'],formal_updates=0,
        official_CVIDF1=official['CVIDF1'],official_CVMA=official['CVMA'],
        frozen_prior_integrity_audit=ref(REPORTS/'FINAL_PRIOR_INTEGRITY.json')),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-phase15-refs',action='store_true')
    main(p.parse_args().all_phase15_refs)
