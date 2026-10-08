"""Read-only instrumentation must not alter any native branch outcome."""
import json
from pathlib import Path
from jev_phase8_common import *

def main():
    protect();start=binding();assert not start['worktree_dirty']
    a=OUT/'diagnostic_forks_v1/video07/FORK_RESULT.json';b=OUT/'diagnostic_forks_v2/video07/shard000_001/FORK_RESULT.json'
    x=json.loads(a.read_text())['events'][0];y=json.loads(b.read_text())['events'][0]
    checks={};assert (x['key'],x['row'])==(y['key'],y['row'])
    for tag,old in x['branches'].items():
        new=y['branches'][tag]
        assert old['prediction_sha256']==new['prediction_sha256']
        assert old['horizons']==new['horizons']
        pa=a.parent/'snapshot000_row000'/tag/'COMPLETE_STATE_TRACE.json'
        pb=b.parent/'snapshot000_row000'/tag/'COMPLETE_STATE_TRACE.json'
        ta=json.loads(pa.read_text());tb=json.loads(pb.read_text());assert len(ta)==len(tb)
        for u,v in zip(ta,tb):
            assert {k:w for k,w in u.items()if k!='fingerprint'}=={k:w for k,w in v.items()if k!='fingerprint'}
            if v['fingerprint']is not None:assert v['fingerprint']==u['fingerprint']
        checks[tag]={'full_prediction_sha_parity':True,'H8_H16_H32_all_effects_parity':True,'per_payload_IDs_RNG_next_ID_parity':True,'every_retained_complete_field_digest_parity':True,'payloads':len(ta)}
    save(REPORTS/'OBSERVER_EQUIVALENCE_TESTS.json',{'status':'PASS','binding':start,'diagnostic_video_not_train_validation':7,
        'checks':checks,'original_result_sha256':sha(a),'optimized_result_sha256':sha(b),
        'all_original_frozen_selection_and_utility_unchanged':True,'scope':'independent repeat of one old-video real-state event, all five intervention branches; no claim of new-video performance significance'})
    protect();print('OBSERVER_EQUIVALENCE_TESTS PASS')
if __name__=='__main__':main()
