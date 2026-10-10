"""All declared posterior interventions, with identical checkpoint/prefix pairs."""
import gzip
import collections
from jev_phase15_common import *


def main():
    comparisons=[];aggregate=collections.Counter();changed_rows=0
    for video in TRAIN:
        a=OUT/'pilot_native_v1/F_full/pilot'/f'video{video:02d}'/'RESULT.json'
        b=OUT/'posterior_masked_native_v1/F_full/pilot'/f'video{video:02d}'/'RESULT.json'
        if not b.exists():return
        left=read(a);right=read(b);assert left['binding']['checkpoints']==right['binding']['checkpoints']
        assert len(left['cases'])==len(right['cases'])==6
        for old,new in zip(left['cases'],right['cases']):
            assert (old['key'],old['row'],old['original_native_state_SHA256'])==(new['key'],new['row'],new['original_native_state_SHA256'])
            l=list(map(json.loads,gzip.open(old['trace']['path'],'rt')));r=list(map(json.loads,gzip.open(new['trace']['path'],'rt')))
            assert [(i['key'],i['row']) for i in l]==[(i['key'],i['row']) for i in r]
            changed=sum(i['identity']!=j['identity'] for i,j in zip(l,r));changed_rows+=changed
            delta={}
            for horizon in ['8','16','32']:
                x=old['H'][horizon]['delta_vs_original_entire_pi_multi'];y=new['H'][horizon]['delta_vs_original_entire_pi_multi']
                delta[horizon]={k:y[k]-x[k] for k in x}
            if video!=14:aggregate.update(delta['32'])
            comparisons.append(dict(key=old['key'],row=old['row'],qualified=video!=14,changed_actual_commits=changed,
                masked_minus_feedback=delta,original=ref(a),intervention=ref(b)))
    report=dict(status='COMPLETE',binding=binding(seed=20261009),cases=comparisons,changed_actual_commits_in_all24_overlapping_windows=changed_rows,
        eligible18_H32_overlapping_delta_sum=dict(aggregate),whole_video_gain_claimed=False,
        attribution_scope='same weights, native prefix, RNG, VFCE and future; only previous purity/posterior-available actor input channels masked; actual mutated-state subsequent decisions',
        input_support='both channels constant (.5,0) in frozen corpus; audit proves actual support separately',
        mechanism_diagnosis='NO_EFFECT_ON_ACTUAL_IDS' if changed_rows==0 else 'POSTERIOR_FEEDBACK_CHANGES_NATIVE_IDS_NOT_AUTOMATICALLY_BENEFICIAL',
        feature_family_GT_free=True,intervention_tests=ref(OUT/'posterior_feedback_contract_v1/RESULT.json'))
    save(REPORTS/'POSTERIOR_FEEDBACK_ATTRIBUTION.json',report)
    print('PHASE15_POSTERIOR_ATTRIBUTION',changed_rows,dict(aggregate),flush=True)


if __name__=='__main__':main()
