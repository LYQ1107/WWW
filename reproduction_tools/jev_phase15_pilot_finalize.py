"""Keep every bounded version, with separate label and native-safety gates."""
from jev_phase15_common import *


def main(version=1):
    protect();source=binding(seed=20261009,evaluator='frozen TRAIN assessment plus real native futures',
        scope='bounded research pilot; never a three-seed formal tracking result')
    tiny=OUT/f'training_full_payload_v{version}'/'F_full/seed20261009/tiny/RESULT.json'
    pilot=OUT/f'training_full_payload_v{version}'/'F_full/seed20261009/pilot/RESULT.json'
    prior=read(REPORTS/'TINY_RESULTS.json') if (REPORTS/'TINY_RESULTS.json').exists() else {'versions':[]}
    if tiny.exists():
        result=read(tiny);assert result['status']=='COMPLETE'
        entry=dict(version=version,result=ref(tiny),checkpoint=result['checkpoint'],actual_updates=result['actual_updates'],
            loss_before=result['loss_before'],loss_last=result['loss_last'],gradient_health=result['gradient_health'],assessment=result['final'],
            finite_gradient_learnability=result['loss_last']<result['loss_before'] and all(result['gradient_health'].get(key,0)>0 for key in ['identity','question','option','reliability']))
        prior['versions']=[item for item in prior['versions'] if item['version']!=version]+[entry]
        save(REPORTS/'TINY_RESULTS.json',dict(status='COMPLETE',binding=source,versions=prior['versions'],tracking_quality_claimed=False))
    if not pilot.exists():return
    trained=read(pilot);assert trained['status']=='COMPLETE'
    native=[];adverse=[];pending=[]
    for video in TRAIN:
        path=OUT/f'pilot_native_v{version}'/'F_full/pilot'/f'video{video:02d}'/'RESULT.json'
        if not path.exists():pending.append(video);continue
        result=read(path);assert result['status']=='COMPLETE';native.append(ref(path))
        if video==14:continue
        for item in result['cases']:
            for horizon,metrics in item['H'].items():
                delta=metrics['delta_vs_original_entire_pi_multi']
                if delta['new_cross_GT_gallery_mix']>0 or delta['false_birth']>0:
                    adverse.append(dict(key=item['key'],row=item['row'],H=horizon,delta=delta,result=ref(path)))
    passed=not pending and trained['label_assessment_qualified'] and not adverse
    path=REPORTS/'PILOT_RESULTS.json';old=read(path) if path.exists() else {'versions':[]}
    entry=dict(version=version,trained=ref(pilot),checkpoint=trained['checkpoint'],actual_updates=trained['actual_updates'],
        assessment=trained['final'],label_assessment_qualified=trained['label_assessment_qualified'],native= native,
        adverse_native_cases=adverse,pending_native_videos=pending,full_pilot_qualified=bool(passed),
        reason='PASS' if passed else 'PENDING_NATIVE' if pending else 'failed frozen label/continuation/correction or real native merge/birth safety gate')
    versions=[item for item in old['versions'] if item['version']!=version]+[entry]
    save(path,dict(status='COMPLETE' if not pending else 'RUNNING_NATIVE_AUDIT',binding=source,versions=versions,
        current_version=version,full_pilot_qualified=bool(passed),formal_qualified=bool(passed),
        all_versions_and_failures_retained=True,no_formal_on_unreliable_pilot=True))
    print('PHASE15_PILOT_GATES',version,trained['label_assessment_qualified'],len(adverse),pending,'qualified',passed,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);a=p.parse_args();main(a.version)
