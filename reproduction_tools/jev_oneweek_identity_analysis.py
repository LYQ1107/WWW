"""Causal past-index diagnostics on actual IDs, duplicate GT UNKNOWN, censoring explicit."""
import argparse
import collections
import gzip
from jev_oneweek_common import *


def analyse(arm,seed,phase,video):
    folder=OUT/'online_v1'/phase/arm/f'seed{seed}'/f'video{video:02d}'
    result=read(folder/'RESULT.json');descriptor=result['raw_predictions']
    assert sha(descriptor['path'])==descriptor['SHA256']
    from jev_phase7_offline import IdentityEvaluator
    evaluator=IdentityEvaluator(video);rows=evaluator.align(read(descriptor['path']))
    bad={}
    for image in evaluator.images.values():
        counts=collections.Counter(a['instance_id'] for a in evaluator.gt[image['id']])
        bad[image['frame_id']-1,image['view_id']-1]={g for g,n in counts.items() if n>1 or g<0}
    duplicate_masked=0
    for (frame,view,row),value in rows.items():
        if value['gt'] in bad[frame,view]:value['gt']=None;duplicate_masked+=1
    groups=collections.defaultdict(list)
    for key,value in rows.items():groups[key[:2]].append((key,value))
    boot_view=max([0,1],key=lambda view:(len(groups[0,view]),view))
    keys=sorted(groups,key=lambda key:(key[0],0 if key[0]==0 and key[1]==boot_view else 1 if key[0]==0 else key[1]))
    owner={};ever_person=set();seen_ids=set();index=collections.defaultdict(collections.Counter)
    all_past_person=collections.Counter();counts=collections.Counter();statuses={};episodes=[];active={}
    def close(person_view,end,reason,censored):
        event=active.pop(person_view)
        event.update(end=event['last'],duration_camera_frames=event['last']-event['start']+1,
                     right_censored=censored,end_reason=reason)
        episodes.append(event)
    for key in keys:
        frame,view=key
        for rowkey,value in sorted(groups[key]):
            identity=value['id'];gt=value['gt'];counts['rows']+=1
            new=identity not in seen_ids;seen_ids.add(identity)
            if gt is None:
                counts['UNKNOWN_rows']+=1
                statuses[rowkey]=dict(known=False,id=identity,gt=None,owner=owner.get(identity),correct=None,new_native_ID=new)
                continue
            owner.setdefault(identity,gt);correct=owner[identity]==gt
            counts['known_rows']+=1;counts['immutable_owner_correct' if correct else 'immutable_owner_wrong']+=1
            fragment=new and gt in ever_person;counts['previous_person_new_native_ID_births']+=fragment
            statuses[rowkey]=dict(known=True,id=identity,gt=gt,owner=owner[identity],correct=correct,
                                 new_native_ID=new,previous_person_new_native_ID=fragment)
            pv=gt,view;old=active.get(pv)
            if old and (frame>old['last']+1 or correct):
                close(pv,frame,'known owner-correct next observation' if frame==old['last']+1 else 'UNKNOWN/unobserved observation gap',frame!=old['last']+1)
            if not correct:
                if pv not in active:active[pv]=dict(gt=gt,view=view,start=frame,last=frame,wrong_ids=[])
                active[pv]['last']=frame
                if identity not in active[pv]['wrong_ids']:active[pv]['wrong_ids'].append(identity)
            returned=index[identity,1-view]
            if all_past_person[gt,1-view]:
                counts['cross_camera_queries_with_prior_target']+=1
                counts['cross_known_relevant_prior_observations']+=all_past_person[gt,1-view]
                counts['cross_known_correct_returns']+=returned[gt]
                counts['cross_known_wrong_returns']+=sum(n for g,n in returned.items() if g!=gt and g is not None)
                counts['cross_UNKNOWN_returns']+=returned[None]
                counts['cross_empty_or_wrong_ID_queries']+=not bool(returned[gt])
                counts['cross_polluted_native_ID_returns']+=len([g for g,n in returned.items() if g is not None and n])>1
            ever_person.add(gt)
        # Entire camera payload is queried before inserting its observations,
        # exactly as native scoring happens before same-camera row commits.
        for _,value in groups[key]:
            index[value['id'],view][value['gt']]+=1
            if value['gt'] is not None:all_past_person[value['gt'],view]+=1
    for pv in list(active):close(pv,result['frames'],'video end',True)
    tp=counts['cross_known_correct_returns'];fp=counts['cross_known_wrong_returns']
    analysis=dict(status='COMPLETE',arm=arm,seed=seed,phase=phase,video=video,
                  source=ref(folder/'RESULT.json'),counts=dict(counts),duplicate_GT_rows_masked=duplicate_masked,
                  wrong_owner_episodes=episodes,
                  observed_error_duration_camera_frames=sum(e['duration_camera_frames'] for e in episodes),
                  right_censored_episodes=sum(e['right_censored'] for e in episodes),
                  cross_past_index_precision=tp/(tp+fp) if tp+fp else None,
                  cross_past_index_recall=tp/counts['cross_known_relevant_prior_observations'] if counts['cross_known_relevant_prior_observations'] else None,
                  index_definition='Actual predicted Global ID lookup into already committed other-camera observations; no whole-video GT-selected ID mapping; UNKNOWN retrieved observations excluded from known precision and counted separately',
                  owner_definition='Immutable first known observed GT per native ID; duplicate GT ids per camera/frame UNKNOWN; not a future-safety certificate',
                  duration_definition='Observed contiguous owner errors, in camera frames; gaps/end right-censored. Not CLEAR IDSW, not fps-converted seconds',
                  full_mutated_state_scope=True,paired_same_state_causal_interventions=False)
    save(folder/'STRICT_CAUSAL_IDENTITY_ANALYSIS.json',analysis)
    return analysis,statuses


def main(arm,seed,phase,video):
    protect();actual,after=analyse(arm,seed,phase,video);baseline,before=analyse('parent',seed,phase,video)
    assert set(after)==set(before)
    comparison=collections.Counter()
    for key,value in after.items():
        old=before[key]
        if not value['known'] or not old['known']:comparison['UNKNOWN_comparison_rows']+=1;continue
        assert value['gt']==old['gt']
        a,b=int(old['correct']),int(value['correct']);comparison[f'N{a}{b}']+=1
        if a==0 and b==1:
            comparison['N01_new_native_ID_birth']+=value['new_native_ID']
            comparison['N01_existing_native_ID']+=not value['new_native_ID']
    result=dict(status='COMPLETE',actual=actual,baseline=baseline,comparison=dict(comparison),
                N01_N10_scope='Independent full cold-start mutated-state runs, identical detection rows, each run own immutable owner namespace; not same-state local causal effects',
                current_owner_correction_not_proof_original_ID_preserved=True,
                official_metrics_unmodified=True)
    save(REPORTS/f'IDENTITY_{arm}_{phase}_SEED{seed}_VIDEO{video:02d}.json',result)
    print('STRICT_IDENTITY_ANALYSIS',arm,video,dict(comparison),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--phase',default='pilot');p.add_argument('--video',type=int,required=True);a=p.parse_args()
    main(a.arm,a.seed,a.phase,a.video)
