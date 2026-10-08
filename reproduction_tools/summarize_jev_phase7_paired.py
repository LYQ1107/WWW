"""Offline paired externality audit; unknown target labels never become training truth."""
from collections import Counter,defaultdict
import json
import numpy as np
import torch
from jev_phase7_common import *
from jev_phase7_offline import IdentityEvaluator
from jev_phase6_offline_utility import prefix_identity


def window_audit(rows,prefix,key,h,next_id):
    mapping,_=prefix_identity(prefix,key[1:]);window={k:r for k,r in rows.items()if k[:2]>=key[1:]and k[0]<=key[1]+h}
    for k,r in sorted(window.items()):
        if r['id']>next_id and r['gt']is not None:mapping.setdefault(r['id'],r['gt'])
    timelines=defaultdict(dict);counts=Counter();correctness={}
    for k,r in sorted(window.items()):
        if r['gt']is None:counts['unknown_GT']+=1;continue
        counts['known_GT']+=1;known_identity=r['id']in mapping;correct=known_identity and mapping[r['id']]==r['gt']
        counts['correct']+=correct;counts['wrong']+=known_identity and not correct;counts['unanchored_identity']+=not known_identity
        correctness[k]=bool(correct)if known_identity else None
        if known_identity:timelines[(r['gt'],k[1])][k[0]]=not correct or timelines[(r['gt'],k[1])].get(k[0],False)
    episodes=[]
    for (g,v),timeline in sorted(timelines.items()):
        active=[];previous=None
        for frame,bad in sorted(timeline.items()):
            if active and (not bad or frame!=previous+1):
                episodes.append({'GT':g,'camera':v,'start':active[0],'end':active[-1],'duration_frames':len(active),'right_censored':frame!=previous+1});active=[]
            if bad:active.append(frame)
            previous=frame
        if active:episodes.append({'GT':g,'camera':v,'start':active[0],'end':active[-1],'duration_frames':len(active),'right_censored':True})
    return {'counts':dict(counts),'wrong_ID_episodes':episodes,'wrong_ID_camera_frames':sum(e['duration_frames']for e in episodes)},correctness


def summarize():
    details=[];aggregates=defaultdict(Counter);parity=[]
    for video in (9,10,11):
        path=OUT/'paired'/f'video{video:02d}/PAIRED_AUDIT.json';data=json.loads(path.read_text());assert data['status']=='COMPLETE'
        evaluator=IdentityEvaluator(video)
        factual=evaluator.align(json.loads((OUT/'closed_loop'/f'video{video:02d}/B2/tracking_predictions/jev.json').read_text()))
        groups=defaultdict(list)
        for r in data['records']:groups[r['snapshot']].append(r)
        for source,branches in groups.items():
            capture=torch.load(source,map_location='cpu');key=tuple(capture['key']);prefix=capture['state'];horizons=branches[0]['live_horizons']
            target=next(r['offline_GT']for r in branches if r['condition']=='CONTROL')
            rows={r['condition']:evaluator.align(json.loads(Path(r['result']['predictions']).read_text()))for r in branches}
            control=next(r for r in branches if r['condition']=='CONTROL');b2=next(r for r in branches if r['condition']=='B2')
            assert control['fingerprints']==b2['fingerprints']and control['commits']==b2['commits']
            for r in branches:
                assert r['initial_state_fingerprint']==capture['fingerprint']
                assert r['first_mechanism']['score_sha256']==capture['mechanism']['score_sha256']
                d={'video':video,'key':list(key),'condition':r['condition'],'offline_target_GT':target,
                   'unknown_target_zero_supervision_weight':target is None,'same_start_and_frozen_score_matrix':True,
                   'native_state_diverged_from_control':r['fingerprints']!=control['fingerprints'],
                   'committed_ids_diverged_from_control':r['commits']!=control['commits'],'horizons':{}}
                count=aggregates[r['condition']];count['events']+=1;count['unknown_target_events']+=target is None
                count['state_divergence_events']+=d['native_state_diverged_from_control'];count['commit_divergence_events']+=d['committed_ids_diverged_from_control']
                for h in horizons:
                    audit,correct=window_audit(rows[r['condition']],factual,key,h,prefix.next_id)
                    ref,ref_correct=window_audit(rows['CONTROL'],factual,key,h,prefix.next_id)
                    paired=Counter()
                    for k,c in correct.items():
                        old=ref_correct.get(k)
                        if c is not None and old is not None:
                            paired['N01_corrections']+=not old and c;paired['N10_regressions']+=old and not c
                    audit.update(paired);audit['wrong_ID_camera_frames_delta_vs_control']=audit['wrong_ID_camera_frames']-ref['wrong_ID_camera_frames']
                    d['horizons'][str(h)]=audit
                    count[f'H{h}_N01']+=paired['N01_corrections'];count[f'H{h}_N10']+=paired['N10_regressions']
                    count[f'H{h}_wrong_camera_frames_delta']+=audit['wrong_ID_camera_frames_delta_vs_control']
                if r['condition'].endswith('_EQUAL_CURRENT_BUDGET'):
                    actual=sum(a=='REASSOCIATE'for a in r['first_mechanism']['first_actions'].values())
                    expected=sum(a=='REASSOCIATE'for a in capture['mechanism']['first_actions'].values());assert actual==expected
                    d['exact_current_trigger_budget']={'actual':actual,'reference':expected}
                details.append(d)
            parity.append({'video':video,'key':list(key),'same_prefix_RNG_and_frozen_scores_all_branches':True,
                           'CONTROL_B2_identity_and_legacy_state_fingerprint_parity':True,'source_sha256':sha(source)})
    result={'status':'COMPLETE','binding':binding(),'events':len(parity),'live_horizon_events':sum(bool(d['horizons'])for d in details if d['condition']=='CONTROL'),
        'branches':len(details),'aggregate':{k:dict(v)for k,v in aggregates.items()},'records':details,'parity':parity,
        'target_selection':'factual B2 REASSOCIATE triggers; all eleven target detections are GT-unmatched in this run',
        'scope':'causal current-payload intervention then common live B2 continuation; enriched overlapping windows, not independent samples or population rates',
        'temporal_definition':'fixed prefix identity, first known anchor for new IDs within each horizon, consecutive camera-frames; gaps/end censored; no seconds',
        'state_fingerprint_scope':'legacy lab digest includes identity/history/RNG/prototypes/gallery/metadata; counters and cumulative archival assignments omitted',
        'original_target_utility_preserved':True,'new_externality_diagnostics_are_offline_only':True,'official_test_read':False}
    save(REPORTS/'PAIRED_REASSOCIATION_AUDIT.json',result)
    print(json.dumps({'status':'COMPLETE','events':len(parity),'branches':len(details),'summary':result['aggregate']}))

if __name__=='__main__':summarize()
