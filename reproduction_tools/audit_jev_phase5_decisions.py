"""GT-only offline paired-trajectory diagnostics; no local causal claim."""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction_tools'))
from audit_jev_phase5 import OUT,TRAIN,key,sha,save
HORIZONS=(1,2,4,8,16,32)


def xyxy(b):return np.asarray([b[0],b[1],b[0]+b[2],b[1]+b[3]])


def overlaps(a,b):
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(hi-lo,0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--variants',nargs='+',choices=['A1','A2','A3','A4','B1','B2'],default=['A1','A2','A3','A4']);parser.add_argument('--prefix',default='');args=parser.parse_args()
    variants=['A0']+args.variants
    ann=json.loads(TRAIN.read_text());images={x['id']:x for x in ann['images'] if x['video_id']==1};gt=defaultdict(list)
    for a in ann['annotations']:
        if a['image_id'] in images and a.get('conf',1)>0 and a.get('category_id',1)==1:gt[a['image_id']].append(a)
    rows={};decisions={};counts={};maps={};timelines={};writes={}
    for variant in variants:
        method='gmt_off' if variant=='A0' else 'jev';root=OUT/('minimal_tracking' if variant.startswith('B') else 'ablations')/variant
        pred=json.loads((root/'tracking_predictions'/f'{method}.json').read_text());byimage=defaultdict(list)
        for p in pred:byimage[p['image_id']].append(p)
        aligned={};matrix=defaultdict(Counter);duplicates=0
        for image_id,ps in byimage.items():
            image=images[image_id];frame=image['frame_id']-1;view=image['view_id']-1;targets=gt[image_id]
            target_by_row={}
            if targets:
                ious=overlaps(np.asarray([xyxy(p['bbox']) for p in ps]),np.asarray([xyxy(t['bbox']) for t in targets]))
                # Maximize the number of feasible matches first, then IoU.
                rr,cc=linear_sum_assignment(-((ious>=.5).astype(float)*1000+ious))
                target_by_row={int(r):int(targets[c]['instance_id']) for r,c in zip(rr,cc) if ious[r,c]>=.5}
            duplicates+=len(ps)-len({p['track_id'] for p in ps})
            for row,p in enumerate(ps):
                target=target_by_row.get(row);aligned[(frame,view,row)]={'id':p['track_id'],'gt':target,'box':p['bbox'],'score':p['score']}
                if target is not None:matrix[view][(p['track_id'],target)]+=1
        mapping={};purity=[]
        for view,c in matrix.items():
            ids=sorted({x[0] for x in c});targets=sorted({x[1] for x in c});m=np.asarray([[c[(i,t)] for t in targets] for i in ids]);rr,cc=linear_sum_assignment(-m)
            mapping[view]={ids[r]:targets[c] for r,c in zip(rr,cc) if m[r,c]>0}
            for i,values in zip(ids,m):
                total=int(values.sum());top=int(values.max());purity.append({'view':view,'track_id':i,'observations':total,'minority_identity_observations':total-top,'purity':top/total,'gt_counts':{str(t):int(n) for t,n in zip(targets,values) if n}})
        timeline=defaultdict(list)
        for (frame,view,row),r in aligned.items():
            r['correct']=r['gt'] is not None and mapping.get(view,{}).get(r['id'])==r['gt']
            if r['gt'] is not None:timeline[(view,r['gt'])].append((frame,r['id'],int(r['correct'])))
        for values in timeline.values():values.sort()
        online={key(d):d for d in (json.loads(l) for l in (root/'online_decisions.jsonl').open())}
        write_conflicts=0;memory_targets={};write_events=Counter()
        for k,d in sorted(online.items()):
            if d['question']!='MEMORY_DECISION' or d['action']!='WRITE_MEMORY':continue
            r=aligned.get((k[0],k[1],k[3]));c=d['context'];track=c.get('track_id',c.get('proposal_track_id'))
            if r and r['gt'] is not None and track is not None:
                old=memory_targets.get(int(track));write_conflicts+=old is not None and old!=r['gt'];write_events[(k[0],k[1],r['gt'])]+=int(old is not None and old!=r['gt']);memory_targets[int(track)]=r['gt']
        counts[variant]={'predicted_ids':len({r['id'] for r in aligned.values()}),'gt_matched_observations':sum(r['gt'] is not None for r in aligned.values()),'correct_gt_mapped_observations':sum(r['correct'] for r in aligned.values()),'wrong_gt_mapped_observations':sum(r['gt'] is not None and not r['correct'] for r in aligned.values()),'duplicate_predicted_ids_within_image':duplicates,'mixed_identity_tracks':sum(p['minority_identity_observations']>0 for p in purity),'minority_identity_observations':sum(p['minority_identity_observations'] for p in purity),'offline_write_target_conflicts':write_conflicts,'track_purity':purity}
        rows[variant]=aligned;maps[variant]=mapping;timelines[variant]=timeline;decisions[variant]=online;writes[variant]=write_events
    assert all(set(r)==set(rows['A0']) for r in rows.values())
    assert all(r[k]['box']==rows['A0'][k]['box'] and r[k]['score']==rows['A0'][k]['score'] for v,r in rows.items() for k in r)
    summaries={};temporal={};rawpath=OUT/(args.prefix+'paired_decisions.jsonl');raw=rawpath.open('w')
    for variant in args.variants:
        old=decisions['A0'];new=decisions[variant];transitions={};unmatched=Counter();damage=defaultdict(lambda:defaultdict(float));first=None;examples=[]
        for k in sorted(set(old)|set(new)):
            if k not in old or k not in new:
                unmatched[('only_GMT:' if k not in new else 'only_JEV:')+k[2]]+=1;continue
            a=old[k];b=new[k];transition=k[2]+':'+a['action']+'->'+b['action'];t=transitions.setdefault(transition,{'count':0,'N01':0,'N10':0,'both_correct':0,'both_wrong':0,'GT_unmatched':0,'horizons':{str(h):defaultdict(float) for h in HORIZONS}});t['count']+=1
            rowkey=(k[0],k[1],k[3]);ro=rows['A0'].get(rowkey);rn=rows[variant].get(rowkey)
            if not ro or ro['gt'] is None:t['GT_unmatched']+=1;continue
            n01=int(not ro['correct'] and rn['correct']);n10=int(ro['correct'] and not rn['correct']);t['N01']+=n01;t['N10']+=n10;t['both_correct']+=ro['correct'] and rn['correct'];t['both_wrong']+=not ro['correct'] and not rn['correct']
            divergent=a['action']!=b['action'];windows={}
            if divergent:
                if first is None:first=list(k)
                targetkey=(k[1],ro['gt'])
                for h in HORIZONS:
                    measures=[]
                    for v in ('A0',variant):
                        allvalues=timelines[v][targetkey];values=[x for x in allvalues if k[0]<=x[0]<=k[0]+h];prefix=[x for x in allvalues if x[0]<k[0]];id_before=prefix[-1][1] if prefix else None
                        ids=[x[1] for x in values];switch=sum(x!=y for x,y in zip(([id_before] if id_before is not None else [])+ids,ids if id_before is not None else ids[1:]));prefix_ids={x[1] for x in prefix}
                        correct=sum(x[2] for x in values);wrong=len(values)-correct;recovery=next((x[0]-k[0] for x in values if x[2]),None)
                        measures.append({'correct_identity_duration':correct,'wrong_identity_duration':wrong,'identity_switches':switch,'identity_fragmentation_proxy':max(0,len(set(ids))-1),'new_id_proliferation':len(set(ids)-prefix_ids),'identity_continuity':int(bool(ids) and len(set(ids))==1),'recovery_latency':recovery,'offline_write_target_conflicts':sum(n for (f,w,g),n in writes[v].items() if w==k[1] and g==ro['gt'] and k[0]<=f<=k[0]+h)})
                    delta={m:measures[1][m]-measures[0][m] for m in measures[0] if measures[0][m] is not None and measures[1][m] is not None}
                    windows[str(h)]={'GMT':measures[0],'JEV':measures[1],'delta':delta}
                    for m,x in delta.items():t['horizons'][str(h)][m]+=x;damage[str(h)][m]+=x
                    damage[str(h)]['divergences']+=1
                entry={'variant':variant,'key':list(k),'transition':transition,'GMT_id':ro['id'],'JEV_id':rn['id'],'gt':ro['gt'],'N01':n01,'N10':n10,'future':windows}
                raw.write(json.dumps(entry,sort_keys=True)+'\n')
                if n10 and len(examples)<8:examples.append(entry)
        for t in transitions.values():t['net_corrected']=t['N01']-t['N10']
        # Fill required transitions even when none occurred.
        for q,a,b in [('MATCH_DECISION','ACCEPT_CURRENT','START_NEW'),('MATCH_DECISION','ACCEPT_CURRENT','REASSOCIATE'),('MATCH_DECISION','START_NEW','ACCEPT_CURRENT'),('MATCH_DECISION','START_NEW','REASSOCIATE'),('MEMORY_DECISION','WRITE_MEMORY','SKIP_MEMORY'),('MEMORY_DECISION','SKIP_MEMORY','WRITE_MEMORY'),('REACTIVATION_DECISION','START_NEW','REACTIVATE_OLD')]:transitions.setdefault(q+':'+a+'->'+b,{'count':0,'N01':0,'N10':0,'net_corrected':0,'horizons':{}})
        summaries[variant]={'by_transition':transitions,'unpaired_dynamic_events':dict(unmatched),'first_action_divergence':first,'harmful_examples':examples}
        temporal[variant]={'aggregate_divergence_windows':damage,'aggregation_warning':'Overlapping future windows are counted per divergence; totals are not unique frames or whole-run causal effects.'}
    raw.close()
    save(args.prefix+'PAIRED_DECISION_AUDIT.json',{'status':'COMPLETE','definition':'Per-image IoU>=0.5 one-to-one GT matching; per-camera whole-run maximum-count one-to-one predicted-ID/GT-ID mapping. N01/N10 classify paired assignment observations, not isolated causal action outcomes.','GT_only_offline':True,'all_prediction_geometry_identical':True,'comparisons':summaries,'full_run_identity_statistics':counts,'mapping':maps,'raw_paired_decisions_sha256':sha(rawpath),'memory_immediate_N01_N10_not_action_attribution':True})
    save(args.prefix+'TEMPORAL_DAMAGE.json',{'status':'COMPLETE','horizons':list(HORIZONS),'comparisons':temporal,'identity_fragmentation_proxy_is_not_TrackEval_Frag':True,'write_target_conflict_is_not_native_bank_contamination':True,'stale_recovery_scope':'First future GT-mapped correct observation of the target; raw Reactivation decisions separately counted.','causal_local_attribution_proven':False})
    print(json.dumps({'status':'COMPLETE','identity_stats':{v:{k:n for k,n in c.items() if k!='track_purity'} for v,c in counts.items()}}))
if __name__=='__main__':main()
