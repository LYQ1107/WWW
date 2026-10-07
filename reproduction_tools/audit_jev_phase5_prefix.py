"""Offline identity-prefix and actual native-bank sample-purity audit."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'reproduction_tools'))
from audit_jev_phase5 import OUT,TRAIN,save,key
from audit_jev_phase5_decisions import xyxy,overlaps


def main():
    ann=json.loads(TRAIN.read_text());images={i['id']:i for i in ann['images'] if i['video_id']==1};gt=defaultdict(list)
    for a in ann['annotations']:
        if a['image_id'] in images and a.get('conf',1)>0 and a.get('category_id',1)==1:gt[a['image_id']].append(a)
    original=json.loads((OUT/'ablations/A0/tracking_predictions/gmt_off.json').read_text());byimage=defaultdict(list)
    for p in original:byimage[p['image_id']].append(p)
    matched={}
    for image_id,ps in byimage.items():
        image=images[image_id];targets=gt[image_id]
        for row in range(len(ps)):matched[(image['frame_id']-1,image['view_id']-1,row)]=None
        if targets:
            ious=overlaps(np.asarray([xyxy(p['bbox']) for p in ps]),np.asarray([xyxy(t['bbox']) for t in targets]));rr,cc=linear_sum_assignment(-((ious>=.5).astype(float)*1000+ious))
            for r,c in zip(rr,cc):
                if ious[r,c]>=.5:matched[(image['frame_id']-1,image['view_id']-1,int(r))]=int(targets[c]['instance_id'])
    results={}
    for v in ('A0','A1','A2','A3','A4'):
        root=OUT/'ablations'/v;method='gmt_off' if v=='A0' else 'jev';pred=json.loads((root/'tracking_predictions'/f'{method}.json').read_text());payloads={};row_counts=Counter()
        for p in pred:
            image=images[p['image_id']];k=(image['frame_id']-1,image['view_id']-1);row=row_counts[k];row_counts[k]+=1;payloads.setdefault(k,[]).append((row,p['track_id'],matched[(*k,row)]))
        online=defaultdict(list)
        for line in (root/'online_decisions.jsonl').open():
            d=json.loads(line);online[(d['context']['frame'],d['context']['view'])].append(d)
        memory=defaultdict(list);bank={};last_gt={};stats=Counter();react=[];errors=[];question=defaultdict(Counter)
        for k,assignments in payloads.items():
            decisions=online[k];current={r:(t,g) for r,t,g in assignments}
            # All decisions at one frame/view precede this key's memory writes.
            for d in decisions:
                context=d['context'];state=context['tracker_state_before'];row=context['detection_index'];track,target=current[row];q=d['question'];question[q]['events']+=1
                for t,n in state['memory_lengths'].items():
                    if len(memory[int(t)])!=n and len(errors)<12:errors.append({'key':k,'track_id':t,'reconstructed':len(memory[int(t)]),'logged':n})
                if target is not None:
                    old=last_gt.get(track);question[q]['GT_matched']+=1;question[q]['prefix_existing_identity_conflicts']+=old is not None and old!=target
                if q!='REACTIVATION_DECISION':continue
                for t in state['old_reid_ids']:
                    if t not in bank:
                        n=state['memory_bank_size'];labels=memory[t][-n:];bank[t]=list(labels);stats['bank_promotions']+=1;stats['mixed_GT_bank_promotions']+=len({g for g in labels if g is not None})>1
                candidate=context['track_id'];labels=bank.get(candidate,[]);known=[g for g in labels if g is not None];old=last_gt.get(candidate)
                status='NOT_REACTIVATED'
                if d['action']=='REACTIVATE_OLD':
                    if target is None:status='CURRENT_DETECTION_GT_UNMATCHED'
                    elif old is None:status='NO_GT_MATCHED_PREFIX_FOR_ID'
                    else:status='FALSE_REACTIVATION' if old!=target else 'PREFIX_IDENTITY_RECOVERED'
                    stats[status]+=1
                react.append({'key':list(k)+[row],'action':d['action'],'off_action':d['off_action'],'candidate_id':candidate,'final_committed_id':track,'candidate_score':context['reactivation_score'],'gt':target,'prefix_gt_for_candidate':old,'classification':status,'native_bank_sample_GT_counts':dict(Counter('UNKNOWN' if g is None else str(g) for g in labels)),'mixed_bank_sample_identities':len(set(known))>1,'known_bank_samples_wrong_for_current_target':sum(g!=target for g in known) if target is not None else None})
            for d in decisions:
                if d['question']=='MEMORY_DECISION' and d['action']=='WRITE_MEMORY':
                    row=d['context']['detection_index'];track,target=current[row];assert track==d['context']['track_id'];memory[track].append(target)
                elif d['question']=='REACTIVATION_DECISION' and d['action']=='REACTIVATE_OLD':bank.pop(d['context']['track_id'],None)
            for row,track,target in assignments:
                if target is not None:last_gt[track]=target
        assert not errors,errors
        results[v]={'memory_write_history_lengths_match_logged_state':True,'question_counts':dict(question),'bank_and_reactivation_counts':dict(stats),'reactivation_events':react,'final_memory_recent10_purity':[{'track_id':t,'known_GT_counts':dict(Counter(str(x) for x in labels[-10:] if x is not None)),'unknown_samples':sum(x is None for x in labels[-10:])} for t,labels in memory.items()]}
    save('PREFIX_IDENTITY_AND_NATIVE_BANK_AUDIT.json',{'status':'COMPLETE','GT_only_offline':True,'memory_history_reconstructed_from_actual_online_writes':True,'native_bank_semantics':'last bank_size raw-ReID write samples at first appearance in logged old_reid_ids; persistent until reactivated; only GT labels attached offline','false_reactivation_definition':'At an actual REACTIVATE_OLD, current one-to-one IoU-matched GT differs from last GT-matched observation for that candidate ID before this frame/view commit. Unknown GT/prefix is unassessed, not counted false.','wrong_commit_definition':'Final committed existing ID changes its last observed GT identity across key boundary; observational prefix conflict, not counterfactual action utility.','results':results})
    print(json.dumps({'status':'COMPLETE','counts':{v:r['bank_and_reactivation_counts'] for v,r in results.items()}}))
if __name__=='__main__':main()
