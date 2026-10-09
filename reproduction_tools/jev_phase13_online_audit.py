"""Offline causal identity audit; invoked only after immutable predictions exist."""
import gzip,collections,math
import numpy as np
import torch
from jev_phase13_common import *
from gtr.modeling.jev_stage2.assignment import lawful_choice

def audit(video,raw,journal,commits,reader):
    from build_dense_jev_stage2_dataset import OfflineLabels
    labels=OfflineLabels(video,reader);past=collections.defaultdict(set);ever=set();count=collections.Counter();loss=[];brier=[];conf=[];correct=[];observed=[]
    questions=collections.defaultdict(list)
    with gzip.open(journal,'rt') as h:
        for line in h:
            r=json.loads(line);questions[tuple(r['key'][1:])].append(r)
    with gzip.open(commits,'rt') as h:stream=list(map(json.loads,h))
    assert stream;first=stream[0];view=1-first['key'][2];image_ids={i['id']:i for i in labels.images.values()};boot=[p for p in raw if image_ids[p['image_id']]['frame_id']==1 and image_ids[p['image_id']]['view_id']==view+1]
    for p,gt in zip(boot,labels.current(0,view)):
        if gt is not None:past[p['track_id']].add(gt);ever.add(gt)
    for event in stream:
        key=tuple(event['key'][1:]);gts=labels.current(*key);ids=event['ids']
        for record in questions[key]:
            task=record['task'];refs=record['refs'];rows=record['rows'];z=torch.tensor(record['logits'],dtype=torch.float32);k=len(refs);legal=torch.tensor(record['legal'],dtype=torch.bool).reshape(len(rows),k);choice=lawful_choice(z,legal)
            for rr,(row,col) in enumerate(zip(rows,choice)):
                gt=gts[row];count['MATCH_questions' if task==0 else 'REACT_questions']+=1
                native=event['events'][row];actual_col=refs.index(native['identity']) if native['action']==('ASSOCIATE_EXISTING' if task==0 else 'REACTIVATE') else -1;assert col==actual_col,('saved decisions differ from native commit',key,task,row,col,actual_col)
                if gt is None:count['GT_UNKNOWN']+=1;continue
                known=np.array([len(past[t])==1 for t in refs]+[False]);positive=np.array([past[t]=={gt} for t in refs]+[False]);ambig=any(len(past[t])>1 and gt in past[t] for t in refs)
                if positive[:-1].any():known[-1]=True
                elif task==0 and not ambig:known[-1]=True;positive[-1]=True
                elif task==1 and gt not in ever and not any(len(v)>1 and gt in v for v in past.values()):known[-1]=True;positive[-1]=True
                else:count['label_UNKNOWN']+=1;continue
                selected=k if col<0 else col;scope='MATCH' if task==0 else 'REACT';count[scope+'_certified_rows']+=1
                if not known[selected]:count[scope+'_UNKNOWN_selected']+=1
                elif positive[selected]:count[scope+'_certified_correct']+=1
                else:count[scope+'_certified_wrong']+=1
                count['normal_supported_rows']+=int(task==0 and positive[:-1].any());count['normal_supported_correct']+=int(task==0 and positive[:-1].any() and col>=0 and positive[selected])
                if task==0:
                    lp=torch.log_softmax(z[rr].masked_fill(~torch.tensor(known),-1e4),-1);prob=lp.exp().numpy();mass=float(prob[positive].sum());loss.append(-math.log(max(mass,1e-30)));target=positive.astype(float)/max(1,positive.sum());brier.append(float(((prob-target)**2*known).sum()));best=int(prob.argmax());conf.append(float(prob[best]));correct.append(bool(positive[best]))
                observed.append({'key':record['key'],'task':scope,'row':row,'action':native['action'],'known':bool(known[selected]),'correct':bool(positive[selected]) if known[selected] else None})
        for row,(ref,gt) in enumerate(zip(ids,gts)):
            if gt is None:continue
            if event['events'][row]['action']=='START_NEW':count['GT_known_births']+=1;count['new_birth_previously_seen_GT']+=int(gt in ever)
            if event['events'][row]['action']=='REACTIVATE':
                count['GT_known_recoveries']+=1
                if len(past[ref])==1:count['certified_recovery_wrong']+=int(past[ref]!={gt})
                else:count['recovery_history_UNKNOWN']+=1
            past[ref].add(gt);ever.add(gt)
    count['contaminated_Gallery_IDs']=sum(len(v)>1 for v in past.values());count['known_Gallery_IDs']=sum(bool(v) for v in past.values());ece=0.;risk=[];bins=[]
    if conf:
        a=np.asarray(conf);c=np.asarray(correct)
        for lo in np.linspace(0,1,11)[:-1]:
            m=(a>=lo)&(a<lo+.100000001)
            if m.any():ece+=float(m.mean()*abs(a[m].mean()-c[m].mean()));bins.append({'lo':float(lo),'count':int(m.sum()),'confidence':float(a[m].mean()),'accuracy':float(c[m].mean())})
        order=np.argsort(-a,kind='stable');risk=[{'coverage':p,'risk':float(1-c[order[:max(1,math.ceil(p*len(c)))]].mean())} for p in [.1,.25,.5,.75,1.]]
    return {'status':'COMPLETE','counts':dict(count),'normal_association_keep_rate':count['normal_supported_correct']/count['normal_supported_rows'] if count['normal_supported_rows'] else None,'calibration':{'NLL':float(np.mean(loss)) if loss else None,'Brier':float(np.mean(brier)) if brier else None,'ECE':ece if conf else None,'reliability_bins':bins,'risk_coverage':risk,'scope':'conditional certified options only; no calibration claim for contaminated/UNKNOWN or untrained REACT fallback'},'Gallery_contamination_fraction':count['contaminated_Gallery_IDs']/count['known_Gallery_IDs'] if count['known_Gallery_IDs'] else None,'saved_logits_assignment_matches_actual_native_events':True,'GT_actor_inputs':False,'action_outcomes':observed}
