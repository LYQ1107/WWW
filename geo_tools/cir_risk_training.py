"""Offline TRAIN-only risk supervision; never imported by native inference."""
from collections import Counter
import gzip
import hashlib
import json
import numpy as np
import torch


def read_risk_training(path,expected_sha256,allowed_scenes):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Sealed risk training SHA mismatch')
    features=[];keys=[];known=[];labels=[];people=[];seen=set();all_scenes=Counter()
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line);key=tuple(row['key']);x=row['features'];label=row['risk_label']
            if len(key)!=3 or any(type(v)!=int for v in key) or key[0] not in allowed_scenes or key[1]<1 or key[2]<0:
                raise ValueError('Foreign or invalid physical TRAIN source')
            if key in seen:raise ValueError('Repeated query is not an independent training record')
            if len(x)!=12 or not np.isfinite(x).all():raise ValueError('Finite native-risk12 evidence required')
            if row['source_state']!='Original GMT history + same-context C1 primary joint proposal, no mutated future/veto utility yet':
                raise ValueError('Risk labels must not be relabelled as veto utility')
            seen.add(key);all_scenes[key[0]]+=1;features.append(x);keys.append(key)
            if label is None:continue
            if type(label)!=int or label not in (0,1) or type(row['person'])!=int or row['person']<1:
                raise ValueError('Only certified binary labels; UNKNOWN is masked')
            known.append(len(features)-1);labels.append(label);people.append((key[0],row['person']))
    if set(all_scenes)!=set(allowed_scenes):raise ValueError('Missing or extra declared training scenes')
    x=torch.tensor(features,dtype=torch.float32);index=torch.tensor(known,dtype=torch.long)
    y=torch.tensor(labels,dtype=torch.float32);groups=Counter((scene,person,label) for (scene,person),label in zip(people,labels))
    class_groups=Counter(label for scene,person,label in groups)
    if set(class_groups)!={0,1}:raise ValueError('Both certified classes required')
    weights=torch.tensor([len(labels)/(2*class_groups[label]*groups[(scene,person,label)])
        for (scene,person),label in zip(people,labels)],dtype=torch.float32)
    stats=dict(all_queries=len(keys),known_queries=len(known),UNKNOWN_queries=len(keys)-len(known),
        correct_queries=labels.count(0),wrong_queries=labels.count(1),
        wrong_scene_person_groups=class_groups[1],correct_scene_person_groups=class_groups[0],
        wrong_scenes=sorted({scene for (scene,person),label in zip(people,labels) if label==1}),
        training_video_ids=sorted({scene for scene,person in people}),all_source_video_ids=sorted(all_scenes),
        per_scene_queries=dict(all_scenes),weight_mean=float(weights.mean()))
    return dict(all_features=x,known_indices=index,features=x[index],labels=y,weights=weights,people=people,stats=stats)


def calibrate_risk_threshold(scores,labels,maximum_FPR=.001):
    """Frozen FIT F1 criterion, strict logit>threshold; ties are conservative."""
    scores=np.asarray(scores,dtype=np.float64);labels=np.asarray(labels)
    if scores.ndim!=1 or scores.shape!=labels.shape or not np.isfinite(scores).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Finite paired risk scores and binary labels required')
    positives=int(labels.sum());negatives=len(labels)-positives
    if not positives or not negatives:raise ValueError('Both classes required for threshold calibration')
    order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[ends];fp=ends+1-tp;f1=2*tp/(positives+tp+fp)
    valid=fp<=int(np.floor(maximum_FPR*negatives+1e-12));best=np.argmax(np.where(valid,f1,-1))
    threshold=float(s[0])
    if valid[best] and f1[best]>0:
        end=int(ends[best]);threshold=float(s[end+1]) if end+1<len(s) else float(s[-1]-1.)
    pred=scores>threshold;TP=int(((labels==1)&pred).sum());FP=int(((labels==0)&pred).sum());FN=positives-TP;TN=negatives-FP
    return dict(threshold=threshold,strict_comparison='logit>threshold',TP=TP,FP=FP,FN=FN,TN=TN,
        F1=2*TP/(2*TP+FP+FN),correct_record_FPR=FP/negatives,wrong_record_recall=TP/positives,
        maximum_correct_record_FPR=maximum_FPR,scope='in-sample FIT risk classification, not action utility or online tracking')


def dependence_metrics(scores,labels,people,threshold):
    grouped={}
    for score,label,person in zip(scores,labels,people):
        counts=grouped.setdefault(person,Counter());pred=score>threshold
        counts['TP' if label and pred else 'FN' if label else 'FP' if pred else 'TN']+=1
    recall=[c['TP']/(c['TP']+c['FN']) for c in grouped.values() if c['TP']+c['FN']]
    fpr=[c['FP']/(c['FP']+c['TN']) for c in grouped.values() if c['FP']+c['TN']]
    return dict(macro_wrong_person_group_recall=float(np.mean(recall)),macro_correct_person_group_FPR=float(np.mean(fpr)),
        per_scene_person=[dict(scene=s,person=p,counts=dict(c)) for (s,p),c in sorted(grouped.items())],
        caveat='Dependence groups are retained; no independent-trial significance claim')
