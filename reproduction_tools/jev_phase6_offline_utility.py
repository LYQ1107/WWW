"""Annotation-only evaluation; this module is never imported by live policy."""
from collections import Counter, defaultdict
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_jev_phase5_decisions import xyxy, overlaps
from jev_phase6_common import save


class OfflineIdentityAudit:
    def __init__(self, video):
        from run_early_pilot_tracking import ANNOTATIONS
        annotations=json.loads(ANNOTATIONS.read_text())
        self.images={i['id']:i for i in annotations['images'] if int(i['video_id'])==video}
        self.gt=defaultdict(list)
        for a in annotations['annotations']:
            if a['image_id'] in self.images and a.get('conf',1)>0 and a.get('category_id',1)==1:
                self.gt[a['image_id']].append(a)

    def align(self, predictions):
        grouped=defaultdict(list)
        for p in predictions:grouped[p['image_id']].append(p)
        rows={}
        for image_id, detections in grouped.items():
            image=self.images[image_id]; targets=self.gt[image_id]; matched={}
            if targets and detections:
                ious=overlaps(np.array([xyxy(d['bbox']) for d in detections]),np.array([xyxy(t['bbox']) for t in targets]))
                rr,cc=linear_sum_assignment(-((ious>=.5).astype(float)*1000+ious))
                matched={int(r):int(targets[c]['instance_id']) for r,c in zip(rr,cc) if ious[r,c]>=.5}
            for row,d in enumerate(detections):
                rows[(int(image['frame_id'])-1,int(image['view_id'])-1,row)]={
                    'id':int(d['track_id']),'gt':matched.get(row),'bbox':d['bbox'],'score':d['score']}
        return rows


def prefix_identity(rows, key):
    counts=defaultdict(Counter)
    for k,r in rows.items():
        if k[:2]<key[:2] and r['gt'] is not None: counts[r['id']][r['gt']]+=1
    mapping={t:c.most_common(1)[0][0] for t,c in counts.items()}
    purity={str(t):dict(c) for t,c in counts.items()}
    return mapping,purity


def whole_run_correct(rows):
    counts=defaultdict(Counter); mapping={}
    for (_,view,_),r in rows.items():
        if r['gt'] is not None:counts[view][(r['id'],r['gt'])]+=1
    for view,c in counts.items():
        ids=sorted({t for t,g in c}); targets=sorted({g for t,g in c})
        matrix=np.array([[c[(t,g)] for g in targets] for t in ids]); rr,cc=linear_sum_assignment(-matrix)
        mapping[view]={ids[r]:targets[c] for r,c in zip(rr,cc) if matrix[r,c]>0}
    return {k:r['gt'] is not None and mapping.get(k[1],{}).get(r['id'])==r['gt'] for k,r in rows.items()}


def consequence(rows, prefix_rows, start_key, target, horizon, next_id):
    """Paired fixed-prefix identity utility, allowing correct new identities.

    New IDs inherit their first offline matched target; old IDs retain their
    pre-intervention majority identity. Fragmentation/switches charge births.
    Unknown observations are censored rather than declared correct/wrong.
    """
    mapping,_=prefix_identity(prefix_rows,start_key)
    ordered=sorted((k,r) for k,r in rows.items() if k[:2]>=start_key[:2] and k[0]<=start_key[0]+horizon)
    born={}
    for _,r in ordered:
        if r['id']>next_id and r['id'] not in born and r['gt'] is not None:born[r['id']]=r['gt']
    mapping.update(born)
    target_rows=[(k,r) for k,r in ordered if target is not None and r['gt']==target]
    correct=sum(mapping.get(r['id'])==target for k,r in target_rows)
    wrong=sum(r['id'] in mapping and mapping[r['id']]!=target for k,r in target_rows)
    unknown=len(target_rows)-correct-wrong
    timelines=defaultdict(list); prefix_last={}
    for k,r in sorted(prefix_rows.items()):
        if k[:2]<start_key[:2] and r['gt']==target:prefix_last[k[1]]=r['id']
    for k,r in target_rows:timelines[k[1]].append((k[0],r['id']))
    switches=0
    for view,values in timelines.items():
        ids=([prefix_last[view]] if view in prefix_last else [])+[t for f,t in values]
        switches+=sum(a!=b for a,b in zip(ids,ids[1:]))
    target_ids={r['id'] for k,r in target_rows}; new_ids={t for t in target_ids if t>next_id}
    contamination=sum(r['gt'] is not None and r['id'] in mapping and mapping[r['id']]!=r['gt'] for k,r in ordered)
    collisions=0
    perkey=defaultdict(list)
    for k,r in ordered:perkey[k[:2]].append(r['id'])
    collisions=sum(len(ids)-len(set(ids)) for ids in perkey.values())
    latency=next((k[0]-start_key[0] for k,r in target_rows if mapping.get(r['id'])==target),None)
    value={'correct_identity_duration':correct,'wrong_identity_duration':wrong,'unknown_identity_duration':unknown,
           'identity_switches':switches,'new_id_creation':len(new_ids),'identity_fragmentation_proxy':max(0,len(target_ids)-1),
           'identity_contamination':contamination,'duplicate_identity_collisions':collisions,
           'identity_continuity':bool(target_rows and len(target_ids)==1),'recovery_latency':latency}
    value['utility']=correct-wrong-.5*switches-.25*len(new_ids)-.5*collisions-contamination
    return value
