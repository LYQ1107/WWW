"""Freeze160 actual paired images before new feature results; reuse native error index."""
import collections
import itertools
import json
from pathlib import Path
from PIL import Image
from omega_corr_common import ROOT,DATA,FIT,MECHANISM,role,save_new,sha256,verified_native


def choose(errors,occupied,scenes,limit):
    buckets=collections.defaultdict(lambda:collections.defaultdict(list))
    for e in errors:
        if e['key'][0] in scenes and e['synchronous_pure_positive_tracks']:
            buckets[e['key'][0]][e['query_gt_identity']].append(e)
    queues={}
    for scene,people in sorted(buckets.items()):
        per_person=[]
        for person,events in sorted(people.items()):
            selected=[];times=[]
            for e in sorted(events,key=lambda x:(x['frame'],x['key'])):
                if (scene,e['frame']) not in occupied and all(abs(e['frame']-t)>=30 for t in times):selected.append(e);times.append(e['frame'])
            per_person.append(selected)
        queues[scene]=[e for chunk in itertools.zip_longest(*per_person) for e in chunk if e is not None]
    answer=[]
    while len(answer)<limit and any(queues.values()):
        for scene,q in sorted(queues.items()):
            while q:
                e=q.pop(0)
                if (scene,e['frame']) not in occupied:
                    answer.append(e);occupied.add((scene,e['frame']));break
            if len(answer)==limit:break
    return answer


def main():
    plan_path=ROOT/'manifests/omega_corr_preregistration_v1.json';plan=json.loads(plan_path.read_text())
    for p,h in plan['frozen_inputs'].items():assert sha256(ROOT/p)==h
    old=json.loads((ROOT/'manifests/omega_fixed_pair_cohort_v1.json').read_text())
    scope=json.loads((ROOT/'results/omega_native_synchronous_scope_v1.json').read_text())
    pairs=[dict(p,role=role(p['scene']),omega_corr_selection='RETAINED_ALL_PREVIOUS96') for p in old['pairs']]
    occupied={(p['scene'],p['group']) for p in pairs}
    extra=choose(scope['errors'],occupied,FIT+MECHANISM,56)+choose(scope['errors'],occupied,(12,),8)
    data=json.loads((DATA/'annotations/train.json').read_text())
    images=collections.defaultdict(list)
    for i in data['images']:images[i['video_id'],i['frame_id']].append(i)
    for e in extra:
        scene,group=e['key'][0],e['frame'];items=sorted(images[scene,group],key=lambda i:i['view_id']);assert len(items)==2
        pair_images=[]
        for i in items:
            path=DATA/'train'/i['file_name']
            with Image.open(path) as image:w,h=image.size
            assert (w,h)==(i['width'],i['height'])
            pair_images.append(dict(camera_id=i['view_id'],image_id=i['id'],path=str(path),sha256=sha256(path),original_width=w,original_height=h))
        pairs.append(dict(pair_id=f'scene{scene}_group{group}',scene=scene,group=group,role=role(scene),
            omega_corr_selection='ADDED_PREEXISTING_NATIVE_SYNC_PURE_ERROR',images=pair_images))
    assert len(pairs)<=160 and len({p['pair_id'] for p in pairs})==len(pairs)
    pair_ids={p['pair_id'] for p in pairs}
    excluded={(e['key'][0],e['frame']) for e in scope['errors'] if f"scene{e['key'][0]}_group{e['frame']}" not in pair_ids}
    local=[f"scene{e['key'][0]}_group{e['frame']}" for e in choose(scope['errors'],excluded,FIT+MECHANISM+(12,),32)]
    for p in pairs:
        if len(local)==32:break
        if p['pair_id'] not in local:local.append(p['pair_id'])
    _,diagnostic_receipt=verified_native(12)
    labels=[e for e in scope['errors'] if f"scene{e['key'][0]}_group{e['frame']}" in pair_ids]
    # Labels live in a separate file; production extraction reads only paired inputs.
    save_new(ROOT/'manifests/omega_corr_cohort_v1.json',dict(schema='omega-corr-paired-inputs/v1',status='FROZEN_BEFORE_TOKEN_OUTPUTS',pairs=pairs,
        pair_count=len(pairs),retained_previous96=True,additional_pairs=len(extra),roles=plan['roles'],local_crop_pair_ids=local,
        source_cohort_sha256=sha256(ROOT/'manifests/omega_fixed_pair_cohort_v1.json'),native_source_receipts=old['native_source_receipts']+[diagnostic_receipt],
        note='All queries and legal references in these groups, not just these offline selected errors; F validity never determines membership. Diagnostic12 receipt verified separately before consumption.'))
    save_new(ROOT/'results/omega_corr_cohort_offline_labels_v1.json',dict(status='OFFLINE_ONLY_ALREADY_EXPOSED_TRAIN_LABELS',selected_known_errors=labels,
        scene_counts=dict(collections.Counter(p['scene'] for p in pairs)),role_counts=dict(collections.Counter(p['role'] for p in pairs)),
        all12_native_applicability=scope['all12_counts'],no_new_GT_matching=True,no_GMT_inference=True,no_new_Omega_inference=True))
    print('fixed pairs',len(pairs),'additional',len(extra),'known errors',len(labels),'local',len(local))

if __name__=='__main__':main()
