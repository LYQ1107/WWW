"""Versioned TRAIN certificates from an ID's actual recent committed segment."""
import collections
import gzip
import torch
from jev_phase15_common import *
from jev_phase14_artifacts import load_dense
from jev_phase13_learning import atomic_torch


def recent_owner(history,frame):
    if len(history)<3:return None
    last=list(history)[-3:]
    if any(gt is None for _,gt in last):return None
    if len({gt for _,gt in last})!=1:return None
    if any(b[0]-a[0]!=1 for a,b in zip(last,last[1:])):return None
    if not 0<frame-last[-1][0]<=2:return None
    return last[-1][1]


def patch_record(r,recent):
    # WHO ownership of a globally mixed ID remains UNKNOWN. Only its current
    # commitment is unsafe when a separate, recent causal certificate exists.
    patch={};stats=collections.Counter();q=len(r['rows']);k=len(r['refs']);frame,view=r['key'][1:]
    commit=r['commit_positive'].clone();kind=r['commit_kind'].clone()
    known=r['known_options'].clone();safety=r['safety'].clone()
    feature=r['inputs']['commitment_features'][0];legal=r['inputs']['legal'][0]
    for row,gt in enumerate(r['GT_labels_OFFLINE_ONLY']):
        if gt is None or not k or int(kind[row])!=0 or not r['positive'][row,:k].any():continue
        col=int(feature[row,:,0].argmax());refid=r['refs'][col]
        owner=recent_owner(recent.get((refid,view),[]),frame)
        if not legal[row,col] or owner is None or owner==gt or r['known_options'][row,col]:continue
        stats['recent_wrong_segment_with_pure_alternative']+=1
        if feature[row,col,0]<=.5:continue
        commit[row]=r['positive'][row];kind[row]=2
        known[row,col]=True;safety[row,col]=0
        stats['additional_necessary_correction']+=1
    if stats['additional_necessary_correction']:
        patch=dict(commit_positive=commit,commit_kind=kind,commit_known_options=known,safety=safety)
    return patch,stats


def main():
    protect();torch.set_num_threads(1)
    protocol=read(REPORTS/'RECENT_OWNER_LABEL_PROTOCOL.json');assert protocol['status']=='FROZEN_BEFORE_COUNTS'
    folder=OUT/'recent_owner_labels_v2';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'RESULT.json').exists()
    patches={};counts=collections.Counter();by_video={};sources=[];examples=[]
    for video in TRAIN:
        corpus=read(OUT/'commitment_dataset_v2'/f'video{video:02d}'/'RESULT.json')
        descriptor=corpus['DATASET'];assert sha(descriptor['path'])==descriptor['SHA256']
        commits=OUT/'commitment_dataset_v2'/f'video{video:02d}'/'COMMITS.jsonl.gz'
        events={tuple(e['key']):e for e in map(json.loads,gzip.open(commits,'rt'))}
        data=load_dense(descriptor['path']);recent=collections.defaultdict(lambda:collections.deque(maxlen=3));local=collections.Counter()
        previous_key=None
        for r in data['records']:
            assert r['key'][0]==video and (previous_key is None or tuple(r['key'])>previous_key)
            previous_key=tuple(r['key']);frame,view=r['key'][1:]
            patch,statistics=patch_record(r,recent)
            split='audit' if r['audit_block'] else 'train'
            for key,value in statistics.items():local[split+'_'+key]+=value
            if patch:
                patches[tuple(r['key'])]=patch
                for row in torch.where((patch['commit_kind']==2)&(r['commit_kind']!=2))[0].tolist():
                    col=int(r['inputs']['commitment_features'][0,row,:,0].argmax())
                    if len(examples)<32:examples.append(dict(key=r['key'],row=row,audit_block=r['audit_block'],
                        causal_predecessor_ID=r['refs'][col],past_segment=list(recent[r['refs'][col],view]),
                        current_GT_OFFLINE_ONLY=r['GT_labels_OFFLINE_ONLY'][row],
                        predecessor_support=float(r['inputs']['commitment_features'][0,row,col,0]),
                        safe_alternative_IDs=[r['refs'][i] for i in torch.where(r['positive'][row,:len(r['refs'])])[0].tolist()]))
            event=events[tuple(r['key'][1:])];assert len(event['ids'])==len(r['rows'])
            for identity,gt in zip(event['ids'],r['GT_labels_OFFLINE_ONLY']):recent[identity,view].append((frame,gt))
        by_video[str(video)]=dict(local);counts.update(local)
        sources.append(dict(video=video,dataset=descriptor,commits=ref(commits),collection=ref(OUT/'commitment_dataset_v2'/f'video{video:02d}'/'RESULT.json')))
        print('PHASE15_RECENT_LABEL_COUNTS',video,dict(local),flush=True)
        del data
    artifact=folder/'LABEL_PATCH.pth';atomic_torch(artifact,dict(patches=patches,protocol=ref(REPORTS/'RECENT_OWNER_LABEL_PROTOCOL.json'),sources=sources))
    source=binding(seed=20261009,dataset=sources,evaluator='past-only actual native recent-segment certificates',scope='TRAIN label identifiability; no DEV or actor GT')
    result=dict(status='COMPLETE',binding=source,counts=dict(counts),by_video=by_video,changed_payloads=len(patches),
        patch=ref(artifact),examples=examples,raw_data_and_native_inputs_unchanged=True,WHO_mixed_ownership_remains_UNKNOWN=True,
        no_future_labels=True,reserved_audit_never_gradients=True,
        new_reserved_correction_support=counts.get('audit_additional_necessary_correction',0))
    save(folder/'RESULT.json',result);save(REPORTS/'RECENT_OWNER_LABEL_IDENTIFIABILITY.json',result)


if __name__=='__main__':main()
