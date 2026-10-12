"""Offline labels on actual past committed IDs; actor tensors are never relabelled.

Cross-view compatibility and continuity of the previously certified Global ID
are different targets. Neither target is a future safe-action/value certificate.
Only compact new labels are saved; original dense inputs are reused in place.
"""
import collections
import gc
import gzip
import traceback
import torch
from jev_oneweek_common import *
from jev_oneweek_common import ref as asset_ref


def certify(votes,total):
    n=sum(votes.values())
    if n>=3 and n/max(total,1)>=.8 and len(votes)==1:
        return next(iter(votes))
    return None


def main():
    protect();torch.set_num_threads(1)
    protocol_path=REPORTS/'LABEL_PROTOCOL.json'
    assert not (REPORTS/'SUPERVISION_AUDIT.json').exists(),'Completed audit is immutable'
    protocol=dict(status='FROZEN_BEFORE_NEW_LABEL_COUNTS',
                  per_view_certificate=dict(min_known=3,min_known_coverage=.8,owner='exactly one observed GT identity; mixtures UNKNOWN'),
                  global_certificate='Identical certificate, pooled actual past views; checked against frozen source known_options/positive',
                  cross_target='Current known GT equals certified other-camera historical GT; current identity duplicates UNKNOWN',
                  stability_target='If latest same-GT own-camera past observation <=8 frames has a certified pure compatible legal Global ID, label its continuation 1 and other certified choices 0. Otherwise UNKNOWN; current continuity only, not future action safety',
                  row_dependence='same person/video and scene/time blocks dependent; no IID claims',
                  two_question_differences_required=True,training_videos=[12,13,14],audit_only_video=16,
                  dataset_inference_source='Frozen Phase XIV Multi seed20261009 real predicted-state Phase XV commitment_dataset_v2',
                  no_synthetic_events=True,GT_actor_inputs=False)
    if protocol_path.exists():
        assert read(protocol_path)==protocol,'Engineering retry must retain the exact label protocol'
    else:
        save(protocol_path,protocol)
    from jev_phase14_artifacts import load_dense,save_dense
    from jev_phase13_runtime import cache_inputs
    from build_dense_jev_stage2_dataset import OfflineLabels
    all_labels=[];summaries=[]
    for video in TRAIN:
        folder=PHASE15/'commitment_dataset_v2'/f'video{video:02d}'
        manifest=read(folder/'RESULT.json')
        assert sha(manifest['DATASET']['path'])==manifest['DATASET']['SHA256']
        data=load_dense(manifest['DATASET']['path']);records=data['records']
        by_key={tuple(r['key'][1:]):r for r in records}
        assert len(by_key)==len(records)
        with gzip.open(folder/'COMMITS.jsonl.gz','rt') as stream:
            commits=[json.loads(line) for line in stream]
        _,frames,reader=cache_inputs(video);offline=OfflineLabels(video,reader)
        # Apply the identical duplicate-GT UNKNOWN rule already used by source.
        duplicate_rows=0
        for key,image in offline.images.items():
            counts=collections.Counter(a['instance_id'] for a in offline.gt[image['id']])
            bad={g for g,n in counts.items() if n>1}
            if bad:
                actual=offline.current(*key)
                duplicate_rows+=sum(g in bad for g in actual if g is not None)
                offline.aligned[key]=[None if g in bad else g for g in actual]
        views=collections.defaultdict(collections.Counter);view_total=collections.Counter()
        global_votes=collections.defaultdict(collections.Counter);global_total=collections.Counter()
        latest={};count=collections.Counter();persons=collections.defaultdict(set)
        def update(frame,view,ids,targets):
            assert len(ids)==len(targets) and len(ids)==len(set(ids))
            for ref,gt in zip(ids,targets):
                view_total[ref,view]+=1;global_total[ref]+=1
                if gt is not None:
                    views[ref,view][gt]+=1;global_votes[ref][gt]+=1
                    latest[gt,view]=(frame,ref);persons[gt].add((frame//64,view))
        first=commits[0];first_view=int(first['key'][1]);boot_view=1-first_view
        bootstrap=offline.current(0,boot_view)
        update(0,boot_view,list(range(1,len(bootstrap)+1)),bootstrap)
        for event in commits:
            frame,view=map(int,event['key']);key=(frame,view);r=by_key.get(key)
            targets=offline.current(frame,view)
            if r is not None:
                assert targets==r['GT_labels_OFFLINE_ONLY'],(video,key,'GT source mismatch')
                q,k=len(targets),len(r['refs']);cross=torch.full((q,k),-1,dtype=torch.int8)
                stable=cross.clone();global_owners=[]
                for ref in r['refs']:
                    global_owners.append(certify(global_votes[ref],global_total[ref]))
                for row,gt in enumerate(targets):
                    if gt is None:continue
                    previous=latest.get((gt,view));previous_col=None
                    if previous and 0<=frame-previous[0]<=8 and previous[1] in r['refs']:
                        col=r['refs'].index(previous[1])
                        if global_owners[col]==gt and bool(r['inputs']['legal'][0,row,col]):
                            previous_col=col
                    for col,ref in enumerate(r['refs']):
                        legal=bool(r['inputs']['legal'][0,row,col])
                        owner=global_owners[col]
                        assert bool(r['known_options'][row,col])==(owner is not None)
                        assert bool(r['positive'][row,col])==(owner==gt and legal)
                        other=certify(views[ref,1-view],view_total[ref,1-view])
                        if legal and owner is not None and other is not None:
                            cross[row,col]=int(other==gt)
                        if legal and owner is not None and previous_col is not None:
                            stable[row,col]=int(col==previous_col)
                both=(cross>=0)&(stable>=0)
                count['joint_known_edges']+=int(both.sum())
                count['independent_target_disagreements']+=int(((cross!=stable)&both).sum())
                for name,labels in [('cross',cross),('stability',stable)]:
                    for target in (-1,0,1):count[f'{name}_{target}']+=int((labels==target).sum())
                    count[f'{name}_positive_rows']+=int((labels==1).any(-1).sum())
                count['current_known_rows']+=sum(gt is not None for gt in targets)
                count['current_UNKNOWN_rows']+=sum(gt is None for gt in targets)
                evidence=r['inputs']['pair_evidence'][0]
                pos=r['positive'][:,:k]&r['inputs']['legal'][0]
                wrong=r['known_options'][:,:k]&~pos&r['inputs']['legal'][0]
                scores=evidence[:,:,[0,1,2,3]].max(-1).values
                hard=(wrong&(scores>=.75)).any(-1)&pos.any(-1)
                count['natural_hard_rows']+=int(hard.sum());count['payloads']+=1
                count['multi_compatible_identity_rows']+=int((pos.sum(-1)>1).sum())
                all_labels.append(dict(key=r['key'],cross=cross,stability=stable,hard_rows=hard,
                                       train_gradient=video in [12,13,14] and not r['audit_block']))
            update(frame,view,event['ids'],targets)
        summary=dict(video=video,counts=dict(count),duplicate_current_GT_rows_masked=duplicate_rows,
                     person_groups=len(persons),person_time_view_groups=sum(len(v) for v in persons.values()),
                     source=asset_ref(folder/'RESULT.json'),commits=asset_ref(folder/'COMMITS.jsonl.gz'),
                     exact_original_known_positive_masks=True,source_tensor_unchanged=True,
                     neural_GT_or_labels=False)
        summaries.append(summary)
        save(OUT/'supervision/PROGRESS.json',dict(status='RUNNING',complete_videos=summaries))
        print('ONEWEEK_LABEL_VIDEO',video,dict(count),flush=True)
        del data,records,by_key,offline,reader,commits
        gc.collect()
    path=OUT/'supervision/LABELS.pth.xz'
    save_dense(path,dict(labels=all_labels,protocol=asset_ref(protocol_path)),reserve=30*2**30)
    totals=collections.Counter()
    for row in summaries:totals.update(row['counts'])
    result=dict(status='COMPLETE',binding=binding('Actual natural TRAIN past-history two-question label audit'),
                counts=dict(totals),videos=summaries,labels=asset_ref(path),original_dense_inputs_copied=False,
                independent_questions_observable=totals['independent_target_disagreements']>0,
                current_association_supervision_only=True,future_safe_action_gate_qualified=False,
                official_TEST=False,sealed20_21_22=True,synthetic_samples=0)
    save(REPORTS/'SUPERVISION_AUDIT.json',result)
    registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
    registry['jobs']['TRAIN_supervision_audit'].update(status='COMPLETE',results=asset_ref(REPORTS/'SUPERVISION_AUDIT.json'))
    save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)


if __name__=='__main__':
    try:main()
    except Exception:
        failure('supervision',traceback.format_exc());raise
