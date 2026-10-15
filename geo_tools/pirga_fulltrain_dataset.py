"""Lazy GT-free native actor loader; separately pinned TRAIN-only supervision.

The source store is immutable. No raw GT matcher, detector, teacher or video
reader is imported. Same-scene-frame refs stay in the native matrix but are
excluded from old C2/C3/C4 history projections under the inherited contract.
"""
from collections import OrderedDict
import gzip
import json
from pathlib import Path
import torch
from pirga_indexed_store import StoreReader, dense_matrix, epoch_order, sha

FIT={4,5,6,7,8,9,13,17,20,21,24}
LABEL_FIELDS={'event_key','GT_person','candidate_persons','actual_primary_state'}


class NativeQueryDataset:
    def __init__(self,root,manifest_sha,call_cache_size=8):
        if call_cache_size<1:raise ValueError('Positive bounded cache required')
        self.store=StoreReader(root,manifest_sha);self.cache=OrderedDict();self.cache_size=call_cache_size
    def _call(self,scene,cid):
        key=(scene,cid)
        if key not in self.cache:
            row=self.store.db.execute('SELECT body FROM calls WHERE scene=? AND call_id=?',key).fetchone()
            if row is None:raise ValueError('Missing native call')
            self.cache[key]=json.loads(row[0])
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]
    def metadata(self,fid):
        row=self.store.db.execute('SELECT scene,image,detection,frame,camera FROM features WHERE id=?',(fid,)).fetchone()
        if row is None:raise ValueError('Missing native physical feature')
        return row
    def actor(self,event_key):
        key=list(event_key)
        row=self.store.db.execute('SELECT scene,call_id,row_id FROM queries WHERE source=?',(json.dumps(key,separators=(',',':')),)).fetchone()
        if row is None:raise ValueError('Query not recorded by original solver')
        scene,cid,qi=row;c=self._call(scene,cid);qo=self.metadata(c['query_features'][qi]);hist=[[] for _ in c['candidate_ids']];owners={t:i for i,t in enumerate(c['candidate_ids'])}
        for fid,track in zip(c['reference_features'],c['reference_tracks']):
            ro=self.metadata(fid);lag=qo[3]-ro[3]
            if lag>0:hist[owners[track]].append(dict(camera=ro[4],lag=lag,feature=torch.from_numpy(self.store.feature(fid)),cross_view=ro[4]!=qo[4]))
        return dict(event_key=key,video=scene,frame=qo[3],camera=qo[4],query_feature=torch.from_numpy(self.store.feature(c['query_features'][qi])),
                    native_candidate_ids=list(c['candidate_ids']),native_scores=torch.from_numpy(dense_matrix(c['matrix_shape'],c['matrix_entries'])[qi].copy()),past_histories=hist)
    def keys(self):
        for (key,) in self.store.db.execute('SELECT source FROM queries ORDER BY scene,call_id,row_id'):yield tuple(json.loads(key))


def load_supervision(path,expected_sha,dataset):
    """Explicit offline API; UNKNOWN/impurity remains masked, aliases multi-positive."""
    path=Path(path).resolve()
    if path.is_relative_to(dataset.store.root.resolve()):raise ValueError('Supervision must be physically outside inference store')
    if sha(path)!=expected_sha:raise ValueError('Unpinned offline supervision')
    result={}
    with gzip.open(path,'rt') as f:
        for line in f:
            label=json.loads(line)
            if set(label)!=LABEL_FIELDS:raise ValueError('Unknown offline schema')
            key=tuple(label['event_key'])
            if key[0] not in FIT or key in result:raise ValueError('Duplicate/development/out of scope TRAIN label')
            actor=dataset.actor(key);people=label['candidate_persons'];person=label['GT_person']
            if len(people)!=len(actor['native_candidate_ids']):raise ValueError('Label-created candidate')
            if any(p is not None and (type(p)!=int or p<0) for p in people+[person]):raise ValueError('Invalid certified identity')
            positives=[i for i,p in enumerate(people) if person is not None and p==person]
            result[key]=dict(GT_person=person,certified_target_indices=positives,candidate_known_mask=[p is not None for p in people],actual_primary_state=label['actual_primary_state'])
    if set(result)!=set(dataset.keys()):raise ValueError('Full TRAIN label/query coverage incomplete')
    return result


def eligible_epoch(dataset,supervision,seed,epoch):
    """Coverage over actually eligible supervision, never a with-replacement schedule."""
    from cv_pil_v2_model import tensorize,training_eligible
    by_scene={}
    for key,label in supervision.items():
        if training_eligible(tensorize(dataset.actor(key)),label):by_scene.setdefault(key[0],[]).append(key)
    for key in epoch_order(by_scene,seed,epoch):yield key,dataset.actor(key),dict(supervision[key])
