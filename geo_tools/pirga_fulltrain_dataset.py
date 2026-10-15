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
        self.store=StoreReader(root,manifest_sha);self.input_roots=(Path(root).resolve(),);self.manifest_bindings={str(Path(root).resolve()):manifest_sha};self.cache=OrderedDict();self.cache_size=call_cache_size
        self.locations={tuple(json.loads(k)):(scene,cid,row) for k,scene,cid,row in self.store.db.execute('SELECT source,scene,call_id,row_id FROM queries')}
        self.feature_metadata={fid:(scene,image,detection,frame,camera) for fid,scene,image,detection,frame,camera in self.store.db.execute('SELECT id,scene,image,detection,frame,camera FROM features')}
        self.variant_features={fid for fid,variant in self.store.db.execute('SELECT id,variant FROM features') if variant is not None}
    def _call(self,scene,cid):
        key=(scene,cid)
        if key not in self.cache:
            row=self.store.db.execute('SELECT body FROM calls WHERE scene=? AND call_id=?',key).fetchone()
            if row is None:raise ValueError('Missing native call')
            self.cache[key]=json.loads(row[0])
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]
    def metadata(self,fid):
        row=self.feature_metadata.get(fid)
        if row is None:raise ValueError('Missing native physical feature')
        return row
    def actor(self,event_key):
        key=list(event_key)
        row=self.locations.get(tuple(key))
        if row is None:raise ValueError('Query not recorded by original solver')
        scene,cid,qi=row;c=self._call(scene,cid);qo=self.metadata(c['query_features'][qi]);hist=[[] for _ in c['candidate_ids']];owners={t:i for i,t in enumerate(c['candidate_ids'])}
        for fid,track in zip(c['reference_features'],c['reference_tracks']):
            ro=self.metadata(fid);lag=qo[3]-ro[3]
            if lag>0:hist[owners[track]].append(dict(camera=ro[4],lag=lag,feature=torch.from_numpy(self.store.feature(fid)),cross_view=ro[4]!=qo[4]))
        return dict(event_key=key,video=scene,frame=qo[3],camera=qo[4],query_feature=torch.from_numpy(self.store.feature(c['query_features'][qi])),
                    native_candidate_ids=list(c['candidate_ids']),native_scores=torch.from_numpy(dense_matrix(c['matrix_shape'],c['matrix_entries'])[qi].copy()),past_histories=hist)
    def uncertified_variant_columns(self,event_key):
        row=self.locations.get(tuple(event_key))
        c=self._call(row[0],row[1]);bad=set()
        for fid,track in zip(c['reference_features'],c['reference_tracks']):
            if fid in self.variant_features:bad.add(c['candidate_ids'].index(track))
        return bad
    def candidate_ids(self,key):
        scene,cid,_=self.locations[tuple(key)];return list(self._call(scene,cid)['candidate_ids'])
    def keys(self):
        yield from self.locations



class FullTrainDataset:
    def __init__(self,stores):
        if set(map(int,stores))!=FIT:raise ValueError('All11 fit stores required, DEV12 excluded')
        self.scenes={int(scene):NativeQueryDataset(spec['root'],spec['manifest_sha256']) for scene,spec in stores.items()}
        self.manifest_bindings={root:digest for ds in self.scenes.values() for root,digest in ds.manifest_bindings.items()}
        self.input_roots=tuple(root for ds in self.scenes.values() for root in ds.input_roots)
    def actor(self,key):return self.scenes[key[0]].actor(key)
    def uncertified_variant_columns(self,key):return self.scenes[key[0]].uncertified_variant_columns(key)
    def candidate_ids(self,key):return self.scenes[key[0]].candidate_ids(key)
    def keys(self):
        for scene in sorted(self.scenes):yield from self.scenes[scene].keys()


def load_supervision(path,expected_sha,dataset):
    """Explicit offline API; UNKNOWN/impurity remains masked, aliases multi-positive."""
    path=Path(path).resolve()
    if any(path.is_relative_to(root) for root in dataset.input_roots):raise ValueError('Supervision must be physically outside inference store')
    if sha(path)!=expected_sha:raise ValueError('Unpinned offline supervision')
    result={}
    with gzip.open(path,'rt') as f:
        for line in f:
            label=json.loads(line)
            if set(label)!=LABEL_FIELDS:raise ValueError('Unknown offline schema')
            key=tuple(label['event_key'])
            if key[0] not in FIT or key in result:raise ValueError('Duplicate/development/out of scope TRAIN label')
            candidates=dataset.candidate_ids(key);people=label['candidate_persons'];person=label['GT_person']
            if len(people)!=len(candidates):raise ValueError('Label-created candidate')
            if any(p is not None and (type(p)!=int or p<0) for p in people+[person]):raise ValueError('Invalid certified identity')
            if any(people[i] is not None for i in dataset.uncertified_variant_columns(key)):
                raise ValueError('Native averaged/version descriptor lacks separate support certificate; mask identity offline, retain inference candidate')
            positives=[i for i,p in enumerate(people) if person is not None and p==person]
            result[key]=dict(GT_person=person,certified_target_indices=positives,candidate_known_mask=[p is not None for p in people],actual_primary_state=label['actual_primary_state'])
    if set(result)!=set(dataset.keys()):raise ValueError('Full TRAIN label/query coverage incomplete')
    return result


def eligible_epoch(dataset,supervision,seed,epoch,admission_path,admission_sha):
    """Coverage over actually eligible supervision, never a with-replacement schedule."""
    from cv_pil_v2_model import tensorize,training_eligible
    if sha(admission_path)!=admission_sha:raise ValueError('Unpinned full training admission')
    admission=json.loads(Path(admission_path).read_text())
    if (admission.get('status')!='FULLTRAIN_DATA_CONTRACT_PASS' or set(admission['fit_scenes'])!=FIT
        or admission['source_store_manifests']!=dataset.manifest_bindings
        or set(k[0] for k in dataset.keys())!=FIT or len(supervision)!=154911):
        raise ValueError('Partial/engineering source cannot launch full training')
    eligibility=admission['ranking_eligibility'];ep=Path(eligibility['path'])
    if sha(ep)!=eligibility['sha256']:raise ValueError('Frozen ranking eligibility changed')
    keys=json.loads(ep.read_text());by_scene={}
    if len(keys)!=admission['counts']['ranking_eligible']:raise ValueError('Common supervised domain changed')
    for key in keys:
        key=tuple(key)
        if key not in supervision:raise ValueError('Eligibility outside complete TRAIN labels')
        by_scene.setdefault(key[0],[]).append(key)
    for key in epoch_order(by_scene,seed,epoch):yield key,dataset.actor(key),dict(supervision[key])
