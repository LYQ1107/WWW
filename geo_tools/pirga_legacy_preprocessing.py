"""Faithful historical Top8/latest4 inputs for full-data coverage attribution.

Candidate selection is stable original native-score Top8. Latest4 sorting is
(frame,camera,physical_source), BEFORE excluding same-frame histories, exactly
as frozen selected-V3 capture and normalize_event. No GT selects inputs.
"""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from cv_pil_v2_model import ActorBatch,training_eligible
from cv_pil_native_handoff import safe_torch_load
from pirga_indexed_store import sha
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision


class LegacyTrainingCache:
    def __init__(self,manifest_path,expected_sha):
        path=Path(manifest_path)
        if sha(path)!=expected_sha:raise ValueError('Legacy input manifest changed')
        self.manifest=json.loads(path.read_text());m=self.manifest
        if m['GT_in_cache'] is not False or sha(m['data_path'])!=m['data_sha256']:raise ValueError('GT-containing or changed cache')
        self.base=NativeTrainingCache(m['full_cache']['path'],m['full_cache']['sha256'])
        self.calls=safe_torch_load(m['data_path']);self.locations={}
        for ci,c in enumerate(self.calls):
            for row,key in enumerate(c['keys']):self.locations[tuple(key)]=(ci,row)
        if set(self.locations)!=set(self.base.locations):raise ValueError('Complete physical query coverage changed')
        self.base.calls=[]
    def columns(self,key):
        ci,row=self.locations[tuple(key)];return self.calls[ci]['top8'][row].tolist()
    def label(self,key,label):
        columns=self.columns(key);positive=set(label['certified_target_indices'])
        return dict(GT_person=label['GT_person'],actual_primary_state=label['actual_primary_state'],
            certified_target_indices=[i for i,c in enumerate(columns) if c in positive],candidate_known_mask=[label['candidate_known_mask'][c] for c in columns])
    def batch(self,key):
        ci,row=self.locations[tuple(key)];c=self.calls[ci];features=self.base.features[c['scene']];positions=c['positions'][row];positions=positions[positions>=0]
        ids=c['histories'][positions];owners=c['owners8'][row][:len(positions)];q=F.normalize(features[c['query_features'][row]].float(),dim=0)
        h=F.normalize(features.index_select(0,ids).float(),dim=-1)
        return ActorBatch(q,h,owners,c['age'][positions],c['cross'][positions],c['native_scores'][row,c['top8'][row]].clone(),c['camera'])


def prepare(protocol_path,protocol_sha,out):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Unpinned source preparation')
    if out.exists():raise FileExistsError(out)
    p=json.loads(protocol_path.read_text());torch.set_num_threads(1);start=time.monotonic()
    if sha(p['admission']['path'])!=p['admission']['sha256']:raise ValueError('Full admission changed')
    ad=json.loads(Path(p['admission']['path']).read_text());ds=FullTrainDataset(ad['stores']);native=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);packed=[]
    for ci,c in enumerate(native.calls):
        scene=c['scene'];original=ds.scenes[scene]._call(scene,c['call_id']);metadata=ds.scenes[scene].feature_metadata
        frame=metadata[c['query_features'][0]][3];by={t:[] for t in original['candidate_ids']}
        for fid,t in zip(original['reference_features'],original['reference_tracks']):by[t].append(fid)
        ids=[];age=[];cross=[];offsets=[0]
        for t in original['candidate_ids']:
            ordered=sorted(by[t],key=lambda f:(metadata[f][3],metadata[f][4],(metadata[f][1],metadata[f][2])))[-4:]
            if len({(metadata[f][1],metadata[f][2]) for f in ordered})!=len(ordered):raise ValueError('Historical selected-V3 schema cannot represent duplicate physical history')
            for fid in ordered:
                lag=frame-metadata[fid][3]
                if lag<0:raise ValueError('Future native history')
                if lag==0:continue
                ids.append(fid);age.append(min(1.,lag/400.));cross.append(metadata[fid][4]!=c['camera'])
            offsets.append(len(ids))
        top=torch.argsort(c['native_scores'],dim=1,descending=True,stable=True)[:,:8]
        positions=[];owners=[]
        for cols in top.tolist():
            positions.append([j for col in cols for j in range(offsets[col],offsets[col+1])]);owners.append([i for i,col in enumerate(cols) for _ in range(offsets[col+1]-offsets[col])])
        width=max(map(len,positions),default=0);pos=torch.full((len(top),width),-1,dtype=torch.long);own=torch.full_like(pos,-1)
        for i,(pp,oo) in enumerate(zip(positions,owners)):
            pos[i,:len(pp)]=torch.tensor(pp,dtype=torch.long);own[i,:len(oo)]=torch.tensor(oo,dtype=torch.long)
        packed.append(dict(scene=scene,call_id=c['call_id'],keys=c['keys'],query_features=c['query_features'],camera=c['camera'],
            histories=torch.tensor(ids,dtype=torch.long),age=torch.tensor(age,dtype=torch.float32),cross=torch.tensor(cross,dtype=torch.bool),
            positions=pos,owners8=own,top8=top,native_scores=c['native_scores']))
        if ci%3000==0:print(json.dumps(dict(prepared_calls=ci,GT_read=False)),flush=True)
    inputs=out/'indexed_inputs';inputs.mkdir(parents=True);data=inputs/'calls.pt';torch.save(packed,data)
    manifest=dict(status='COMPLETE_OLD_TOP8_LATEST4_INPUTS_PENDING_PARITY',GT_in_cache=False,queries=len(native.locations),calls=len(packed),
        full_cache=p['cache'],data_path=str(data.resolve()),data_sha256=sha(data),physical_vectors_duplicated_on_disk=False,
        selection='Stable native Top8 columns; sort actual history(frame,camera,physical_source),take latest4 BEFORE strict frame-past filtering; all1152 dims and original FP16 representation')
    mp=inputs/'MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    cache=LegacyTrainingCache(mp,sha(mp));labels=load_supervision(ad['supervision']['path'],ad['supervision']['sha256'],ds);eligible=[];predict=0;persons=set();difficulty={}
    for key in cache.locations:
        label=cache.label(key,labels[key]);positive=set(label['certified_target_indices']);ci,row=cache.locations[key];c=cache.calls[ci];positions=c['positions'][row];positions=positions[positions>=0];owners=c['owners8'][row][:len(positions)]
        valid=bool(positive and len(positions) and any(v and i not in positive for i,v in enumerate(label['candidate_known_mask'])))
        if valid:
            eligible.append(key);persons.add((key[0],label['GT_person']));difficulty[label['actual_primary_state']]=difficulty.get(label['actual_primary_state'],0)+1
            predict+=any(bool(c['cross'][j]) and int(owner) in positive for j,owner in zip(positions.tolist(),owners.tolist()))
    offline=out/'offline_supervision';offline.mkdir();ep=offline/'ranking_eligible_keys.json';ep.write_text(json.dumps(sorted(eligible),separators=(',',':'))+'\n')
    result=dict(status='COMPLETE_HISTORICAL_PREPROCESSING_PENDING_OLD3000_PARITY',input_manifest=dict(path=str(mp.resolve()),sha256=sha(mp)),
        eligibility=dict(path=str(ep.resolve()),sha256=sha(ep)),queries=len(cache.locations),ranking_eligible=len(eligible),prediction_eligible_within_rank=predict,person_scene_groups=len(persons),difficulty_counts=difficulty,
        original_supervision=ad['supervision'],full_admission=p['admission'],wall_seconds=time.monotonic()-start,GPU_wall_hours=0,DEV_read=False,training_updates=0,
        reason='Necessary coverage/evidence-scope attribution control. Existing full-native fits retained; this reconstruction preserves old Top8/latest4 without re-exporting video or GT matching.')
    (out/'ADMISSION_PENDING_PARITY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.protocol,a.protocol_sha,a.out)
