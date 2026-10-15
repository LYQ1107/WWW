"""Fast native batch assembly without duplicating physical vectors on disk.

Only causal reference indices, original scores and time/camera tensors are
cached. The immutable native FP16 vector files are reused by copy-on-write
memory maps; outputs are fresh normalized tensors. Candidate-major history
order matches the original tensorize() loss and floating-point reduction order.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from cv_pil_native_handoff import safe_torch_load
from cv_pil_v2_model import ActorBatch
from pirga_indexed_store import StoreReader,sha,dense_matrix


class NativeTrainingCache:
    def __init__(self,manifest_path,expected_sha,device='cpu'):
        manifest_path=Path(manifest_path)
        if sha(manifest_path)!=expected_sha:raise ValueError('Unpinned training input cache')
        self.manifest=json.loads(manifest_path.read_text());m=self.manifest
        if m['GT_in_cache'] is not False:raise ValueError('GT must remain outside actor cache')
        if sha(m['calls_path'])!=m['calls_sha256']:raise ValueError('Native call cache changed')
        self.calls=safe_torch_load(m['calls_path']);self.locations={};self.features={};self.device=torch.device(device)
        for scene,spec in m['stores'].items():
            scene=int(scene);root=Path(spec['root'])
            if sha(root/'manifest.json')!=spec['manifest_sha256']:raise ValueError('Native source manifest changed')
            source=json.loads((root/'manifest.json').read_text())
            if sha(root/'native_vectors.bin')!=source['artifacts']['native_vectors.bin']:raise ValueError('Native source vectors changed')
            # Private mapping protects the immutable file even from an accidental
            # in-place operation. No new vector archive is written.
            array=np.memmap(root/'native_vectors.bin',mode='c',dtype=source['stored_dtype'],shape=(source['features'],1152))
            self.features[scene]=torch.from_numpy(array).to(self.device)
        for ci,c in enumerate(self.calls):
            for row,key in enumerate(c['keys']):self.locations[tuple(key)]=(ci,row)
            for name in ('histories','owners','age','cross','native_scores'):
                c[name]=c[name].to(self.device)

    def batch(self,key):
        ci,row=self.locations[tuple(key)];c=self.calls[ci];features=self.features[c['scene']]
        q=F.normalize(features[c['query_features'][row]].float(),dim=0)
        h=F.normalize(features.index_select(0,c['histories']).float(),dim=-1)
        return ActorBatch(q,h,c['owners'],c['age'],c['cross'],c['native_scores'][row].clone(),c['camera'])


def prepare(admission_path,admission_sha,out):
    started=time.monotonic();torch.set_num_threads(1)
    if sha(admission_path)!=admission_sha:raise ValueError('Unpinned full input admission')
    a=json.loads(Path(admission_path).read_text())
    if a['status']!='FULLTRAIN_DATA_CONTRACT_PASS':raise ValueError('Full training contract required')
    out.mkdir(parents=True,exist_ok=False);packed=[];queries=0
    for scene,spec in a['stores'].items():
        scene=int(scene);r=StoreReader(spec['root'],spec['manifest_sha256'])
        meta={fid:(frame,camera) for fid,frame,camera in r.db.execute('SELECT id,frame,camera FROM features')}
        for c in r.calls():
            if c['stage']=='run_memory_tracker' or not c['query_keys']:continue
            frames={meta[f][0] for f in c['query_features']};cameras={meta[f][1] for f in c['query_features']}
            if len(frames)!=1 or len(cameras)!=1:raise ValueError('Heterogeneous actual query image')
            frame=next(iter(frames));camera=next(iter(cameras));by={t:[] for t in c['candidate_ids']}
            for fid,t in zip(c['reference_features'],c['reference_tracks']):
                if meta[fid][0]<frame:by[t].append(fid)
            hids=[];owners=[];age=[];cross=[]
            for owner,t in enumerate(c['candidate_ids']):
                for fid in by[t]:hids.append(fid);owners.append(owner);age.append(min(1.,(frame-meta[fid][0])/400.));cross.append(meta[fid][1]!=camera)
            packed.append(dict(scene=scene,call_id=c['call_id'],keys=c['query_keys'],query_features=c['query_features'],camera=camera,
                               histories=torch.tensor(hids,dtype=torch.long),owners=torch.tensor(owners,dtype=torch.long),age=torch.tensor(age,dtype=torch.float32),
                               cross=torch.tensor(cross,dtype=torch.bool),native_scores=torch.from_numpy(dense_matrix(c['matrix_shape'],c['matrix_entries']))))
            queries+=len(c['query_keys'])
        print(json.dumps(dict(prepared_scene=scene,calls=len(packed),queries=queries)),flush=True)
    if queries!=154911:raise ValueError('Missing full actual queries from cache')
    p=out/'native_calls.pt';torch.save(packed,p)
    manifest=dict(status='PREPARED_INPUT_ONLY_NOT_TRAINED',GT_in_cache=False,source_admission_sha256=admission_sha,stores=a['stores'],calls_path=str(p.resolve()),calls_sha256=sha(p),
                  calls=len(packed),queries=queries,physical_vectors_duplicated_on_disk=False,history_order='candidate-major then original native order, strict scene-past, inherited C2/C3/C4 preprocessing',
                  native_features_dtype='original storedFP16 convertedFP32 and normalized by original tensorize operations',wall_seconds=time.monotonic()-started,GPU_wall_hours=0)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--admission-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.admission,a.admission_sha,a.out)
