"""GT-free exact joint opportunity costs against fixed C1 other query rows.

For a current row and legal column, solve the full remaining assignment. All
other rows/columns stay present. These are training input scalars, not labels.
No claim is made about future Bank/Gallery states or fully learned other rows.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from pirga_indexed_store import sha
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset
from pirga_candidate_state import CandidateStates


def opportunity(scores,row):
    a=np.asarray(scores,dtype=np.float64)
    if a.ndim!=2 or not 0<=row<len(a) or not np.isfinite(a).all():raise ValueError('Full finite native matrix required')
    other=np.delete(a,row,axis=0)
    def solve(x):
        rr,cc=linear_sum_assignment(-x);return float(x[rr,cc].sum())
    best=solve(other);values=np.asarray([solve(np.delete(other,col,axis=1))-best for col in range(a.shape[1])],dtype=np.float32)
    return torch.from_numpy(values)


def prepare(protocol_path,protocol_sha,out):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Unpinned context preparation')
    if out.exists():raise FileExistsError(out)
    p=json.loads(protocol_path.read_text());torch.set_num_threads(1);start=time.monotonic()
    repo=Path(__file__).resolve().parents[1]
    for name,digest in p['source_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Context source changed')
    if sha(p['admission']['path'])!=p['admission']['sha256']:raise ValueError('Full admission changed')
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);states=CandidateStates(cache,p['horizon_boundaries'],1)
    ad=json.loads(Path(p['admission']['path']).read_text());ds=FullTrainDataset(ad['stores']);packed=[]
    for ci,c in enumerate(cache.calls):
        original=ds.scenes[c['scene']]._call(c['scene'],c['call_id']);threshold=torch.tensor(original['thresholds'],dtype=torch.float32)
        st=states.state(ci);q=F.normalize(cache.features[c['scene']][c['query_features']].float(),dim=-1)
        base=c['native_scores']+2*(q@st.mean.T)
        ids=original['candidate_ids'];accepted=original['accepted_ids']
        accepted_columns=torch.tensor([ids.index(i) if i>=0 else -1 for i in accepted],dtype=torch.long)
        opp=torch.stack([opportunity(base.numpy(),row) for row in range(len(c['keys']))])
        packed.append(dict(scene=c['scene'],call_id=c['call_id'],keys=c['keys'],thresholds=threshold,original_accepted_columns=accepted_columns,opportunity=opp))
        if ci%2000==0:print(json.dumps(dict(prepared_calls=ci,GT_read=False)),flush=True)
    out.mkdir(parents=True);target=out/'context.pt';torch.save(packed,target)
    manifest=dict(status='COMPLETE_GT_FREE_FIXED_C1_JOINT_CONTEXT',calls=len(packed),queries=sum(len(c['keys']) for c in packed),
        data_path=str(target.resolve()),data_sha256=sha(target),cache_sha256=p['cache']['sha256'],admission_sha256=p['admission']['sha256'],
        GT_in_cache=False,physical_vectors_duplicated=False,wall_seconds=time.monotonic()-start,GPU_wall_hours=0,protocol_sha256=protocol_sha,
        qualification='Exact fixed-C1 other-row assignment opportunity, not full learned-matrix differentiation or online state propagation.')
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.protocol,a.protocol_sha,a.out)
