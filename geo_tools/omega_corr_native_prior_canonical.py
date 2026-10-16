"""Preserve inherited canonicalNativePrior +2/cap2 in same native contexts, no NN calls."""
import collections
import gzip
import itertools
import json
import numpy as np
import torch
from omega_corr_common import ROOT,RUNTIME,sha256,save_new,provenance,DATA
from dev12_native_schema import activate
from omega_modules import geometry_module
PairScoreAdapter=geometry_module('pair_score_adapter').PairScoreAdapter


def main():
    source=provenance();assert not source['source_status'].strip()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((RUNTIME/'native_context_full_v1/manifest.json').read_text());assert m['status']=='MATCHED_NATIVE_CONTEXT_PARITY_PASS'
    assert sha256(m['output']['path'])==m['output']['sha256'] and sha256(m['proposal_cache']['path'])==m['proposal_cache']['sha256']
    cache=torch.load(m['proposal_cache']['path'],map_location='cpu',weights_only=True)
    from read_dev12_native_release import NativeRelease
    release=NativeRelease('/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1','ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4');assert release.labels is None
    data=json.loads((DATA/'annotations/train.json').read_text());meta={i['id']:i for i in data['images']};frames=collections.defaultdict(set)
    for i in data['images']:frames[i['video_id']].add(i['frame_id'])
    positions={scene:{f:i for i,f in enumerate(sorted(ff))} for scene,ff in frames.items()}
    valid_cache={}
    def valid(source):
        key=tuple(source)
        if key in valid_cache:return valid_cache[key]
        v=cache[source[0]]['fields']['reid_features'][source[1]] if source[0] in cache else release.observations[tuple(source)]['feature']
        result=bool(torch.isfinite(v).all() and v.norm()>0);valid_cache[key]=result;return result
    path=RUNTIME/'native_prior_canonical2_v1.jsonl.gz';calls=0;queries=0;maximum=0.
    with gzip.open(m['output']['path'],'rt') as f,gzip.open(path,'xt') as out:
        for line in f:
            row=json.loads(line);nq,nr=len(row['queries']),len(row['references']);raw=torch.tensor(row['raw_logits'],device='cuda',dtype=torch.float32).reshape(nq,nr)
            tracks=torch.tensor([r['track'] for r in row['references']],device='cuda',dtype=torch.long);unique=torch.tensor(row['unique_tracks'],device='cuda',dtype=torch.long)
            membership=(tracks[:,None]==unique[None,:]).float();blocks=[len(list(g)) for _,g in itertools.groupby(row['references'],key=lambda r:r['source'][0])]
            def scores(logits):return activate(logits,blocks)@membership if blocks else logits.new_zeros((nq,len(unique)))
            mask_np=np.zeros((nq,nr),bool)
            for i,q in enumerate(row['queries']):
                qm=meta[q[0]]
                for j,r in enumerate(row['references']):
                    rm=meta[r['source'][0]]
                    mask_np[i,j]=qm['view_id']!=rm['view_id'] and abs(positions[qm['video_id']][qm['frame_id']]-positions[rm['video_id']][rm['frame_id']])<=20 and valid(q) and valid(r['source'])
            mask=raw.new_tensor(mask_np,dtype=torch.bool)
            # Control reconstruction must also reproduce both captured bounded
            # amplitudes; canonical2 is not a newly tuned weaker baseline.
            for weight in [.1,.3]:
                changed=PairScoreAdapter(enabled=True,feature_weight=weight,max_residual=1.).apply(raw,mask.float(),None,mask)
                actual=scores(changed);expected=actual.new_tensor(row['NativePrior'][str(weight)]['track_scores']).reshape(actual.shape)
                error=float((actual-expected).abs().max()) if actual.numel() else 0.;maximum=max(maximum,error);assert error<=2e-6
            changed=PairScoreAdapter(enabled=True,feature_weight=2.,max_residual=2.).apply(raw,mask.float(),None,mask);sc=scores(changed)
            rank=unique[torch.argsort(sc,dim=1,descending=True,stable=True)].cpu().tolist()
            out.write(json.dumps(dict(pair_id=row['pair_id'],queries=row['queries'],unique_tracks=row['unique_tracks'],track_scores=sc.cpu().tolist(),ranked_tracks=rank,feature_weight=2.,max_residual=2.,age_observed_groups=20),allow_nan=False,separators=(',',':'))+'\n')
            calls+=1;queries+=nq
    save_new(RUNTIME/'native_prior_canonical2_v1.json',dict(status='CANONICAL_NATIVE_PRIOR_SAME_CONTEXT_READY',source=source,calls=calls,queries=queries,
        canonical_parameter_source=dict(path='manifests/native_prior_main_control_v1.json',sha256=sha256(ROOT/'manifests/native_prior_main_control_v1.json')),
        feature_weight=2.,max_residual=2.,age=20,extra_amplitudes_point1_point3_are_not_canonical=True,
        source_implementation_sha256=sha256(ROOT/'gtr/geometry/native_prior.py'),max_captured_amplitude_reconstruction_error=maximum,
        no_model_forward_calls=True,no_GT_input=True,no_tracking_replay=True,scope='frozen sameOFFnativehistory row-control, not propagated NativePrior fullvideo baseline',
        CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),output=dict(path=str(path),sha256=sha256(path))))
    print('PASS canonical prior2',calls,queries,'amplitude parity error',maximum,flush=True)

if __name__=='__main__':main()
