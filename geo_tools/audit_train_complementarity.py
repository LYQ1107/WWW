"""Explicit TRAIN-only potential gate relative to both GMT score sources."""
import argparse,numpy as np
from common import *

def normalized(array):
    x=array.astype(float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True);args=parser.parse_args()
    source=json.loads(args.samples.read_text());split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if source['provenance'].get('split')!='train':raise ValueError('TRAIN-only audit')
    rows=[];totals=dict(queries=0,gmt_reid_errors=0,vg_fix_gmt_reid=0,vg_harm_gmt_reid=0)
    artifact_hashes={}
    for sample in source['samples']:
        if sample['video_id'] not in split['association_train_video_ids']:raise ValueError('non-TRAIN video')
        path=Path(sample['path']);feature_path=args.features/path.name
        arr=np.load(path);feature=np.load(feature_path);artifact_hashes[str(path)]=sha256(path);artifact_hashes[str(feature_path)]=sha256(feature_path)
        keep0=arr['labels_0']>0;keep1=arr['labels_1']>0;ids0=arr['labels_0'][keep0];ids1=arr['labels_1'][keep1]
        gmt=normalized(arr['gmt_1'][keep1])@normalized(arr['gmt_0'][keep0]).T
        base=arr['gmt_logits'][np.ix_(keep1,keep0)].astype(float)
        key='layer_17/roi_align'
        originals=[feature[key+f'/{v}'][keep0 if v==0 else keep1] for v in range(2)]
        reversed_vectors=[feature['reverse_'+key+f'/{v}'][keep0 if v==0 else keep1] for v in range(2)]
        x0,x1=[x.astype(float) for x in originals];r0,r1=[x.astype(float) for x in reversed_vectors]
        vg=(x1@x0.T+r1@r0.T)/2
        for q,identity in enumerate(ids1):
            positive=ids0==identity;negative=~positive
            if not positive.any() or not negative.any():continue
            totals['queries']+=1
            gmt_ok=bool(positive[gmt[q].argmax()]);vg_ok=bool(positive[vg[q].argmax()]);base_ok=bool(positive[base[q].argmax()])
            totals['gmt_reid_errors']+=int(not gmt_ok)
            totals['vg_fix_gmt_reid']+=int(vg_ok and not gmt_ok)
            totals['vg_harm_gmt_reid']+=int(gmt_ok and not vg_ok)
            if vg_ok and not gmt_ok:
                # Conservative full-ULP bound on FP16 storage rounding, including
                # subnormals. It does not bound encoder arithmetic variability.
                def error(v):
                    eps=np.finfo(np.float16).eps;sub=float(np.nextafter(np.float16(0),np.float16(1)))
                    return (eps*np.linalg.norm(v,axis=-1)+v.shape[-1]**.5*sub)/(1-eps)
                def dot_error(a,b):
                    ea,eb=error(a),error(b)
                    return ea*np.linalg.norm(b,axis=-1)+eb*np.linalg.norm(a,axis=-1)+ea*eb
                bound=(dot_error(x1[q],x0)+dot_error(r1[q],r0))/2
                positive_index=np.flatnonzero(positive)[vg[q,positive].argmax()]
                bound_gap=float(bound[positive_index]+bound[negative].max())
                gap=float(vg[q,positive].max()-vg[q,negative].max())
                rows.append(dict(video_id=sample['video_id'],frame_id=sample['frame_id'],query_gt_identity=int(identity),gmt_reid_correct=gmt_ok,gmt_proposal_correct=base_ok,vggt_correct=vg_ok,
                    gmt_reid_positive_minus_negative=float(gmt[q,positive].max()-gmt[q,negative].max()),gmt_proposal_positive_minus_negative=float(base[q,positive].max()-base[q,negative].max()),
                    vggt_positive_minus_negative=gap,fp16_storage_margin_error_bound=bound_gap,storage_rounding_cannot_reverse_rank=gap>bound_gap,
                    candidate_gt_identities=ids0.tolist(),sample=str(path),feature_artifact=str(feature_path)))
    report=dict(status='LIMITED_TRAIN_PAIR_POTENTIAL' if rows else 'NO_TRAIN_COMPLEMENTARITY',scope='fixed TRAIN same-time opposite-camera GT-eligible proposal retrieval; no claim of tracking gain, novel geometry or DEV generalization',
        feature='symmetric native VGGT layer17 ROI',totals=totals,corrections=rows,sample_manifest_sha256=sha256(args.samples),artifact_sha256=artifact_hashes,
        limitations=['only one GMT failure in this small TRAIN sample; many VGGT harms','weak head-prototype gate only, not a robust scientific-success gate','FP16 bound concerns storage quantization only'],provenance=provenance())
    save_json(ROOT/'results/train_complementarity_native_v1.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
