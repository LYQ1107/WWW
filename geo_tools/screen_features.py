"""Compact frozen target features on the exact same detected candidates as GMT."""
import argparse,collections,csv,json,time,numpy as np,torch
from common import *
from feature_metrics import normalize,evaluate_scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=RUNTIME/'samples/screen_v1r2/samples.json');p.add_argument('--weights',type=Path,default=RUNTIME/'weights/vggt_1b.pt');p.add_argument('--tag',default='feature_v1');p.add_argument('--dim',type=int,default=256);p.add_argument('--limit',type=int);a=p.parse_args()
    sys.path.insert(0,str(RUNTIME/'sources/vggt'))
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.roi_feature_extractor import PadMapping
    from gtr.geometry.vggt_adapter import FrozenVGGT
    source=json.loads(a.samples.read_text());samples=source['samples'][:a.limit] if a.limit else source['samples']
    out=RUNTIME/'features'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True); seed_all(20261014);adapter=FrozenVGGT(a.weights);matrix_items=collections.defaultdict(list); diagnostics=[]
    model_sha=sha256(a.weights);source_provenance=provenance(); vgsha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME/'sources/vggt',text=True).strip()
    for sample in samples:
        arrays=np.load(sample['path']);paths=[v['path'] for v in sample['views']];boxes=[torch.from_numpy(arrays[f'boxes_{v}']) for v in range(2)]
        mapped=[PadMapping(v['width'],v['height']).boxes(b) for v,b in zip(sample['views'],boxes)]
        start=time.perf_counter();images=load_and_preprocess_images(paths,mode='pad');read=time.perf_counter()-start
        torch.cuda.synchronize();start=time.perf_counter();result,cov,info=adapter.extract(images,mapped,dim=a.dim);torch.cuda.synchronize();joint_time=time.perf_counter()-start
        # Reversed joint pair uses identical image content, then restores view IDs.
        torch.cuda.synchronize();start=time.perf_counter();rev,_,_=adapter.extract(images.flip(0),mapped[::-1],dim=a.dim);torch.cuda.synchronize();rev_time=time.perf_counter()-start
        singles=[];torch.cuda.synchronize();start=time.perf_counter()
        for v in range(2): singles.append(adapter.extract(images[v:v+1],[mapped[v]],dim=a.dim)[0])
        torch.cuda.synchronize();single_time=time.perf_counter()-start
        features={}; sample_items={}
        def add(key,s):
            entry=dict(score=s,ids1=arrays['labels_0'],ids2=arrays['labels_1'],base=arrays['gmt_logits']);matrix_items[key].append(entry);sample_items[key]=s
        gmt0=normalize(arrays['gmt_0']);gmt1=normalize(arrays['gmt_1']);add('gmt_reid_cosine',gmt1@gmt0.T);add('gmt_proposal_logits',arrays['gmt_logits'])
        for key,vs in result.items():
            v0,v1=[v.numpy() for v in vs];r0,r1=[v.numpy() for v in rev[key][::-1]]
            s0,s1=[singles[v][key][0].numpy() for v in range(2)]
            add(f'joint/{key}',v1@v0.T);add(f'single/{key}',s1@s0.T);add(f'symmetric_joint/{key}',(v1@v0.T+r1@r0.T)/2)
            features[f'{key}/0']=v0.astype(np.float16);features[f'{key}/1']=v1.astype(np.float16)
        for mode in ['mean','center','roi_align']:
            sims=[sample_items[f'joint/layer_{l}/{mode}'] for l in [4,11,17,23]]
            add(f'joint/multilayer/{mode}',np.mean(sims,axis=0))
        artifact=out/Path(sample['path']).name;np.savez_compressed(artifact,**features)
        row=dict(video_id=sample['video_id'],frame_id=sample['frame_id'],read_seconds=read,joint_seconds=joint_time,reversed_seconds=rev_time,separate_seconds=single_time,
            patch_start_idx=info['patch_start_idx'],layer_shapes=info['layer_shapes'],detections=sum(len(b) for b in boxes),zero_patch_targets=sum(int((v==0).sum()) for v in cov.values()),
            under_one_patch_area=sum(int((((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))<196).sum()) for b in mapped),
            camera_swap_max_score_change=max(float(np.max(np.abs(2*(sample_items[f'symmetric_joint/{k}']-sample_items[f'joint/{k}'])))) if sample_items[f'joint/{k}'].size else 0 for k in result),
            artifact=str(artifact),artifact_sha256=sha256(artifact),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        diagnostics.append(row);save_json(out/'diagnostics.json',diagnostics);print(json.dumps(row),flush=True)
    rows=[dict(setting=k,**evaluate_scores(v)) for k,v in matrix_items.items()]
    with (ROOT/'results/feature_screening.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    save_json(ROOT/'results/feature_screening_manifest.json',dict(provenance=source_provenance,model_sha256=model_sha,vggt_source_sha=vgsha,sample_manifest_sha256=sha256(a.samples),
        dim=a.dim,preprocess='official pad RGB [0,1], 518, internal normalization',parameters=adapter.parameters,cache='target ROI vectors only',
        independent_dinov2_control=False,gmt_baseline='actual first-pair candidate logits on fixed detection candidates',metrics_scope='TRAIN development only; feature retrieval is NOT tracking accuracy',diagnostics=str(out/'diagnostics.json'),rows=len(rows)))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
