"""Independent public DINOv2 ViT-L/14 register weights; target pooling matches VGGT."""
import argparse,collections,csv,time,numpy as np,torch
from common import *
from feature_metrics import normalize,evaluate_scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=RUNTIME/'samples/screen_clean_dev_v2/samples.json');p.add_argument('--tag',default='dino_clean_dev_v1');a=p.parse_args()
    sys.path[:0]=[str(RUNTIME/'sources/dinov2'),str(RUNTIME/'sources/vggt')]
    from dinov2.hub.backbones import dinov2_vitl14_reg
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.roi_feature_extractor import PadMapping,patch_map,pool_roi,compress
    path=RUNTIME/'weights/dinov2_vitl14_reg4.pth';model=dinov2_vitl14_reg(pretrained=False)
    model.load_state_dict(torch.load(path,map_location='cpu'),strict=True);model=model.cuda().eval().requires_grad_(False)
    data=json.loads(a.samples.read_text());out=RUNTIME/'features'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);metrics=collections.defaultdict(list);diagnostics=[]
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for sample in data['samples']:
        arrays=np.load(sample['path']);images=load_and_preprocess_images([v['path'] for v in sample['views']],mode='pad').cuda()
        mapped=[PadMapping(v['width'],v['height']).boxes(torch.from_numpy(arrays[f'boxes_{i}']).cuda()) for i,v in enumerate(sample['views'])]
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):tokens=model.forward_features((images-mean)/std)['x_norm_patchtokens']
        torch.cuda.synchronize();dt=time.perf_counter()-t;feature={}
        for mode in ['mean','center','roi_align']:
            vs=[]
            for v in range(2):
                fm=patch_map(tokens[v],0,(518,518));pooled,count=pool_roi(fm,mapped[v],mode);vs.append(compress(pooled).cpu().numpy())
                feature[f'{mode}/{v}']=vs[-1].astype(np.float16)
            score=vs[1]@vs[0].T
            metrics[mode].append(dict(score=score,ids1=arrays['labels_0'],ids2=arrays['labels_1'],base=arrays['gmt_logits']))
        target=out/Path(sample['path']).name;np.savez_compressed(target,**feature);diagnostics.append(dict(video_id=sample['video_id'],frame_id=sample['frame_id'],inference_seconds=dt,artifact=str(target),sha256=sha256(target)))
        print(json.dumps(diagnostics[-1]),flush=True)
    rows=[dict(setting='independent_dinov2/'+k,**evaluate_scores(v)) for k,v in metrics.items()]
    with (ROOT/'results/dinov2_screening.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_json(ROOT/'results/dinov2_screening_manifest.json',dict(weights_sha256=sha256(path),sample_manifest_sha256=sha256(a.samples),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME/'sources/dinov2',text=True).strip(),diagnostics=diagnostics,independent_dinov2_control=True,parameters=sum(p.numel() for p in model.parameters()),used_dense_cache=False))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
