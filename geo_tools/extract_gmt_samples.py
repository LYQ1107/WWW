"""Fresh GMT detector and proposal evidence on fixed TRAIN development timestamps."""
import argparse, collections, time, numpy as np, torch
from scipy.optimize import linear_sum_assignment
from common import *

def match_gt(boxes, anns):
    if not len(boxes) or not anns: return np.full(len(boxes),-1,dtype=np.int64)
    gt=np.array([a['bbox'] for a in anns],dtype=float); gt[:,2:]+=gt[:,:2]
    b=boxes.astype(float); lo=np.maximum(b[:,None,:2],gt[None,:,:2]); hi=np.minimum(b[:,None,2:],gt[None,:,2:]); inter=np.clip(hi-lo,0,None).prod(-1)
    area=np.clip(b[:,2:]-b[:,:2],0,None).prod(-1); ga=np.clip(gt[:,2:]-gt[:,:2],0,None).prod(-1)
    iou=inter/np.maximum(area[:,None]+ga[None,:]-inter,1e-12)
    ii,jj=linear_sum_assignment(-iou); labels=np.full(len(b),-1,dtype=np.int64)
    for i,j in zip(ii,jj):
        if iou[i,j]>=.5: labels[i]=anns[j]['instance_id']
    return labels

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs-per-scene',type=int,default=10);parser.add_argument('--tag',default='screen_v1');parser.add_argument('--split',choices=['dev','train'],default='dev');args=parser.parse_args()
    out=RUNTIME/'samples'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg,model=load_gmt(); data=train_data(); anns=collections.defaultdict(list)
    for a in data['annotations']:anns[a['image_id']].append(a)
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text()); samples=[]; source=provenance()
    from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
    source['split']=args.split
    source['split_manifest_sha256']=sha256(ROOT/'manifests/geo_split_v1.json')
    video_ids=split['minimal_development_video_ids'] if args.split=='dev' else [5,8,12]
    for vid in video_ids:
        records=video_records(data,vid); frames=sorted({x['frame_id'] for x in records})
        chosen=np.linspace(0,len(frames)-1,args.pairs_per_scene,dtype=int)
        for frame in [frames[k] for k in chosen]:
            seed_all(20261014); pair=[r for r in records if r['frame_id']==frame]; batch=map_records(cfg,pair); inst=[]
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.no_grad():
                for v,r in enumerate(batch):inst.extend(model.inference([r],len(pair),[frame,frame,frame],v,do_postprocess=False))
                inst[0].track_ids=torch.arange(1,len(inst[0])+1,device=inst[0].reid_features.device)
                asso,_,_,Np,_=model.get_asso(inst,k=1)
                gmt_logits=asso[0][:,:Np].cpu().numpy()
            torch.cuda.synchronize(); dt=time.perf_counter()-start
            processed=CustomRCNN._postprocess(inst,pair,[(0,0)]*len(pair),not_clamp_box=cfg.INPUT.NOT_CLAMP_BOX)
            arrays={'gmt_logits':gmt_logits}; views=[]
            for v,(r,p) in enumerate(zip(pair,processed)):
                i=p['instances']; boxes=i.pred_boxes.tensor.cpu().numpy();labels=match_gt(boxes,anns[r['image_id']])
                arrays.update({f'boxes_{v}':boxes,f'labels_{v}':labels,f'gmt_{v}':i.reid_features.cpu().numpy(),f'scores_{v}':i.scores.cpu().numpy()})
                views.append(dict(image_id=r['image_id'],view_id=r['view_id'],path=r['file_name'],width=r['width'],height=r['height'],detections=len(boxes),matched_gt=int((labels>0).sum()),gt_count=len(anns[r['image_id']])))
            path=out/f'v{vid:02d}_f{frame:06d}.npz';np.savez_compressed(path,**arrays)
            sample=dict(video_id=vid,frame_id=frame,views=views,path=str(path),sha256=sha256(path),detector_and_gmt_seconds=dt)
            samples.append(sample);save_json(out/'samples.json',dict(provenance=source,samples=samples));print(json.dumps(sample),flush=True)
    print('DONE',len(samples),flush=True)
if __name__=='__main__':main()
