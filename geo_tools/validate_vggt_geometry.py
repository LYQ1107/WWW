"""Joint VGGT camera geometry validated by independent masked background matches."""
import argparse,csv,time,numpy as np,torch,cv2
from common import *
from gtr.geometry.geometry_alignment import fundamental_from_cameras,symmetric_epipolar_distance,soft_box_evidence
from gtr.geometry.geometry_confidence import estimate_background
from gtr.geometry.roi_feature_extractor import PadMapping

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=RUNTIME/'samples/screen_clean_dev_v2/samples.json');p.add_argument('--tag',default='vggt_geometry_v1');p.add_argument('--limit',type=int);a=p.parse_args()
    sys.path.insert(0,str(RUNTIME/'sources/vggt'))
    from gtr.geometry.vggt_adapter import FrozenVGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    source=json.loads(a.samples.read_text());adapter=FrozenVGGT(RUNTIME/'weights/vggt_1b.pt');rows=[];cv2.setNumThreads(4)
    for sample in source['samples'][:a.limit] if a.limit else source['samples']:
        arr=np.load(sample['path']);paths=[v['path'] for v in sample['views']];boxes=[torch.from_numpy(arr[f'boxes_{v}']) for v in range(2)]
        mappings=[PadMapping(v['width'],v['height']) for v in sample['views']];mapped=[m.boxes(b) for m,b in zip(mappings,boxes)]
        images=load_and_preprocess_images(paths,mode='pad');gray=[cv2.imread(p,cv2.IMREAD_GRAYSCALE) for p in paths]
        background=estimate_background(*gray,*[b.numpy() for b in boxes],method='sift',mask_targets=True)
        for masked in [False,True]:
            inputs=images.clone()
            if masked:
                for v,b in enumerate(mapped):
                    for x1,y1,x2,y2 in b.round().int().tolist():inputs[v,:,max(0,y1):min(518,y2),max(0,x1):min(518,x2)]=1.
            start=time.perf_counter();torch.cuda.synchronize()
            _,_,meta=adapter.extract(inputs,mapped,modes=(),camera=True);torch.cuda.synchronize();dt=time.perf_counter()-start
            pose=meta['pose'];E,K=pose_encoding_to_extri_intri(pose,(518,518));E=E.numpy()[0];K=K.numpy()[0];F=None
            try:
                Fp=fundamental_from_cameras(E[0],E[1],K[0],K[1]);As=[]
                for m in mappings:
                    w,h=m.resized;As.append(np.array([[w/m.width,0,(m.target-w)//2],[0,h/m.height,(m.target-h)//2],[0,0,1.]]))
                F=As[1].T@Fp@As[0];F/=np.linalg.norm(F)
                res=symmetric_epipolar_distance(F,background['points1'],background['points2'])
            except (ValueError,np.linalg.LinAlgError):res=np.array([])
            finite=bool(len(res)>=20 and np.isfinite(res).all());median=float(np.median(res)) if finite else None;ratio=float((res<=12.).mean()) if finite else 0.
            # 12 original-image pixels ≈ 3 VGGT pixels for 1920px inputs.
            valid=finite and background['coverage']>=.01 and ratio>=.25 and median<=12.
            evidence=soft_box_evidence(F,boxes[0].numpy(),boxes[1].numpy(),valid,ratio)
            positive=arr['labels_1'][:,None]==arr['labels_0'][None,:];known=(arr['labels_1'][:,None]>0)&(arr['labels_0'][None,:]>0)
            pos=positive&known;neg=~positive&known
            row=dict(video_id=sample['video_id'],frame_id=sample['frame_id'],method='vggt_camera',masked=masked,valid=bool(valid),seconds=dt,
                background_matches=background['matches'],background_reference_valid=background['valid'],median_epipolar_original_px=median,
                inlier_ratio_at_12px=ratio,coverage=background['coverage'],positive_score=float(evidence[pos].mean()) if pos.any() else None,negative_score=float(evidence[neg].mean()) if neg.any() else None)
            rows.append(row);print(json.dumps(row),flush=True)
    path=ROOT/'results/vggt_geometry_validation.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_json(ROOT/'results/vggt_geometry_manifest.json',dict(sample_manifest_sha256=sha256(a.samples),weights_sha256=sha256(RUNTIME/'weights/vggt_1b.pt'),validity_uses_gt=False,
        validity='>=20 finite background matches, footprint >=1%, >=25% matches within12 originalpx, median<=12px',independent_world_coordinates_compared=False,model_source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME/'sources/vggt',text=True).strip(),provenance=provenance()))
if __name__=='__main__':main()
