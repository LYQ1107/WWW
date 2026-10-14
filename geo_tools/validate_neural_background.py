"""Failure-recovery diagnostic: learned background matches, independent validity.

All dense maps are transient. Labels are neither loaded nor used by the gates.
No geometry is inserted into tracking until pixel correspondence validation.
"""
import argparse,csv,time,cv2,numpy as np,torch
from common import *
from gtr.geometry.roi_feature_extractor import PadMapping
from gtr.geometry.geometry_confidence import estimate_background
from gtr.geometry.background_patch_matcher import background_descriptors,mutual_ratio_matches,validate_candidate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=Path,nargs='+',required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args();out=RUNTIME/'geometry'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path.insert(0,str(RUNTIME/'sources/vggt'))
    from gtr.geometry.vggt_adapter import FrozenVGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    adapter=FrozenVGGT(RUNTIME/'weights/vggt_1b.pt',layers=[17]);rows=[];cv2.setNumThreads(4)
    for manifest in args.samples:
        source=json.loads(manifest.read_text())
        for sample in source['samples']:
            # Read detector boxes only. Offline labels in the same file are not
            # accessed by this correspondence or confidence diagnostic.
            arr=np.load(sample['path']);boxes=[torch.tensor(arr[f'boxes_{v}']) for v in range(2)]
            mappings=[PadMapping(v['width'],v['height']) for v in sample['views']]
            paths=[v['path'] for v in sample['views']];images=load_and_preprocess_images(paths,mode='pad')
            mapped=[mapping.boxes(box) for mapping,box in zip(mappings,boxes)]
            for view in range(2):
                for box in mapped[view]:
                    x1,y1,x2,y2=box.round().int().tolist()
                    images[view,:,max(0,y1-4):min(518,y2+4),max(0,x1-4):min(518,x2+4)]=1.
            gray=[cv2.imread(path,cv2.IMREAD_GRAYSCALE) for path in paths]
            reference=estimate_background(*gray,*[box.numpy() for box in boxes],method='sift',mask_targets=True)
            torch.cuda.synchronize();start=time.perf_counter()
            _,_,first=adapter.extract(images,mapped,modes=(),return_patch_maps=True)
            _,_,reverse=adapter.extract(images.flip(0),mapped[::-1],modes=(),return_patch_maps=True)
            torch.cuda.synchronize();encoding_seconds=time.perf_counter()-start
            for feature in ['dino_internal','layer_17']:
                descriptors=[];points=[]
                for view in range(2):
                    a,p=background_descriptors(first['patch_maps'][feature][view],mappings[view],boxes[view])
                    b,q=background_descriptors(reverse['patch_maps'][feature][1-view],mappings[view],boxes[view])
                    if not torch.equal(p,q):raise RuntimeError('inconsistent symmetric patch masks')
                    descriptors.append(torch.cat([a,b],dim=1)/2**.5);points.append(p)
                p1,p2=mutual_ratio_matches(*descriptors,*points)
                # Sampling precision is one patch at original scale. This loose
                # candidate fit never relaxes the independent 12px validity gate.
                threshold=max(max(m.patch*m.width/m.resized[0],m.patch*m.height/m.resized[1]) for m in mappings)
                for kind in ['F','H']:
                    cv2.setRNGSeed(20261014);matrix=None;inside=None
                    if len(p1)>=8:
                        if kind=='F':matrix,inside=cv2.findFundamentalMat(p1.astype(np.float32),p2.astype(np.float32),cv2.FM_RANSAC,threshold,.99)
                        else:matrix,inside=cv2.findHomography(p1.astype(np.float32),p2.astype(np.float32),cv2.RANSAC,threshold)
                    validation=validate_candidate(matrix,kind,reference['points1'],reference['points2'],[g.shape for g in gray])
                    inliers=int(inside.sum()) if inside is not None else 0
                    if inliers<20:validation['valid']=False
                    row=dict(split=source['provenance'].get('split'),video_id=sample['video_id'],frame_id=sample['frame_id'],feature=feature,geometry=kind,mutual_matches=len(p1),candidate_inliers=inliers,candidate_ransac_original_px=threshold,encoding_seconds_two_orders=encoding_seconds,**validation)
                    rows.append(row);print(json.dumps(row),flush=True)
            del first,reverse,descriptors,points
    path=out/'validation.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=dict(status='DIAGNOSTIC_COMPLETE',provenance=provenance(),samples={str(p):sha256(p) for p in args.samples},weights_sha256=sha256(RUNTIME/'weights/vggt_1b.pt'),dense_maps_cached=False,gt_used=False,actual_vggt_calls=2*len(rows)//4,
        candidate_rule='foreground-input masking; exclude foreground footprint and padding; symmetric two-order unit descriptor concat; bidirectional ratio0.8 and mutual nearest neighbor; original-pixel quantization-aware RANSAC',
        validity_rule='>=20 neural fit inliers plus independent masked SIFT: >=20 pixel inliers, >=25% within12 originalpx, median<=12px and convex-hull coverage>=1% in both views',
        counts={f'{split}/{feature}/{kind}':dict(total=sum(r['split']==split and r['feature']==feature and r['geometry']==kind for r in rows),valid=sum(r['valid'] for r in rows if r['split']==split and r['feature']==feature and r['geometry']==kind)) for split in sorted({r['split'] for r in rows}) for feature in ['dino_internal','layer_17'] for kind in ['F','H']},
        csv=str(path),csv_sha256=sha256(path),runtime_scope='diagnostic encoding only; not end-to-end tracker throughput')
    save_json(out/'manifest.json',report);save_json(ROOT/f'results/{args.tag}_manifest.json',report)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
