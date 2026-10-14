import argparse,json,time,cv2,csv,numpy as np
from common import *
from gtr.geometry.geometry_confidence import estimate_background
from gtr.geometry.geometry_alignment import symmetric_epipolar_distance,soft_box_evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=RUNTIME/'samples/screen_v1r2/samples.json');a=p.parse_args()
    source=json.loads(a.samples.read_text());rows=[]
    cv2.setRNGSeed(20261014);cv2.setNumThreads(4)
    for sample in source['samples']:
        arrays=np.load(sample['path']);images=[cv2.imread(v['path'],cv2.IMREAD_GRAYSCALE) for v in sample['views']]
        for method in ['orb','sift']:
            for masked in [False,True]:
                cv2.setRNGSeed(20261014);start=time.perf_counter()
                result=estimate_background(*images,arrays['boxes_0'],arrays['boxes_1'],method,masked)
                row=dict(video_id=sample['video_id'],frame_id=sample['frame_id'],method=method,masked=masked,seconds=time.perf_counter()-start,
                    **{k:result[k] for k in ['valid','keypoints1','keypoints2','matches','inliers','inlier_ratio','median_residual','coverage']})
                boxes1,boxes2=arrays['boxes_0'],arrays['boxes_1']; labels1,labels2=arrays['labels_0'],arrays['labels_1']
                score=soft_box_evidence(result['F'],boxes1,boxes2,result['valid'],result['inlier_ratio'])
                valid=(labels2[:,None]>0)&(labels1[None,:]>0);pos=valid&(labels2[:,None]==labels1[None,:]);neg=valid&~pos
                row.update(positive_pairs=int(pos.sum()),negative_pairs=int(neg.sum()),positive_score=float(score[pos].mean()) if pos.any() else None,negative_score=float(score[neg].mean()) if neg.any() else None)
                rows.append(row);print(json.dumps(row),flush=True)
    with (ROOT/'results/geometry_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_json(ROOT/'results/geometry_reference_manifest.json',dict(samples_sha256=sha256(a.samples),geometry_confidence_uses_gt=False,identity_diagnostics_use_train_dev_gt_only=True,source_commit=source['provenance']['commit'],rows=len(rows),metrics_source=str(ROOT/'results/geometry_validation.csv')))
if __name__=='__main__':main()
