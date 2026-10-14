import argparse,csv,time,cv2,numpy as np
from common import *
from gtr.geometry.geometry_confidence import estimate_background
from gtr.geometry.geometry_alignment import soft_plane_evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=RUNTIME/'samples/screen_clean_dev_v2/samples.json');a=p.parse_args();rows=[];cv2.setNumThreads(4)
    source=json.loads(a.samples.read_text())
    for sample in source['samples']:
        arrays=np.load(sample['path']);images=[cv2.imread(v['path'],cv2.IMREAD_GRAYSCALE) for v in sample['views']]
        for method in ['orb_homography','sift_homography']:
            cv2.setRNGSeed(20261014);t=time.perf_counter();r=estimate_background(*images,arrays['boxes_0'],arrays['boxes_1'],method,True)
            score=soft_plane_evidence(r['H'],arrays['boxes_0'],arrays['boxes_1'],r['H_valid'],r['H_inlier_ratio'])
            labels1,labels2=arrays['labels_0'],arrays['labels_1'];known=(labels2[:,None]>0)&(labels1[None,:]>0);pos=known&(labels2[:,None]==labels1[None,:]);neg=known&~pos
            row=dict(video_id=sample['video_id'],frame_id=sample['frame_id'],method=method,seconds=time.perf_counter()-t,valid=r['H_valid'],inliers=r['H_inliers'],inlier_ratio=r['H_inlier_ratio'],median_residual=r['H_median_residual'],coverage=r['H_coverage'],
                positive_pairs=int(pos.sum()),negative_pairs=int(neg.sum()),positive_score=float(score[pos].mean()) if pos.any() else None,negative_score=float(score[neg].mean()) if neg.any() else None)
            rows.append(row);print(json.dumps(row),flush=True)
    with (ROOT/'results/plane_geometry_validation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_json(ROOT/'results/plane_geometry_manifest.json',dict(samples_sha256=sha256(a.samples),provenance=provenance(),ground_plane_assumption=True,validity_uses_gt=False,gt_used_only_in_offline_candidate_diagnostics=True))
if __name__=='__main__':main()
