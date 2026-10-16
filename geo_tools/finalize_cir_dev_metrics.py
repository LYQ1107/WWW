"""Reuse exact completed metrics; append the final invalid-GT scene explicitly."""
import argparse
import json
from pathlib import Path
from common import sha256,save_json
from evaluate_cir_fulltrain import collect
from cir_evaluation import validate_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--interim',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    source=json.loads(a.runs.read_text());runs=collect(source);prior=json.loads(a.interim.read_text())
    if source['required_video_ids']!=[1,2,3,10,11,14,15,16,18,19,22,23]:raise ValueError('Complete registered split required')
    if prior['required_video_ids']!=source['required_video_ids'][:-1]:raise ValueError('Interim completion scope changed')
    for vid,item in prior['per_video'].items():
        current=runs[int(vid)]
        if item['run']['sha256']!=current['manifest']['sha256'] or item['raw_gt_sha256']!=current['raw_gt_sha256']:
            raise ValueError('Previously evaluated trajectory or original GT changed')
    latest=runs[23];reason=None
    try:
        for name,rows in latest['gt'].items():validate_rows(rows,name+' raw GT')
    except ValueError as error:reason=str(error)
    if reason is None:raise ValueError('New valid scene would require recomputing native combined metrics')
    prior['per_video']['23']=dict(status='BLOCKED_RAW_GT_INVALID',reason=reason,run=latest['manifest'],raw_gt_sha256=latest['raw_gt_sha256'],
        **{kind:dict(**{k:None for k in ('HOTA','AssA','CVIDF1','IDF1','IDSW','DetA','MOTA','Frag')},FPS=latest['run']['full_image_FPS']) for kind in ('raw','stock')})
    for vid,item in prior['per_video'].items():
        if item['status']=='BLOCKED_RAW_GT_INVALID':
            for kind in ('raw','stock'):item[kind]['FPS']=runs[int(vid)]['run']['full_image_FPS']
    prior.update(status='COMPLETE_ALL12_INFERENCE_AND_EXPLICIT_METRIC_GT_BLOCKS',required_video_ids=source['required_video_ids'],
        invalid_raw_gt_video_ids=sorted(set(prior['invalid_raw_gt_video_ids']+[23])),
        full_inference_images_all12=sum(len(v['run']['processed_image_ids']) for v in runs.values()),
        full_image_FPS_all12=sum(len(v['run']['processed_image_ids']) for v in runs.values())/sum(v['run']['full_image_seconds'] for v in runs.values()),
        source_results=dict(interim=dict(path=str(a.interim),sha256=sha256(a.interim)),full_runs_manifest=dict(path=str(a.runs),sha256=sha256(a.runs))),
        exact_prior_native_combination_reused=True,
        interpretation='All12 complete; metrics on predeclared5 legal raw-GT videos. Remaining7 predictions retained, metric values null because raw GT invalid. Full12 FPS includes every image; no GT/evaluator modifications.')
    save_json(a.out,prior);print(json.dumps(dict(status=prior['status'],blocked=prior['invalid_raw_gt_video_ids'],images=prior['full_inference_images_all12'],FPS_all12=prior['full_image_FPS_all12'])),flush=True)


if __name__=='__main__':main()
