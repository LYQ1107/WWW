#!/usr/bin/env python3
"""Complete-video official evaluation; never substitutes custom metrics."""
import argparse,json
from pathlib import Path
from worldmot.evaluation.official_fullvideo import prepare,trackeval,matlab
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack');p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--audit',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--scenes',nargs='+',type=int)
    p.add_argument('--box-source',default='tracks_2d_by_camera');p.add_argument('--skip-matlab',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[2];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    gt,pred,lengths,groups,blocked,coverage=prepare(a.dataset_root,a.split,a.audit,a.predictions,a.scenes,a.box_source)
    report={'status':'OFFICIAL_EVALUATION_RUNNING','split':a.split,'blocked':blocked,'coverage':coverage,'evaluated_scenes':list(groups),'custom_metrics_used':False,'box_source':a.box_source,'raw_GT_modified':False}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    if not groups:raise RuntimeError('No complete valid sequences; no official score fabricated')
    report['TrackEval']=trackeval(gt,pred,lengths,out/'trackeval',repo/'TrackEval',a.split,groups)
    if not a.skip_matlab:
        try:report['VisionTrack_MATLAB']=matlab(gt,pred,groups,out/'matlab',repo/'MOTChallengeEvalKit_cv_test/matlab_devkit','/data1/liuyeqiang/MATLAB/R2020a/bin/matlab',Path(__file__).with_name('official_worldmot_metrics.m'))
        except Exception as e:report['VisionTrack_MATLAB']={'status':'BLOCKED','reason':f'{type(e).__name__}: {e}'}
    report['status']='COMPLETE_VALID_SCOPE';(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['TrackEval','VisionTrack_MATLAB']},indent=2))
