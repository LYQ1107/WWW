"""Fresh complete-TRAIN online prediction evaluation, never cached-score HOTA.

All requested scenes must be present. Invalid raw GT is retained and explicitly
blocked rather than repaired, dropped silently, or evaluated with a lax subclass.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from common import DATA,ROOT,sha256,save_json,train_data,video_records
from cir_evaluation import read_mot,validate_rows,evaluate_trackeval,evaluate_matlab

METRICS=('HOTA','AssA','CVIDF1','IDF1','IDSW','DetA','MOTA','Frag','FPS')


def collect(source,split='train'):
    if split!='train':raise ValueError('TEST needs the final frozen-model evaluation gate')
    data=train_data();required=source['required_video_ids'];runs=source['runs']
    if sorted(map(int,runs))!=sorted(required) or len(set(required))!=len(required):raise ValueError('Missing/extra registered complete videos')
    result={};bindings=[]
    for vid in required:
        path=Path(runs[str(vid)]);run=json.loads(path.read_text())
        if run['status']!='COMPLETE' or run['split']!=split or run['video_id']!=vid or not run['full_video']:
            raise ValueError('Not a complete cold-start run '+str(path))
        records=video_records(data,vid);index={r['image_id']:r for r in records}
        if sorted(run['processed_image_ids'])!=sorted(index):raise ValueError('Full physical image coverage mismatch')
        if run['checkpoint_sha256']!='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8':raise ValueError('Wrong GMT weight')
        scene=next(v['file_name'] for v in data['videos'] if v['id']==vid)
        gt={};lengths={};rawsha={}
        for view in sorted({r['view_id'] for r in records}):
            name=f'{scene}_View{view}';gtpath=DATA/split/name/'gt/gt.txt'
            gt[name]=read_mot(gtpath);rawsha[str(gtpath)]=sha256(gtpath)
            lengths[name]=max(r['frame_id'] for r in records if r['view_id']==view)
        predictions={}
        for kind in ('raw','stock'):
            spec=run['predictions'][kind];pp=Path(spec['path'])
            if sha256(pp)!=spec['sha256']:raise ValueError('Prediction drift '+str(pp))
            byview={name:[] for name in gt}
            for r in json.loads(pp.read_text()):
                im=index[r['image_id']];name=f"{scene}_View{im['view_id']}"
                byview[name].append([im['frame_id'],r['track_id'],*r['bbox'],r['score'],-1,-1,-1])
            predictions[kind]=byview
        result[vid]=dict(run=run,scene=scene,gt=gt,lengths=lengths,predictions=predictions,raw_gt_sha256=rawsha,
                         manifest=dict(path=str(path),sha256=sha256(path)))
        bindings.append(tuple(run[k] for k in ('method','checkpoint_sha256','config_sha256','seed','inference_protocol')))
    if len(set(bindings))!=1:raise ValueError('Mixed method/weights/config/seed/inference protocol in one suite')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    source=json.loads(a.runs.read_text());runs=collect(source);a.out.mkdir(parents=True)
    per_video={};valid=[];blocked=[]
    for vid,spec in runs.items():
        try:
            for name,rows in spec['gt'].items():validate_rows(rows,name+' raw GT')
        except ValueError as error:
            blocked.append(vid);per_video[str(vid)]=dict(status='BLOCKED_RAW_GT_INVALID',reason=str(error),
                raw={k:None for k in METRICS},stock={k:None for k in METRICS},run=spec['manifest'],raw_gt_sha256=spec['raw_gt_sha256'])
            continue
        valid.append(vid);per_video[str(vid)]=dict(status='COMPLETE',run=spec['manifest'],raw_gt_sha256=spec['raw_gt_sha256'])
        for kind in ('raw','stock'):
            track=evaluate_trackeval(spec['gt'],spec['predictions'][kind],spec['lengths'],a.out/f'video{vid}'/kind/'trackeval')
            per_video[str(vid)][kind]=dict(**track['combined'],FPS=spec['run']['full_image_FPS'],
                per_camera=track['per_sequence'],CVIDF1=None,official_MATLAB='pooled call below',
                FPS_scope=spec['run']['timing_scope'])
    pooled={}
    for kind in ('raw','stock'):
        gt={};pred={};lengths={};scenes={}
        for vid in valid:
            spec=runs[vid];gt.update(spec['gt']);pred.update(spec['predictions'][kind]);lengths.update(spec['lengths']);scenes[spec['scene']]=list(spec['gt'])
        if not valid:
            pooled[kind]={k:None for k in METRICS};continue
        track=evaluate_trackeval(gt,pred,lengths,a.out/'pooled'/kind/'trackeval')
        official=evaluate_matlab(scenes,gt,pred,a.out/'pooled'/kind/'official_matlab')
        pooled[kind]=dict(**track['combined'],**official['combined'],
            FPS=sum(len(runs[v]['run']['processed_image_ids']) for v in valid)/sum(runs[v]['run']['full_image_seconds'] for v in valid),
            actual_evaluation_video_ids=valid,scope='native combine_sequences and summed official counts; excludes explicitly blocked invalid raw GT; not a full24 score',
            TrackEval_source_sha256=track['formula_sources'],official_MATLAB=official)
        for vid in valid:
            per_video[str(vid)][kind].update(official['per_scene'][runs[vid]['scene']])
    result=dict(status='COMPLETE_ALL_REQUESTED_PREDICTIONS_EVALUATED_WITH_EXPLICIT_GT_BLOCKS' if blocked else 'COMPLETE',
                required_video_ids=source['required_video_ids'],invalid_raw_gt_video_ids=blocked,
                per_video=per_video,pooled=pooled,runs_manifest_sha256=sha256(a.runs),
                raw_GT_modified=False,metric_equations_modified=False,GT_in_inference=False,
                scientific_positive=None,interpretation='Compare the registered same-scope original run; no historical different-sequence scores substituted')
    save_json(a.out/'RESULT.json',result)
    print(json.dumps(dict(status=result['status'],valid=valid,blocked=blocked,
                         pooled={k:{m:v.get(m) for m in METRICS} for k,v in pooled.items()})),flush=True)


if __name__=='__main__':main()
