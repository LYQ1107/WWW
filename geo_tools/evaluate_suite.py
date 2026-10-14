"""Combined development evaluation via native TrackEval sequence combination.

Never averages already-normalized per-video metrics. Cross-view identities are
flattened within each scene, and scenes remain separate sequences.
"""
import argparse,collections,csv,time,numpy as np
from common import *
sys.path.insert(0,str(ROOT/'TrackEval'))
import trackeval

def rows_from(path):
    return [[s.strip() for s in r.split(',')] for r in path.read_text().splitlines() if r.strip()]

def validate_control_flags(runs):
    controls={}
    for flag in ('causal_online','validation_corrected_gmt_camera_index','validation_reversed_vggt_order','validation_constant_head_evidence','validation_native_output_prior','compute_neutral'):
        default=None if flag=='validation_constant_head_evidence' else False
        values={run.get(flag,default) for run in runs}
        if len(values)!=1:raise ValueError('mixed protocol/validation/neutral flag within one suite: '+flag)
        controls[flag]=values.pop()
    return controls

def evaluate(gt,trackers,lengths,base):
    gtroot=base/'gt';trroot=base/'trackers'/'GMT'/'data';trroot.mkdir(parents=True)
    for name,values in gt.items():
        path=gtroot/name/'gt/gt.txt';path.parent.mkdir(parents=True)
        with path.open('w',newline='') as f:csv.writer(f).writerows(values)
        with (trroot/f'{name}.txt').open('w',newline='') as f:csv.writer(f).writerows(trackers[name])
    ec=trackeval.Evaluator.get_default_eval_config();ec.update(USE_PARALLEL=False,PRINT_CONFIG=False,PRINT_RESULTS=False,PLOT_CURVES=False,OUTPUT_FOLDER=str(base/'output'))
    dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config();dc.update(GT_FOLDER=str(gtroot),TRACKERS_FOLDER=str(base/'trackers'),TRACKERS_TO_EVAL=['GMT'],BENCHMARK='VisionTrack',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,SEQ_INFO=lengths,PRINT_CONFIG=False,OUTPUT_FOLDER=str(base/'output'))
    dataset=trackeval.datasets.MotChallenge2DBox(dc)
    result,message=trackeval.Evaluator(ec).evaluate([dataset],[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()])
    value=result[dataset.get_name()]['GMT']
    if value is None:raise RuntimeError(message)
    combined=value['COMBINED_SEQ']['pedestrian']
    def native(x):
        if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return x
    save_json(base/'combined_raw_metrics.json',native(combined))
    return dict(HOTA=float(np.mean(combined['HOTA']['HOTA'])*100),DetA=float(np.mean(combined['HOTA']['DetA'])*100),AssA=float(np.mean(combined['HOTA']['AssA'])*100),
        IDF1=float(combined['Identity']['IDF1']*100),MOTA=float(combined['CLEAR']['MOTA']*100),IDSW=int(combined['CLEAR']['IDSW']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--report-output',type=Path);a=p.parse_args()
    source=json.loads(a.sources.read_text());paths={int(v):Path(path) for v,path in source['video_predictions'].items()}
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if sorted(paths)!=sorted(split['strict_development_video_ids']):raise ValueError('suite must cover every declared strict development video')
    out=RUNTIME/'suites'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);data=train_data();gt={};trackers={};lengths={};scenes={};runs={};hashes={};total_images=0;start=time.perf_counter()
    for vid,path in sorted(paths.items()):
        recordlist=video_records(data,vid);index={r['image_id']:r for r in recordlist};total_images+=len(recordlist)
        manifest_path=path.with_name(path.stem+'_manifest.json')
        if not manifest_path.exists():manifest_path=path.parent/'manifest.json'
        run=json.loads(manifest_path.read_text())
        if not isinstance(run,dict) or not run.get('full_video'):raise ValueError('not a complete-video run')
        if run.get('checkpoint_sha256')!='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8':raise ValueError('checkpoint mismatch')
        expected=run.get('prediction_sha256',run.get('predictions_sha256'))
        if expected!=sha256(path):raise ValueError('prediction/run hash mismatch')
        runs[str(vid)]=dict(path=str(manifest_path),sha256=sha256(manifest_path),causal_online=run['causal_online'],commit=run['commit'],validation_corrected_gmt_camera_index=run.get('validation_corrected_gmt_camera_index',False),
                          validation_reversed_vggt_order=run.get('validation_reversed_vggt_order',False),
                          validation_constant_head_evidence=run.get('validation_constant_head_evidence'),validation_native_output_prior=run.get('validation_native_output_prior',False),compute_neutral=run.get('compute_neutral',False))
        scene=next(v['file_name'] for v in data['videos'] if v['id']==vid);scenes[scene]=[];byview=collections.defaultdict(list)
        for prediction in json.loads(path.read_text()):
            r=index[prediction['image_id']];byview[r['view_id']].append([r['frame_id'],prediction['track_id'],*prediction['bbox'],prediction['score'],-1,-1,-1])
        for view in sorted({r['view_id'] for r in recordlist}):
            name=f'{scene}_View{view}';gtpath=DATA/'train'/name/'gt/gt.txt';values=rows_from(gtpath)
            counts=collections.Counter((int(float(x[0])),int(float(x[1]))) for x in values)
            if any(n>1 for n in counts.values()):raise ValueError('strict suite has repeated raw GT identities')
            gt[name]=values;trackers[name]=byview[view];lengths[name]=max(r['frame_id'] for r in recordlist if r['view_id']==view);scenes[scene].append(name);hashes[str(gtpath)]=sha256(gtpath)
    controls=validate_control_flags(list(runs.values()))
    metrics=evaluate(gt,trackers,lengths,out/'per_view')
    for convention in ['sequential','interleaved']:
        gtflat={};trflat={};flatlengths={}
        for scene,names in scenes.items():
            gtflat[scene]=[];trflat[scene]=[];offset=0
            for v,name in enumerate(sorted(names)):
                def frame(x):return int(float(x))+offset if convention=='sequential' else int(float(x))*len(names)+v
                gtflat[scene].extend([[frame(r[0]),*r[1:]] for r in gt[name]])
                trflat[scene].extend([[frame(r[0]),*r[1:]] for r in trackers[name]])
                offset+=lengths[name]
            flatlengths[scene]=max(int(r[0]) for r in gtflat[scene]+trflat[scene])
        result=evaluate(gtflat,trflat,flatlengths,out/convention)
        if convention=='sequential':metrics['CVIDF1']=result['IDF1'];metrics['CVMA_sequential']=result['MOTA']
        else:metrics['CVMA']=result['MOTA']
    report=dict(status='PASS',scope='all declared strict TRAIN development scenes; original raw GT unchanged',video_ids=sorted(paths),images=total_images,full_video=True,
        **controls,metrics=metrics,source_manifest_sha256=sha256(a.sources),prediction_sha256={str(p):sha256(p) for p in paths.values()},run_manifests=runs,
        raw_gt_sha256=hashes,combination='native TrackEval combine_sequences; not mean of per-video normalized scores',evaluator='TrackEval, not official MATLAB',
        seconds=time.perf_counter()-start,provenance=provenance())
    save_json(out/'metrics.json',report);save_json(a.report_output or ROOT/f'results/{a.tag}_metrics.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
