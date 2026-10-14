"""Fresh TRAIN evaluation, original raw GT and versioned TrackEval equations."""
import argparse, collections, csv, json, shutil, numpy as np
from common import *
sys.path.insert(0,str(ROOT/'TrackEval'))
import trackeval

def read_rows(path):
    return [[x.strip() for x in line.split(',')] for line in path.read_text().splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path,required=True);p.add_argument('--video-id',type=int,required=True);p.add_argument('--allow-duplicate-gt',action='store_true');p.add_argument('--run-manifest',type=Path);args=p.parse_args()
    report_path=args.prediction.with_suffix('.raw_diagnostic.metrics.json' if args.allow_duplicate_gt else '.metrics.json')
    if report_path.exists():raise FileExistsError(report_path)
    data=train_data(); records=video_records(data,args.video_id); index={x['image_id']:x for x in records}
    pred=json.loads(args.prediction.read_text()); by_view=collections.defaultdict(list)
    for row in pred:
        image=index[row['image_id']]; by_view[image['view_id']].append([image['frame_id'],row['track_id'],*row['bbox'],row['score'],-1,-1,-1])
    work=args.prediction.with_suffix('.eval_raw_diagnostic' if args.allow_duplicate_gt else '.eval'); work.mkdir(); scene=next(v['file_name'] for v in data['videos'] if v['id']==args.video_id)
    sources={}; trackers={}; lengths={}; duplicates={}
    for view in sorted({x['view_id'] for x in records}):
        name=f'{scene}_View{view}'; path=DATA/'train'/name/'gt/gt.txt'; gt=read_rows(path)
        keys=collections.Counter((int(float(r[0])),int(float(r[1]))) for r in gt)
        duplicates[name]=sum(n-1 for n in keys.values() if n>1)
        sources[name]=gt;trackers[name]=by_view[view];lengths[name]=max(x['frame_id'] for x in records if x['view_id']==view)
    if any(duplicates.values()) and not args.allow_duplicate_gt:
        save_json(args.prediction.with_suffix('.gt_audit.json'),dict(duplicate_rows=duplicates,status='STRICT_EVAL_BLOCKED_RAW_GT_DUPLICATES'))
        raise ValueError(f'raw GT duplicate identities {duplicates}; retain all rows, explicit diagnostic mode required')
    class RawDuplicateDiagnostic(trackeval.datasets.MotChallenge2DBox):
        @staticmethod
        def _check_unique_ids(data,after_preproc=False):return None
    def evaluate(gt_rows,tracker_rows,seq_lengths,label):
        base=work/label; gtdir=base/'gt'; trdir=base/'trackers'/'GMT'/'data'; trdir.mkdir(parents=True)
        for name,rows in gt_rows.items():
            path=gtdir/name/'gt/gt.txt';path.parent.mkdir(parents=True)
            with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
            with (trdir/f'{name}.txt').open('w',newline='') as f:csv.writer(f).writerows(tracker_rows[name])
        ec=trackeval.Evaluator.get_default_eval_config();ec.update(USE_PARALLEL=False,PRINT_CONFIG=False,PRINT_RESULTS=False,PLOT_CURVES=False,OUTPUT_FOLDER=str(base/'output'))
        dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config();dc.update(GT_FOLDER=str(gtdir),TRACKERS_FOLDER=str(base/'trackers'),TRACKERS_TO_EVAL=['GMT'],BENCHMARK='VisionTrack',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,SEQ_INFO=seq_lengths,PRINT_CONFIG=False,OUTPUT_FOLDER=str(base/'output'))
        cls=RawDuplicateDiagnostic if args.allow_duplicate_gt else trackeval.datasets.MotChallenge2DBox;ds=cls(dc)
        result,messages=trackeval.Evaluator(ec).evaluate([ds],[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()])
        values=result[ds.get_name()]['GMT']
        if values is None:raise RuntimeError(messages)
        c=values['COMBINED_SEQ']['pedestrian']
        metrics=dict(HOTA=float(np.mean(c['HOTA']['HOTA'])*100),DetA=float(np.mean(c['HOTA']['DetA'])*100),AssA=float(np.mean(c['HOTA']['AssA'])*100),IDF1=float(c['Identity']['IDF1']*100),MOTA=float(c['CLEAR']['MOTA']*100),IDSW=int(c['CLEAR']['IDSW']))
        return metrics
    metrics=evaluate(sources,trackers,lengths,'per_view')
    for convention in ['sequential','interleaved']:
        gtflat=[];trflat=[];offset=0;views=len(sources)
        for v,name in enumerate(sorted(sources)):
            def frame(f):return int(float(f))+offset if convention=='sequential' else int(float(f))*views+v
            gtflat.extend([[frame(r[0]),*r[1:]] for r in sources[name]]);trflat.extend([[frame(r[0]),*r[1:]] for r in trackers[name]])
            offset+=lengths[name]
        n=max(int(r[0]) for r in gtflat+trflat)
        values=evaluate({scene:gtflat},{scene:trflat},{scene:n},convention)
        if convention=='sequential': metrics['CVIDF1']=values['IDF1'];metrics['CVMA_sequential']=values['MOTA']
        else:metrics['CVMA']=values['MOTA']
    source_manifest=args.run_manifest or args.prediction.with_name(args.prediction.stem+'_manifest.json')
    meta=json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
    expected=meta.get('prediction_sha256',meta.get('predictions_sha256'))
    if expected is not None and expected!=sha256(args.prediction):raise ValueError('prediction hash differs from run manifest')
    report=dict(status='PASS',video_id=args.video_id,metrics=metrics,prediction_sha256=sha256(args.prediction),raw_gt_sha256={n:sha256(DATA/'train'/n/'gt/gt.txt') for n in sources},
        duplicate_gt_rows=duplicates,allow_duplicate_gt=args.allow_duplicate_gt,gt_policy='all raw rows retained',causal_online=meta.get('causal_online',False),full_video=meta.get('full_video',False),
        evaluator='TrackEval equations; CVIDF1=Identity on sequential view blocks; CVMA=CLEAR on interleaved views; not official MATLAB',
        evaluator_source_hashes={str(x.relative_to(ROOT)):sha256(x) for x in (ROOT/'TrackEval/trackeval').rglob('*.py')})
    save_json(report_path,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
