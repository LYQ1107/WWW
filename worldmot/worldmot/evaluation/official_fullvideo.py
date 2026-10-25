"""WorldMOT format adapter to unchanged TrackEval and VisionTrack MATLAB.

The sequential/interleaved conventions follow the existing repository's
geo_tools/cir_evaluation.py converter.  No tracker/model from GMT is imported.
Only this posthoc process opens original GT.  No duplicate removal is allowed.
"""
from __future__ import annotations
import collections,hashlib,json,os,signal,subprocess,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_mot(p):return [[float(x) for x in line.strip().split(',')] for line in Path(p).read_text().splitlines() if line.strip()]
def validate(rows,name):
    counts=collections.Counter((int(r[0]),int(r[1])) for r in rows)
    if any(n>1 for n in counts.values()):raise ValueError('duplicate frame/identity in '+name)
    if any(len(r)<6 or not np.isfinite(r[:6]).all() or r[0]<1 or r[4]<0 or r[5]<0 for r in rows):raise ValueError('invalid MOT rows in '+name)
def write_mot(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in sorted(rows,key=lambda r:(r[0],r[1])):
            tail=list(r[6:10])+[-1]*max(0,10-len(r))
            # Round-trip source coordinates exactly, including original GT.
            # Two-decimal presentation rounding must not change scoring boxes.
            f.write(f'{int(r[0])},{int(r[1])},'+','.join(format(float(x),'.17g') for x in r[2:6])+','+','.join(map(str,tail))+'\n')

def native_tracking_sources(kit):
    """Hash the native tracking dependency tree, not the unused detector toolbox."""
    kit=Path(kit);utils=kit/'utils'
    paths=set()
    for directory in (kit,utils,utils/'camera',utils/'external/iniconfig'):
        for suffix in ('*.m','*.cpp','*.mexa64'):paths.update(directory.glob(suffix))
    if (utils/'external/PrintTable.m').is_file():paths.add(utils/'external/PrintTable.m')
    return sorted(path for path in paths if path.is_file())
def prepare(dataset_root,split,audit,inference_root,scenes=None,box_source="tracks_2d_by_camera"):
    audit=json.loads(Path(audit).read_text())[split];root=Path(inference_root)
    gt={};pred={};lengths={};groups={};blocked=[];coverage=[]
    for scene in audit['scenes']:
        sid=scene['scene']
        if scenes and sid not in scenes:continue
        p=root/f'scene_{sid:02d}'/'tracks.jsonl'
        if not p.is_file():blocked.append({'scene':sid,'reason':'PREDICTIONS_MISSING'});continue
        rows=[json.loads(line) for line in p.open() if line.strip()];times=[int(r['timestamp']) for r in rows]
        if any(box_source not in row for row in rows):raise RuntimeError('requested prediction box source missing: '+box_source+' in '+str(p))
        expected=list(range(scene['first_frame'],scene['last_frame']+1))
        if times!=expected or len(rows)!=scene['synchronized_frames']:
            blocked.append({'scene':sid,'reason':'INCOMPLETE_OR_NONCAUSAL_SEQUENCE','actual':len(rows),'expected':scene['synchronized_frames']});continue
        coverage.append({'scene':sid,'frames':len(rows),'predictions_sha256':sha(p)})
        if not scene['official_gt_legal']:
            blocked.append({'scene':sid,'reason':'RAW_GT_DUPLICATE_OR_VERSION_INVALID','gt_audit':scene['raw_gt']});continue
        names=[]
        for raw in scene['raw_gt']:
            v=raw['view'];name=f'{scene["name"]}_View{v}';names.append(name);g=read_mot(raw['path']);pr=[]
            if sha(raw['path'])!=raw['sha256']:raise RuntimeError('original GT changed since audit')
            for row in rows:
                for obj in row.get(box_source,{}).get(str(v),[]):
                    if obj.get('evaluation_eligible',True) and obj.get('visible',True) and obj.get('bbox_xywh') is not None:
                        pr.append([row['timestamp'],obj['global_id'],*obj['bbox_xywh'],obj.get('score',1.),-1,-1,-1])
            validate(g,name+' raw GT');validate(pr,name+' predictions');gt[name]=g;pred[name]=pr;lengths[name]=scene['last_frame']
        groups[scene['name']]=names
    return gt,pred,lengths,groups,blocked,coverage

def trackeval(gt,pred,lengths,output,trackeval_root,split,groups=None,*,authorization=None,method_id=None):
    from ..runtime.experiment_authorization import require_experiment_authorization
    require_experiment_authorization(authorization,'evaluate',split,method_id)
    sys.path.insert(0,str(trackeval_root));import trackeval
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for name,rows in gt.items():write_mot(out/'gt'/name/'gt/gt.txt',rows);write_mot(out/'trackers/WorldMOT/data'/f'{name}.txt',pred[name])
    ec=trackeval.Evaluator.get_default_eval_config();ec.update(USE_PARALLEL=False,PRINT_CONFIG=False,PRINT_RESULTS=False,PLOT_CURVES=False,OUTPUT_FOLDER=str(out/'output'))
    dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config();dc.update(GT_FOLDER=str(out/'gt'),TRACKERS_FOLDER=str(out/'trackers'),TRACKERS_TO_EVAL=['WorldMOT'],BENCHMARK='VisionTrack',SPLIT_TO_EVAL=split,SKIP_SPLIT_FOL=True,SEQ_INFO=lengths,PRINT_CONFIG=False,OUTPUT_FOLDER=str(out/'output'))
    dataset=trackeval.datasets.MotChallenge2DBox(dc)
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]
    result,messages=trackeval.Evaluator(ec).evaluate([dataset],metrics)
    values=result[dataset.get_name()]['WorldMOT']
    if values is None:raise RuntimeError(str(messages))
    def extract(row):
        return {'HOTA':float(np.mean(row['HOTA']['HOTA'])*100),'AssA':float(np.mean(row['HOTA']['AssA'])*100),'DetA':float(np.mean(row['HOTA']['DetA'])*100),'IDF1':float(row['Identity']['IDF1']*100),'MOTA':float(row['CLEAR']['MOTA']*100),'IDSW':int(row['CLEAR']['IDSW']),'Frag':int(row['CLEAR']['Frag']),'CLR_TP':int(row['CLEAR']['CLR_TP']),'CLR_FN':int(row['CLEAR']['CLR_FN']),'CLR_FP':int(row['CLEAR']['CLR_FP'])}
    per_scene={}
    for scene,names in (groups or {}).items():
        combined={metric.get_name():metric.combine_sequences({name:values[name]['pedestrian'][metric.get_name()] for name in names}) for metric in metrics}
        per_scene[scene]=extract(combined)
    return {'combined':extract(values['COMBINED_SEQ']['pedestrian']),'per_scene':per_scene,'per_camera':{name:extract(row['pedestrian']) for name,row in values.items() if name!='COMBINED_SEQ'},'tool_sha256':{str(p.relative_to(trackeval_root)):sha(p) for p in sorted((Path(trackeval_root)/'trackeval').rglob('*.py'))}}

def matlab(gt,pred,groups,out,kit,executable,driver,*,authorization=None,method_id=None,split='test'):
    from ..runtime.experiment_authorization import require_experiment_authorization
    require_experiment_authorization(authorization,'evaluate',split,method_id)
    kit=Path(kit);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    native_sources=native_tracking_sources(kit)
    if not Path(executable).is_file() or not any(p.suffix=='.mexa64' for p in native_sources):raise RuntimeError('Native MATLAB/MEX unavailable')
    jobs=[]
    for scene,names in groups.items():
        names=sorted(names)
        for convention in ['sequential','interleaved']:
            parts={'gt':[],'prediction':[]};offset=0
            for camera,name in enumerate(names):
                length=max([int(r[0]) for r in gt[name]+pred[name]]+[0])+1
                for rows,key in [(gt[name],'gt'),(pred[name],'prediction')]:
                    for r in rows:
                        frame=int(r[0])+offset if convention=='sequential' else int(r[0])*len(names)+camera
                        parts[key].append([frame,*r[1:6],-1,-1,-1,-1])
                offset+=length
            paths={}
            for key,rows in parts.items():paths[key]=out/convention/f'{scene}_{key}.txt';write_mot(paths[key],rows)
            directory=paths['gt'].with_suffix('');directory.mkdir(exist_ok=True)
            total=max([int(r[0]) for rows in parts.values() for r in rows]+[1]);(directory/'seqinfo.ini').write_text(f'[Sequence]\nname={scene}\nimDir=img1\nframeRate=30\nseqLength={total}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n')
            jobs.append({'scene':scene,'convention':convention,'groundtruth':str(paths['gt'].resolve()),'prediction':str(paths['prediction'].resolve()),'gt_directory':str(directory.resolve())})
    config=out/'CONFIG.json';native=out/'NATIVE.json';config.write_text(json.dumps({'kit':str(kit.resolve()),'jobs':jobs,'output':str(native.resolve()),'benchmark':'VisionTrack'}))
    before={str(p.relative_to(kit)):sha(p) for p in native_sources}
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','WORLDMOT_MATLAB_CONFIG':str(config.resolve()),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    env.pop('LD_LIBRARY_PATH',None);env.pop('MKL_THREADING_LAYER',None)
    with (out/'matlab.log').open('w') as f:
        process=subprocess.Popen([str(executable),'-batch',"run('"+str(Path(driver).resolve())+"')"],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            code=process.wait(timeout=1800)
            if code:raise RuntimeError('Native MATLAB failed with exit code '+str(code))
        except subprocess.TimeoutExpired:
            # Only the dedicated process group just created by this adapter.
            if process.poll() is None and os.getpgid(process.pid)==process.pid:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise RuntimeError('Native MATLAB timeout after 1800 seconds; partial jobs retained')
    data=json.loads(native.read_text());after={str(p.relative_to(kit)):sha(p) for p in native_sources}
    if before!=after:raise RuntimeError('Official MATLAB toolkit modified')
    def combine(rows):
        ids=[x['Identity'] for x in rows if x['convention']=='sequential'];clear=[x['CLEAR'] for x in rows if x['convention']=='interleaved']
        ic={k:sum(r[k] for r in ids) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']};cc={k:sum(r[k] for r in clear) for k in ['fn','fp','id_switches','tp']}
        return {'CVIDF1':200*ic['IDTP']/max(1,ic['n_gt']+ic['n_tr']),'CVMA':100*(1-(cc['fn']+cc['fp']+cc['id_switches'])/max(1,ic['n_gt'])),'identity_counts':ic,'CLEAR_counts':cc}
    return {'combined':combine(data['results']),'per_scene':{s:combine([x for x in data['results'] if x['scene']==s]) for s in groups},'MATLAB_version':data['version'],'native_function':data['native_official_function'],'toolkit_sha256':before,'adapter_sha256':sha(driver),'native_report_sha256':sha(native)}
