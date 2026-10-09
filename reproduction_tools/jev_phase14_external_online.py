"""Frozen external controllers, strict native commits before any identity evaluation."""
import argparse
import collections
import gzip
import time
import numpy as np
import torch
from jev_phase14_common import *
from jev_phase13_runtime import build_tracker,run
from jev_phase14_online import load_policy
from evaluate_jev_stage2_online import load_policy as load_historical_policy
from jev_phase14_external_prepare import EXTERNAL,SCENE,inputs
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from run_jev_phase10_closed_loop import raw_predictions
from gtr.modeling.jev_native_state import fingerprint

class EmptySafePolicy(torch.nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def forward(self,x):
        if x['detection_visual'].shape[1]==0:return x['detection_visual'].new_empty(1,0,x['identity_mask'].shape[1]+1)
        return self.net(x)

def evaluate_predictions(prediction_path,clipped=False):
    sys.path.insert(0,str(ROOT/'TrackEval'))
    for name,typ in [('float',float),('int',int),('bool',bool)]:
        if name not in np.__dict__:setattr(np,name,typ)
    import trackeval
    annotation=read(EXTERNAL/('ANNOTATIONS_CLIPPED.json' if clipped else 'ANNOTATIONS_RAW.json'));images={i['id']:i for i in annotation['images']}
    out=prediction_path.parent/('eval_clippedGT' if clipped else 'eval_rawGT');gt=out/'gt';trackers=out/'trackers';(trackers/'GMT/data').mkdir(parents=True,exist_ok=True)
    gts=collections.defaultdict(list);preds=collections.defaultdict(list)
    for a in annotation['annotations']:
        i=images[a['image_id']];view=i['view_id'];gts[view].append([i['frame_id'],a['instance_id'],*a['bbox'],1,1,1])
    for p in read(prediction_path):
        i=images[p['image_id']];preds[i['view_id']].append([i['frame_id'],p['track_id'],*p['bbox'],p['score'],-1,-1,-1])
    sequences=['WILDTRACK_C1','WILDTRACK_C2'];lengths={s:320 for s in sequences}
    for view,seq in enumerate(sequences,1):
        folder=gt/seq/'gt';folder.mkdir(parents=True,exist_ok=True);np.savetxt(folder/'gt.txt',np.asarray(gts[view]).reshape(-1,9),delimiter=',',fmt='%.8f')
        np.savetxt(trackers/'GMT/data'/(seq+'.txt'),np.asarray(preds[view]).reshape(-1,10),delimiter=',',fmt='%.8f')
        (folder.parent/'seqinfo.ini').write_text('[Sequence]\nname='+seq+'\nimDir=img1\nframeRate=2\nseqLength=320\nimWidth=1920\nimHeight=1080\nimExt=.png\n')
    manifest=out/'manifest.json';save(manifest,dict(sequences=sequences,scenes=['WILDTRACK'],seq_lengths=lengths,trackeval_gt=str(gt),trackeval_trackers=str(trackers),dataset='WILDTRACK',custom_prefix=True))
    cfg=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config();cfg.update(GT_FOLDER=str(gt),TRACKERS_FOLDER=str(trackers),TRACKERS_TO_EVAL=['GMT'],BENCHMARK='WILDTRACK',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,SEQ_INFO=lengths,PRINT_CONFIG=False)
    e=trackeval.Evaluator(dict(USE_PARALLEL=False,PRINT_RESULTS=False,PRINT_CONFIG=False,PLOT_CURVES=False,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True))
    results,messages=e.evaluate([trackeval.datasets.MotChallenge2DBox(cfg)],[trackeval.metrics.HOTA(dict(PRINT_CONFIG=False)),trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False)),trackeval.metrics.Identity(dict(PRINT_CONFIG=False))])
    result=results['MotChallenge2DBox']['GMT']['COMBINED_SEQ']['pedestrian'];h=result['HOTA'];c=result['CLEAR'];identity=result['Identity']
    summary=dict(HOTA=float(np.mean(h['HOTA'])*100),AssA=float(np.mean(h['AssA'])*100),DetA=float(np.mean(h['DetA'])*100),IDF1=float(identity['IDF1']*100),MOTA=float(c['MOTA']*100),IDSW=int(c['IDSW']),Frag=int(c['Frag']))
    save(out/'RESULT.json',dict(status='COMPLETE',binding=binding(),metrics=summary,manifest=dict(path=str(manifest),SHA256=sha(manifest)),predictions_SHA256=sha(prediction_path),GT_convention='clipped sensitivity' if clipped else 'primary raw provided projection coordinates',all_two_cameras_pooled=True,per_video_mean_used_as_pooled=False));return summary,manifest

def official(manifest,out):
    from evaluate_jev_phase13_official_matlab import MATLAB,KIT,prepare
    # The established converter is shared verbatim except its development-only
    # fixed6sequence assertion, which does not apply to this2camera1scene prefix.
    import inspect
    code=inspect.getsource(prepare).replace("    assert len(manifest['sequences']) == 6 and len(manifest['scenes']) == 3\n",'').replace("    assert sorted(manifest['seq_lengths'].values()) == sorted([1200,1200,1029,1029,1052,1052])\n",'')
    namespace=dict(json=__import__('json'),Path=Path,np=np,rows=lambda p:np.loadtxt(p,delimiter=',',ndmin=2) if Path(p).stat().st_size else np.empty((0,10)),reference=lambda p:dict(path=str(p),SHA256=sha(p)))
    exec(code,namespace);jobs,sources=namespace['prepare'](manifest,out);native=out/'NATIVE.json';config=out/'CONFIG.json';log=out/'matlab.log'
    save(config,dict(kit=str(KIT),jobs=jobs,output=str(native),benchmark='VisionTrack'))
    env=os.environ.copy();env.update(JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');script=ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m'
    with log.open('w') as h:subprocess.run([str(MATLAB),'-batch',"run('"+str(script)+"')"],env=env,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,check=True)
    values=read(native);assert values['status']=='COMPLETE';identity=[r['Identity'] for r in values['results'] if r['convention']=='sequential'];clear=[r['CLEAR'] for r in values['results'] if r['convention']=='interleaved'];counts={k:sum(v[k] for v in identity) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']};errors={k:sum(v[k] for v in clear) for k in ['fn','fp','id_switches','tp']}
    return dict(CVIDF1=200*counts['IDTP']/(counts['n_gt']+counts['n_tr']),CVMA=100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']),Identity_counts=counts,interleaved_CLEAR_counts=errors,native_report=dict(path=str(native),SHA256=sha(native)),source_formatting_function_SHA256=sha(ROOT/'reproduction_tools/evaluate_jev_phase13_official_matlab.py'),unchanged_official_metric_functions=True,native_version=values['version'],scope='official MATLAB metric application to custom fixed WILDTRACK C1/C2 prefix; not an official full7camera benchmark')

def main(variant,seed,historical=False):
    protect();source=binding();torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert read(EXTERNAL/'stage1_cache_v1/RESULT.json')['status']=='COMPLETE';assert read(REPORTS/'EXTERNAL_EVALUATION_FORMAT.json')['status']=='FROZEN_BEFORE_EXTERNAL_TRACKING_OUTCOMES'
    policy,trained,temp=(load_historical_policy(variant,seed,'formal') if historical else load_policy(variant,seed,'formal'))
    if historical and policy is not None:policy=EmptySafePolicy(policy)
    name=('phase13_' if historical else 'phase14_')+f'{variant}_seed{seed}';out=EXTERNAL/'native_online_v1'/name;out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    values,frames,images=inputs();model=build_tracker(17,policy=policy,variant='full',temperature=(temp,1.),react_learned=False);model.jev_perception_cache_reader=FrozenPerceptionCache(EXTERNAL/'stage1_cache_v1');ex=model.jev_stage2_executor;ex.memory_factory=CachedIdentityMemory;counts=collections.Counter();begin=time.monotonic()
    with gzip.open(out/'COMMITS.jsonl.gz','wt') as commits:
        def after(**d):
            current=d['instances'][-1];refs=current.track_ids.cpu().tolist();assert len(refs)==len(set(refs));counts['payloads']+=1
            commits.write(__import__('json').dumps(dict(key=[SCENE,d['frame'],d['view']],ids=refs,id_count=d['id_count'],events=d['events']))+'\n')
            if counts['payloads']%64==0:save(out/'PROGRESS.json',dict(status='ACTUAL_NATIVE_RUNNING',last_key=[d['frame'],d['view']],payloads=counts['payloads'],total=639))
        ex.commit_observer=after
        with torch.no_grad():raw,_=run(model,values,frames)
    assert counts['payloads']==639;predictions=raw_predictions(raw,images);save(out/'RAW_PREDICTIONS.json',predictions)
    save(out/'PREDICTIONS_FROZEN.json',dict(status='FROZEN_BEFORE_GT_EVALUATION',binding=source,trained=trained,prediction_SHA256=sha(out/'RAW_PREDICTIONS.json'),commits_SHA256=sha(out/'COMMITS.jsonl.gz'),native_GT_actor_inputs=False))
    primary,manifest=evaluate_predictions(out/'RAW_PREDICTIONS.json');clipped,_=evaluate_predictions(out/'RAW_PREDICTIONS.json',True);cross=official(manifest,out/'official_matlab')
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,variant=variant,seed=seed,historical=historical,trained=trained,temperature=temp,scene_id=SCENE,actual_mutated_state_online=True,GT_actor_inputs=False,GTA_throw=True,
        metrics=primary,clipped_GT_sensitivity=clipped,official_MATLAB=cross,predictions=dict(path=str(out/'RAW_PREDICTIONS.json'),SHA256=sha(out/'RAW_PREDICTIONS.json')),commits_SHA256=sha(out/'COMMITS.jsonl.gz'),
        frontend_manifest_SHA256=sha(EXTERNAL/'stage1_cache_v1/RESULT.json'),data_manifest_SHA256=sha(EXTERNAL/'DATA_MANIFEST.json'),counts=dict(counts),seconds=time.monotonic()-begin,
        independent_scope='known VisionTrack training-list external scene transfer; initializer absolute exposure unknown; fixed320 observed2FPS samples, not complete7camera benchmark'))
    print('PHASE14_EXTERNAL_NATIVE_COMPLETE',name,primary,cross['CVIDF1'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--historical',action='store_true');a=p.parse_args();main(a.variant,a.seed,a.historical)
