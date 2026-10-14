"""Frozen TrackEval and official MATLAB; only own reproducible temporaries removed."""
import argparse,gzip,numpy as np
from gmt_jev_common import *
def evaluate(predictions,folder):
 from run_jev_phase10_closed_loop import metrics
 from jev_oneweek_matlab import prepare
 folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=False)
 scratch=folder/'REGENERABLE_EVAL';scratch.mkdir();p=scratch/'RAW.json';p.write_text(json.dumps(predictions))
 scores,evaluated=metrics(p,[16],scratch/'TrackEval')
 official_metric=folder/'TRACKEVAL_NATIVE.json';official_metric.write_bytes((evaluated/'metrics.json').read_bytes())
 jobs,inputs=prepare(evaluated.parent/'prepared/manifest.json',scratch/'cv_inputs')
 kit=pathlib.Path('/home/liuyeqiang/WWW_jev_phase14/MOTChallengeEvalKit_cv_test/matlab_devkit')
 old=read(ROOT/'reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json')['cases'][0]['result'];expected=read(old['path'])['official_toolkit_SHA256']
 for name,digest in expected.items():assert sha(kit/name)==digest,name
 config=scratch/'CV_CONFIG.json';native=folder/'MATLAB_NATIVE.json';log=folder/'MATLAB.log'
 save(config,dict(kit=str(kit),jobs=jobs,output=str(native),benchmark='VisionTrack'))
 env=dict(os.environ,JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with log.open('w') as f:subprocess.run(['/data1/liuyeqiang/MATLAB/R2020a/bin/matlab','-batch',"run('"+str(ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m')+"')"],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 d=read(native);assert d['status']=='COMPLETE'
 identity=[r['Identity'] for r in d['results'] if r['convention']=='sequential'];clear=[r['CLEAR'] for r in d['results'] if r['convention']=='interleaved']
 cv_counts={k:sum(r[k] for r in identity) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']};cv_error={k:sum(r[k] for r in clear) for k in ['fn','fp','id_switches','tp']}
 result=dict(metrics=scores,CVIDF1=200*cv_counts['IDTP']/(cv_counts['n_gt']+cv_counts['n_tr']),CVMA=100*(1-(cv_error['fn']+cv_error['fp']+cv_error['id_switches'])/cv_counts['n_gt']),
  CV_counts=cv_counts,CV_errors=cv_error,TrackEval=ref(official_metric),MATLAB=ref(native),unchanged_official_toolkit=True,
  converter_input_SHA_before_regenerable_cleanup=inputs,temporary_inputs_regenerable_from_retained_predictions=True)
 save(folder/'RESULT.json',result)
 # Generated annotation copies/MOT text are temporary, never old evidence.
 assert scratch.resolve().is_relative_to(OUT.resolve());shutil.rmtree(scratch)
 return result
def duration(predictions):
 from jev_phase7_offline import IdentityEvaluator
 import collections
 e=IdentityEvaluator(16);rows=e.align(predictions);mapping,counts=e.mapping(rows);summary=collections.Counter();timelines=collections.defaultdict(dict)
 ids_by_gt=collections.defaultdict(set);persons_by_id=collections.defaultdict(set)
 for (f,v,r),d in sorted(rows.items()):
  g=d['gt']
  if g is None:summary['UNKNOWN_GT_observations']+=1;continue
  ids_by_gt[g].add(d['id']);persons_by_id[d['id']].add(g)
  bad=mapping.get(d['id'])!=g;unmapped=d['id'] not in mapping
  summary['known_GT_observations']+=1;summary['unmapped_fragment_observations']+=unmapped;summary['mapped_wrong_person_observations']+=bad and not unmapped
  timelines[g,v][f]=bad or timelines[g,v].get(f,False)
 episodes=[]
 for key,line in timelines.items():
  first=None;last=None
  for f,bad in sorted(line.items()):
   if first is not None and (not bad or f!=last+1):episodes.append(last-first+1);first=None
   if bad and first is None:first=f
   last=f
  if first is not None:episodes.append(last-first+1)
 summary['extra_native_IDs_per_person']=sum(max(0,len(s)-1) for s in ids_by_gt.values());summary['mixed_person_native_IDs']=sum(len(s)>1 for s in persons_by_id.values())
 return dict(counts=dict(summary),identity_inconsistency_duration_camera_frames=sum(episodes),episode_p50_p95_max=list(map(float,np.quantile(episodes,[.5,.95,1]))) if episodes else [0.,0.,0.],
  definition='Fixed complete-video one-to-one native-ID/person map from unchanged IoU>=.5 alignment; consecutive observed person/camera inconsistency includes wrong-person IDs and unmapped fragments. Descriptive retrospective time, not an online owner/future-safety or causal claim. Not seconds.')
def main():
 p=read(REPORTS/'BASELINE_PARITY.json')['predictions'];assert sha(p['path'])==p['SHA256']
 with gzip.open(p['path'],'rt') as f:pred=json.load(f)
 folder=OUT/'metrics'/'GMT_raw';r=evaluate(pred,folder);r.update(status='COMPLETE_NATIVE_RAW_ORIGINAL_GMT',predictions=p,identity=duration(pred),source=binding(),independent_validation=False,full_image_FPS=None)
 save(REPORTS/'ORIGINAL_GMT_METRICS.json',r);print('GMT_BASELINE_METRICS',r['metrics'],r['CVIDF1'],r['identity']['counts'],flush=True)
if __name__=='__main__':main()
