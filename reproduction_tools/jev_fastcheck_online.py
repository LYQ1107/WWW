"""Complete cold-start native TRAIN16 validation and separate live-image timing."""
import argparse,collections,gzip,gc,torch,numpy as np
from jev_fastcheck_common import *
def actor_for(tag,steps):
 if tag.startswith('direct_'):
  from jev_fastcheck_direct_train import direct_actor
  arm=tag.split('_')[1];actor,cp=direct_actor(arm);m=read(REPORTS/f'TRAIN_direct_{arm}_D1_STEPS1000.json')
  assert m['training_signal_eligible'];cp=m['checkpoint'];assert sha(cp['path'])==cp['SHA256']
  missing,unexpected=actor.load_state_dict(torch.load(cp['path'],map_location='cuda:0')['new_modules'],strict=False)
  assert not unexpected and all(k.startswith('identity.') for k in missing)
  return actor.eval(),cp
 if tag in ['parent','threshold','mlp','attention_initial','jev_initial']:
  arm=tag.replace('_initial','');return load_policy('jev',False) if arm=='parent' else load_policy(arm)
 arm,dataset=tag.split('_');actor,cp=load_policy(arm)
 m=read(REPORTS/f'TRAIN_{arm}_{dataset}_STEPS{steps}.json');assert m['training_signal_eligible'];cp=m['checkpoint'];assert sha(cp['path'])==cp['SHA256']
 stored=torch.load(cp['path'],map_location='cuda:0');missing,unexpected=actor.load_state_dict(stored['new_modules'],strict=False)
 assert not unexpected and all(k.startswith('identity.') for k in missing)
 return actor.eval(),cp
def official_cv(result,folder):
 from jev_oneweek_matlab import prepare
 metric=pathlib.Path(result['metrics_source']['path']);manifest=metric.parent.parent/'prepared/manifest.json'
 jobs,inputs=prepare(manifest,folder/'cv_inputs')
 toolkit=pathlib.Path('/home/liuyeqiang/WWW_jev_phase14/MOTChallengeEvalKit_cv_test/matlab_devkit')
 source=read(ROOT/'reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json')['cases'][0]['result'];prior=read(source['path'])['official_toolkit_SHA256']
 for name,digest in prior.items():assert sha(toolkit/name)==digest,name
 config=folder/'CV_CONFIG.json';output=folder/'CV_NATIVE.json';log=folder/'CV_MATLAB.log'
 save(config,dict(kit=str(toolkit),jobs=jobs,output=str(output),benchmark='VisionTrack'))
 env=dict(os.environ,JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with log.open('w') as f:subprocess.run(['/data1/liuyeqiang/MATLAB/R2020a/bin/matlab','-batch',"run('"+str(ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m')+"')"],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 native=read(output);assert native['status']=='COMPLETE'
 identities=[r['Identity'] for r in native['results'] if r['convention']=='sequential'];clear=[r['CLEAR'] for r in native['results'] if r['convention']=='interleaved']
 counts={k:sum(r[k] for r in identities) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']};errors={k:sum(r[k] for r in clear) for k in ['fn','fp','id_switches','tp']}
 return dict(CVIDF1=200*counts['IDTP']/(counts['n_gt']+counts['n_tr']),CVMA=100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']),CV_counts=counts,CV_errors=errors,source=ref(output),converter_inputs=inputs,toolkit_hash_matches_frozen=True)
def main(tag,steps,mode):
 from jev_oneweek_common import seed_all
 from jev_phase13_runtime import build_tracker,cache_inputs,run
 from gtr.modeling.jev_recovery.native import attach_jevtrack
 from jev_phase14_latency import prune
 from gtr.modeling.jev_native_state import fingerprint
 video=16;seed_all(SEED);values,frames,reader=cache_inputs(video);folder=OUT/'online'/tag/f'steps{steps}'/mode;folder.mkdir(parents=True,exist_ok=False)
 actor,cp=actor_for(tag,steps);model=build_tracker(video,actor,react_learned=False,live=mode=='live')
 original_count=sum(p.numel() for p in model.parameters());pruned=prune(model);executor=attach_jevtrack(model)
 parameters=dict(original_instantiated_system=original_count,new_controller=sum(p.numel() for p in actor.parameters() if not getattr(p,'frozen',False)),new_controller_trainable=sum(p.numel() for p in actor.parameters() if p.requires_grad),deployed_with_controller=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in actor.parameters()),dormant_pruned=pruned)
 if mode=='live':
  assert model.jev_perception_cache_reader is None
  uuid=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();trials=[];digests=[]
  for repeat in range(3):
   with torch.no_grad():warm,_=run(model,values,frames,stop=7)
   del warm;seed_all(SEED);torch.cuda.synchronize();start=time.monotonic()
   before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   assert not any(line.startswith(uuid+',') and int(line.split(',')[1])!=os.getpid() for line in before.splitlines())
   with torch.no_grad():raw,_=run(model,values,frames)
   torch.cuda.synchronize();seconds=time.monotonic()-start
   after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   assert not any(line.startswith(uuid+',') and int(line.split(',')[1])!=os.getpid() for line in after.splitlines())
   digests.append(fingerprint(raw[:frames*2]));latency=[r['total_stage2_ms'] for r in executor.latency[16:]]
   trials.append(dict(repeat=repeat,full_scene_FPS=frames/seconds,seconds=seconds,scene_frames=frames,stage2_p50_p95_ms=list(map(float,np.quantile(latency,[.5,.95])))))
   save(folder/'PROGRESS.json',dict(status='FULL_IMAGE_TIMING',trials=trials));print('FASTCHECK_LIVE_FPS',tag,repeat,trials[-1],flush=True);del raw
  assert len(set(digests))==1
  result=dict(status='COMPLETE',tag=tag,checkpoint=cp,video=video,trials=trials,parameters=parameters,hardware=torch.cuda.get_device_name(),GPU_UUID=uuid,
   source=binding('Three full TRAIN16 live-image native restarts'),mean_full_scene_FPS=float(np.mean([r['full_scene_FPS'] for r in trials])),native_prediction_SHA=digests[0],independent_validation=False)
  save(folder/'RESULT.json',result);save(REPORTS/f'LIVE_{tag}_STEPS{steps}.json',result);return
 from jev_fastcheck_rollout import LabelState
 state=LabelState(video,reader);counts=collections.Counter();trace=[];pending={};last=[0.];journal=gzip.open(folder/'COMMITS.jsonl.gz','wt');begin=time.monotonic()
 def before(**d):
  c=d['context'];key=(c['frame'],c['view'])
  if c.get('first'):state.boot(0,1-c['view'])
  if d['task']!=0 or not len(d['logits']):return
  r=state.record(d,actor.runtime_details);pending[key]=r;k=len(r['refs'])
  for row,col in enumerate(r['native_choice']):
   a=r.get('base_choice',r['plans'][0])[row];a=k if a<0 else a;b=k if col<0 else col;counts['MATCH_rows']+=1
   has=bool(r['positive'][row,:k].any());counts['certified_active_covered']+=has;counts['correct_identity_retained']+=int(has and r['positive'][row,b])
   counts['correct_active_outside_plans']+=int(has and not any(p[row]>=0 and r['positive'][row,p[row]] for p in r['plans']))
   if a!=b:
    counts['changed']+=1
    if r['known_options'][row,a] and r['known_options'][row,b]:counts['N01']+=int(not r['positive'][row,a] and r['positive'][row,b]);counts['N10']+=int(r['positive'][row,a] and not r['positive'][row,b])
    else:counts['UNKNOWN_changed']+=1
 def after(**d):
  frame,view=d['frame'],d['view'];ids=d['instances'][-1].track_ids.cpu().tolist();assert len(ids)==len(set(ids))
  if d['first']:state.boot(0,1-view)
  gt=state.labels.current(frame,view)
  for row,(ref_id,g) in enumerate(zip(ids,gt)):
   wrong=g is not None and ref_id in state.owner and state.owner[ref_id]!=g
   trace.append(dict(frame=frame,camera=view,row=row,gt=g,wrong=wrong,known=g is not None and ref_id in state.owner))
  journal.write(json.dumps(dict(key=[video,frame,view],ids=ids,events=d['events']))+'\n');pending.pop((frame,view),None);state.update(frame,view,ids)
  if time.monotonic()-last[0]>=15:
   save(folder/'PROGRESS.json',dict(status='COMPLETE_VIDEO_NATIVE_TRACKING',frame=frame,frames=frames,counts=dict(counts),seconds=time.monotonic()-begin));journal.flush();print('FASTCHECK_ONLINE_PROGRESS',tag,frame,frames,dict(counts),flush=True);last[0]=time.monotonic()
 executor.observer=before;executor.commit_observer=after
 try:
  with torch.no_grad():raw,_=run(model,values,frames)
 finally:journal.close()
 from run_jev_phase10_closed_loop import raw_predictions,metrics
 predictions=raw_predictions(raw[:frames*2],state.labels.images);save(folder/'RAW_PREDICTIONS.json',predictions)
 score,evaluated=metrics(folder/'RAW_PREDICTIONS.json',[16],folder/'strict_eval')
 from jev_phase7_offline import IdentityEvaluator
 e=IdentityEvaluator(video);identity=e.summarize(e.align(predictions));save(folder/'IDENTITY.json',identity)
 result=dict(status='COMPLETE',tag=tag,steps=steps,video=video,frames=frames,checkpoint=cp,raw_online_metrics=score,
  metrics_source=ref(evaluated/'metrics.json'),raw_predictions=ref(folder/'RAW_PREDICTIONS.json'),counts=dict(counts),
  identity_summary={k:v for k,v in identity.items() if k not in ['wrong_ID_episodes','identity_index_query']},
  correct_identity_preservation=counts['correct_identity_retained']/max(1,counts['certified_active_covered']),
  UNKNOWN_change_fraction=counts['UNKNOWN_changed']/max(1,counts['changed']),
  certified_change_scope='individual changed-row first-past certificate; not fully certified whole-plan causal attribution; changed UNKNOWN counterparts are counted separately',
  certified_first_past_owner_wrong_observations=sum(t['wrong'] for t in trace),
  parameters=parameters,actual_mutated_state=True,GT_actor_inputs=False,full_image_FPS=None,independent_validation=False,binding=binding('Full cold-start gradient-free TRAIN16 validation'),seconds=time.monotonic()-begin)
 cv=official_cv(result,folder);result.update(cv)
 save(folder/'RESULT.json',result);save(REPORTS/f'ONLINE_{tag}_STEPS{steps}.json',result)
 print('FASTCHECK_ONLINE_COMPLETE',tag,score,cv['CVIDF1'],dict(counts),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--mode',choices=['cached','live'],default='cached');a=p.parse_args();main(a.tag,a.steps,a.mode)
