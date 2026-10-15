"""Three same-GPU complete image-to-stock-tracks trials, with no GT observer."""
import argparse,gc,torch,numpy as np
from gmt_jev_runtime import *
def main(kind,gpu):
 from gtr.modeling.gmt_jev import IdentityCorrection
 from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
 from gtr.modeling.jev_native_state import fingerprint
 from run_jev_phase10_closed_loop import raw_predictions
 assert int(os.environ['CUDA_VISIBLE_DEVICES'])==gpu
 folder=OUT/'speed'/kind;folder.mkdir(parents=True,exist_ok=False)
 values,frames,images=metadata(16);seed();m,cfg,_=model();actor=None;checkpoint=None
 if kind!='original':
  if kind=='rule_cached':
   assert read(REPORTS/'FEATURE_CACHE_PARITY.json')['status']=='PASS_EXACT_FULL_NATIVE_RULE'
  if kind.endswith('_learned'):
   from gmt_jev_learned_checkpoint import load_actor
   actor,checkpoint=load_actor(kind.replace('_learned',''))
  else:actor=IdentityCorrection(kind.replace('_zero','').replace('_cached',''),cache_features=kind.endswith('_cached')).cuda().eval()
  m.gmt_identity_correction=actor
  def committed(**d):actor.committed(d['instances'][-1],d['view'],d['frame'])
  m.jev_candidate_commit_observer=committed
 assert m.jev_perception_cache_reader is None
 uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 def clean():
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
  assert not any(r.startswith(uuid+',') and int(r.split(',')[1])!=os.getpid() for r in rows)
 clean();stage=[];start=[None];get=m.get_asso
 def get_asso(*a,**kw):
  if start[0] is None:torch.cuda.synchronize();start[0]=time.perf_counter()
  return get(*a,**kw)
 m.get_asso=get_asso
 def wrap(fn):
  def completed(*a,**kw):
   result=fn(*a,**kw);torch.cuda.synchronize();assert start[0] is not None
   stage.append((time.perf_counter()-start[0])*1000);start[0]=None;return result
  return completed
 m.run_first_tracker_plus=wrap(m.run_first_tracker_plus);m.run_global_tracker_plus=wrap(m.run_global_tracker_plus)
 trials=[];digests=[]
 for repeat in range(3):
  seed()
  if actor:actor.reset()
  with torch.no_grad():warm,_=run(m,values,frames,7)
  del warm;seed();stage.clear();start[0]=None
  if actor:actor.reset()
  clean();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
  with torch.no_grad():raw,_=run(m,values,frames)
  # Unchanged original stock final output processing is inside full FPS.
  stock=m._remove_short_track(raw) if m.min_track_len>0 else raw
  if m.roi_heads.delay_cls:stock=m._delay_cls(stock,video_id=16)
  batch=[values[v*frames+f] for f in range(frames) for v in range(2)]
  stock=CustomRCNN._postprocess(stock,batch,[(0,0)]*len(batch),not_clamp_box=m.not_clamp_box)
  torch.cuda.synchronize();seconds=time.monotonic()-begin;peak=torch.cuda.max_memory_allocated()/2**20;clean()
  assert len(stage)==2*frames-1
  digests.append(fingerprint([r['instances'] for r in stock]))
  trial=dict(repeat=repeat,frames=frames,full_scene_FPS=frames/seconds,seconds=seconds,stage2_p50_p95_ms=list(map(float,np.quantile(stage[16:],[.5,.95]))),peak_VRAM_MiB=peak)
  trials.append(trial);save(folder/'PROGRESS.json',dict(status='FULL_IMAGE_TIMING',trials=trials));print('GMT_LIVE_FPS',kind,trial,flush=True)
  if repeat==0:
   import gzip
   path=folder/'STOCK_PREDICTIONS.json.gz'
   with gzip.open(path,'wt') as f:json.dump(raw_predictions([r['instances'] for r in stock],images),f)
  del raw,stock;gc.collect()
 assert len(set(digests))==1
 result=dict(status='COMPLETE',kind=kind,trials=trials,mean_full_scene_FPS=float(np.mean([r['full_scene_FPS'] for r in trials])),hardware=torch.cuda.get_device_name(),GPU_UUID=uuid,physical_GPU=gpu,
  full_system_parameters=sum(p.numel() for p in m.parameters()),extra_trainable_parameters=sum(p.numel() for p in actor.parameters()) if actor else 0,
  deployed_parameters=sum(p.numel() for p in m.parameters()),no_GTA_or_RPCE_pruning=True,all_native_states_reset_between_trials=True,model_object_reused=True,
  no_GT_or_eval_in_timed_loop=True,no_perception_cache=True,stock_predictions=ref(folder/'STOCK_PREDICTIONS.json.gz'),stock_prediction_SHA=digests[0],
  untrained_prototype=kind.endswith('_zero'),cache_features=bool(actor and actor.cache_features),extra_neural_FLOPs='NOT_MEASURED_YET' if kind.endswith('_zero') else 0,source=binding())
 if checkpoint:
  result.update(trained_checkpoint=checkpoint,risk_mode='learned',training_steps=300,
   extra_neural_FLOPs='Same head operators/parameter shapes as measured zero prototypes; full-video FLOPs not measured')
 save(REPORTS/f'SPEED_{kind}.json',result);print('GMT_SPEED_COMPLETE',kind,result['mean_full_scene_FPS'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['original','threshold','rule','rule_cached','confidence','mlp_zero','attention_zero','cic_zero','mlp_learned','attention_learned','cic_learned'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();main(a.kind,a.gpu)
