"""Exact complete-GMT source/config/logit/score/commit parity; no GT in actor."""
import argparse,gc,torch
from gmt_jev_runtime import *
def main(video,stop,zero=False):
 from gtr.modeling.jev_native_state import fingerprint
 values,frames,images=metadata(video);cases=[];limit=frames if stop is None else min(frames,stop)
 folder=OUT/'baseline'/f'video{video:02d}_frames{limit}{"_zero_cic" if zero else ""}';folder.mkdir(parents=True,exist_ok=False)
 for original in [True,False]:
  seed();m,cfg,source=model(original);events=[];logits=[];scores=[];rpce=[];tc={};last=[0.];actor=None
  if zero and not original:
   from gtr.modeling.gmt_jev import IdentityCorrection
   actor=IdentityCorrection('cic').cuda().eval();m.gmt_identity_correction=actor
  seed()
  def committed(**d):
   inst=d['instances'][-1];ids=inst.track_ids.cpu().tolist();assert len(ids)==len(set(ids))
   full=d['frame']<32 or d['frame']==limit-1
   event=dict(key=[video,d['frame'],d['view']],ids=ids,id_count=d['id_count'],hits=d['hits'],gallery_lengths={t:len(g) for t,g in d['galleries'].items()},possible_ids=sorted(d['possible_ids']),old_ids=sorted(d['old_ids']))
   # Deep immutable digest now, rather than holding mutable state references.
   event=dict(key=event['key'],native_discrete_state_SHA=fingerprint(event),gallery_bank_tensor_SHA=fingerprint(dict(galleries=d['galleries'],old_reids=d['old_reids']),tensor_cache=tc) if full else None)
   events.append(event)
   if actor is not None:actor.committed(inst,d['view'],d['frame'])
   if time.monotonic()-last[0]>20:
    save(folder/'PROGRESS.json',dict(status='STRICT_COMPLETE_GMT_PARITY',original=original,key=event['key'],limit=limit,frames=frames));print('GMT_BASELINE_PROGRESS',original,event['key'],limit,flush=True);last[0]=time.monotonic()
  m.jev_candidate_commit_observer=committed
  transform=m.roi_heads._forward_transformer
  def transformed(*a,**kw):
   out=transform(*a,**kw);logits.append(fingerprint(out[0]));return out
  m.roi_heads._forward_transformer=transformed
  initial=m._candidate_initial_pairs
  def pairs(x):scores.append(fingerprint(x));return initial(x)
  m._candidate_initial_pairs=pairs
  hook=m.roi_heads.s_t_head.register_forward_hook(lambda mod,args,out:rpce.append(list(out.shape)))
  begin=time.monotonic()
  with torch.no_grad():raw,_=run(m,values,frames,None if stop is None else limit-1)
  hook.remove()
  raw=raw[:limit*2];assert all(x.reid_features.shape[-1]==1152 for x in raw)
  data=dict(original_unmodified_frozen_source=original,source_snapshot=ref(source) if source else None,config=cfg.dump(),config_SHA256=hashlib.sha256(cfg.dump().encode()).hexdigest(),
   full_system_parameters=sum(p.numel() for p in m.parameters()),GTA_calls=len(logits),RPCE_calls=len(rpce),RPCE_shapes=rpce,logits=logits,association_scores=scores,commits=events,raw_prediction_SHA=fingerprint(raw),seconds=time.monotonic()-begin)
  assert logits and rpce
  save(folder/('ORIGINAL.json' if original else 'ADAPTER_OFF.json'),data);cases.append(data)
  if original:
   from run_jev_phase10_closed_loop import raw_predictions
   predictions=raw_predictions(raw,images)
   # Only compact geometry/IDs retained; no images or feature corpus.
   import gzip,json
   with gzip.open(folder/'RAW_PREDICTIONS.json.gz','wt') as f:json.dump(predictions,f)
  del m,raw;gc.collect();torch.cuda.empty_cache()
 a,b=cases;checks={k:a[k]==b[k] for k in ['config_SHA256','logits','association_scores','commits','raw_prediction_SHA']}
 checks['parameter_increment']=b['full_system_parameters']-a['full_system_parameters']==(4962 if zero else 0)
 result=dict(status='PASS_EXACT' if all(checks.values()) else 'FAIL_RETAINED',checks=checks,video=video,scene_frames=limit,complete_video=limit==frames,
  cases=[ref(folder/n) for n in ['ORIGINAL.json','ADAPTER_OFF.json']],predictions=ref(folder/'RAW_PREDICTIONS.json.gz'),checkpoint=ref(STAGE2),config_SHA=a['config_SHA256'],parameters=a['full_system_parameters'],GTA_calls=a['GTA_calls'],RPCE_calls=a['RPCE_calls'],
  source=binding(),GPU_UUID=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),hardware=torch.cuda.get_device_name(),GT_actor_input=False,full_image_FPS=None,scope='Original project source frozen at04324354, complete native GMT, all old JEV modes disabled; parity instrumentation is not FPS',independent_validation=False)
 save(REPORTS/('ZERO_RESIDUAL_NATIVE_PARITY.json' if zero else 'BASELINE_PARITY.json' if stop is None else 'BASELINE_PARITY_PREFIX.json'),result)
 print('GMT_BASELINE_PARITY',checks,flush=True);assert all(checks.values()),checks
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,default=16);p.add_argument('--frames',type=int);p.add_argument('--zero',action='store_true');a=p.parse_args();main(a.video,a.frames,a.zero)
