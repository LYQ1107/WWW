"""Small real-image component latency diagnostic; explicitly not full FPS."""
import numpy as np
from gmt_jev_runtime import *

def main():
 from gmt_jev_learned_checkpoint import load_actor
 evidence=read(REPORTS/'LEARNED_GATE_RESULTS.json')['methods']['mlp']
 assert evidence['joint_measured_quality_pass']
 seed();m,cfg,_=model();actor,checkpoint=load_actor('mlp');m.gmt_identity_correction=actor
 samples={k:[] for k in ['features','head','committed']};shapes=[]
 def timed(name,fn):
  def call(*args,**kwargs):
   torch.cuda.synchronize();begin=time.perf_counter();out=fn(*args,**kwargs);torch.cuda.synchronize()
   samples[name].append((time.perf_counter()-begin)*1000)
   if name=='features':shapes.append(list(args[0].shape))
   return out
  return call
 actor.features=timed('features',actor.features)
 actor.head.forward=timed('head',actor.head.forward)
 actor.committed=timed('committed',actor.committed)
 def committed(**d):actor.committed(d['instances'][-1],d['view'],d['frame'])
 m.jev_candidate_commit_observer=committed
 values,frames,_=metadata(16);seed()
 with torch.no_grad():raw,_=run(m,values,frames,23)
 assert len(samples['features'])==47 and len(samples['head'])==47
 result=dict(status='COMPLETE_24_SCENE_REAL_IMAGE_COMPONENT_DIAGNOSTIC',video=16,scene_frames=24,
  trained_checkpoint=checkpoint,source=binding(),hardware=torch.cuda.get_device_name(),
  physical_GPU=int(os.environ['CUDA_VISIBLE_DEVICES']),actual_candidate_shapes=shapes,
  component_ms={k:dict(calls=len(v),sum=float(sum(v)),p50_p95_max=list(map(float,np.quantile(v,[.5,.95,1])))) for k,v in samples.items()},
  full_image_FPS=None,GT_actor_input=False,new_gradient_updates=0,no_persistent_corpus=True,
  limitations=['Synchronized component boundaries perturb execution; not whole-video or deployment FPS.',
   'Only 24 real scene frames and their native states; no extrapolated full-video cost claim.',
   'Measures current learned MLP, not all heads. Does not change policy, features, checkpoint or native states.'])
 save(REPORTS/'LEARNED_MLP_COMPONENT_PROFILE.json',result)
 print('GMT_LEARNED_COMPONENT_PROFILE',result['component_ms'],flush=True)

if __name__=='__main__':main()
