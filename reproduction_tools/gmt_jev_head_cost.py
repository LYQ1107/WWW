"""Real native-prefix head matrix-operation audit; separate from FPS and training."""
import argparse
from gmt_jev_runtime import *

def main():
 from gtr.modeling.gmt_jev import IdentityCorrection
 values,frames,_=metadata(16);results={}
 for kind in ['mlp','attention','cic']:
  seed();m,_,_=model();actor=IdentityCorrection(kind).cuda().eval();m.gmt_identity_correction=actor
  def committed(**d):actor.committed(d['instances'][-1],d['view'],d['frame'])
  m.jev_candidate_commit_observer=committed
  events=[];current={};handles=[]
  def before(module,args):
   q,k=args[0].shape[:2];current.clear();current.update(rows=q,candidates=k,linear_MACs=0)
  def linear(module,args,out):current['linear_MACs']+=out.numel()*module.in_features
  def after(module,args,out):
   q,k=current['rows'],current['candidates']
   # q@k.T and attention_weights@values; each uses inner dimension32.
   attention_MACs=2*q*k*32 if kind!='mlp' and q and k else 0
   assert not bool(out[1].ne(0).any()),'Prototype must remain zero residual'
   events.append(dict(current,attention_MACs=attention_MACs,
    head_dense_matmul_FLOPs=2*(current['linear_MACs']+attention_MACs),
    feature_cosine_matmul_FLOPs=2*(2*q*k*1024)))
  handles.append(actor.head.register_forward_pre_hook(before))
  for layer in actor.head.modules():
   if isinstance(layer,torch.nn.Linear):handles.append(layer.register_forward_hook(linear))
  handles.append(actor.head.register_forward_hook(after))
  with torch.no_grad():raw,_=run(m,values,frames,63)
  assert len(events)==127
  results[kind]=dict(parameters=sum(p.numel() for p in actor.parameters()),events=events,
   head_dense_matmul_FLOPs_total=sum(e['head_dense_matmul_FLOPs'] for e in events),
   head_dense_matmul_FLOPs_p50_p95_max=list(map(float,np.quantile([e['head_dense_matmul_FLOPs'] for e in events],[.5,.95,1]))),
   feature_cosine_matmul_FLOPs_total=sum(e['feature_cosine_matmul_FLOPs'] for e in events),untrained_zero_residual=True)
  for handle in handles:handle.remove()
  del m,actor,raw;gc.collect();torch.cuda.empty_cache()
 result=dict(status='COMPLETE_REAL_NATIVE_PREFIX_AUDIT',video=16,scene_frames=64,
  no_gradient_updates=True,no_GT_actor_input=True,not_FPS=True,methods=results,
  convention='Two FLOPs per multiply-accumulate; observed nn.Linear input/output sizes plus explicit attention matrix products. Separately counts the two 1024-d cosine matrix products.',
  exclusions=['bias additions','normalization','GELU','softmax','reductions','elementwise residual arithmetic','Python history maintenance','memory transfers'],
  scope='Dense matrix operations only, not total extra module FLOPs or whole-video cost. Native candidate sizes/state observed independently for each zero-residual head.',source=binding())
 save(REPORTS/'HEAD_COST_PREFIX.json',result);print('HEAD_COST_COMPLETE',{k:v['head_dense_matmul_FLOPs_total'] for k,v in results.items()},flush=True)

if __name__=='__main__':main()
