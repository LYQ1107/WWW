"""One conditional matched direct-readout pair; same D1 schedule/data/loss."""
import argparse,collections,random,torch
from torch.nn import functional as F
from jev_fastcheck_common import *
def direct_actor(arm):
 from gtr.modeling.jev_fastcheck.model import DirectAssociation
 actor,cp=load_policy(arm);actor.__class__=DirectAssociation
 del actor.plan_value
 return actor,cp
def main(arm):
 from jev_oneweek_common import seed_all
 from jev_recovery_learning import batch
 from jev_fastcheck_train import data,pools,evaluate
 seed_all(SEED)
 gate=read(REPORTS/'P3_ACTIVATION_GATE.json');assert gate['execute_P3'] is True
 assert read(REPORTS/'DIRECT_CONTRACT_TESTS.json')['status']=='PASS'
 train,audit,own,sources=data();combined=train+own;oh,normal=pools(train);sh,_=pools(own)
 keys=[sorted(oh),sorted(sh),sorted(normal)];rng=random.Random(SEED);schedule=[]
 # Match the exact four-slot draw order used in P2 D1.
 for _ in range(1000):
  part=[]
  for slot in range(4):
   if slot==0:k=rng.choice(keys[0]);part.append(('original',rng.choice(oh[k])))
   elif slot==1:k=rng.choice(keys[1]);part.append(('own',rng.choice(sh[k])))
   else:k=rng.choice(keys[2]);part.append(('original',rng.choice(normal[k])))
  schedule.append(part)
 weights={}
 for name in ['cross','stability']:
  pos=sum(int((r['oneweek_labels'][name]==1).sum()) for r in combined);neg=sum(int((r['oneweek_labels'][name]==0).sum()) for r in combined);weights[name]=neg/max(1,pos)
 pos=sum(int(r['positive'][:,:-1].sum()) for r in combined);neg=sum(int((r['known_options'][:,:-1]&~r['positive'][:,:-1]).sum()) for r in combined);weights['compatibility']=neg/max(1,pos)
 actor,cp=direct_actor(arm);folder=OUT/'direct_training'/arm;folder.mkdir(parents=True,exist_ok=False);optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=.0001,weight_decay=.01)
 before=evaluate(actor,audit);groups=set();history=[];digest=hashlib.sha256();begin=time.monotonic()
 for step,indices in enumerate(schedule):
  part=[(train if source=='original' else own)[index] for source,index in indices];assert all(r['key'][0] in [12,13,14] and not r['audit_block'] for r in part)
  digest.update(json.dumps(indices).encode());x,y=batch(part);actor.train();optimizer.zero_grad(set_to_none=True);details=actor.details(x);z=details['edge_logits'];known=y['known_options'].clone();known[:,:,:-1]&=x['legal'];positive=y['positive']&known;valid=x['question_mask']&positive.any(-1)
  lp=F.log_softmax(z.masked_fill(~known,-1e4),-1);mass=torch.logsumexp(lp.masked_fill(~positive,-1e4),-1);ce=(-mass[valid]).mean() if valid.any() else details['features'].sum()*0
  margin=z[:,:,:-1]-z[:,:,-1:];mask=known[:,:,:-1]&x['question_mask'][:,:,None]
  edge=F.binary_cross_entropy_with_logits(margin[mask],positive[:,:,:-1][mask].float(),pos_weight=z.new_tensor(weights['compatibility'])) if mask.any() else ce*0
  aux=[]
  for name,key in [('cross','cross_logits'),('stability','stability_logits')]:
   labels=y[name];valid_aux=labels>=0;aux.append(F.binary_cross_entropy_with_logits(details[key][valid_aux],labels[valid_aux].float(),pos_weight=z.new_tensor(weights[name])) if valid_aux.any() else ce*0)
  value=ce+.1*edge+.25*sum(aux);assert torch.isfinite(value)
  gradient=torch.autograd.grad(ce+.1*edge,actor.edge_value.weight,retain_graph=True,allow_unused=True)[0]
  if gradient is not None and gradient.abs().max()>0:
   for r in part:
    for g in set(r['GT_labels_OFFLINE_ONLY']):
     if g is not None:groups.add((r['key'][0],g,r['key'][2],r['key'][1]//64))
  value.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);optimizer.step()
  if step%100==0 or step==999:
   event=dict(step=step+1,loss=float(value.detach()),direct_choice_CE=float(ce.detach()),groups=len(groups),seconds=time.monotonic()-begin);history.append(event);save(folder/'PROGRESS.json',dict(status='DIRECT_ASSOCIATION_TRAINING',**event));print('DIRECT_TRAIN',arm,event,flush=True)
 final=evaluate(actor,audit);p=folder/'NEW_MODULES.pth';torch.save(dict(new_modules={k:v.cpu() for k,v in actor.state_dict().items() if not k.startswith('identity.')},arm=arm,initializer=cp,updates=1000),p)
 result=dict(status='COMPLETE',arm='direct_'+arm,dataset='D1',steps=1000,phase='P3',checkpoint=ref(p),initializer=cp,training_signal_eligible=len(groups)>=10,local_gradient_groups=len(groups),schedule_SHA=digest.hexdigest(),trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
  before=before,final=final,history=history,one_actual_MATCH_solve=True,all_legal_candidates=True,auxiliary_outputs_not_inference_inputs=True,loss_weights=weights,independent_validation=False,binding=binding('One direct-readout matched architecture repair'),seconds=time.monotonic()-begin)
 save(folder/'RESULT.json',result);save(REPORTS/f'TRAIN_direct_{arm}_D1_STEPS1000.json',result);print('DIRECT_TRAIN_COMPLETE',arm,final,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['jev','attention'],required=True);main(p.parse_args().arm)
