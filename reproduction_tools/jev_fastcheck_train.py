"""Fixed-schedule 2x2 D0/D1 adaptation; architecture unchanged in P2."""
import argparse,collections,math,random,torch
from jev_fastcheck_common import *
def data():
 from jev_fastcheck_distribution import records
 learning=configure_paths();train,audit,sources=learning.load_records(3)
 own=[]
 for v in [12,13,14]:
  result=read(REPORTS/f'ROLLOUT_self_VIDEO{v:02d}.json')
  own.extend(r for r in records(result) if not r['audit_block'] and r['rows'])
  sources.extend(result['record_chunks'])
 return train,audit,own,sources
def pools(records):
 hard=collections.defaultdict(list);protect=collections.defaultdict(list)
 for i,r in enumerate(records):
  video,frame,view=r['key'];people=tuple(sorted(g for g in set(r['GT_labels_OFFLINE_ONLY']) if g is not None))
  key=(video,people,view,frame//64)
  # Exactly the same GT-defined pools for both architectures. The normal half
  # remains broad enough to protect already-correct actions, not error-only.
  protect[key].append(i)
  known=r['known_options'][:,:-1]&r['inputs']['legal'][0]
  positive=r['positive'][:,:-1]&known
  margin=r['inputs']['pair_evidence'][0,:,:,:4].max(-1).values
  difficult=positive.any(-1)&((known&~positive)&(margin>=.75)).any(-1)
  if difficult.any() or r['oneweek_labels'].get('hard_rows',torch.zeros(1,dtype=torch.bool)).any():hard[key].append(i)
 if not hard:hard=protect
 return hard,protect
@torch.no_grad()
def evaluate(model,records,limit=128):
 from jev_recovery_learning import batch
 from jev_oneweek_learning import comparisons
 model.eval();counts=collections.Counter()
 chosen=[records[i] for i in torch.linspace(0,len(records)-1,min(limit,len(records))).long().tolist()]
 for start in range(0,len(chosen),4):
  part=chosen[start:start+4];x,y=batch(part);detail=model.details(x);pairs=comparisons(detail,y,part)
  for i,a,b,sign,n in pairs:
   counts['rank_pairs']+=1;counts['rank_pair_correct']+=int(sign*float(detail['plan_scores'][i][a]-detail['plan_scores'][i][b])>0)
  for i,r in enumerate(part):
   k=len(r['refs']);plans=detail['plans'][i];choice=plans[int(detail['plan_scores'][i].argmax())];base=plans[0]
   if detail.get('direct_readout'):
    from gtr.modeling.jev_stage2.assignment import lawful_choice
    q=len(r['rows']);base=lawful_choice(torch.cat([detail['base_logits'][i,:q,:k],detail['base_logits'][i,:q,-1:]],-1),x['legal'][i,:q,:k])
   for row,(old,new) in enumerate(zip(base,choice)):
    a=k if old<0 else old;b=k if new<0 else new;counts['rows']+=1
    counts['known_selected']+=int(r['known_options'][row,b]);counts['correct_selected']+=int(r['positive'][row,b]);counts['UNKNOWN_selected']+=int(not r['known_options'][row,b])
    if a!=b:
     counts['changed_rows']+=1
     if r['known_options'][row,a] and r['known_options'][row,b]:
      counts['N01']+=int(not r['positive'][row,a] and r['positive'][row,b]);counts['N10']+=int(r['positive'][row,a] and not r['positive'][row,b])
     else:counts['UNKNOWN_changed']+=1
 return dict(counts=dict(counts),net_certified_N01_minus_N10=counts['N01']-counts['N10'],rank_accuracy=counts['rank_pair_correct']/counts['rank_pairs'] if counts['rank_pairs'] else None,scope='deterministic pre-exposed TRAIN16/source-state diagnostic; not full native or independent')
def main(arm,dataset,steps):
 from jev_oneweek_common import seed_all
 from jev_recovery_learning import batch
 from jev_oneweek_learning import loss
 seed_all(SEED);train,audit,own,sources=data();original=list(train);combined=original+own
 original_hard,normal=pools(original);self_hard,_=pools(own)
 groups=[sorted(original_hard),sorted(self_hard),sorted(normal)];rng=random.Random(SEED)
 schedule=[]
 for step in range(1000):
  part=[]
  for slot in range(4):
   if slot<2:
    if dataset=='D1' and slot==1:key=rng.choice(groups[1]);part.append(('own',rng.choice(self_hard[key])))
    else:key=rng.choice(groups[0]);part.append(('original',rng.choice(original_hard[key])))
   else:key=rng.choice(groups[2]);part.append(('original',rng.choice(normal[key])))
  schedule.append(part)
 weights={}
 for name in ['cross','stability']:
  pos=sum(int((r['oneweek_labels'][name]==1).sum()) for r in combined);neg=sum(int((r['oneweek_labels'][name]==0).sum()) for r in combined);weights[name]=neg/max(1,pos)
 pos=sum(int(r['positive'][:,:-1].sum()) for r in combined);neg=sum(int((r['known_options'][:,:-1]&~r['positive'][:,:-1]).sum()) for r in combined);weights['compatibility']=neg/max(1,pos)
 actor,cp=load_policy(arm);folder=OUT/'training'/f'{arm}_{dataset}'/f'steps{steps}';folder.mkdir(parents=True,exist_ok=False)
 optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=.0001,weight_decay=.01);start=0
 if steps==1000:
  pilot=read(folder.parent/'steps300/RESULT.json');assert pilot['training_signal_eligible'];p=pilot['checkpoint'];assert sha(p['path'])==p['SHA256'];stored=torch.load(p['path'],map_location='cuda:0');actor.load_state_dict(stored['new_modules'],strict=False);optimizer.load_state_dict(stored['optimizer']);start=300
 before=evaluate(actor,audit);history=[];rank_batches=0;gradient_groups=set();digest=hashlib.sha256();begin=time.monotonic()
 for step in range(start,steps):
  part=[(original if corpus=='original' else own)[index] for corpus,index in schedule[step]]
  assert all(r['key'][0] in [12,13,14] and not r['audit_block'] for r in part)
  digest.update(json.dumps(schedule[step]).encode());x,y=batch(part);actor.train();optimizer.zero_grad(set_to_none=True);details=actor.details(x)
  value,terms,pairs=loss(details,x,y,part,weights);assert torch.isfinite(value)
  if pairs:
   ranking=torch.stack([torch.nn.functional.softplus(-sign*(details['plan_scores'][i][a]-details['plan_scores'][i][b])/math.sqrt(n)) for i,a,b,sign,n in pairs]).mean()
   grad=torch.autograd.grad(ranking,actor.edge_value.weight,retain_graph=True,allow_unused=True)[0]
   if grad is not None and bool(grad.abs().max()>0):
    rank_batches+=1
    for i,a,b,sign,n in pairs:
     r=part[i];p=details['plans'][i];changed=[row for row,(u,v) in enumerate(zip(p[a],p[b])) if u!=v];people=tuple(sorted({r['GT_labels_OFFLINE_ONLY'][row] for row in changed if r['GT_labels_OFFLINE_ONLY'][row] is not None}));ids=tuple(sorted({r['refs'][c] for pp in [p[a],p[b]] for row,c in enumerate(pp) if row in changed and c>=0}));gradient_groups.add((r['key'][0],people,ids,r['key'][2],r['key'][1]//64))
  value.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);optimizer.step()
  if (step+1)%50==0 or step==start:
   event=dict(step=step+1,loss=float(value.detach()),terms=terms,rank_batches=rank_batches,gradient_groups=len(gradient_groups),seconds=time.monotonic()-begin);history.append(event);save(folder/'PROGRESS.json',dict(status='TRAINING',**event));print('FASTCHECK_TRAIN',arm,dataset,event,flush=True)
 final=evaluate(actor,audit);checkpoint=folder/'NEW_MODULES.pth'
 torch.save(dict(new_modules={k:v.cpu() for k,v in actor.state_dict().items() if not k.startswith('identity.')},optimizer=optimizer.state_dict(),updates=steps,arm=arm,initializer=cp),checkpoint)
 result=dict(status='COMPLETE',arm=arm,dataset=dataset,steps=steps,actual_new_updates=steps-start,initializer=cp,checkpoint=ref(checkpoint),
  original_events=len(original),self_events=len(own),source=sources,schedule_SHA=digest.hexdigest(),
  common_loss_weights=weights,rank_gradient_batches=rank_batches,local_gradient_groups=len(gradient_groups),
  training_signal_eligible=rank_batches>0 and len(gradient_groups)>=10,before=before,final=final,history=history,
  trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),new_architecture=False,GT_actor_input=False,independent_validation=False,seconds=time.monotonic()-begin,binding=binding('Equal-schedule ordinary/JEV data intervention'))
 save(folder/'RESULT.json',result);save(REPORTS/f'TRAIN_{arm}_{dataset}_STEPS{steps}.json',result)
 print('FASTCHECK_TRAIN_COMPLETE',arm,dataset,steps,result['training_signal_eligible'],final,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['jev','attention'],required=True);p.add_argument('--dataset',choices=['D0','D1'],required=True);p.add_argument('--steps',type=int,choices=[300,1000],default=300);a=p.parse_args();main(a.arm,a.dataset,a.steps)
