"""Tiny -> fresh pilot -> cumulative formal, equal TRAIN conflict replay for all arms."""
import argparse,collections,random,math,traceback,torch
from jev_recovery_common import *
from jev_recovery_learning import *
def main(arm,phase,seed,revision,base):
 protect();protocol=read(REPORTS/'PREREGISTRATION.json');budget=protocol['training'];assert seed in SEEDS
 contract=REPORTS/('MODEL_AND_NATIVE_CONTRACT_TESTS.json' if revision==1 else f'R{revision}_MODEL_AND_NATIVE_CONTRACT_TESTS.json')
 assert read(contract)['status']=='PASS'
 if phase not in ('tiny','tiny8') and revision in (1,2):assert read(REPORTS/f'R{revision}_8PLAN_CONTRACT_TESTS.json')['status']=='PASS'
 target={'tiny':128,'tiny8':128,'pilot':1000,'formal':budget['formal_total_updates']}[phase]
 folder=OUT/f'training_R{revision}'/base/arm/f'seed{seed}'/phase
 folder.mkdir(parents=True,exist_ok=False)
 train,validation,sources=load_records(revision);groups=conflict_groups(train);assert groups,'No certified train conflict groups'
 semantic_groups=question_conflict_groups(train) if model_revision(revision)>=2 else {}
 keys=sorted(groups);model,initializer=initialize(arm,seed,revision,base)
 rng=random.Random(seed);ordinary=collections.defaultdict(list)
 for r in train:ordinary[r['key'][0],r['key'][1]//64].append(r)
 block_keys=sorted(ordinary);weights={}
 for name in ('cross','stability'):
  p=sum(int((r['oneweek_labels'][name]==1).sum()) for r in train);n=sum(int((r['oneweek_labels'][name]==0).sum()) for r in train);weights[name]=n/max(p,1)
 p=sum(int(r['positive'][:,:-1].sum()) for r in train);n=sum(int((r['known_options'][:,:-1]&~r['positive'][:,:-1]).sum()) for r in train);weights['compatibility']=n/max(p,1)
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=budget['learning_rate'],weight_decay=budget['weight_decay'])
 start=0
 if phase=='formal':
  pilot=read(folder.parent/'pilot/RESULT.json');assert pilot['engineering_training_pass']
  cp=pilot['checkpoint'];assert sha(cp['path'])==cp['SHA256'];s=torch.load(cp['path'],map_location='cuda:0')
  model.load_state_dict(s['new_modules'],strict=False);optimizer.load_state_dict(s['optimizer']);start=s['updates'];assert start==1000
 fixture_keys=[next(k for k in keys if k[0]==v) for v in [12,13,14] if any(k[0]==v for k in keys)]
 fixture_keys+=(keys[:4]);fixture_keys=fixture_keys[:4]
 fixture=[groups[k][0] for k in fixture_keys];before=assess(model,validation,128)
 x,y=batch(fixture);initial=float(loss(model.details(x),x,y,fixture,weights)[0].detach())
 save(folder/'START.json',dict(binding=binding('Bounded dependence-aware TRAIN conflict learning'),arm=arm,revision=revision,base=base,seed=seed,phase=phase,total_updates=target,start_updates=start,initializer=initializer,sources=sources,TRAIN_independent_conflict_groups=len(groups),TRAIN_question_conflict_groups=len(semantic_groups),sampling='R1:50% certified conflicts/50% ordinary; R2:25% certified association conflicts/25% true question disagreements/50% ordinary. Group-balanced, not repeated-edge-balanced.',pos_weights=weights,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),DEV_gradient=False))
 history=[];rank_batches=0;nonzero_batches=0;seen_groups=set();actual_gradient_groups=set();all_keys=hashlib.sha256();gradients=collections.Counter();begin=time.monotonic()
 # Advance identical RNG draws on formal continuation to avoid pilot replay.
 def sampled():
  part=[];samplegroups=[]
  for slot in range(4):
   if slot==1 and semantic_groups:
    key=rng.choice(sorted(semantic_groups));part.append(rng.choice(semantic_groups[key]))
   elif slot<2:
    key=rng.choice(keys);part.append(rng.choice(groups[key]));samplegroups.append(key)
   else:
    video=rng.choice([12,13,14]);blocks=[k for k in block_keys if k[0]==video];key=rng.choice(blocks);part.append(rng.choice(ordinary[key]))
  return part,samplegroups
 if phase=='formal':
  for _ in range(start):sampled()
 for step in range(start+1,target+1):
  part,sg=(fixture,fixture_keys) if phase in ('tiny','tiny8') else sampled()
  assert all(r['key'][0] in [12,13,14] and not r['audit_block'] for r in part)
  all_keys.update(json.dumps([r['key'] for r in part]).encode());x,y=batch(part);model.train();optimizer.zero_grad(set_to_none=True)
  d=model.details(x);value,components,pairs=loss(d,x,y,part,weights);assert torch.isfinite(value)
  # Audit whether the certified ranking term actually reaches a scoring head.
  # This is separate from auxiliary/edge BCE gradients.
  ranking=torch.stack([torch.nn.functional.softplus(-sign*(d['plan_scores'][i][a]-d['plan_scores'][i][b])/math.sqrt(n)) for i,a,b,sign,n in pairs]).mean() if pairs else None
  rank_gradient=False
  if ranking is not None:
   heads=[model.edge_value.weight]
   if hasattr(model,'plan_value') and model.recovery_arm!='no_interaction':heads.append(model.plan_value.weight)
   rr=torch.autograd.grad(ranking,heads,retain_graph=True,allow_unused=True)
   rank_gradient=any(g is not None and bool(g.abs().max()>0) for g in rr)
  if rank_gradient:
   for i,a,b,sign,n in pairs:
    record=part[i];pp=d['plans'][i];changed=[row for row,(u,v) in enumerate(zip(pp[a],pp[b])) if u!=v]
    people=tuple(sorted({record['GT_labels_OFFLINE_ONLY'][row] for row in changed if record['GT_labels_OFFLINE_ONLY'][row] is not None}))
    ids=tuple(sorted({record['refs'][c] for p in (pp[a],pp[b]) for row,c in enumerate(p) if row in changed and c>=0}))
    actual_gradient_groups.add((record['key'][0],people,ids,record['key'][1]//64,record['key'][2]))
  value.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm)
  for name,p in model.named_parameters():
   if p.grad is not None:
    assert torch.isfinite(p.grad).all();gradients[name]=max(gradients[name],float(p.grad.abs().max()))
  optimizer.step();rank_batches+=bool(pairs);nonzero_batches+=components['current_plan_ranking']>0
  if pairs:seen_groups.update(sg)
  if step%32==0 or step in (start+1,target):
   row=dict(step=step,total_steps=target,sampling_epoch_equivalent=step*4/len(train),rank_supervised_batches_so_far=rank_batches,rank_nonzero_batch_fraction_so_far=nonzero_batches/(step-start),loss=float(value.detach()),components=components,certified_rank_pairs=len(pairs),gradient_norm=float(norm),seconds=time.monotonic()-begin)
   history.append(row);save(folder/'PROGRESS.json',dict(status='TRAINING',**row));print('RECOVERY_TRAIN',revision,base,arm,phase,seed,row,flush=True)
  if step%128==0:protect()
 model.eval();x,y=batch(fixture)
 with torch.no_grad():last=float(loss(model.details(x),x,y,fixture,weights)[0])
 final=assess(model,validation,128);cp=folder/'NEW_MODULES.pth'
 torch.save(dict(new_modules={k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('identity.')},optimizer=optimizer.state_dict(),initializer=initializer,updates=target,arm=arm,seed=seed,revision=revision,base=base),cp)
 result=dict(status='COMPLETE',binding=binding('TRAIN-only learned recovery residual'),arm=arm,phase=phase,seed=seed,revision=revision,base=base,maximum_plans=model.maximum,total_updates=target,actual_new_updates=target-start,checkpoint=ref(cp),initializer=initializer,before=before,final=final,history=history,fixture_loss_before=initial,fixture_loss_after=last,rank_supervised_batches=rank_batches,rank_nonzero_batch_fraction=nonzero_batches/(target-start),sampled_conflict_groups_with_ranking=len(seen_groups),certified_changed_row_groups_with_nonzero_ranking_head_gradient=len(actual_gradient_groups),ranking_gradient_group_definition='Exact changed-row person/native-ID/camera/64-frame cluster; nonzero derivative into edge or plan scoring head, not merely positive softplus loss',TRAIN_conflict_groups=len(groups),batch_key_digest=all_keys.hexdigest(),max_abs_gradients=dict(gradients),engineering_training_pass=bool(gradients) and rank_batches>0 and (phase not in ('tiny','tiny8') or last<initial),supervision_eligible='CURRENT_MATCH_ONLY',future_safety_qualified=False,independent_validation=False,seconds=time.monotonic()-begin)
 save(folder/'RESULT.json',result);save(REPORTS/f'TRAIN_R{revision}_{base}_{arm}_{phase}_SEED{seed}.json',result);print('RECOVERY_TRAIN_COMPLETE',arm,phase,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--phase',choices=['tiny','tiny8','pilot','formal'],required=True);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--revision',type=int,default=1);p.add_argument('--base',default='XIV');a=p.parse_args()
 try:main(a.arm,a.phase,a.seed,a.revision,a.base)
 except Exception:failure('training',traceback.format_exc());raise
