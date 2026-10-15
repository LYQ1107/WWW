"""One preregistered bounded smoke: ephemeral original-GMT samples, frozen GMT."""
import collections, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from gmt_jev_runtime import *
from gmt_jev_opportunity import collect, native_assignment

CAP=.05
MARGIN=.005
KINDS=['mlp','attention','cic']

def assignment(scores):
 if not scores.numel():return [],[]
 r,c=linear_sum_assignment(-scores.detach().cpu().numpy())
 return r.tolist(),c.tolist()

def forced_assignment(scores,targets):
 """Same rectangular assignment constraints, with certified training targets fixed."""
 rows=torch.where(targets>=0)[0].tolist();cols=targets[targets>=0].tolist()
 assert len(cols)==len(set(cols))
 rr=[r for r in range(scores.shape[0]) if r not in rows]
 cc=[c for c in range(scores.shape[1]) if c not in cols]
 if rr and cc:
  a,b=assignment(scores[rr][:,cc]);rows += [rr[r] for r in a];cols += [cc[c] for c in b]
 return rows,cols

def prepare(s):
 targets=s['targets'].clone();scores=s['raw_scores'];lengths=s['lengths'];correct=s['GMT_current_correct']
 counts=collections.Counter();counts['rows']=len(targets)
 counts['certified_unique_rows']=int((targets>=0).sum());counts['UNKNOWN_or_no_active_target_rows']=int((targets<0).sum())
 counts['E1_certified_original_wrong']=int(((targets>=0)&~correct).sum())
 counts['E2_certified_original_correct']=int(((targets>=0)&correct).sum())
 multiplicity=collections.Counter(targets[targets>=0].tolist())
 for r,t in enumerate(targets.tolist()):
  if t<0:continue
  if multiplicity[t]>1:
   counts['masked_target_column_conflict']+=1;targets[r]=-1
  elif scores[r,t]+CAP*lengths[t] <= scores.new_tensor((.1+MARGIN)*float(lengths[t])):
   counts['masked_capped_threshold_margin_unreachable']+=1;targets[r]=-1
 # A capped same-state witness qualifies wrong rows. It is never a runtime actor.
 signs=torch.zeros_like(scores)
 for r,t in enumerate(targets.tolist()):
  if t>=0:signs[r]=-1;signs[r,t]=1
 witness=native_assignment(scores+CAP*signs*lengths[None],lengths)
 for r,t in enumerate(targets.tolist()):
  if t>=0 and not correct[r] and witness[r]!=t:
   targets[r]=-1;counts['masked_E1_without_joint_capped_assignment_witness']+=1
 counts['supervised_E1_witness_rows']=int(((targets>=0)&~correct).sum())
 counts['supervised_preserve_rows']=int(((targets>=0)&correct).sum())
 counts['masked_rows']=int((targets<0).sum())
 item=dict(key=tuple(s['key']),features=s['features'],raw_scores=scores,lengths=lengths,
           targets=targets,correct=correct)
 assert not any(v.requires_grad for v in item.values() if isinstance(v,torch.Tensor))
 assert all(torch.isfinite(v).all() for v in item.values() if isinstance(v,torch.Tensor))
 return item,counts

def size(item):
 # Tensor storage plus Python containers/objects; leave additional allocation reserve.
 return sys.getsizeof(item)+sum(sys.getsizeof(v)+(v.numel()*v.element_size() if isinstance(v,torch.Tensor)
   else sum(sys.getsizeof(x) for x in v) if isinstance(v,tuple) else 0) for v in item.values())

class BoundedBuffer:
 def __init__(self):self.parts={k:[] for k in ['E1','PRESERVE']};self.peak=0;self.seen=collections.Counter()
 def add(self,item):
  if not (item['targets']>=0).any():return
  group='E1' if ((item['targets']>=0)&~item['correct']).any() else 'PRESERVE'
  weight=size(item);self.seen[group]+=1
  if weight>14*2**20:return
  priority=hashlib.sha256(repr((SEED,item['key'])).encode()).hexdigest()
  part=self.parts[group];part.append((priority,weight,item));part.sort(key=lambda x:x[0])
  while sum(x[1] for x in part)>14*2**20:part.pop()
  held=sum(x[1] for p in self.parts.values() for x in p)+sum(sys.getsizeof(p)+sum(sys.getsizeof(x)+sys.getsizeof(x[0]) for x in p) for p in self.parts.values())
  self.peak=max(self.peak,held);assert held<32*2**20
 def summary(self):
  return dict(peak_accounted_buffer_bytes=self.peak,limit_bytes=32*2**20,
   sampling='Deterministic bottom SHA256 priority within E1 and preserve strata, each <=14MiB; full native matrices retained, no row/candidate pruning.',
   seen_matrices=dict(self.seen),retained_matrices={k:len(p) for k,p in self.parts.items()},
   retained_rows={k:sum(len(x[2]['targets']) for x in p) for k,p in self.parts.items()},
   retained_supervised_E1=sum(int(((x[2]['targets']>=0)&~x[2]['correct']).sum()) for p in self.parts.values() for x in p),
   retained_supervised_preserve=sum(int(((x[2]['targets']>=0)&x[2]['correct']).sum()) for p in self.parts.values() for x in p),
   retained_keys={k:[list(x[2]['key']) for x in p] for k,p in self.parts.items()})

def loss(head,item):
 device=next(head.parameters()).device
 x=item['features'].to(device);base=item['raw_scores'].to(device);lengths=item['lengths'].to(device);y=item['targets'].to(device);correct=item['correct'].to(device)
 gate,delta=head(x);residual=CAP*gate*delta;new=base+residual*lengths[None]
 assert torch.isfinite(new).all() and (residual.abs()<=CAP).all()
 known=torch.where(y>=0)[0];assert len(known)
 augmented=new.detach().clone();augmented[known,y[known]]-=MARGIN*lengths[y[known]]
 pr,pc=assignment(augmented);rr,rc=forced_assignment(new.detach(),y)
 penalty=sum(float(MARGIN*lengths[y[r]]) for r in known.tolist() if r not in pr or pc[pr.index(r)]!=int(y[r]))
 structured=torch.relu(new[pr,pc].sum()-new[rr,rc].sum()+penalty)/lengths[y[known]].sum().clamp_min(1)
 threshold=torch.relu(.1+MARGIN-new[known,y[known]]/lengths[y[known]]).mean()
 preserve=residual[correct].square().mean() if correct.any() else residual.sum()*0
 regular=residual.square().mean() if residual.numel() else residual.sum()*0
 total=structured+threshold+10*preserve+regular
 return total,dict(structured=float(structured.detach()),threshold=float(threshold.detach()),
                   preserve=float(preserve.detach()),regular=float(regular.detach()),total=float(total.detach()))

def gradient_contract(item):
 from gtr.modeling.gmt_jev.correction import ResidualHead
 reports={}
 for kind in KINDS:
  seed();head=ResidualHead(kind).cuda();total,parts=loss(head,item);total.backward()
  grads=[p.grad for p in head.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
  norm=sum(float(g.square().sum()) for g in grads)**.5;assert norm>0
  reports[kind]=dict(parameters=sum(p.numel() for p in head.parameters()),gradient_norm=norm,loss=parts,
    parameter_updates=0,zero_delta_initialization=True)
  del head
 return reports

def main():
 from gtr.modeling.gmt_jev.correction import ResidualHead
 assert read(REPORTS/'LEARNED_GATE_ZERO_NATIVE_PARITY.json')['status']=='PASS_EXACT_ALL_THREE_HEADS_32_SCENE_PREFIX'
 buffer=BoundedBuffer();cases=[];totals={k:collections.Counter() for k in ['TRAIN','TEMPORAL_HELDOUT']}
 for video in [12,13,14]:
  _,frames,_=metadata(video);buckets={k:collections.Counter() for k in totals}
  def observe(s):
   item,c=prepare(s);key='TEMPORAL_HELDOUT' if s['audit_block'] else 'TRAIN';buckets[key].update(c)
   if not s['audit_block']:buffer.add(item)
  samples,result=collect(video,frames=frames,records=False,report=False,diagnostic_sample_observer=observe)
  assert samples==[] and result['counts']==read(REPORTS/f'FULL_TRAIN_OPPORTUNITY_VIDEO{video:02d}.json')['counts']
  assert result['GMT_requires_grad_parameters']==0
  for k,c in buckets.items():totals[k].update(c)
  case=dict(video=video,frames=frames,counts={k:dict(v) for k,v in buckets.items()},original_counter_parity=True,
            frozen_GMT_parameters=result['frozen_GMT_parameters'],GMT_requires_grad_parameters=0)
  cases.append(case)
  save(REPORTS/'LEARNED_GATE_SUPERVISION_PROGRESS.json',dict(status='SAMPLING_ORIGINAL_GMT_NO_UPDATES',cases=cases,buffer=buffer.summary()))
 e1=[x[2] for x in buffer.parts['E1']];keep=[x[2] for x in buffer.parts['PRESERVE']]
 assert e1 and keep, 'No witnessed corrective or preserve supervision; do not train'
 assert all(not ((i['key'][1]//64)%5==4) for i in e1+keep)
 gradients=gradient_contract(e1[0]);precheck=dict(status='PASS_NATIVE_ZERO_AND_SUPERVISION_CONTRACT',
  cases=cases,counts={k:dict(v) for k,v in totals.items()},buffer=buffer.summary(),gradient_checks=gradients,
  gradient_check_parameter_updates=0,heldout_matrices_used_for_gradients=0,GT_actor_input=False,
  no_persistent_corpus=True,ORACLE_witness_is_not_online_gain=True,source=binding())
 save(REPORTS/'LEARNED_GATE_SUPERVISION_PRECHECK.json',precheck)
 # Alternating witnessed E1 and preserve matrices, identical keys/order for all heads.
 rng=np.random.RandomState(SEED);schedule=[(e1 if step%2==0 else keep)[rng.randint(len(e1 if step%2==0 else keep))] for step in range(300)]
 schedule_sha=hashlib.sha256(json.dumps([list(i['key']) for i in schedule]).encode()).hexdigest()
 results={};folder=OUT/'learned_gate_smoke';folder.mkdir(parents=True,exist_ok=False)
 for kind in KINDS:
  seed();head=ResidualHead(kind).cuda();optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);logs=[]
  for step,item in enumerate(schedule,1):
   optim.zero_grad(set_to_none=True);total,parts=loss(head,item);assert torch.isfinite(total)
   total.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);assert torch.isfinite(norm)
   optim.step()
   if step==1 or step%25==0:
    logs.append(dict(step=step,gradient_norm_before_clip=float(norm),**parts))
    save(REPORTS/'LEARNED_GATE_TRAINING_PROGRESS.json',dict(status='TRAINING_SINGLE_FIXED_300_ENDPOINT',kind=kind,step=step,completed_heads=results,log=logs))
    print('GMT_LEARNED_SMOKE_PROGRESS',kind,step,parts,flush=True)
  path=folder/(kind+'_step300.pth');reserve()
  torch.save(dict(head=head.cpu().state_dict(),kind=kind,risk_mode='learned',steps=300,seed=SEED,
   schedule_SHA256=schedule_sha,feature_names=__import__('gtr.modeling.gmt_jev.correction',fromlist=['FEATURE_NAMES']).FEATURE_NAMES,
   cap=CAP,precheck=ref(REPORTS/'LEARNED_GATE_SUPERVISION_PRECHECK.json')),path)
  results[kind]=dict(checkpoint=ref(path),parameters=sum(p.numel() for p in head.parameters()),steps=300,logs=logs,
   optimizer='AdamW',lr=.001,weight_decay=.0001,clip_grad_norm=1,seed=SEED,schedule_SHA256=schedule_sha,
   selection='Only fixed step300 endpoint; no TRAIN16 selection',online_quality=None)
  del head,optim;torch.cuda.empty_cache()
  save(REPORTS/'LEARNED_GATE_SMOKE_TRAINING.json',dict(status='COMPLETE' if len(results)==3 else 'PARTIAL_FIXED_ENDPOINTS',
   cases=results,precheck=ref(REPORTS/'LEARNED_GATE_SUPERVISION_PRECHECK.json'),source=binding(),scientific_success=False))

if __name__=='__main__':main()
