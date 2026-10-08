"""Fixed real TRAIN tiny set; identical optimization and legal-support tests."""
import collections,json,time
import torch
from jev_phase10_learning import *
from gtr.modeling.jev_native_state import fingerprint


def main(device):
 rows,manifest=load_data();source=binding();assert not source['dirty'];stats=normalization(rows);chosen=[];selectedgroups=[]
 if(OUT/'tiny_v1/RESULT.json').exists():
  prior=json.loads((OUT/'tiny_v1/RESULT.json').read_text());assert prior['binding']==source and prior['status']=='PASS';print('ALL_THREE_TINY_ALREADY_PASS');return
 for video in TRAIN:
  pool=sorted([r for r in rows if r['key'][0]==video and r['CE_eligible']],key=lambda r:(r['key'][1],r['key'][2],r['row']));groups=[]
  for r in pool:
   if r['group']not in groups:groups.append(r['group'])
  for g in groups[:2]:chosen.extend([r for r in pool if r['group']==g][:3]);selectedgroups.append(g)
 assert len(chosen)>=6 and len(set(selectedgroups))>=4
 out=OUT/'tiny_v1';out.mkdir(exist_ok=True);selection={'dataset_SHA256':manifest['dataset_SHA256'],'groups':selectedgroups,'rows':[{'key':r['key'],'row':r['row'],'group':r['group'],'prefix_SHA256':r['native_prefix_SHA256']}for r in chosen],'selection':'first two chronological known-positive TRAIN groups per video; max3 rows/group','heldout_sealed':True};p=out/'SELECTION.json'
 if p.exists():assert json.loads(p.read_text())==selection
 else:save(p,selection)
 # Contradictory exactly identical observations must be surfaced, not trained
 # away with extra epochs. Runtime IDs never enter this comparison.
 collisions=[];seen={}
 for r in chosen:
  columns=sorted(range(len(r['candidate_ids'])),key=lambda c:fingerprint(r['evidence12'][c]));sig=fingerprint([r['state64'],r['evidence12'][columns],r['legal_mask'][columns]]);labels=(r['known_mask'][columns].tolist(),r['positive_mask'][columns].tolist())
  if sig in seen and labels!=seen[sig]:collisions.append({'key':r['key'],'row':r['row'],'fingerprint':sig})
  seen[sig]=labels
 assert not collisions,('contradictory visible tiny inputs',collisions)
 state,e,mask,known,pos,q,qm,w=collate(chosen,device);models={};optimizers={};initial={};history=collections.defaultdict(list);steps={}
 for name in ['CandidateMLP','CandidateDeepSets','CandidateJEV']:
  torch.manual_seed(20261008);m=build_candidate_model(name).to(device);install_normalization(m,stats);models[name]=m;optimizers[name]=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4);initial[name]=evaluate(m,chosen,'correctness',device=device);steps[name]=0;p=out/(name+'_RESUME.pth')
  if p.exists():
   saved=torch.load(p,map_location=device);assert saved['binding']==source and saved['dataset_SHA256']==manifest['dataset_SHA256'];m.load_state_dict(saved['model']);optimizers[name].load_state_dict(saved['optimizer']);steps[name]=saved['steps'];initial[name]=saved['initial'];history[name]=saved['history']
 budgets=[2000]if max(steps.values())>500 else[500,2000];done=0;passed=False
 for budget in budgets:
  for name,m in models.items():
   m.train();optimizer=optimizers[name]
   for step in range(steps[name],budget):
    optimizer.zero_grad(set_to_none=True);logits=m(state,e,mask);ce,ok,_,_=losses(logits,known,pos,q,qm);loss=(ce*w).sum()/w.sum()+logits.sum()*0;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step()
    steps[name]=step+1
    if steps[name]%100==0:
     p=out/(name+'_RESUME.pth');tmp=p.with_suffix('.pth.tmp');torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict(),'steps':steps[name],'initial':initial[name],'history':history[name],'binding':source,'dataset_SHA256':manifest['dataset_SHA256']},tmp);tmp.replace(p);save(out/'PROGRESS.json',{'model':name,'steps':steps[name],'budget':budget,'status':'RUNNING','source_commit':source['source_commit']})
  result=evaluate(m,chosen,'correctness',device=device);history[name].append({'steps':budget,'rank1':result['group_weighted_rank1'],'loss':result['correctness_loss'],'unknown_choice_rate':result['unknown_choice_rate']});print('TINY_FIT',name,history[name][-1],flush=True)
  p=out/(name+'_RESUME.pth');tmp=p.with_suffix('.pth.tmp');torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict(),'steps':steps[name],'initial':initial[name],'history':history[name],'binding':source,'dataset_SHA256':manifest['dataset_SHA256']},tmp);tmp.replace(p)
  done=budget;passed=all(history[name][-1]['rank1']>=.95 and history[name][-1]['loss']<max(initial[name]['correctness_loss'],1e-8)for name in models)
  if passed:break
  # Every method extends together only after finite/collision/masking checks.
  assert all(torch.isfinite(p).all()for m in models.values()for p in m.parameters())
 outputs={}
 for name,m in models.items():
  m.eval();order=torch.randperm(e.shape[1],device=device)
  with torch.no_grad():normal=m(state,e,mask);permuted=m(state,e[:,order],mask[:,order]);before=normal[:,:,0].masked_fill(~mask,-torch.inf).argmax(1);after=permuted[:,:,0].masked_fill(~mask[:,order],-torch.inf).argmax(1);mapped=order[after];finite=torch.isfinite(normal).all();error=float((permuted-normal[:,order]).abs().max())
  assert finite and torch.equal(before,mapped),'trained tiny permutation changed legal choices'
  script=torch.jit.script(CandidateScorer(m));sv,_=script(state,e,mask);assert torch.equal(sv,normal[:,:,0]),'scripted normalization differs from training'
  path=out/(name+'.pth');payload={'model':m.state_dict(),'normalization':stats,'dataset_SHA256':manifest['dataset_SHA256'],'source':source,'tiny_only':True}
  if path.exists():assert fingerprint(torch.load(path,map_location='cpu'))==fingerprint(payload),'existing immutable tiny checkpoint differs'
  else:
   tmp=path.with_suffix('.pth.tmp');torch.save(payload,tmp);tmp.replace(path)
  outputs[name]={'initial':{k:v for k,v in initial[name].items()if k!='records'},'history':history[name],'final':{k:v for k,v in evaluate(m,chosen,'correctness',device=device).items()if k!='records'},'permutation_candidate_choices_exact':True,'max_output_permutation_abs_error':error,'scripted_training_normalization_exact':True,'checkpoint_SHA256':sha(path),'checkpoint_path':str(path)}
 save(out/'RESULT.json',{'status':'PASS'if passed else'FAIL_TINY_FIT','steps_per_model':done,'selection':selection,'selection_SHA256':sha(out/'SELECTION.json'),'models':outputs,'exact_input_label_collisions':collisions,'mask_and_unknown_supervision_test':'SUPERVISION_CONTRACT_TESTS.json','binding':source,'heldout_sealed':True,'formal_training_starts_only_if_PASS':True});assert passed,'tiny science/optimization contract not passed after bounded inspection and equal budgets';print('ALL_THREE_TINY_PASS',done,flush=True)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)
