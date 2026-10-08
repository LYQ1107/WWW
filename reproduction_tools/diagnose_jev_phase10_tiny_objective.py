"""Bounded known-edge BCE diagnostic, preserving every original gate/label."""
import json,collections,time
import torch
import torch.nn.functional as F
from jev_phase10_learning import *
from gtr.modeling.jev_native_state import fingerprint


def main(objective='binary_CE'):
 protocol_file='TINY_JOINT_DIAGNOSTIC_PROTOCOL.json'if objective=='joint'else'TINY_OBJECTIVE_DIAGNOSTIC_PROTOCOL.json'
 allrows,manifest=load_data();source=binding();assert not source['dirty'];protocol=json.loads((REPORTS/protocol_file).read_text());selection=json.loads((OUT/'tiny_v1/SELECTION.json').read_text());assert sha(OUT/'tiny_v1/SELECTION.json')==protocol['fixed_selection_SHA256'];wanted={tuple(r['key'])+(r['row'],)for r in selection['rows']};rows=[r for r in allrows if tuple(r['key'])+(r['row'],)in wanted];stats=normalization(allrows);st,e,mask,known,pos,q,qm,w=collate(rows,'cuda:0');out=OUT/('tiny_joint_diagnostic_v1'if objective=='joint'else'tiny_binary_ce_diagnostic_v1');assert not out.exists();out.mkdir();models={};optimizers={};history=collections.defaultdict(list);steps=0;active='joint'if objective=='joint'else'correctness'
 for name in ['CandidateMLP','CandidateDeepSets','CandidateJEV']:
  torch.manual_seed(20261008);m=build_candidate_model(name).to('cuda:0');install_normalization(m,stats);models[name]=m;optimizers[name]=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4)
 for budget in [500,2000]:
  for name,m in models.items():
   m.train();optimizer=optimizers[name]
   for step in range(steps,budget):
    optimizer.zero_grad(set_to_none=True);logits=m(st,e,mask)
    if objective=='joint':
     ce,co,rank,ro=losses(logits,known,pos,q,qm);loss=(ce*w).sum()/w.sum()+(rank*w).sum()/(w*ro).sum().clamp_min(1e-12)
    else:
     edge=F.binary_cross_entropy_with_logits(logits[:,:,0],pos.float(),reduction='none');ce=(edge*known).sum(1)/known.sum(1).clamp_min(1);loss=(ce*w).sum()/w.sum()
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step()
   result=evaluate(m,rows,active,device='cuda:0');history[name].append({'steps':budget,'full_legal_confirmed_rank1':result['group_weighted_rank1'],'unknown_choice_rate':result['unknown_choice_rate'],'known_conditional_NLL':result['selection_NLL'],'actual_training_loss':float(loss)});save(out/'PROGRESS.json',{'source':source,'history':history,'model':name,'steps':budget});print('TINY_OBJECTIVE_DIAGNOSTIC',objective,name,history[name][-1],flush=True)
  steps=budget
  if all(history[name][-1]['full_legal_confirmed_rank1']>=.95 for name in models):break
 outputs={}
 for name,m in models.items():
  m.eval();permutation=torch.randperm(e.shape[1],device='cuda:0')
  with torch.no_grad():v=m(st,e,mask);vp=m(st,e[:,permutation],mask[:,permutation]);chosen=preference(v,active).masked_fill(~mask,-torch.inf).argmax(1);mapped=permutation[preference(vp,active).masked_fill(~mask[:,permutation],-torch.inf).argmax(1)]
  assert torch.equal(chosen,mapped)and torch.isfinite(v).all();script=torch.jit.script(CandidateScorer(m,active));sv,_=script(st,e,mask);assert torch.equal(sv,preference(v,active));path=out/(name+'.pth');torch.save({'model':m.state_dict(),'normalization':stats,'source':source,'dataset_SHA256':manifest['dataset_SHA256'],'diagnostic_only':True},path);outputs[name]={'history':history[name],'final':{k:v for k,v in evaluate(m,rows,active,device='cuda:0').items()if k!='records'},'checkpoint_path':str(path),'checkpoint_SHA256':sha(path),'permutation_choices_exact':True,'scripted_training_normalization_exact':True}
 passed=all(outputs[name]['history'][-1]['full_legal_confirmed_rank1']>=.95 for name in models);result={'status':'PASS_TINY_OBJECTIVE_DIAGNOSTIC'if passed else'FAIL_TINY_OBJECTIVE_DIAGNOSTIC','objective':objective,'binding':source,'JIT_runtime':JIT_RUNTIME,'protocol_SHA256':sha(REPORTS/protocol_file),'selection_SHA256':sha(OUT/'tiny_v1/SELECTION.json'),'dataset_SHA256':manifest['dataset_SHA256'],'steps_per_model':steps,'models':outputs,'gate_unchanged_rank1_point95':True,'unknown_labels_or_utilities_not_filled':True,'heldout_unopened':True,'formal_training_authorized_by_this_probe':False};save(out/'RESULT.json',result);print('BOUNDED_OBJECTIVE_DIAGNOSTIC_COMPLETE',result['status'],flush=True)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--objective',choices=['binary_CE','joint'],default='binary_CE');main(p.parse_args().objective)
