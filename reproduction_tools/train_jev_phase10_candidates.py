"""Actual matched-budget training, immutable completed results and resume."""
import argparse,collections,json,math,random,time,shutil
import numpy as np
import torch
from jev_phase10_learning import *


def atomic_torch(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def capacity(model):
 macs=sum(m.in_features*m.out_features for m in model.modules()if isinstance(m,torch.nn.Linear))
 return {'parameters':sum(p.numel()for p in model.parameters()),'trainable_parameters':sum(p.numel()for p in model.parameters()if p.requires_grad),'linear_MACs_per_edge_upper_bound':macs,'FLOPs_per_edge_upper_bound':2*macs,'scope':'conservative sum of all linear layers per candidate; row/context layers actually amortized across candidates; pooling/nonlinear operations separate'}


def train(name,supervision,seed,epochs,device):
 rows,manifest=load_data();assert json.loads((REPORTS/'TINY_OVERFIT.json').read_text())['status']=='PASS';source=binding();source['controller_training_seed']=seed;assert not source['dirty']
 trainrows=[r for r in rows if r['partition']=='train'];valrows=[r for r in rows if r['partition']=='validation'];stats=normalization(rows);out=OUT/'fair_training_v1'/name/supervision/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);done=out/f'RESULT_{epochs:03d}.json'
 if done.exists():assert json.loads(done.read_text())['binding']==source;print('TRAIN_ALREADY_COMPLETE',name,supervision,seed,epochs);return
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=build_candidate_model(name).to(device);install_normalization(model,stats);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);start_epoch=0;history=[];best=float('inf');updates=0
 last=out/'LAST.pth';bestpath=out/'BEST.pth'
 if last.exists():
  saved=torch.load(last,map_location=device);assert saved['binding']==source and saved['dataset_SHA256']==manifest['dataset_SHA256'];model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start_epoch=saved['epoch'];history=saved['history'];best=saved['best_selection_NLL'];updates=saved['updates']
 groups=collections.defaultdict(list)
 for r in trainrows:groups[r['group']].append(r)
 ceweight=sum(r['weight']for r in trainrows if r['CE_eligible']);rankweight=sum(r['weight']for r in trainrows if r['ranking_eligible']);assert ceweight>0 and rankweight>0;nbatches=math.ceil(len(trainrows)/32);begin=time.monotonic()
 for epoch in range(start_epoch,epochs):
  model.train();generator=random.Random(seed+100000+epoch);keys=sorted(groups);generator.shuffle(keys);ordered=[]
  for k in keys:
   within=list(groups[k]);generator.shuffle(within);ordered.extend(within)
  for start in range(0,len(ordered),32):
   st,e,mask,known,pos,q,qm,w=collate(ordered[start:start+32],device);optimizer.zero_grad(set_to_none=True);logits=model(st,e,mask);assert torch.isfinite(logits).all();ce,ceok,rank,rankok=losses(logits,known,pos,q,qm);celoss=(ce*w).sum()*nbatches/ceweight;rankloss=(rank*w).sum()*nbatches/rankweight;loss=logits.sum()*0+(celoss if supervision=='correctness'else rankloss if supervision=='H32_ranking'else celoss+rankloss);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();updates+=1
  tr=evaluate(model,trainrows,supervision,device=device);va=evaluate(model,valrows,supervision,device=device);selection=va['selection_NLL'];assert selection is not None and math.isfinite(selection);item={'epoch':epoch+1,'TRAIN':{k:v for k,v in tr.items()if k!='records'},'VALIDATION':{k:v for k,v in va.items()if k!='records'},'updates':updates};history.append(item)
  checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,'best_selection_NLL':min(best,selection),'binding':source,'dataset_SHA256':manifest['dataset_SHA256'],'model_name':name,'supervision':supervision,'seed':seed,'updates':updates,'normalization':stats}
  if selection<best:best=selection;atomic_torch(bestpath,checkpoint)
  atomic_torch(last,checkpoint);save(out/'PROGRESS.json',{'status':'RUNNING','epoch':epoch+1,'target_epochs':epochs,'updates':updates,'TRAIN_rank1':tr['group_weighted_rank1'],'validation_selection_NLL':selection,'source_commit':source['source_commit']});print('TRAIN_EPOCH',name,supervision,seed,epoch+1,round(selection,5),flush=True)
 saved=torch.load(bestpath,map_location=device);model.load_state_dict(saved['model']);grid=json.loads((REPORTS/'PHASE10_MODEL_AND_CONTROL_PROTOCOL.json').read_text())['calibration_temperature_grid'];candidates=[(evaluate(model,valrows,supervision,temperature=t,device=device),t)for t in grid];calibrated,temp=min(candidates,key=lambda pair:(pair[0]['selection_NLL'],pair[1]));training=evaluate(model,trainrows,supervision,temperature=temp,device=device);scriptpath=out/f'MODEL_CALIBRATED_{epochs:03d}.pt';script=torch.jit.script(CandidateScorer(model.cpu(),supervision,temp));scripttmp=scriptpath.with_suffix('.pt.tmp');script.save(str(scripttmp));scripttmp.replace(scriptpath);model.to(device)
 frozenbest=out/f'BEST_{epochs:03d}.pth';frozenlast=out/f'LAST_{epochs:03d}.pth';
 for original,frozen in [(bestpath,frozenbest),(last,frozenlast)]:
  if frozen.exists():assert sha(frozen)==sha(original),'unpublished budget snapshot differs'
  else:shutil.copyfile(original,frozen)
 result={'status':'COMPLETE','binding':source,'JIT_runtime':JIT_RUNTIME,'model':name,'supervision':supervision,'seed':seed,'budget_epochs':epochs,'best_epoch':saved['epoch'],'updates':updates,'dataset_SHA256':manifest['dataset_SHA256'],'capacity':capacity(model),'history':history,'TRAIN':training,'VALIDATION':calibrated,'temperature':temp,'temperature_grid':grid,'best_checkpoint':{'path':str(frozenbest),'SHA256':sha(frozenbest)},'last_checkpoint':{'path':str(frozenlast),'SHA256':sha(frozenlast)},'scripted_calibrated_model':{'path':str(scriptpath),'SHA256':sha(scriptpath)},'elapsed_seconds_this_resume':time.monotonic()-begin,'heldout_sealed':True,'fixed_NEW':True,'common_assignment':'assign_candidate_values(values mode)','common_optimizer_order_and_update_budget':True};save(done,result);save(out/'PROGRESS.json',{'status':'COMPLETE','epoch':epochs,'updates':updates,'best_epoch':saved['epoch'],'temperature':temp});print('TRAIN_RUN_COMPLETE',name,supervision,seed,epochs,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--supervision',choices=['correctness','H32_ranking','joint'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(a.model,a.supervision,a.seed,a.epochs,a.device)
