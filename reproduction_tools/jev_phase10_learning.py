"""Shared v3 normalization, masked supervision and evaluation for all models."""
import collections,json,math
import torch
import torch.nn.functional as F
from jev_phase10_common import *
from gtr.modeling.jev_candidate_models import build_candidate_model,CandidateScorer


def load_data():
 protect();gate=json.loads((REPORTS/'DATA_ELIGIBILITY.json').read_text());assert gate['gate_pass'],'no learning without scientific data eligibility'
 m=json.loads((REPORTS/'NATIVE_CANDIDATE_V3_MANIFEST.json').read_text());assert sha(m['dataset_path'])==m['dataset_SHA256'];data=torch.load(m['dataset_path'],map_location='cpu');return data['rows'],m


def normalization(rows):
 selected=[r for r in rows if r['partition']=='train'];sw=sum(r['weight']for r in selected)
 mean=sum(r['weight']*r['state64']for r in selected)/sw
 var=sum(r['weight']*(r['state64']-mean).square()for r in selected)/sw
 ev=[r for r in selected if r['legal_mask'].any()];ew=sum(r['weight']for r in ev)
 em=sum(r['weight']*r['evidence12'][r['legal_mask']].mean(0)for r in ev)/ew
 evv=sum(r['weight']*(r['evidence12'][r['legal_mask']]-em).square().mean(0)for r in ev)/ew
 return {'state_mean':mean,'state_std':var.sqrt().clamp_min(1e-8),'evidence_mean':em,'evidence_std':evv.sqrt().clamp_min(1e-8),'state_constant':var.sqrt()<1e-8,'evidence_constant':evv.sqrt()<1e-8}


def install_normalization(model,stats):
 with torch.no_grad():
  for k,v in stats.items():getattr(model,k).copy_(v)


def collate(rows,device='cpu'):
 n=len(rows);c=max((len(r['candidate_ids'])for r in rows),default=0);state=torch.stack([r['state64']for r in rows]).to(device);e=state.new_zeros((n,c,12));mask=torch.zeros(n,c,dtype=torch.bool,device=device);known=mask.clone();positive=mask.clone();qm=mask.clone();q=state.new_full((n,c),float('nan'));weight=state.new_tensor([r['weight']for r in rows])
 for i,r in enumerate(rows):
  k=len(r['candidate_ids']);e[i,:k]=r['evidence12'].to(device);mask[i,:k]=r['legal_mask'].to(device);known[i,:k]=r['known_mask'].to(device);positive[i,:k]=r['positive_mask'].to(device);qm[i,:k]=r['utility_mask'].to(device);q[i,:k]=r['utility'].to(device)
 return state,e,mask,known,positive,q,qm,weight


def losses(logits,known,positive,q,qm):
 ce=logits.new_zeros(len(logits));ceok=positive.any(1)&known.any(1)
 if ceok.any():
  ce[ceok]=torch.logsumexp(logits[ceok,:,0].masked_fill(~known[ceok],-torch.inf),1)-torch.logsumexp(logits[ceok,:,0].masked_fill(~positive[ceok],-torch.inf),1)
 rank=logits.new_zeros(len(logits));rankok=torch.zeros(len(logits),dtype=torch.bool,device=logits.device)
 for i in range(len(logits)):
  pair=qm[i,:,None]&qm[i,None,:]&(q[i,:,None]-q[i,None,:]>1e-6)
  if pair.any():
   delta=logits[i,:,1,None]-logits[i,None,:,1];rank[i]=F.softplus(-delta[pair]).mean();rankok[i]=True
 return ce,ceok,rank,rankok


def preference(logits,supervision):
 if supervision=='H32_ranking':return logits[...,1]
 if supervision=='joint':return logits.sum(-1)
 return logits[...,0]


@torch.no_grad()
def evaluate(model,rows,supervision,temperature=1.,device='cpu'):
 model.eval();records=[];tot=collections.defaultdict(float);strata=collections.defaultdict(lambda:collections.defaultdict(float));pairhits=pairden=0
 for start in range(0,len(rows),32):
  chunk=rows[start:start+32];st,e,mask,known,pos,q,qm,w=collate(chunk,device);logits=model(st,e,mask).cpu();mask=mask.cpu();known=known.cpu();pos=pos.cpu();q=q.cpu();qm=qm.cpu();assert torch.isfinite(logits).all();v=preference(logits,supervision)/temperature;ce,ceok,rank,rankok=losses(logits,known,pos,q,qm)
  for i,r in enumerate(chunk):
   k=len(r['candidate_ids']);legal=mask[i,:k];weight=r['weight'];order=torch.argsort(v[i,:k].masked_fill(~legal,-torch.inf),descending=True,stable=True).tolist();chosen=order[0]if legal.any()else None;eligible=bool(pos[i,:k].any());correct=bool(pos[i,chosen])if chosen is not None else False;unknown=chosen is not None and not bool(known[i,chosen]);ranks=[j+1 for j,col in enumerate(order)if bool(pos[i,col])];mrr=1/min(ranks)if ranks else 0.;nll=brier=conf=calcorrect=None
   if eligible:
    probs=torch.softmax(v[i,:k].masked_fill(~known[i,:k],-torch.inf),0);nll=float(torch.logsumexp(v[i,:k].masked_fill(~known[i,:k],-torch.inf),0)-torch.logsumexp(v[i,:k].masked_fill(~pos[i,:k],-torch.inf),0));target=pos[i,:k].float()/pos[i,:k].sum();brier=float((probs-target).square()[known[i,:k]].sum());selected=int(probs.argmax());conf=float(probs[selected]);calcorrect=float(pos[i,selected]);tot['eval_weight']+=weight;tot['rank1']+=weight*correct;tot['MRR']+=weight*mrr;tot['NLL']+=weight*nll;tot['Brier']+=weight*brier;tot['unknown_choice']+=weight*unknown
    if r['factual_identity_correct']is False:tot['corrective_den']+=weight;tot['corrective']+=weight*correct
    if r['factual_identity_correct']is True:tot['regression_den']+=weight;tot['regression']+=weight*(chosen is not None and bool(known[i,chosen])and not correct)
    for d in r['distribution']:
     stt=strata[d];stt['weight']+=weight;stt['rank1']+=weight*correct;stt['MRR']+=weight*mrr;stt['NLL']+=weight*nll
   if bool(ceok[i]):tot['ce_weight']+=weight;tot['correctness_loss']+=weight*float(ce[i])
   if bool(rankok[i]):tot['ranking_weight']+=weight;tot['ranking_loss']+=weight*float(rank[i]);pair=qm[i,:,None]&qm[i,None,:]&(q[i,:,None]-q[i,None,:]>1e-6);dif=v[i,:,None]-v[i,None,:];pairhits+=float((dif[pair]>0).float().sum());pairden+=int(pair.sum())
   record={'key':r['key'],'row':r['row'],'group':r['group'],'bundle':r['temporal_dependency_bundle'],'weight':weight,'eligible':eligible,'rank1':correct,'MRR':mrr,'unknown_choice':unknown,'NLL':nll,'Brier':brier,'confidence':conf,'calibration_correct':calcorrect,'chosen_candidate_reference':r['candidate_ids'][chosen]if chosen is not None else None,'chosen_utility_known':chosen is not None and bool(qm[i,chosen]),'chosen_H32_utility':float(q[i,chosen])if chosen is not None and bool(qm[i,chosen])else None};records.append(record)
 ew=tot['eval_weight'];ece=0.
 for lo in range(10):
  selected=[r for r in records if r['eligible']and r['confidence']is not None and lo/10<=r['confidence']<((lo+1)/10 if lo<9 else 1.000001)]
  mass=sum(r['weight']for r in selected)
  if mass:ece+=mass/max(ew,1e-12)*abs(sum(r['weight']*r['confidence']for r in selected)/mass-sum(r['weight']*r['calibration_correct']for r in selected)/mass)
 return {'rows':len(rows),'eligible_rows':sum(r['eligible']for r in records),'group_weighted_rank1':tot['rank1']/ew if ew else None,'group_weighted_MRR':tot['MRR']/ew if ew else None,'selection_NLL':tot['NLL']/ew if ew else None,'Brier':tot['Brier']/ew if ew else None,'ECE10':ece,'unknown_choice_rate':tot['unknown_choice']/ew if ew else None,'corrective_recall':tot['corrective']/tot['corrective_den']if tot['corrective_den']else None,'wrong_correction_rate':tot['regression']/tot['regression_den']if tot['regression_den']else None,'correctness_loss':tot['correctness_loss']/tot['ce_weight']if tot['ce_weight']else None,'ranking_loss':tot['ranking_loss']/tot['ranking_weight']if tot['ranking_weight']else None,'executed_pair_order_accuracy':pairhits/pairden if pairden else None,'executed_pair_count':pairden,'strata':{k:{'rank1':v['rank1']/v['weight'],'MRR':v['MRR']/v['weight'],'NLL':v['NLL']/v['weight']}for k,v in strata.items()if v['weight']},'records':records}
