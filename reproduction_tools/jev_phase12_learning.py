"""Strict separation of current runtime tensors and partial offline supervision."""
import collections,math,dataclasses
import torch
import torch.nn.functional as F
from jev_phase12_common import *
from gtr.modeling.visual_jev_mcmot import *
from gtr.modeling.visual_jev_mcmot.schemas import masked_softmax
from gtr.modeling.visual_jev_mcmot.calibration import known_distribution

def load_data():
    protect();manifest=json.loads((OUT/'visual_dataset_v1/MANIFEST.json').read_text())
    assert manifest['status']=='PASS' and sha(manifest['path'])==manifest['SHA256']
    data=torch.load(manifest['path'],map_location='cpu');return data['rows'],manifest

def gates():
    names=['STRUCTURAL_TESTS','QUESTION_CONDITIONING_TESTS','NATIVE_PARITY','SHADOW_MODE_TESTS','SUPERVISION_CONTRACT_TESTS']
    for name in names:
        assert json.loads((REPORTS/(name+'.json')).read_text())['status']=='PASS',name
    return {name:sha(REPORTS/(name+'.json')) for name in names}

def normalization(rows):
    from jev_phase10_learning import normalization as numeric_stats
    stats=numeric_stats(rows)
    selected=[r for r in rows if r['partition']=='train']
    scales=[]
    for h in range(3):
        values=torch.cat([r['consequence_targets'][:,h][r['consequence_mask'][:,h]] for r in selected])
        scales.append(float(values.std(unbiased=False).clamp_min(1.)))
    stats['consequence_scale']=torch.tensor(scales)
    return stats

class LegacyNumericAdapter(torch.nn.Module):
    """Original numerical architecture with a shared multihorizon output adapter.

    This is the added-input comparison, not the same-visual capacity comparison.
    Phase X two-head network unchanged; 2->3 auxiliary consequence projection
    gives exactly the same executed horizon label support. No visual NN inputs.
    """
    def __init__(self,name):
        super().__init__()
        from gtr.modeling.jev_candidate_models import build_candidate_model
        self.network=build_candidate_model(name)
        self.consequence=torch.nn.Linear(2,3)
        self.variant=name
    def forward(self,state,questions,options):
        b,q,k=options.mask.shape
        s=questions.context.reshape(b*q,64);e=options.evidence.reshape(b*q,k,12);mask=options.mask.reshape(b*q,k)
        logits=self.network(s,e,mask).reshape(b,q,k,2)
        return {'choice_logits':logits[...,0],'consequences':self.consequence(logits)}

class NumericalOnlyVisualAdapter(VisualJev):
    """Same full reader architecture/capacity with visual tensors removed."""
    def __init__(self):
        super().__init__('full');self.variant='numerical_only'
    def forward(self,state,questions,options):
        state=dataclasses.replace(state,visual=torch.zeros_like(state.visual))
        questions=dataclasses.replace(questions,visual=torch.zeros_like(questions.visual))
        options=dataclasses.replace(options,visual=torch.zeros_like(options.visual))
        return super().forward(state,questions,options)

def build_network(name,stats):
    model=LegacyNumericAdapter(name) if name.startswith('Candidate') else NumericalOnlyVisualAdapter() if name=='numerical_only' else VisualJev(name)
    numeric=model.network if isinstance(model,LegacyNumericAdapter) else model
    with torch.no_grad():
        for field,value in stats.items():
            if hasattr(numeric,field):getattr(numeric,field).copy_(value)
    return model

def collate(rows,device='cpu'):
    b=len(rows);k=max(len(r['candidate_ids']) for r in rows);t=max(r['visual_state'].visual.shape[1] for r in rows)
    visual=torch.zeros(b,t,1152);metadata=torch.zeros(b,t,8);sm=torch.zeros(b,t,dtype=torch.bool)
    qv=torch.zeros(b,1,1152);qc=torch.stack([r['state64'] for r in rows])[:,None]
    ov=torch.zeros(b,1,k,3,1152);om=torch.zeros(b,1,k,3,dtype=torch.bool);e=torch.zeros(b,1,k,12);legal=torch.zeros(b,1,k,dtype=torch.bool)
    known=torch.zeros(b,k,dtype=torch.bool);positive=known.clone();targets=torch.full((b,k,3),float('nan'));qm=torch.zeros(b,k,3,dtype=torch.bool)
    for i,row in enumerate(rows):
        tt=row['visual_state'].visual.shape[1];kk=len(row['candidate_ids'])
        visual[i,:tt]=row['visual_state'].visual[0];metadata[i,:tt]=row['visual_state'].metadata[0];sm[i,:tt]=row['visual_state'].mask[0]
        qv[i]=row['visual_question'].visual[0]
        opts=row['visual_options'];ov[i,:,:kk]=opts.visual[0];om[i,:,:kk]=opts.visual_mask[0];e[i,:,:kk]=opts.evidence[0];legal[i,:,:kk]=opts.mask[0]
        known[i,:kk]=row['known_mask'];positive[i,:kk]=row['positive_mask'];targets[i,:kk]=row['consequence_targets'];qm[i,:kk]=row['consequence_mask']
    move=lambda x:x.to(device)
    state=OnlineVisualState(move(visual),move(metadata),move(sm))
    question=QuestionDescriptor(move(qv),move(qc),torch.zeros(b,1,device=device,dtype=torch.long),torch.ones(b,1,device=device,dtype=torch.bool))
    options=OptionTensors(move(ov),move(om),move(e),torch.zeros(b,1,k,device=device,dtype=torch.long),move(legal))
    # Targets are a separate return value, never supplied to model.forward.
    labels={'known':move(known),'positive':move(positive),'targets':move(targets),'utility_mask':move(qm),'weight':move(torch.tensor([r['weight'] for r in rows])),'legal':move(legal[:,0])}
    return (state,question,options),labels

def preference(output,supervision,variant):
    logits=output['choice_logits'][:,0];q=output['consequences'][:,0,:,2]
    if supervision=='H32':return q
    if supervision=='joint' and variant not in {'no_H32','no_consequence'}:return logits+.25*q
    return logits

def losses(output,labels,stats,supervision,variant):
    logits=output['choice_logits'][:,0];known=labels['known'];positive=labels['positive'];targets=labels['targets'];qm=labels['utility_mask']
    scale=stats['consequence_scale'].to(logits.device)
    ce=logits.new_zeros(len(logits));ceok=positive.any(-1)&known.any(-1)
    if ceok.any():ce[ceok]=torch.logsumexp(logits[ceok].masked_fill(~known[ceok],-torch.inf),-1)-torch.logsumexp(logits[ceok].masked_fill(~positive[ceok],-torch.inf),-1)
    rank=logits.new_zeros(len(logits));rankok=torch.zeros(len(logits),device=logits.device,dtype=torch.bool)
    rank_values=output['consequences'][:,0,:,2] if supervision=='H32' else preference(output,supervision,variant)
    for i in range(len(logits)):
        values=targets[i,:,2];mask=qm[i,:,2]
        pairs=mask[:,None]&mask[None,:]&(values[:,None]-values[None,:]>1e-6)
        if pairs.any():rank[i]=F.softplus(-(rank_values[i,:,None]-rank_values[i,None,:])[pairs]).mean();rankok[i]=True
    qloss=logits.new_zeros(len(logits));qok=qm.flatten(1).any(-1)
    predicted=output['consequences'][:,0]
    for i in range(len(logits)):
        if qok[i]:qloss[i]=F.smooth_l1_loss(predicted[i][qm[i]],(targets[i]/scale)[qm[i]])
    active='CE' if variant=='no_H32' else supervision
    total=ce if active=='CE' else rank+.1*qloss if active=='H32' else ce+rank+(0 if variant=='no_consequence' else .1*qloss)
    return total,{'ce':ce,'ce_eligible':ceok,'ranking':rank,'ranking_eligible':rankok,'Q':qloss,'Q_eligible':qok}

@torch.no_grad()
def evaluate(model,rows,stats,supervision,device='cpu',temperature=1.):
    model.eval();records=[];total=collections.defaultdict(float);pair_correct=pair_total=0
    for start in range(0,len(rows),16):
        chunk=rows[start:start+16];args,labels=collate(chunk,device);output=model(*args)
        assert torch.isfinite(output['choice_logits']).all() and torch.isfinite(output['consequences']).all()
        pref=preference(output,supervision,model.variant)/temperature;legal=labels['legal'];known=labels['known'];positive=labels['positive'];target=labels['targets'][:,:,2];qm=labels['utility_mask'][:,:,2]
        _,parts=losses(output,labels,stats,supervision,model.variant)
        for i,row in enumerate(chunk):
            kk=len(row['candidate_ids']);mask=legal[i,:kk];kknown=known[i,:kk];pos=positive[i,:kk];w=row['weight']
            order=torch.argsort(pref[i,:kk].masked_fill(~mask,-torch.inf),descending=True,stable=True).tolist();chosen=order[0] if mask.any() else None
            certificate='CORRECT' if chosen is not None and bool(pos[chosen]) else 'WRONG' if chosen is not None and bool(kknown[chosen]) else 'UNKNOWN'
            total['weight']+=w;total[certificate]+=w
            ranks=[rank+1 for rank,col in enumerate(order) if bool(pos[col])];mrr=1/min(ranks) if ranks else None
            nll=brier=confidence=calcorrect=None
            if pos.any():
                probabilities,nlls,briers=known_distribution(pref[i:i+1,:kk],kknown[None],pos[None]);nll=float(nlls[0]);brier=float(briers[0]);knownbest=int(probabilities[0].argmax());confidence=float(probabilities[0,knownbest]);calcorrect=bool(pos[knownbest])
                total['eligible_weight']+=w;total['rank1']+=w*(certificate=='CORRECT');total['MRR']+=w*mrr;total['NLL']+=w*nll;total['Brier']+=w*brier
            regret=None
            if chosen is not None and qm[i,chosen]:
                regret=float(target[i][qm[i]].max()-target[i,chosen]);total['regret_weight']+=w;total['regret']+=w*regret
            pairs=qm[i,:,None]&qm[i,None,:]&(target[i,:,None]-target[i,None,:]>1e-6)
            if pairs.any():
                differences=pref[i,:,None]-pref[i,None,:];pair_correct+=int((differences[pairs]>0).sum());pair_total+=int(pairs.sum())
            for name,eligibility in [('ce','ce_eligible'),('ranking','ranking_eligible'),('Q','Q_eligible')]:
                if bool(parts[eligibility][i]):total[name+'_weight']+=w;total[name]+=w*float(parts[name][i])
            records.append({'key':row['key'],'row':row['row'],'group':row['group'],'bundle':row['temporal_dependency_bundle'],'weight':w,'certificate':certificate,'chosen_ref':row['candidate_ids'][chosen] if chosen is not None else None,'rank1':certificate=='CORRECT','MRR':mrr,'known_NLL':nll,'known_Brier':brier,'known_confidence':confidence,'known_calibration_correct':calcorrect,'executed_H32_regret':regret,'selected_Q_unknown':regret is None})
    weighted=lambda field,den:total[field]/total[den] if total[den] else None
    ew=total['eligible_weight'];ece=0.
    for index in range(10):
        selected=[r for r in records if r['known_confidence'] is not None and index/10<=r['known_confidence']<((index+1)/10 if index<9 else 1.000001)];mass=sum(r['weight'] for r in selected)
        if mass:ece+=mass/max(ew,1e-12)*abs(sum(r['weight']*r['known_confidence'] for r in selected)/mass-sum(r['weight']*r['known_calibration_correct'] for r in selected)/mass)
    return {'rows':len(rows),'CertifiedCorrect':weighted('CORRECT','weight'),'CertifiedWrong':weighted('WRONG','weight'),'UNKNOWNChoice':weighted('UNKNOWN','weight'),'rank1':weighted('rank1','eligible_weight'),'MRR':weighted('MRR','eligible_weight'),'known_NLL':weighted('NLL','eligible_weight'),'known_Brier':weighted('Brier','eligible_weight'),'known_ECE10':ece if ew else None,'executed_H32_regret':weighted('regret','regret_weight'),'executed_regret_label_coverage':weighted('regret_weight','weight'),'executed_pair_order_accuracy':pair_correct/pair_total if pair_total else None,'executed_pairs':pair_total,'CE_loss':weighted('ce','ce_weight'),'ranking_loss':weighted('ranking','ranking_weight'),'Q_loss':weighted('Q','Q_weight'),'records':records}
