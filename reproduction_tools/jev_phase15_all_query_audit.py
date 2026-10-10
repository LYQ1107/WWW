"""Post-freeze coverage audit includes queries excluded by WHO certification."""
import collections
import numpy as np
import torch
from jev_phase15_common import *
from jev_phase15_train import load_records,batch,initialize
from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy
from gtr.modeling.jev_stage2.assignment import lawful_choice


@torch.no_grad()
def audit(model,records,limit=256):
    model.eval();counts=collections.Counter();scored=[];examples=[]
    chosen=set(np.linspace(0,len(records)-1,min(limit,len(records)),dtype=int).tolist())
    corrections=[i for i,r in enumerate(records) if (r['commit_kind']==2).any()]
    for i in np.linspace(0,len(corrections)-1,min(limit,len(corrections)),dtype=int).tolist() if corrections else []:chosen.add(corrections[i])
    items=[records[i] for i in sorted(chosen)]
    for start in range(0,len(items),4):
        part=items[start:start+4];x,y=batch(part);details=model.details(x)
        for index,r in enumerate(part):
            q=len(r['rows']);k=len(r['refs']);z=torch.cat([details['logits'][index,:q,:k],details['logits'][index,:q,-1:]],-1)
            choices=lawful_choice(z,x['legal'][index,:q,:k]);confidence=z.softmax(-1).max(-1).values.cpu().tolist()
            for row,col in enumerate(choices):
                col=k if col<0 else col;gt=r['GT_labels_OFFLINE_ONLY'][row];counts['all_query_rows']+=1
                counts['DEFER_all_rows']+=int(col==k)
                if gt is None:counts['current_GT_UNKNOWN_rows']+=1;continue
                counts['current_GT_known_rows']+=1;certified=bool(r['supervised'][row])
                counts['WHO_certified_query_rows']+=certified;known=bool(r['known_options'][row,col]);correct=bool(r['positive'][row,col])
                counts['selected_UNKNOWN_all_GT_known_rows']+=int(not known)
                counts['selected_UNKNOWN_uncertified_queries']+=int(not known and not certified)
                counts['selected_certified_wrong_all_GT_known_rows']+=int(known and not correct)
                counts['selected_certified_correct_all_GT_known_rows']+=int(known and correct)
                mixed=col<k and int(r['trust'][col])==0;counts['selected_globally_mixed_all_GT_known_rows']+=mixed
                pure_support=bool(r['positive'][row,:k].any());counts['pure_supported_queries']+=pure_support
                counts['no_certified_pure_correct_candidate_queries']+=int(not pure_support)
                counts['mixed_selected_despite_pure_alternative']+=int(mixed and pure_support)
                scored.append((confidence[row],known,correct,mixed))
                if mixed and pure_support and len(examples)<16:examples.append(dict(key=r['key'],row=row,selected_ID=r['refs'][col],confidence=confidence[row],
                    safe_pure_alternative_IDs=[r['refs'][c] for c in torch.where(r['positive'][row,:k])[0].tolist()]))
    curves=[]
    for threshold in [.5,.7,.9]:
        selected=[v for v in scored if v[0]>=threshold];known=[v for v in selected if v[1]]
        curves.append(dict(confidence_threshold=threshold,all_GT_known_query_coverage=len(selected)/max(1,len(scored)),
            selected_UNKNOWN=sum(not v[1] for v in selected),selected_mixed=sum(v[3] for v in selected),
            certified_selection_support=len(known),certified_selection_risk=sum(not v[2] for v in known)/len(known) if known else None))
    return dict(counts=dict(counts),risk_coverage=curves,examples=examples,
        UNKNOWN_is_not_certified_wrong=True,scope='reserved real TRAIN queries, including uncertified WHO queries; UNKNOWN current GT unassessed')


def main(version=1):
    protect();torch.set_num_threads(1);torch.manual_seed(20261009)
    train,reserved,sources=load_records(label_version=2 if version>=2 else 1,onpolicy_pilot=version>=3)
    constants=collections.Counter()
    for r in train+reserved:
        f=r['inputs']['commitment_features'];constants['payloads']+=1
        constants['posterior_input_nonconstant_payloads']+=int(bool((f[...,16]!=.5).any() or (f[...,17]!=0).any()))
    path=OUT/f'training_full_payload_v{version}/F_full/seed20261009/pilot/RESULT.json';result=read(path);ck=result['checkpoint']
    assert sha(ck['path'])==ck['SHA256']
    weights=torch.load(ck['path'],map_location='cpu');full=PersistentIdentityPolicy('F_full').cuda().eval();full.load_state_dict(weights['model'],strict=True)
    full.posterior_feedback=weights.get('posterior_feedback','native')
    original,oldck,oldsource=initialize('A_original')
    findings=dict(original= audit(original,reserved),F_full=audit(full,reserved))
    source=binding(seed=20261009,checkpoints=[ck,oldck],dataset=sources,evaluator='post-freeze all-query constrained assignment',scope='coverage repair; no optimizer updates, no DEV')
    value=dict(status='COMPLETE',binding=source,version=version,results=findings,training_input_support=dict(constants),
        old_conditional_assessment_retained=True,no_extra_optimizer_updates=True)
    save(OUT/f'all_query_audit_v{version}/RESULT.json',value);save(REPORTS/f'ALL_QUERY_RELIABILITY_V{version}.json',value)
    print('PHASE15_ALL_QUERY_AUDIT',version,{k:v['counts'] for k,v in findings.items()},dict(constants),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);a=p.parse_args();main(a.version)
