"""Separate certified active-ID recovery from legal private MATCH DEFER."""
import collections,torch
from jev_recovery_common import *
from jev_recovery_learning import load_records,load_model,batch


def main():
    protect();torch.set_num_threads(1)
    _,records,sources=load_records(2);actor,checkpoint,_=load_model('jev',20261009,'pilot',2,'XIV')
    count=collections.Counter();examples=[]
    with torch.no_grad():
        for event,r in enumerate(records):
            x,_=batch([r]);d=actor.details(x);pp=d['plans'][0]
            before=pp[0];after=pp[int(d['plan_scores'][0].argmax())];k=len(r['refs'])
            for row,(u,v) in enumerate(zip(before,after)):
                a,b=k if u<0 else u,k if v<0 else v
                if bool(r['known_options'][row,a]) and bool(r['known_options'][row,b]):
                    aa,bb=int(r['positive'][row,a]),int(r['positive'][row,b])
                    if aa!=bb:
                        tag=f'N{aa}{bb}';count[tag]+=1
                        count[tag+('_private_MATCH_DEFER' if v<0 else '_existing_active_ID')]+=1
                if v<0 or not bool(r['known_options'][row,b]) or bool(r['positive'][row,b]):continue
                cross=int(r['oneweek_labels']['cross'][row,v]);persistent=int(r['oneweek_labels']['stability'][row,v])
                if cross<0 or persistent<0:continue
                if bool(d['cross_logits'][0,row,v]>=0)!=bool(cross) or bool(d['stability_logits'][0,row,v]>=0)!=bool(persistent):continue
                count['both_aux_correct_final_wrong']+=1
                count[f'Cross_true{cross}_Persistent_true{persistent}']+=1
                legal=[c for c in range(k) if bool(r['inputs']['legal'][0,row,c]) and bool(r['known_options'][row,c]) and bool(r['positive'][row,c])]
                covered=[c for c in legal if any(p[row]==c for p in pp)]
                defer_correct=bool(r['known_options'][row,k]) and bool(r['positive'][row,k])
                defer_covered=defer_correct and any(p[row]<0 for p in pp)
                if covered:category='certified_active_ID_in_plans'
                elif defer_covered:category='only_certified_private_DEFER_in_plans'
                elif legal:category='certified_active_ID_legal_but_not_in_plans'
                elif defer_correct:category='certified_DEFER_not_in_plans'
                else:category='no_certified_correct_alternative'
                count[category]+=1
                count['any_certified_correct_action_in_plans']+=bool(covered) or defer_covered
                if len(examples)<40:
                    edge=[dict(native_ID=r['refs'][c],cross_target=int(r['oneweek_labels']['cross'][row,c]),
                         persistent_target=int(r['oneweek_labels']['stability'][row,c]),
                         cross_sigmoid_uncalibrated=float(d['cross_logits'][0,row,c].sigmoid()),
                         persistent_sigmoid_uncalibrated=float(d['stability_logits'][0,row,c].sigmoid()),
                         base_score=float(d['base_logits'][0,row,c]),final_edge_score=float(d['edge_logits'][0,row,c])) for c in [v,*covered]]
                    order=d['plan_scores'][0].argsort(descending=True).tolist()
                    examples.append(dict(key=r['key'],row=row,category=category,selected_native_ID=r['refs'][v],
                        certified_correct_legal_native_IDs=[r['refs'][c] for c in legal],
                        certified_correct_covered_native_IDs=[r['refs'][c] for c in covered],
                        private_DEFER_correct=defer_correct,private_DEFER_in_plans=defer_covered,
                        plan_scores=d['plan_scores'][0].tolist(),plan_rank=order,candidate_scores=edge,
                        actual_source_state_native_commit='NOT_REPLAYED_FOR_THIS_SPECIFIC_ROW; separate H64 and complete online real commits'))
            if event%256==0:print('CERTIFIED_ALTERNATIVE_EVENT',event,len(records),flush=True)
    original=read(REPORTS/'QUESTION_TO_ASSOCIATION_DIAGNOSIS.json')['interventions']['full']
    assert count['N01']==original['N01'] and count['N10']==original['N10']
    assert count['both_aux_correct_final_wrong']==original['both_aux_correct_final_wrong']
    assert count['certified_active_ID_in_plans']==original['both_correct_final_wrong_with_legal_correct_alternative']
    save(REPORTS/'QUESTION_CERTIFIED_ALTERNATIVE_AUDIT.json',dict(status='COMPLETE_SAME_FROZEN_CHECKPOINT',
        counts=dict(count),examples=examples,sources=sources,checkpoint=checkpoint,
        original_diagnostic=ref(REPORTS/'QUESTION_TO_ASSOCIATION_DIAGNOSIS.json'),
        clarification='Original 188/475 split means covered certified ACTIVE-ID alternative versus no such covered ACTIVE-ID. The latter does not assert all legal actions including private DEFER are uncertifiable.',
        private_DEFER_not_certified_existing_Global_ID_restoration=True,old_result_preserved=True,
        original_counts_reproduced_exactly=True,independent_validation=False))
    print('CERTIFIED_ALTERNATIVE_COMPLETE',dict(count),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('certified_alternative_audit',traceback.format_exc());raise
