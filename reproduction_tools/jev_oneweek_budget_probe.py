"""Same-state, GT-blind event-budget diagnostic on the immutable TRAIN16 corpus.

Exact budgets force each scorer's best non-base proposal on the same hashed
events. They are diagnostic interventions, never replacement online results.
"""
import collections
import hashlib
import math
import torch
from jev_oneweek_common import *
from jev_oneweek_assess import load_model
from jev_oneweek_learning import load_records,batch


def comparison(record,base,choice):
    c=collections.Counter();k=len(record['refs'])
    for row,(old,new) in enumerate(zip(base,choice)):
        if old==new:continue
        c['reassigned_rows']+=1;c['existing_to_DEFER']+=new<0;c['DEFER_to_existing']+=old<0
        a=k if old<0 else old;b=k if new<0 else new
        if record['known_options'][row,a] and record['known_options'][row,b]:
            u=int(record['positive'][row,a]);v=int(record['positive'][row,b]);c[f'N{u}{v}']+=1
        else:c['UNKNOWN_changed_rows']+=1
    return dict(c)


def main():
    protect();protocol=REPORTS/'BUDGET_PROBE_PROTOCOL.json'
    setup=dict(status='FROZEN_BEFORE_DIAGNOSTIC_OUTPUTS',seed=20261009,video=16,
               rates=[0,.01,.05,.10,.25,1.],event_selection='SHA256 of natural event key, independent of scores/GT; same exact eligible events for all policies',
               eligible='Immutable baseline-state natural TRAIN16 events with >=2 common legal proposals',
               arms=['jev','mlp','attention','deepsets','threshold','fixed_threshold','fixed_fusion','extra_hungarian'],
               force_best_nonbase_only_for_exact_budget_diagnostic=True,online_policy_or_checkpoint_selection=False)
    if protocol.exists():assert read(protocol)==setup
    else:save(protocol,setup)
    train,records,_=load_records();del train
    networks={arm:load_model(arm,20261009,'formal')[0] for arm in setup['arms'][:5]}
    rows=[];original=collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for i,record in enumerate(records):
            x,_=batch([record]);all_details={arm:network.details(x) for arm,network in networks.items()}
            first=all_details['jev'];alternatives=first['plans'][0]
            assert all(d['plans'][0]==alternatives for d in all_details.values())
            assert all(torch.equal(d['base_logits'],first['base_logits']) for d in all_details.values())
            if len(alternatives)<2:continue
            base=alternatives[0];logits=first['base_logits'][0];k=len(record['refs'])
            score_vectors={arm:d['plan_scores'][0].cpu().tolist() for arm,d in all_details.items()}
            for arm in ['fixed_threshold','fixed_fusion','extra_hungarian']:
                modified=logits.clone()
                if arm=='fixed_threshold':modified[:,:-1]-=.1
                elif arm=='fixed_fusion':
                    p=x['pair_evidence'][0];c=x['commitment_features'][0]
                    modified[:,:-1]+=.1*(p[:,:,3]+c[:,:,0]-p[:,:,10:14].square().sum(-1).sqrt())
                score_vectors[arm]=[sum(float(modified[r,k if col<0 else col]) for r,col in enumerate(plan)) for plan in alternatives]
            outcome={};best_nonbase={}
            for arm,scores in score_vectors.items():
                choice=max(range(len(scores)),key=lambda index:scores[index])
                nonbase=max(range(1,len(scores)),key=lambda index:scores[index])
                original[arm]['events']+=1;original[arm]['nonbase_events']+=choice!=0
                original[arm].update(comparison(record,base,alternatives[choice]))
                best_nonbase[arm]=nonbase;outcome[arm]=comparison(record,base,alternatives[nonbase])
            rows.append(dict(key=list(record['key']),event_hash=hashlib.sha256(json.dumps(list(record['key'])).encode()).hexdigest(),
                             outcomes=outcome,proposals=len(alternatives),input_GT=False))
            if i%128==0:
                save(OUT/'budget_probe/PROGRESS.json',dict(status='RUNNING',payload=i,total=len(records),eligible=len(rows)))
                print('BUDGET_DIAGNOSTIC_PROGRESS',i,len(records),len(rows),flush=True)
    order=sorted(rows,key=lambda r:r['event_hash']);budgets=[]
    for rate in setup['rates']:
        size=math.floor(rate*len(order));selected=order[:size];methods={}
        for arm in setup['arms']:
            totals=collections.Counter()
            for row in selected:totals.update(row['outcomes'][arm])
            methods[arm]=dict(exact_nonbase_event_budget=size,counts=dict(totals))
        budgets.append(dict(requested_rate=rate,eligible_events=len(order),selected_event_keys_SHA256=hashlib.sha256(json.dumps([r['key'] for r in selected]).encode()).hexdigest(),methods=methods))
    save(OUT/'budget_probe/LOCAL_EVENT_OUTCOMES.json',rows)
    save(REPORTS/'SAME_STATE_BUDGET_DIAGNOSTIC.json',dict(status='COMPLETE',protocol=ref(protocol),
         natural_raw_policy_results={k:dict(v) for k,v in original.items()},budget_diagnostics=budgets,
         local_event_outcomes=ref(OUT/'budget_probe/LOCAL_EVENT_OUTCOMES.json'),
         identical_starting_state_tensors_candidates_and_base_scores=True,
         current_plan_compatibility_scope_only=True,full_mutated_state_online_metrics=None,
         no_future_or_GT_used_to_select_events_proposals_or_scores=True,
         forcing_low_score_reassociation_is_not_the_original_policy=True,
         UNKNOWN_not_interpreted_as_negative=True))


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('budget_probe',traceback.format_exc());raise
