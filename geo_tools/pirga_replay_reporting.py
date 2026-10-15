"""Common physical-query accounting for PIRGA; GT is offline reporting only."""
from collections import Counter,defaultdict
import math


def method_summary(rows,arms):
    result={}
    for arm in arms:
        covered=[r for r in rows if r[arm+'_rr'] is not None]
        result[arm]=dict(queries=len(rows),positive_covered=len(covered),
            top1=sum(r[arm+'_rr']==1 for r in covered)/len(covered) if covered else None,
            mrr=sum(r[arm+'_rr'] for r in covered)/len(covered) if covered else None,
            primary_known_correct=sum(r[arm+'_primary'] is True for r in rows),primary_known_wrong=sum(r[arm+'_primary'] is False for r in rows),
            primary_unknown=sum(r[arm+'_primary'] is None for r in rows),primary_unknown_fraction=sum(r[arm+'_primary'] is None for r in rows)/len(rows) if rows else None,
            rank_unknown=sum(r[arm+'_rank'] is None for r in rows),primary_accepts=sum(r[arm+'_accepted_id'] is not None for r in rows))
    return result


def mechanism_audit(rows,arms,forecast='A4_uncertainty',controls=('C1','A2_rank','AC_contrastive','C3')):
    if len({tuple(r['key']) for r in rows})!=len(rows):raise ValueError('Duplicate physical query')
    errors=[r for r in rows if r['source_primary_state'] in ('wrong_link','missed_known_positive')]
    correct=[r for r in rows if r['source_primary_state'] in ('correct_link','correct_reject')]
    if any(r['C0_primary'] is not False for r in errors) or any(r['C0_primary'] is not True for r in correct):raise ValueError('Original source outcome disagrees with replay')
    reachable=[r for r in errors if r['positive_covered']];methods={};fixes={};harms={};unknown={};newwrong={};keyed={tuple(r['key']):r for r in rows}
    def group(k):return (keyed[k]['key'][0],keyed[k]['person'])
    for arm in arms:
        fixes[arm]={tuple(r['key']) for r in reachable if r[arm+'_primary'] is True and r[arm+'_accepted_id'] is not None}
        harms[arm]={tuple(r['key']) for r in correct if r[arm+'_primary'] is False}
        unknown[arm]={tuple(r['key']) for r in correct if r[arm+'_primary'] is None}
        newwrong[arm]={tuple(r['key']) for r in rows if r['C0_primary'] is None and r[arm+'_primary'] is False}
        methods[arm]=dict(original_error_repairs=len(fixes[arm]),known_harms_to_original_correct=len(harms[arm]),original_correct_to_unknown=len(unknown[arm]),
            original_unknown_to_known_wrong=len(newwrong[arm]),repair_person_scene_groups=len({group(k) for k in fixes[arm]}),
            net_repairs_minus_known_harms=len(fixes[arm])-len(harms[arm]),
            fixes_by_original_error_type=dict(Counter(keyed[k]['source_primary_state'] for k in fixes[arm])),
            fixed_keys=[list(k) for k in sorted(fixes[arm])],harmed_keys=[list(k) for k in sorted(harms[arm])],
            unknown_to_wrong_keys=[list(k) for k in sorted(newwrong[arm])],
            original_error_rejected_without_legal_positive=sum(not r['positive_covered'] and r[arm+'_accepted_id'] is None for r in errors),
            full_online_identity_repairs=None)
    comparisons={}
    for baseline in controls:
        extra=fixes[forecast]-fixes[baseline];lost=fixes[baseline]-fixes[forecast]
        additional=harms[forecast]-harms[baseline];avoided=harms[baseline]-harms[forecast];groups=defaultdict(Counter)
        for name,subset in [('extra_repairs',extra),('lost_repairs',lost),('additional_harms',additional),('avoided_harms',avoided)]:
            for k in subset:groups[group(k)][name]+=1
        comparisons[forecast+'_vs_'+baseline]=dict(extra_original_repairs=len(extra),lost_original_repairs=len(lost),
            net_original_repairs=len(extra)-len(lost),net_original_repairs_minus_additional_harms=len(extra)-len(lost)-len(additional)+len(avoided),
            additional_known_harms=len(additional),avoided_known_harms=len(avoided),
            additional_original_correct_to_unknown=len(unknown[forecast]-unknown[baseline]),
            additional_original_unknown_to_wrong=len(newwrong[forecast]-newwrong[baseline]),
            net_positive_person_scene_groups=sum(g['extra_repairs']-g['lost_repairs']-g['additional_harms']+g['avoided_harms']>0 for g in groups.values()),
            by_person_scene={str(k):dict(v) for k,v in groups.items()},extra_fix_keys=[list(k) for k in sorted(extra)],lost_fix_keys=[list(k) for k in sorted(lost)],
            primary_action_changes=sum(r[forecast+'_accepted_id']!=r[baseline+'_accepted_id'] for r in rows),
            certification_transitions=dict(Counter(str(r[baseline+'_primary'])+' -> '+str(r[forecast+'_primary']) for r in rows)))
    conditions=dict(reachable_errors_present=bool(reachable))
    for baseline in controls:
        v=comparisons[forecast+'_vs_'+baseline]
        conditions['extra_net_native_repairs_over_'+baseline]=v['net_original_repairs_minus_additional_harms']>0 and v['net_original_repairs']>0
        conditions['no_added_known_harms_over_'+baseline]=len(harms[forecast])<=len(harms[baseline])
        conditions['no_added_unknown_conversion_over_'+baseline]=len(unknown[forecast])<=len(unknown[baseline]) and len(newwrong[forecast])<=len(newwrong[baseline])
    conditions['net_extra_repairs_in_at_least2people_over_C1']=comparisons[forecast+'_vs_C1']['net_positive_person_scene_groups']>=2
    if not reachable:status='INSUFFICIENT_NATIVE_OPPORTUNITY'
    elif not conditions['extra_net_native_repairs_over_C1']:status='NO_PREDICTION_SPECIFIC_GAIN'
    elif not all(conditions['extra_net_native_repairs_over_'+b] for b in controls):status='NO_UNIQUE_PREDICTIVE_EFFECT'
    elif all(conditions.values()):status='NATIVE_REPLAY_POSITIVE_PENDING_STRONG_NATIVE_DECISION_CONTROL'
    else:status='NATIVE_REPLAY_MIXED_NOT_ADMITTED'
    return dict(status=status,original_errors=len(errors),reachable_original_errors=len(reachable),methods=methods,comparisons=comparisons,
        gate_conditions=conditions,phase2_mechanism_gate_pass=all(conditions.values()),online_admitted=False,
        online_gate_pending='Strong ordinary native decision-aware MLP and causal native integration remain required even if Phase2 is positive.',scientific_success=False)


def correlation(xs,ys):
    if len(xs)<2:return None
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys);xx=sum((x-mx)**2 for x in xs);yy=sum((y-my)**2 for y in ys)
    if not xx or not yy:return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(xx*yy)


def uncertainty_report(rows):
    """Rows are per physical query, with positive aliases already averaged."""
    values=[r for r in rows if r['positive_variance'] is not None]
    boundaries=[.05,.1,.25,.5,1.,2.,4.00001];groups=defaultdict(list)
    for r in values:
        b=next((i for i in range(len(boundaries)-1) if boundaries[i]<=r['positive_variance']<boundaries[i+1]),None)
        groups[str(b)].append(r)
    variance=[r['positive_variance'] for r in values];error=[r['positive_cosine_error'] for r in values]
    mean=sum(variance)/len(variance) if variance else None
    std=(sum((x-mean)**2 for x in variance)/len(variance))**.5 if variance else None
    certified=[r for r in values if r['primary_correct'] is not None]
    return dict(physical_queries=len(values),person_scene_groups=len({(r['key'][0],r['person']) for r in values}),
        variance_mean=mean,variance_std=std,variance_effectively_constant=std is not None and std<1e-5,
        error_variance_pearson=correlation(error,variance),variance_vs_primary_wrong_pearson=correlation([r['positive_variance'] for r in certified],[int(r['primary_correct'] is False) for r in certified]),
        variance_bin_boundaries=boundaries,by_variance_bin={k:dict(queries=len(v),mean_variance=sum(x['positive_variance'] for x in v)/len(v),
            mean_cosine_error=sum(x['positive_cosine_error'] for x in v)/len(v),certified_primary_wrong=sum(x['primary_correct'] is False for x in v),
            primary_unknown=sum(x['primary_correct'] is None for x in v)) for k,v in groups.items()},
        qualification='Query-level mean over certified pure positive aliases. Offline GT selects analysis targets only. Variance models forecast error, not calibrated identity probability; one exposed DEV scene is not independent validation.')
