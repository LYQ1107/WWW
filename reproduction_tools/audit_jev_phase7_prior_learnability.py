"""Read-only audit of preserved canonical lifecycle and historical B2 labels."""
from collections import Counter
import json
import numpy as np
from jev_phase7_common import *

def quantiles(values):
    return dict(zip(['p0','p25','p50','p75','p90','p100'],map(float,np.quantile(values,[0,.25,.5,.75,.9,1]))))if values else None

def summarize_labels(paths):
    rows=[json.loads(p.read_text())for p in paths];margins=[];winners=Counter();weights=[];diverged=0;known=0;ties=0
    for r in rows:
        actions=r['legal_actions'];utilities=[]
        for action in actions:
            outcomes=r['branches'][action]['outcomes'];last=max(outcomes,key=int);utilities.append(outcomes[last]['utility'])
        assert all(np.isfinite(utilities));ordered=sorted(utilities,reverse=True);margin=ordered[0]-ordered[1];margins.append(margin)
        known+=r['offline_GT']is not None;ties+=margin<=1e-8
        if r['informative']:
            weights.append(r['sample_weight']);winners.update(r['best_actions'])
        hashes={r['branches'][a]['prediction_sha256']for a in actions};diverged+=len(hashes)>1
    known_margins=[m for m,r in zip(margins,rows)if r['offline_GT']is not None]
    ess=sum(weights)**2/sum(w*w for w in weights)if weights else 0
    return {'events':len(rows),'known_target':known,'informative':sum(r['informative']for r in rows),
        'unique_best':sum(r['offline_GT']is not None and len(r['best_actions'])==1 and r['informative']for r in rows),
        'tie_count':ties,'tie_rate':ties/max(1,len(rows)),'positive_margin_rate_raw_including_unknown':sum(m>1e-8 for m in margins)/max(1,len(rows)),
        'positive_margin_rate_known_target':sum(m>1e-8 for m in known_margins)/max(1,len(known_margins)),
        'known_target_utility_margin_quantiles':quantiles(known_margins),
        'unknown_positive_margin_is_supervision':False,
        'utility_margin_quantiles':quantiles(margins),'best_action_counts_informative':dict(winners),'effective_sample_size':ess,
        'prediction_branch_divergence':diverged,'native_state_divergence':'not preserved as complete-field hashes in these labels; do not infer from prediction hashes',
        'GT_or_future_inputs':any(r.get('GT_or_future_inputs',False)for r in rows),
        'source_files':{str(p):sha(p)for p in paths}}


def main():
    protect();prior=ROOT/'reports/JEV_PHASE6';runtime=Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008/lifecycle_native')
    space=json.loads((prior/'REACTIVATION_ACTION_SPACE_AUDIT.json').read_text());react={};memory={}
    for video in (24,23):
        root=runtime/f'video{video:02d}';events=space['videos'][str(video)]['events']
        raw=[json.loads(l)for l in (root/'canonical_reactivation.jsonl').read_text().splitlines()]
        assert len(raw)==len(events)
        margins=[];scores=[]
        for r in raw:
            values=sorted(r['online_context']['candidate_scores'],reverse=True)
            if len(values)>1:margins.append(values[0]-values[1])
            scores.extend(values)
        labels=summarize_labels(sorted((root/'labels').glob('REACT_*.json')))
        reactive={'events':len(events),'candidate_pool_count_distribution':dict(Counter(e['candidate_count']for e in events)),
            'threshold_valid_stale_count_distribution':dict(Counter(e['valid_stale_count']for e in events)),
            'assessable_prefix_identity':space['videos'][str(video)]['assessable'],'known_current_GT':sum(e['GT']is not None for e in events),
            'assigned_wrong_known_prefix':sum(e['assigned_wrong']for e in events),
            'assigned_correct_known_prefix':sum(e['GT']is not None and e['assigned_prefix_GT']==e['GT']for e in events),
            'correct_assigned_or_valid_alternate_lower_bound':sum(e['GT']is not None and (e['assigned_prefix_GT']==e['GT']or e['alternate_correct_for_assigned'])for e in events),
            'assigned_wrong_with_valid_correct_alternative':sum(e['assigned_wrong_with_valid_correct_alternative']for e in events),
            'top1_top2_score_margin_quantiles':quantiles(margins),'raw_candidate_score_quantiles':quantiles(scores),
            'binary_canonical_labels':labels,'third_action_eligible':False,
            'native_relative_hook_status':'BLOCKED_NATIVE_RELATIVE_REACT_HOOK','canonical_sha256':sha(root/'canonical_reactivation.jsonl')}
        react[str(video)]=reactive
        paths=sorted((root/'labels').glob('MEMORY_*.json'));memory[str(video)]=summarize_labels(paths)
        rows=[json.loads(p.read_text())for p in paths]
        memory[str(video)].update(actual_READ_in_WRITE=sum(r['branches']['WRITE_MEMORY']['first_actual_bank_READ']is not None for r in rows),
            actual_READ_in_SKIP=sum(r['branches']['SKIP_MEMORY']['first_actual_bank_READ']is not None for r in rows),
            READ_enriched_selection=sum(r['read_expected']for r in rows),population_prevalence_claim=False)
    source=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/match_training/compact')
    features=np.load(source/'features.npy',mmap_mode='r');videos=np.load(source/'video_ids.npy',mmap_mode='r')
    legal=np.load(source/'legal_actions.npy',mmap_mode='r');outcomes=np.load(source/'outcomes.npy',mmap_mode='r')
    weight=np.load(source/'sample_weight.npy',mmap_mode='r');manifest=json.loads((source/'manifest.json').read_text())
    utility_index=manifest['outcome_fields'].index('utility');historical={}
    for video in (7,6):
        selected=np.flatnonzero(videos==video);margins=[];informative=[];unique=0
        for index in selected:
            actions=legal[index];actions=actions[actions>=0];u=outcomes[index,actions,utility_index]
            assert np.isfinite(u).all()
            if len(u)<2:continue
            order=sorted(map(float,u),reverse=True);margin=order[0]-order[1];margins.append(margin)
            if margin>1e-8:unique+=1;informative.append(float(weight[index]))
        historical[str(video)]={'events':len(selected),'unique_best':unique,'tie_rate':1-unique/max(1,len(selected)),
            'utility_margin_quantiles':quantiles(margins),'effective_sample_size_informative':sum(informative)**2/sum(w*w for w in informative)if informative else 0,
            'feature_abs_max':float(abs(features[selected]).max()),'legal_action_counts':dict(Counter(int((r>=0).sum())for r in legal[selected])),
            'role':'train'if video==7 else 'validation','eligible_native_causal_truth':False}
    report={'status':'COMPLETE_READ_ONLY_SOURCE_AUDIT','binding':binding(),'historical_MATCH':historical,
        'historical_label_contract':'H8 historical B2 utility, legacy/factual-prefix shortcuts and frozen OFF continuation; cannot become Native Causal MiniSet truth',
        'historical_manifest_sha256':sha(source/'manifest.json'),'native_MEMORY':memory,'native_REACTIVATION':react,
        'REACT_total_events':sum(v['events']for v in react.values()),'react_action_support':['REACTIVATE_OLD','START_NEW'],
        'memory_baseline_evidence_sha256':sha(prior/'MEMORY_BASELINE_COMPARISON.json'),
        'WHAT_DID_WE_LEARN':'historical MATCH labels can support like-for-like P0.5 controls, not new native causal supervision; MEMORY single-write targets tie; REACT has no valid corrective alternate and relative production hook is absent',
        'official_test_read':False}
    save(REPORTS/'PRIOR_CANONICAL_LEARNABILITY_AUDIT.json',report);protect();print(json.dumps({'status':report['status'],'react_events':report['REACT_total_events']}))

if __name__=='__main__':main()
