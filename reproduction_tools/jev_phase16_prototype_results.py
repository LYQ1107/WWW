"""Frozen TRAIN-only reader gate; DEV and duplicate-GT diagnostics separate."""
import collections
from jev_phase16_common import *
from gtr.modeling.jev_phase16.multi_prototype_memory import VARIANTS


def summarize(results):
    counts=collections.defaultdict(collections.Counter);clusters=collections.defaultdict(set)
    for r in results:
        for v,c in r['counts'].items():counts[v].update(c)
        for v,values in r['clusters'].items():clusters[v].update(tuple(x) for x in values)
    output={}
    for variant in VARIANTS:
        key=variant+'_MATCH';c=counts[key];n=c['positive_eligible'];neg=c['negative_denominator_GT_known_legal']
        output[variant]=dict(counts=dict(c),positive_support_recall=c['positive_supported']/n if n else None,
            certified_wrong_activation_rate=c['pure_wrong_activated_queries']/neg if neg else None,
            known_wrong_owner_activation_rate=c['known_wrong_owner_activated_queries']/neg if neg else None,
            independent_eligible_cluster_count=len(clusters[key]),
            new_support_cluster_count_vs_A=len(clusters[key+'_new_support_vs_A']),
            REACT_counts=dict(counts[variant+'_REACT']))
    return output


def main():
    protect();results=[];inputs=[]
    for video in TRAIN+DEV:
        path=OUT/'P1'/f'video{video:02d}/RESULT.json';r=read(path)
        assert r['status']=='COMPLETE' and r['every_original_logit_and_committed_ID_exact_vs_P0']
        results.append(r);inputs.append(ref(path))
    strict=summarize([r for r in results if r['strict_train_gate_eligible']])
    base=strict[VARIANTS[0]];passing=[]
    for v in VARIANTS[1:]:
        item=strict[v]
        item['positive_support_gain_pp']=100*(item['positive_support_recall']-base['positive_support_recall'])
        item['certified_wrong_activation_gain_pp']=100*(item['certified_wrong_activation_rate']-base['certified_wrong_activation_rate'])
        item['P1_gate_pass']=item['positive_support_gain_pp']>=2 and item['certified_wrong_activation_gain_pp']<=1 and item['new_support_cluster_count_vs_A']>=8
        if item['P1_gate_pass']:passing.append(v)
    protocol=read(REPORTS/'P1_PROTOCOL.json');selected=next((v for v in protocol['selection_order_if_multiple_pass'] if v in passing),None)
    weights=[]
    for r in results:
        for ck in r['binding']['checkpoints']:
            if ck not in weights:weights.append(ck)
    common=dict(status='COMPLETE',binding=binding(checkpoints=weights,inputs=inputs,evaluator='frozen protocol all-query paired retrieval',
        scope='strict TRAIN only reader selection, isolated TRAIN14 and reused DEV diagnostic; zero new training'),
        frozen_protocol=ref(REPORTS/'P1_PROTOCOL.json'),strict_TRAIN=strict,
        TRAIN14_diagnostic=summarize([r for r in results if r['video']==14]),
        DEV_diagnostic=summarize([r for r in results if r['video'] in DEV]),
        selected_reader=selected,passing_readers=passing,scientific_status='GO_FROZEN_NATIVE_CAUSAL_TRIAL' if selected else 'NO_GO_PROTOTYPE_DEPLOYMENT',
        actual_training_updates=0,original_native_trajectory_parity=True,safe_recovery_proven=False,
        cached_forward_timing_is_not_real_image_FPS=True,source_results=inputs,
        per_video=[dict(video=r['video'],counts=r['counts'],read_latency_ms=r['read_latency_ms'],
            forward_latency_ms=r['forward_latency_ms'],bounded_memory=r['bounded_memory'],
            original_raw_Gallery_visual_MiB=r['original_raw_Gallery_visual_MiB']) for r in results])
    for name in ['MULTI_PROTOTYPE_RESULTS','MULTI_PROTOTYPE_IDENTITY_MEMORY','HISTORY_SEGMENT_RETRIEVAL','CANDIDATE_RECOVERY_ABLATION']:
        save(REPORTS/(name+'.json'),common)
    print('PHASE16_P1_GATE',common['scientific_status'],selected,{v:(x.get('positive_support_gain_pp'),x.get('certified_wrong_activation_gain_pp')) for v,x in strict.items()},flush=True)


if __name__=='__main__':main()
