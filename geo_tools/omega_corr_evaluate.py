"""Frozen row evidence evaluation. No solver or final-ID improvement is inferred."""
import collections,csv,gzip,json
import numpy as np
from omega_corr_common import ROOT,RUNTIME,sha256,save_new,provenance
from omega_corr_eval_helpers import top_reference,cluster_interval


def main():
    source=provenance();assert not source['source_status'].strip()
    native=json.loads((RUNTIME/'native_candidates_v1.json').read_text());qs=native['queries'];truth=json.loads((RUNTIME/'candidate_certifications_v1.json').read_text())['rows']
    assert [q['query'] for q in qs]==[q['query'] for q in truth]
    qi={tuple(q['query']):i for i,q in enumerate(qs)};meta=json.loads((RUNTIME/'candidate_values_v1.json').read_text());assert sha256(meta['values']['path'])==meta['values']['sha256']
    a=np.load(meta['values']['path']);configs=meta['configurations'];names=[c['config_id'] for c in configs];maps={n:{} for n in names};tracks={}
    for c,q,r,t,v,valid in zip(*(a[k] for k in ['config_index','query_index','reference_source_index','reference_track','score_value','score_valid'])):
        tracks[int(q),int(r)]=int(t)
        if valid:maps[names[int(c)]].setdefault(int(q),{})[int(r)]=float(v)
    refs={s:i for i,s in enumerate(meta['reference_sources'])};cohort=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text())
    pair_order=[p['pair_id'] for p in cohort['pairs']];local=set(cohort['local_crop_pair_ids']);scope={}
    for c in configs:scope[c['config_id']]=[i for i,q in enumerate(qs) if (c['layer']!=4 or q['pair_id'] in pair_order[:16]) and (c['route']!='LOCAL_CROP_VISUAL' or q['pair_id'] in local)]
    def add(name,values,route='NATIVE_SAME_CONTEXT',layer=-1):
        maps[name]=values;configs.append(dict(config_id=name,route=route,backend=name,variant=name,layer=layer,mode='M0',score_kind='primary',crop_side=0));scope[name]=list(range(len(qs)))
    appearance={'NativeAppearance1024':{},'NativeDescriptor1152':{}}
    with gzip.open(RUNTIME/'native_appearance_candidates_v1.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);i=qi[tuple(r['query'])];j=refs[':'.join(map(str,r['reference']))]
            for dim,name in [('1024','NativeAppearance1024'),('1152','NativeDescriptor1152')]:
                if r['cosine'][dim] is not None:appearance[name].setdefault(i,{})[j]=r['cosine'][dim]
    for name,values in appearance.items():add(name,values)
    native_rank={};native_scores={};prior={x:{} for x in ['NativePrior0.1','NativePrior0.3','NativePrior2']}
    for filename,canonical in [('native_context_full_v1/scores.jsonl.gz',False),('native_prior_canonical2_v1.jsonl.gz',True)]:
        with gzip.open(RUNTIME/filename,'rt') as f:
            for line in f:
                r=json.loads(line)
                for k,q in enumerate(r['queries']):
                    i=qi[tuple(q)];ts=r['unique_tracks']
                    if canonical:prior['NativePrior2'][i]=dict(zip(ts,r['track_scores'][k]))
                    else:
                        native_rank[i]=r['native_ranked_tracks'][k];native_scores[i]=dict(zip(ts,r['native_scores'][k]))
                        for w in ['0.1','0.3']:prior['NativePrior'+w][i]=dict(zip(ts,r['NativePrior'][w]['track_scores'][k]))
    for name,values in [('NativeGMT',native_scores),*prior.items()]:
        add(name,{i:{refs[':'.join(map(str,r['source']))]:values[i][r['track']] for r in q['legal_synchronous_references']} for i,q in enumerate(qs)})
    # B10 is auxiliary and restricted to exactly the reliable-F pair subset.
    geometry=json.loads((RUNTIME/'geometry_support_v1.json').read_text());gm={l:{} for l in [11,17,23]}
    with gzip.open(RUNTIME/'geometry_support_v1.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            if row['layer'] in gm and row['B10_score'] is not None:
                i=qi[tuple(row['query'])];j=refs[':'.join(map(str,row['reference']))];gm[row['layer']].setdefault(i,{})[j]=row['B10_score']
    for l,values in gm.items():
        n='B10_cachedF_layer'+str(l);add(n,values,route='RELIABLE_F_AUXILIARY',layer=l);scope[n]=[i for i,q in enumerate(qs) if q['pair_id'] in geometry['reliable_original_pairs']]
    # Track-level baseline/prior decisions use all legal history, not only sync refs.
    global_scores={'NativeGMT':native_scores,**prior}
    def status(i,t):return truth[i]['track_status'].get(str(t),'NO_CANDIDATE' if t is None else 'UNKNOWN_REFERENCE_HISTORY')
    def choose(i,values):
        r=top_reference(values,{j:tracks[i,j] for j in values},native_rank[i]);return tracks[i,r] if r is not None else None
    def global_choose(i,values):
        rank={t:k for k,t in enumerate(native_rank[i])};return min(values,key=lambda t:(-values[t],rank[t])) if values else None
    decisions={};summaries=[];query_path=RUNTIME/'paired_query_results_v1.csv.gz'
    fields=['configuration','query_index','pair_id','scene','role','person','baseline_track','chosen_track','baseline_status','chosen_status','valid_candidates','sync_positive_available','known_correction','known_harm','positive_negative_gap']
    with gzip.open(query_path,'xt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for cid,c in enumerate(configs):
            name=c['config_id'];bygroup=collections.defaultdict(collections.Counter);decisions[name]={}
            for i in scope[name]:
                q=qs[i];t=truth[i];values=maps[name].get(i,{});b=t['native_global_top1'];chosen=global_choose(i,global_scores[name][i]) if name in global_scores else (choose(i,values) if values else b)
                before,after=status(i,b),status(i,chosen);decisions[name][i]=chosen
                pos=[v for j,v in values.items() if status(i,tracks[i,j])=='CORRECT'];neg=[v for j,v in values.items() if status(i,tracks[i,j])=='WRONG']
                gap=max(pos)-max(neg) if pos and neg else None
                row=dict(configuration=cid,query_index=i,pair_id=q['pair_id'],scene=q['scene'],role=q['role'],person=t['query_gt_identity'],baseline_track=b,chosen_track=chosen,baseline_status=before,chosen_status=after,valid_candidates=len(values),sync_positive_available=bool(pos),known_correction=before=='WRONG' and after=='CORRECT',known_harm=before=='CORRECT' and after=='WRONG',positive_negative_gap=gap)
                writer.writerow(row)
                bbox=native['proposals'][':'.join(map(str,q['query']))]['bbox_xyxy'];area=max(0,bbox[2]-bbox[0])*max(0,bbox[3]-bbox[1]);size='SMALL_lt32sq' if area<1024 else ('MEDIUM_lt96sq' if area<9216 else 'LARGE')
                for group in [('role',q['role']),('scene',str(q['scene'])),('person',str(q['scene'])+':'+str(t['query_gt_identity'])),('size',size)]:
                    cts=bygroup[group];cts['queries']+=1;cts['with_sync_references']+=bool(q['legal_synchronous_references']);cts['with_valid_evidence']+=bool(values);cts['with_sync_positive']+=bool(pos);cts['known_correct']+=after=='CORRECT';cts['known_wrong']+=after=='WRONG';cts['unknown_or_no_candidate']+=after not in ['CORRECT','WRONG'];cts['known_corrections']+=row['known_correction'];cts['known_harms']+=row['known_harm'];cts['discriminable_queries']+=gap is not None;cts['positive_gap']+=gap is not None and gap>0
                    if gap is not None:cts['gap_sum']+=gap
            for (kind,value),cts in bygroup.items():summaries.append(dict(configuration=cid,group_kind=kind,group=value,**cts,net_known_row_correction=cts['known_corrections']-cts['known_harms']))
    def name(variant,layer=17,mode='M0',kind='primary',backend='omega',route='FULL_IMAGE_MULTIVIEW',side=0):return '|'.join(map(str,[route,backend,variant,layer,mode,kind,side]))
    def net(n,role):return sum(int(status(i,truth[i]['native_global_top1'])=='WRONG' and status(i,t)=='CORRECT')-int(status(i,truth[i]['native_global_top1'])=='CORRECT' and status(i,t)=='WRONG') for i,t in decisions[n].items() if qs[i]['role']==role)
    selected=min([17,11,23],key=lambda l:(-net(name('F2',l),'FIT'),[17,11,23].index(l)));selected_name=name('F2',selected)
    # Every comparison is intersected by physical candidates, not just by queries.
    comparisons=[]
    for c in configs:
        if c['backend']!='omega' or c['layer'] not in [11,17,23] or c['variant'] not in ['F2','LOCAL_F3'] or c['mode']!='M0' or c['score_kind'] not in ['primary','cycle']:continue
        n=c['config_id'];l=c['layer'];kind=c['score_kind'];local_route=c['route']=='LOCAL_CROP_VISUAL';side=c['crop_side']
        controls=([name('LOCAL_DINO',-1,'M0',kind,'dino','LOCAL_CROP_VISUAL',side)] if local_route else [name('F3',l,kind=kind),name('old_vggt',l,kind=kind,backend='old_vggt'),name('dino',-1,kind=kind,backend='dino')])
        # DINO serialized layer is its actual exported identifier (inspect config).
        if any(x not in maps for x in controls):
            dino=[cc['config_id'] for cc in configs if cc['backend']=='dino' and cc['route']==c['route'] and cc['mode']=='M0' and cc['score_kind']==kind and cc['crop_side']==side];assert len(dino)==1
            controls=[x if x in maps else dino[0] for x in controls]
        controls+=['NativeAppearance1024','NativePrior2']
        all_methods=[n]+controls;records=[];lost=collections.Counter();unique=[]
        for i in scope[n]:
            if qs[i]['role']!='MECHANISM_CHECK':continue
            sets=[set(maps[m].get(i,{})) for m in all_methods];common=set.intersection(*sets);lost['queries_in_scope']+=1;lost['union_candidates']+=len(set.union(*sets));lost['common_candidates']+=len(common)
            if len(common)<2:lost['less_than_two_common_candidates']+=1;continue
            st=[status(i,tracks[i,j]) for j in common]
            if 'CORRECT' not in st or 'WRONG' not in st:lost['no_common_certified_positive_and_negative']+=1;continue
            ts=[choose(i,{j:maps[m][i][j] for j in common}) for m in all_methods];ok=[status(i,t)=='CORRECT' for t in ts];person=truth[i]['query_gt_identity'];assert person is not None
            records.append(dict(query_index=i,scene=qs[i]['scene'],person=person,common_candidates=len(common),tracks=ts,correct=ok))
            if ok[0] and not any(ok[1:]):unique.append(i)
        intervals=[dict(control=control,interval=cluster_interval([(r['scene'],r['person'],int(r['correct'][0])-int(r['correct'][k+1])) for r in records])) for k,control in enumerate(controls)]
        comparisons.append(dict(configuration=n,controls=controls,coverage=dict(lost),eligible_queries=len(records),common_candidate_unique_correct_queries=unique,paired_intervals=intervals,rows=records))
    mechanism=[i for i in scope[selected_name] if qs[i]['role']=='MECHANISM_CHECK'];repairs=[i for i in mechanism if status(i,truth[i]['native_global_top1'])=='WRONG' and status(i,decisions[selected_name][i])=='CORRECT'];harms=[i for i in mechanism if status(i,truth[i]['native_global_top1'])=='CORRECT' and status(i,decisions[selected_name][i])=='WRONG']
    paired=next(c for c in comparisons if c['configuration']==selected_name);unique_repairs=sorted(i for i in set(repairs)&set(paired['common_candidate_unique_correct_queries']) if status(i,decisions['NativePrior2'][i])!='CORRECT')
    conditions=dict(at_least3_known_row_repairs=len(repairs)>=3,repairs_exceed_harms=len(repairs)>len(harms),at_least2_scenes=len({qs[i]['scene'] for i in repairs})>=2,at_least2_people=len({(qs[i]['scene'],truth[i]['query_gt_identity']) for i in repairs})>=2,at_least2_unique_repairs_on_common_candidates=len(unique_repairs)>=2,all_common_candidate_control_CI_lower_positive=all(x['interval'].get('lower95') is not None and x['interval']['lower95']>0 for x in paired['paired_intervals']))
    # Target-vs-ring and N1/N2 still require the complete registered review;
    # a failed necessary condition conclusively stops head training without relaxing it.
    necessary=all(conditions.values())
    result=dict(status='FROZEN_ROW_EVIDENCE_EVALUATED',source=source,configurations=configs,scope_queries=len(qs),summary=summaries,selected_layer_fit_only=selected,fit_layer_nets={str(l):net(name('F2',l),'FIT') for l in [11,17,23]},selected_mechanism=dict(known_repairs=repairs,known_harms=harms,unique_repairs=unique_repairs),gateB_necessary_conditions=conditions,gateB_status='NECESSARY_CONDITIONS_PASS_REQUIRES_TARGET_NEGATIVE_CONTROL_REVIEW' if necessary else 'NO_UNIQUE_CROSSVIEW_TOKEN_EVIDENCE',comparisons=comparisons,actual_solver_status='NOT_EXECUTED',final_identity_status='NOT_ESTABLISHED',efficiency_status='NOT_MEASURED',no_GT_model_input=True,no_solver_gain_inferred_from_row_rank=True,query_output=dict(path=str(query_path),sha256=sha256(query_path)),candidate_values_sha256=meta['values']['sha256'])
    save_new(RUNTIME/'evaluation_v1.json',result);print(json.dumps({k:result[k] for k in ['selected_layer_fit_only','fit_layer_nets','selected_mechanism','gateB_necessary_conditions','gateB_status']},indent=2),flush=True)

if __name__=='__main__':main()
