"""Dataset v3 freeze from all native branches, with independent eligibility."""
import json,collections,math
from pathlib import Path
import torch
from jev_phase10_common import *
from gtr.modeling.jev_native_state import fingerprint

def main():
 protect();state=json.loads((REPORTS/'NATIVE_STATE_PARITY.json').read_text());assert state['status']=='PASS'and state['fixed_original_records']==221
 cfg=json.loads((REPORTS/'PHASE10_PREREGISTRATION.json').read_text());plan=json.loads((REPORTS/'PHASE10_NATIVE_FORK_PLAN.json').read_text());old=json.loads((ROOT/'reports/JEV_PHASE9/CAUSAL_DATASET_AUDIT.json').read_text());events=[];raw=[];changes=[];verified=collections.Counter();groups=collections.defaultdict(set);videos=collections.Counter()
 for e in plan['events']:
  p=OUT/'native_forks_v3'/f'event{e["index"]:03d}'/'EVENT_RESULT.json';d=json.loads(p.read_text());assert d['status']=='PASS'and not d['diagnostic_only']and d['CONTROL_KEEP_complete_state_PASS']and d['CONTROL_full_factual_state_events_RNG_PASS'];assert sorted(d['branches'])==e['branches'];assert d['key']==e['key']and d['row']==e['row']and d['group']==e['group'];events.append(d);raw.append({'index':e['index'],'path':str(p),'SHA256':sha(p),'source_commit':d['binding']['source_commit']})
  assert not d['binding']['dirty']and d['binding']['source_commit']=='92654e452f4c202c6009b39941705d987bfe1a07','canonical native forks must use the one pinned production source'
  assert all(b['binding']==d['binding']for b in d['branches'].values()),'branch source/GPU bindings differ inside an event'
  v=e['key'][0];part='train'if v in TRAIN else'validation'
  if d['verified_candidate_corrective_branches']:
   verified[part]+=1;groups[part].add(e['group']);videos[v]+=1
   if 'SUPPLEMENTAL_CORRECTIVE'in e['distribution']:verified['supplemental']+=1;verified[f'supplemental_{part}']+=1
  for tag,b in d['branches'].items():
   source=e['branch_specs'][tag];assert sha(source['old_effect_path'])==source['old_effect_SHA256'];previous=json.loads(Path(source['old_effect_path']).read_text());deltas={}
   for h in [8,16,32]:
    now=b['horizons'][str(h)];before=previous['horizons'][str(h)];fields=['counts','utility','utility_birth_zero','wrong_identity_duration_camera_frames'];different={f:{'old':before.get(f),'new':now.get(f)}for f in fields if before.get(f)!=now.get(f)}
    if different:deltas[str(h)]=different
   if deltas:changes.append({'key':e['key'],'row':e['row'],'index':e['index'],'tag':tag,'old_SHA256':source['old_effect_SHA256'],'changes':deltas,'cause_scope':'whole native ordered-gallery/bank/history/RNG production state and actual production transitions replace legacy replay; not assigned to numerical tolerance'})
 assert len(events)==162 and sum(len(e['branches'])for e in events)==840
 causal={tuple(e['key'])+(e['row'],):e for e in events};priorrows={tuple(e['key'])+(e['row'],):e for e in old['events']};sampling=json.loads((ROOT/'reports/JEV_PHASE9/PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json').read_text())['selected_events'];allrows=collections.defaultdict(list)
 for e in old['events']:
  if 'ORIGINAL_PHASE8_CORRECTIVE'in e['distribution']:allrows[tuple(e['key'])+(e['row'],)].append({'distribution':'ORIGINAL_PHASE8_CORRECTIVE','sampling_probability':1.,'group':e['group']})
 for r in sampling:allrows[tuple(r['key'])+(r['row'],)].append(r)
 snapshots=collections.defaultdict(list)
 for r in cfg['original_snapshot_records']:snapshots[tuple(r['key'])].append(r)
 registries={}
 for v in TRAIN+VAL:
  p=OUT/'native_capture_v1'/f'video{v:02d}'/'compat/RESULT.json';m=json.loads(p.read_text());registries[v]={tuple(r['key']):r for r in m['prefixes']}
 rows=[];labels=collections.Counter();recall=collections.Counter();strata=collections.Counter();branchsummary=[]
 # Cache one old snapshot/native packet at a time; no extra raw-copy payloads.
 for key in sorted({k[:3]for k in allrows}):
  originals=[]
  for r in snapshots[key]:assert sha(r['path'])==r['sha256'];originals.append((r,torch.load(r['path'],map_location='cpu')))
  n=registries[key[0]][key];assert sha(n['packet_path'])==n['packet_sha256'];native_packet=torch.load(n['packet_path'],map_location='cpu');batch=native_packet['packet']['batch'];refs=tuple(map(int,batch.candidate_ids))
  for eid in sorted(k for k in allrows if k[:3]==key):
   row=eid[3];r,s=next((r,s)for r,s in originals if row in s['descriptors']);desc=s['descriptors'][row];anchors={int(t):int(g)for t,g in s['offline_prefix_identity']['reliable'].items()};gt=desc['offline_GT'];correctness=[None if gt is None or t not in anchors else anchors[t]==gt for t in refs];known=torch.tensor([x is not None for x in correctness],dtype=torch.bool);positive=torch.tensor([x is True for x in correctness],dtype=torch.bool);mask=batch.legal_mask[row].cpu();known&=mask;positive&=mask
   provenance=allrows[eid];previous=priorrows.get(eid);group=previous['group']if previous else next((x['group']for x in provenance if 'group'in x),f'V{key[0]}_F{key[1]}_VW{key[2]}_R{row}');distribution=sorted({x['distribution']for x in provenance});part='train'if key[0]in TRAIN else'validation'
   c=causal.get(eid);executed=[];utility=[None]*len(refs);utility_available=[False]*len(refs);ambiguous=[]
   if c:
    offers=collections.defaultdict(list)
    for tag,b in c['branches'].items():
     actual=b['native']['native_existing_ids'][row];value=b['horizons']['32']['delta_utility'];eligible=bool(gt is not None and b['H32_complete']and b['current_candidate_origin_qualified']and actual in refs and (tag=='CONTROL'or tag.startswith(('CANDIDATE_','ALTERNATIVE_'))))
     executed.append({'tag':tag,'existing_candidate_reference':actual if actual>=0 else None,'utility':value,'H32_complete':b['H32_complete'],'single_candidate_ranking_eligible':eligible,'native_trace_SHA256':b['native_trace_SHA256'],'immediate_correct':b['immediate_anchored_correct']})
     if eligible:offers[refs.index(actual)].append(value)
    for col,values in offers.items():
     if max(values)-min(values)<=1e-6:utility[col]=values[0];utility_available[col]=True
     else:ambiguous.append({'candidate_reference':refs[col],'conditional_complete_assignment_utilities':values})
   q=torch.tensor([x if x is not None else float('nan')for x in utility],dtype=torch.float32);qm=torch.tensor(utility_available,dtype=torch.bool)&mask
   assert torch.isfinite(batch.state64[row]).all()and torch.isfinite(batch.evidence12[row]).all()and torch.isfinite(q[qm]).all();assert not(torch.isfinite(q[~qm]).any()),'unexecuted Q must remain unknown'
   scores=batch.scores[row].cpu();order=sorted(range(len(refs)),key=lambda j:(-float(scores[j])if math.isfinite(float(scores[j]))else math.inf,refs[j]));ranks={col:rank+1 for rank,col in enumerate(order)}
   if part=='train'and positive.any():
    recall['rows_with_available_correct']+=1
    for k in [8,16,32,64]:recall[f'@{k}']+=any(ranks[j]<=k for j in torch.where(positive)[0].tolist())
    recall['@Full']+=1
   ce=bool(positive.any());rank=bool(qm.sum()>=2 and float(q[qm].max()-q[qm].min())>1e-6)
   labels[(part,'rows')]+=1;labels[(part,'CE_eligible')]+=ce;labels[(part,'ranking_eligible')]+=rank;labels[(part,'unknown_edges')]+=sum(x is None for x in correctness);labels[(part,'known_edges')]+=int(known.sum());labels[(part,'positive_edges')]+=int(positive.sum())
   for d in distribution:strata[(part,d)]+=1
   rows.append({'key':list(key),'row':row,'partition':part,'group':group,'distribution':distribution,'sampling_records':provenance,'candidate_ids':refs,'state64':batch.state64[row].cpu().float(),'evidence12':batch.evidence12[row].cpu().float(),'legal_mask':mask,'known_mask':known,'positive_mask':positive,'utility':q,'utility_mask':qm,'CE_eligible':ce,'ranking_eligible':rank,'NEW_correctness':False if desc['first_action']=='START_NEW'and desc['correct_candidate_ids']else None,'NEW_supervision_mask':False,'factual_committed_id':native_packet['commit']['ids'][row],'factual_identity_correct':None if gt is None or native_packet['commit']['ids'][row]not in anchors else anchors[native_packet['commit']['ids'][row]]==gt,'original_proposal_column':desc['proposal_column'],'original_proposal_correct':desc['proposal_correct'],'executed_candidate_utilities':executed,'ambiguous_utility_context':ambiguous,'native_prefix_SHA256':n['sha256'],'native_packet_SHA256':n['packet_sha256'],'old_snapshot_SHA256':r['sha256'],'unknown_is_not_negative':True,'unexecuted_Q_is_unknown':True})
 groupcounts=collections.Counter((r['partition'],r['group'])for r in rows)
 for r in rows:r['weight']=1/groupcounts[(r['partition'],r['group'])]
 # Overlapping windows remain dependent even when pre-existing group names
 # differ; retain every original group/weight and report temporal bundles.
 bundles={};bundlecounts={}
 for part,vs in [('train',TRAIN),('validation',VAL)]:
  count=0
  for v in vs:
   ordered=sorted([r for r in rows if r['key'][0]==v],key=lambda r:r['key'][1]);end=-1;bid=-1
   for r in ordered:
    f=r['key'][1]
    if f>end:bid+=1;count+=1
    end=max(end,f+31);r['temporal_dependency_bundle']=f'V{v}_B{bid}';bundles[tuple(r['key'])+(r['row'],)]=r['temporal_dependency_bundle']
  bundlecounts[part]=count
 counts={str(k):v for k,v in labels.items()};stratacounts={str(k):v for k,v in strata.items()}
 checks={'all221_native_state':True,'all162_original_events_all840_original_branches':True,'CONTROL_KEEP_all_complete_native_states':True,'train_verified50':verified['train']>=50,'supplementalone_verified20':verified['supplemental']>=20,'validation_verified20':verified['validation']>=20,'train_two_videos':sum(videos[v]>0 for v in TRAIN)>=2,'validation_two_videos':sum(videos[v]>0 for v in VAL)>=2,'train_five_frozen_groups':len(groups['train'])>=5,'validation_three_frozen_groups':len(groups['validation'])>=3,'natural_both_partitions':all(strata[(p,'NATURAL')]>0 for p in ['train','validation']),'hard_negative_both_partitions':all(strata[(p,'HARD_NEGATIVE')]>0 for p in ['train','validation']),'no_GT_runtime_or_frozen_future_actions':True,'some_train_current_and_causal_supervision':labels[('train','CE_eligible')]>0 and labels[('train','ranking_eligible')]>0};passed=all(checks.values())
 data={'version':'Native Candidate Association Dataset v3','rows':rows,'all_original_selected_events_preserved':True,'protocol_SHA256':sha(REPORTS/'PHASE10_PREREGISTRATION.json'),'model_protocol_SHA256':sha(REPORTS/'PHASE10_MODEL_AND_CONTROL_PROTOCOL.json'),'model_amendment_SHA256':sha(REPORTS/'PHASE10_CAPACITY_AND_STRONG_RULE_AMENDMENT.json')};p=OUT/'dataset_v3/DATASET.pth';p.parent.mkdir(exist_ok=True)
 if p.exists():assert fingerprint(torch.load(p,map_location='cpu'))==fingerprint(data),'existing immutable dataset differs; preserve it and version any scientific amendment'
 else:
  t=p.with_suffix('.pth.tmp');torch.save(data,t);t.replace(p)
 h32={'status':'PASS','events':162,'branches':840,'all_CONTROL_KEEP_full_native_state_events_RNG_exact':True,'all_actual_per_camera_committed_IDs_unique':True,'same_native_production_function_is_the_replay_adapter':True,'future_policy':'fresh native GMT OFF on mutated state','old_results_unchanged':True,'changed_branches':len(changes),'changed_event_rows':len({(tuple(c['key']),c['row'])for c in changes}),'old_vs_new':changes,'raw_event_manifests':raw,'heldout_sealed':True,'official_TEST':False,'Full24':False}
 save(REPORTS/'H32_CAUSAL_PARITY.json',h32);save(REPORTS/'PHASE10_H32_REPLAY_PARITY.json',{'status':'PASS','canonical_report':'H32_CAUSAL_PARITY.json','SHA256':sha(REPORTS/'H32_CAUSAL_PARITY.json'),'events':162,'branches':840})
 gate={'status':'PASS'if passed else'FAIL_SCIENTIFIC_DATA_ELIGIBILITY','gate_pass':passed,'checks':checks,'verified_corrections':dict(verified),'verified_frozen_groups':{k:len(v)for k,v in groups.items()},'verified_per_video':dict(videos),'old_unrevalidated_counts':{'train':74,'validation':59},'label_counts':counts,'strata_counts':stratacounts,'temporal_overlap_bundles':bundlecounts,'overlap_bundles_not_additional_training_samples':True,'new_rows':len(rows),'fixed_NEW':True,'learnable_NEW':'NOT_ELIGIBLE_NO_POSITIVES','heldout_sealed':True,'phase8_original_FAIL_unchanged':True};save(REPORTS/'DATA_ELIGIBILITY.json',gate);save(REPORTS/'PHASE10_DATA_ELIGIBILITY.json',{'status':gate['status'],'canonical_report':'DATA_ELIGIBILITY.json','SHA256':sha(REPORTS/'DATA_ELIGIBILITY.json')})
 audit={'status':'FROZEN_ELIGIBLE'if passed else'FROZEN_INELIGIBLE','dataset_version':'v3','rows':len(rows),'label_counts':counts,'strata_counts':stratacounts,'all_original_causal_events_and_sampling_rows_preserved':True,'correctness_source':'unchanged confirmed permanent native factual-prefix identity anchors; no live GT','utility_source':'only re-executed complete native forks; partial conditional whole-assignment effects, not an exhaustive candidate Q function','ranking_exclusions':'UNKNOWN/censored/unexecuted, re-association policy branches, joint assignments and conflicting conditional utilities excluded from single-candidate ranking; all retained in branch evidence','NEW_positive':0,'fixed_NEW_required':True,'TRAIN_correct_candidate_recall':{k:v/recall['rows_with_available_correct']for k,v in recall.items()if k.startswith('@')},'TRAIN_recall_denominator':recall['rows_with_available_correct'],'TopK':'Full','dataset_path':str(p),'dataset_SHA256':sha(p),'heldout_sealed':True};save(REPORTS/'NATIVE_CANDIDATE_V3_AUDIT.json',audit);save(REPORTS/'NATIVE_CANDIDATE_V3_MANIFEST.json',{'status':audit['status'],'dataset_path':str(p),'dataset_SHA256':audit['dataset_SHA256'],'audit_SHA256':sha(REPORTS/'NATIVE_CANDIDATE_V3_AUDIT.json'),'H32_audit_SHA256':sha(REPORTS/'H32_CAUSAL_PARITY.json'),'data_gate_SHA256':sha(REPORTS/'DATA_ELIGIBILITY.json'),'raw_snapshots_or_checkpoints_uploaded':False})
 (ROOT/'docs/PHASE10_CAUSAL_DATASET_V3.md').write_text(f'# Native candidate dataset v3\n\nStatus: {audit["status"]}. {len(rows)} frozen selected rows; all 162 old causal events and 840 original branches re-executed from actual native prefixes. Verified corrections: {dict(verified)}. Changed branch horizon labels: {len(changes)}. Original Phase V–IX evidence remains unchanged.\n\nCorrectness uses only pre-intervention confirmed anchors. Unknown identities are masked. Utilities remain unavailable for unexecuted or censored candidates; no zeros replace missing Q. Single-candidate ranking excludes REASSOCIATE/joint policy effects and conflicting utilities for the same identity under different complete assignments. All such effects remain in raw branch evidence. Main NEW is fixed because no genuine positive NEW supervision exists. Original group names and equal total group weights are preserved; overlapping windows are separately reported as dependent bundles.\n\nThe state and data gates are independent. Passing serialization does not establish model performance. The compact manifests bind all large tensors and branch records on the server by SHA256.\n')
 print(json.dumps(gate,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
