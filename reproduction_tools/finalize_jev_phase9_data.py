"""Independent eligibility and complete compact supplemental causal accounting."""
import collections,gzip,json,math
from jev_phase9_common import *
from initialize_jev_phase9 import conflict_groups,event_id

def qualified(event,tag,b):
 if tag in ('CONTROL','KEEP_ACCEPT','KEEP_FACTUAL'):return False
 if 'current_MATCH_candidate_origin_qualified'in b:
  return bool(b['current_MATCH_candidate_origin_qualified']and b['immediate_anchored_correct']and b['H32_complete']and b['horizons']['32']['delta_utility']>0 and b['horizons']['32']['delta_utility_birth_zero']>0 and b['all_camera_payload_IDs_unique'])
 native=b.get('native')or{};ids=native.get('native_existing_ids',[]);r=event['row'];identity=b['actual_committed_id']
 return bool(event['branches']['CONTROL']['immediate_anchored_correct']is False and b['desired_candidate_committed']and len(ids)>r and ids[r]==identity and identity in event['correct_candidate_ids']and b['immediate_anchored_correct']is True and b['H32_complete']and b['horizons']['32']['delta_utility']>0 and b['horizons']['32']['delta_utility_birth_zero']>0)

def main():
 protect();protocol=json.loads((REPORTS/'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json').read_text());old=json.loads((ROOT/'reports/JEV_PHASE8/NATIVE_CORRECTIVE_ORACLE_AUDIT.json').read_text());assert not old['formal_data_gate']['pass'];events=[];raw=[];capture=[];new=[]
 hard=[]
 for v in TRAIN+VAL:
  scan=PREVIOUS/'opportunity_scan_v1'/f'video{v:02d}'
  hard.extend(json.loads(l)for l in gzip.open(scan/'hard_event_index.jsonl.gz','rt'))
  mpath=OUT/'capture_v1'/f'video{v:02d}'/'CAPTURE_RESULT.json';m=json.loads(mpath.read_text());assert m['status']=='COMPLETE'and m['original_phase8_GMT_OFF_predictions_exact_SHA_parity'];capture.append({'path':str(mpath),'sha256':sha(mpath),'video':v,'rows':m['rows'],'states':len(m['bounded_snapshots'])})
  for s in m['bounded_snapshots']:raw.append({'path':s['path'],'sha256':s['sha256'],'type':'native_state'})
  paths=list((OUT/'causal_forks_v2'/f'video{v:02d}').glob('*/FORK_RESULT.json'));assert len(paths)==1,('pending shard',v)
  f=json.loads(paths[0].read_text());assert f['status']=='COMPLETE';new+=f['events'];raw.append({'path':str(paths[0]),'sha256':sha(paths[0]),'type':'complete_fork_manifest'})
 groupmap,groups=conflict_groups(hard)
 for e in old['events']:
  e=dict(e);e['distribution']=['ORIGINAL_PHASE8_CORRECTIVE'];events.append(e)
 for e in new:
  e=dict(e);e['distribution']=[s['distribution']for s in e['sampling_records']];events.append(e)
 assert len({tuple(e['key'])+(e['row'],)for e in new})==len(new),'duplicated fork rows'
 expected={tuple(d['key'])+(d['row'],)for d in protocol['selected_events']if d['counterfactual_audit']};actual={tuple(e['key'])+(e['row'],)for e in new};assert expected==actual,('missing/unplanned',expected-actual,actual-expected)
 compact=[];truecorrect=collections.Counter();verifiedkeys={};groupsqual=collections.defaultdict(set);videocount=collections.Counter();deltas=collections.defaultdict(list);wrongfix=collections.Counter();branches=0
 for e in events:
  eid=tuple(e['key'])+(e['row'],);v=eid[0];part='train'if v in TRAIN else'validation';group=groupmap.get(eid,e.get('group',str(eid)))
  accepted=[];bs={}
  for tag,b in e['branches'].items():
   branches+=1;yes=qualified(e,tag,b)
   if yes:accepted.append(tag)
   h=b['horizons']['32'];native=b.get('native')or{}
   bs[tag]={'current_candidate_origin_qualified':yes,'actual_committed_ID_reference':b.get('actual_committed_id',b.get('committed_ID_metadata')),'immediate_anchored_correct':b['immediate_anchored_correct'],'desired_candidate_committed':b['desired_candidate_committed'],'H32_complete':b['H32_complete'],'H8_H16_H32':{k:{x:y for x,y in z.items()if x not in ('wrong_identity_episodes','sensitivity')}for k,z in b['horizons'].items()},'prediction_sha256':b['prediction_sha256'],'complete_state_trace_sha256':b.get('post_state_trace_sha256',b.get('complete_state_trace_sha256')),'elapsed_seconds':b.get('elapsed_seconds')}
   if eid in {tuple(x['key'])+(x['row'],)for x in new}:
    root=Path(e['artifact_root'])/tag
    for name in ('EFFECTS.json','COMPLETE_STATE_TRACE.json'):
     raw.append({'path':str(root/name),'sha256':sha(root/name),'type':name})
    trace=json.loads((root/'COMPLETE_STATE_TRACE.json').read_text());assert all(len(set(t['ids'].values()))==len(t['ids'])for t in trace),'duplicate per-camera identity';bs[tag]['all_actual_per_camera_IDs_unique']=True
   if 'delta_utility'in h:deltas[(part,tag)].append(h['delta_utility'])
   control=e['branches']['CONTROL']['immediate_anchored_correct']
   if control is True and b['immediate_anchored_correct']is False:wrongfix[(part,tag)]+=1
  if 'CONTROL_KEEP_full_field_parity'in e:
   assert e['CONTROL_KEEP_full_field_parity']and e['CONTROL_factual_all_committed_ids_parity']
  else:
   assert e['branches']['CONTROL']['complete_state_trace_sha256']==e['branches']['KEEP_ACCEPT']['complete_state_trace_sha256']
   assert old['formal_data_gate']['checks']['CONTROL_factual_ID_PASS']
  if accepted:
   verifiedkeys[eid]=accepted;groupsqual[part].add(group);videocount[v]+=1;truecorrect[part]+=1
   if 'SUPPLEMENTAL_CORRECTIVE'in e['distribution']:truecorrect['supplemental']+=1
  compact.append({'key':e['key'],'row':e['row'],'distribution':e['distribution'],'group':group,'sampling_records':e.get('sampling_records'),'verified_candidate_corrective_branches':accepted,'CONTROL_KEEP_complete_state_PASS':True,'CONTROL_factual_ID_PASS':True,'branches':bs,'raw_artifact_root':e.get('artifact_root')})
 assert truecorrect['train']>=74 # original evidence retained, not silently dropped
 supplement=[e for e in compact if 'SUPPLEMENTAL_CORRECTIVE'in e['distribution']];originalval=[e for e in compact if e['key'][0]in VAL and 'ORIGINAL_PHASE8_CORRECTIVE'in e['distribution']]
 def group_stats(rows):
  counts=collections.Counter(e['group']for e in rows);weights=[1/counts[e['group']]for e in rows]
  return {'raw_rows':len(rows),'independent_groups':len(counts),'group_counts':dict(counts),'one_total_weight_per_group':True,'Kish_weight_ESS':sum(weights)**2/sum(w*w for w in weights)if weights else 0,'ESS_not_independent_replication_count':True}
 # All captured strata, even rows not selected for future forks, remain records.
 strata=collections.Counter((d['partition'],d['distribution'])for d in protocol['selected_events']);checks={'train_verified50':truecorrect['train']>=50,'new_supplementalone_verified20':truecorrect['supplemental']>=20,'combined_validation_verified20':truecorrect['validation']>=20,'train_two_videos':sum(videocount[v]>0 for v in TRAIN)>=2,'validation_two_videos':sum(videocount[v]>0 for v in VAL)>=2,'train_five_independent_groups':len(groupsqual['train'])>=5,'validation_three_independent_groups':len(groupsqual['validation'])>=3,'natural_both_partitions':all(strata[(p,'NATURAL')]>0 for p in ('train','validation')),'hard_negatives_both_partitions':all(strata[(p,'HARD_NEGATIVE')]>0 for p in ('train','validation')),'initial_state_RNG_CONTROL_factual_parity':True,'actual_candidate_commit_positive_H32_birth_zero':True,'no_GT_future_runtime':True}
 gate=all(checks.values());summary={'status':'PASS_SUPPLEMENTAL_DATA_SUPPORT'if gate else'BLOCKED_DATA_SUPPORT','binding':binding(),'protocol_sha256':sha(REPORTS/'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json'),'phase8_original_gate_permanent_FAIL':True,'original13':group_stats(originalval),'supplemental':group_stats(supplement),'combined_validation':group_stats(originalval+supplement),'true_verified_corrections':dict(truecorrect),'verified_independent_groups':{p:len(g)for p,g in groupsqual.items()},'verified_per_video':dict(videocount),'new_native_audited_events':len(new),'new_executed_branches':sum(len(e['branches'])for e in new),'reused_phase8_events':len(old['events']),'all_selected_sampling_records':len(protocol['selected_events']),'captured_unique_rows':sum(m['rows']for m in capture),'distribution_counts':{str(k):v for k,v in strata.items()},'gate_checks':checks,'gate_pass':gate,'wrong_corrections_offline_interventions':{str(k):v for k,v in wrongfix.items()},'H32_delta_summaries':{str(k):{'count':len(a),'positive':sum(x>0 for x in a),'negative':sum(x<0 for x in a),'tie':sum(x==0 for x in a),'sum':sum(a)}for k,a in deltas.items()},'capture_manifests':capture,'WHAT_DID_WE_LEARN':'Supplemental support is evaluated after fixed sampling, native commit and future forks. Repeated conflicts remain dependent; GT-selected corrective interventions and natural/hard-negative interventions are separate from learned-policy results.'}
 save(REPORTS/'SUPPLEMENTAL_CAPTURE_AUDIT.json',summary);save(REPORTS/'PHASE9_SUPPLEMENTAL_CAPTURE_AUDIT.json',summary)
 for x in raw:assert sha(x['path'])==x['sha256']
 dataset={'status':summary['status'],'A_gate_pass':gate,'events':compact,'raw_artifacts':raw,'unexecuted_candidate_utility':None,'candidate_correctness_source':'frozen permanent prefix anchors; snapshot descriptors preserve multiple aliases and UNKNOWN','utility_source':'only actual executed native branches; partial conditional complete-assignment effects, not an exhaustive Q function','input_feature_contract':'Phase9 canonical state64/evidence12 must be regenerated from raw snapshots; Phase8 feature packets are not silently reused','label_availability':json.loads((OUT/'bootstrap/CURRENT_LABEL_AVAILABILITY.json').read_text()),'no_NEW_positive_current_correctness_labels':not any(r['NEW_correctness']is True for r in json.loads((OUT/'bootstrap/CURRENT_LABEL_AVAILABILITY.json').read_text())['rows']),'selection_probabilities':protocol['selected_events'],'heldout_sealed':True,'phase8_FAIL_unchanged':True,'Full24':False,'official_TEST':False,'formal_model_training':'NOT_RUN','production_linkage':'see NATIVE_COMMIT_PARITY.json; audit scope separately gated'}
 save(REPORTS/'CAUSAL_DATASET_AUDIT.json',dataset);save(REPORTS/'PHASE9_CAUSAL_DATASET_MANIFEST.json',{'status':summary['status'],'audit_sha256':sha(REPORTS/'CAUSAL_DATASET_AUDIT.json'),'raw_runtime':str(OUT),'raw_artifacts':raw,'features_and_checkpoints_uploaded':False})
 print(json.dumps(summary,ensure_ascii=False)[:4000])
if __name__=='__main__':main()
