"""Forensic attribution of gallery offsets; no new training or causal labels."""
import json,gzip,collections
import torch
from jev_phase9_common import *
def main():
 d=json.loads((REPORTS/'MEMORY_REPRESENTATION_AUDIT.json').read_text());assert d['status']=='FAIL_PREFIX_REPRESENTATION_CONTRACT'
 example=d['failures'][0];key=tuple(example['key']);rec=next(r for r in d['records']if tuple(r['key'])==key);assert sha(rec['path'])==rec['sha256'];s=torch.load(rec['path'],map_location='cpu');v=key[0]
 # Seed IDs can be recovered from the first cached frame's largest camera.
 from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
 cache=FrozenPerceptionCache('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
 counts={view:len(cache.load(v,0,view)['pred_boxes'])for view in (0,1)};seed_view=max(counts,key=lambda view:(counts[view],view));known=set(range(1,counts[seed_view]+1));reacts=[]
 path=PREVIOUS/'opportunity_scan_v1'/f'video{v:02d}'/'natural_event_index.jsonl.gz'
 for x in map(json.loads,gzip.open(path,'rt')):
  if tuple(x['key'])==key:break
  t=int(x['actual_committed_id']);was_known=t in known;known.add(t)
  if was_known and x['first_action']=='START_NEW':reacts.append({'key':x['key'],'row':x['row'],'committed_ID':t,'MATCH_first_action':x['first_action'],'scope':'known identity recovered after unmatched MATCH, not a new monotonic birth'})
 # Full production trace was observed independently with the same frozen
 # perception/seed, and matches all factual ID commits on video12.
 proof=json.loads((REPORTS/'NATIVE_COMMIT_PARITY.json').read_text());trace_path=next(x['path']for x in proof['trace_artifacts']if x['path'].endswith('GMT_OFF_TRACE.json'));trace=json.loads(Path(trace_path).read_text());index=next(i for i,x in enumerate(trace)if x['key']==list(key[1:]));before=trace[index-1]
 comparison={}
 for t,offset in example['tracks'].items():
  comparison[t]={'replay_hits':s['state'].track_hits[int(t)],'replay_writes':len(s['state'].memory.get(int(t),[])),'native_pre_MATCH_gallery_length':before['memory_counts'][t],'missing_observation_count':s['state'].track_hits[int(t)]-len(s['state'].memory.get(int(t),[])),'prior_bank_recoveries':[x for x in reacts if x['committed_ID']==int(t)]}
 r={'status':'CONFIRMED_BIRTH_PLUS_REACTIVATION_WRITE_REPRESENTATION_GAP','binding':binding(),'all_prefix_audit_sha256':sha(REPORTS/'MEMORY_REPRESENTATION_AUDIT.json'),'snapshots':d['snapshots'],'failed_birth_only_translation_prefixes':len(d['failures']),'total_track_prefix_instances':d['per_track_instances_checked'],'offset_histogram':d['offset_histogram'],'example':{'key':list(key),'snapshot_sha256':rec['sha256'],'native_previous_commit_key':before['key'],'track_comparison':comparison},'source_findings':[{'path':'reproduction_tools/jev_counterfactual_v2.py','functions':['seed_production_state_from_payload','CachedPerceptionMutableAssociationV2.step','_update_memory_eligibility'],'finding':'replay initial observation is not in memory writes; eligibility uses len>=bank_size, compensating first-observation offset for ordinary all-WRITE prefixes'},{'path':'reproduction_tools/run_early_pilot_tracking.py','function':'run_method Phase2 MEMORY','finding':'memory actions computed only for final_existing IDs before stale-bank recovery; successfully bank-reactivated rows may gain a hit without a corresponding memory write'},{'path':'gtr/modeling/meta_arch/gtr_rcnn.py','functions':['run_global_tracker_plus','memory_bank','_jev_memory_action'],'finding':'birth initializes id_reid_dict; recovered existing ID proceeds through the native existing-ID memory-write branch after bank recovery'}],'interpretation':'The first birth-vector fix passes one early train16 prefix, but is not a lossless bridge for all causal data. This is an observed input/state-contract gap, not evidence that every existing H32 label is numerically wrong. Full replay/deployed counterfactual equivalence after bank events remains unverified.','formal_learning':'BLOCKED_DEPLOYMENT_CONTRACT','all_existing_labels_and_negative_results_preserved':True,'next_minimal_hypothesis':'Build native production prefix snapshots and H32 forks, or a fully proved state adapter including birth and reactivation writes; repeat identical predetermined intervention selections, preserve current results, then re-check native/label parity before fitting. Do not increase model size or open heldout.'}
 save(REPORTS/'GALLERY_FAILURE_ATTRIBUTION.json',r);print(json.dumps(r['example'],indent=2),flush=True)
if __name__=='__main__':main()
