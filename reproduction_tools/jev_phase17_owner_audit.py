"""Past-only owner evidence from actual first native payloads, offline labels isolated."""
from jev_phase17_common import *
import collections,copy,gzip,argparse
import torch
from torch.nn import functional as F
from jev_phase16_common import load_policy,duplicate_masked_labels
from jev_phase14_artifacts import load_dense
from jev_phase15_pilot_native import converted_prefix
from jev_phase13_runtime import cache_inputs
from gtr.modeling.jev_phase16.evidence_ledger import EvidenceLedger
from jev_phase16_memory_risk import MemoryRisk
from gtr.modeling.jev_stage2.assignment import lawful_choice

def main():
 protect();torch.set_num_threads(1);all_events=[];summary=collections.defaultdict(collections.Counter);sources=[]
 for video in TRAIN:
  values,frames,reader=cache_inputs(video);labels=duplicate_masked_labels(video,reader)
  manifest=read(XV/'train_commitment_prefixes_v1'/f'video{video:02d}/RESULT.json')
  for entry in manifest['counterfactual_prefixes']:
   frame,view=entry['key'][1:];name=f'frame{frame:06d}_view{view}_row{entry["row"]}';folder=OUT/'factorial/prefix/B0'/f'video{video:02d}'/name
   completed=read(folder/'RESULT.json');event=load_dense(folder/'event000.pt',map_location='cpu');x=event['raw_batch'];galleries=event['Gallery'];refs=[]
   with gzip.open(completed['queries']['path'],'rt') as f:first=json.loads(f.readline());refs=first['refs'];assert first['task']==0
   prefix=converted_prefix(load_dense(entry['prefix']['path'],map_location='cpu'));ledger=EvidenceLedger.from_prefix(prefix);ledger.verify_raw_galleries(galleries);votes={};owners={};coverage={}
   for identity,records in ledger.records.items():
    v=collections.Counter(labels.current(*r['key'][:2])[r['key'][2]] for r in records);unknown=v.pop(None,0);votes[identity]=v;coverage[identity]=sum(v.values())/max(1,len(records))
    if len(v)==1 and sum(v.values())>=3 and coverage[identity]>=.8:owners[identity]=next(iter(v))
   current=labels.current(frame,view);legal=x['legal'][0].tolist();logits=first['actual_logits'];choices=first['actual_assignment']
   source=ref(folder/'event000.pt');sources.append(source)
   for row,gt in enumerate(current):
    candidate_refs=[r for j,r in enumerate(refs) if legal[row][j]];content=[r for r,v in votes.items() if gt is not None and gt in v];certified=[r for r,o in owners.items() if gt is not None and o==gt];available=[r for r in candidate_refs if r in certified];aliases=certified
    mixed=[r for r in content if len(votes[r])>1];selected=refs[choices[row]] if choices[row]>=0 else None
    similarities=x['pair_evidence'][0,row,:,:4].max(-1).values.tolist();wrong_competitors=[r for r in candidate_refs if r in owners and owners[r]!=gt];best_correct=max([similarities[refs.index(r)] for r in available],default=None);best_wrong=max([similarities[refs.index(r)] for r in wrong_competitors],default=None);margin=best_correct-best_wrong if best_correct is not None and best_wrong is not None else None
    if gt is None:category='UNKNOWN_CURRENT_GT'
    elif not content:category='C5_NO_OBSERVED_PAST_PERSON_CONTENT'
    elif not available and certified:category='C3_CERTIFIED_PAST_OWNER_NOT_LEGAL'
    elif len(aliases)>1 or mixed:category='C2_ALIAS_OR_MIXED_OWNER_HISTORY'
    elif not available:category='C6_PERSON_CONTENT_WITHOUT_QUALIFIED_OWNER'
    elif margin is not None and margin<=.05:category='C4_NUMERIC_EVIDENCE_AMBIGUITY_PROXY'
    else:category='C1_QUALIFIED_OWNER_NUMERIC_SUPPORT'
    selected_owner=owners.get(selected);known=gt is not None and selected_owner is not None;wrong=known and selected_owner!=gt;correct=known and selected_owner==gt
    ranked=sorted(candidate_refs,key=lambda r:(-similarities[refs.index(r)],r));recall={str(k):bool(set(ranked[:k])&set(available)) if available else None for k in [1,4,8,16,32,len(candidate_refs)]}
    past=[]
    for ref_id in candidate_refs:
     records=ledger.records[ref_id];indices=[len(records)-1];own=[j for j,r in enumerate(records) if r['key'][1]==view];other=[j for j,r in enumerate(records) if r['key'][1]!=view]
     indices=sorted(set(indices+own[-1:]+other[-1:]))
     past.append(dict(identity=ref_id,qualified_owner_GT_OFFLINE_ONLY=owners.get(ref_id),known_history_fraction=coverage[ref_id],mixed_GT_history_offline=len(votes[ref_id])>1,raw_history_length=len(records),visual_cosine=similarities[refs.index(ref_id)],observations=[dict(index=j,key=list(records[j]['key']),normalized_box=records[j]['box'].tolist(),detection_score=records[j]['score']) for j in indices]))
    ev=dict(source_video=video,time_block=frame//64,GT_free_event_key=[video,frame,view,row],identity_alias_grouping_OFFLINE=aliases,conflict_group_OFFLINE=sorted({owners[r] for r in candidate_refs if r in owners}),GT_OFFLINE_ONLY=gt,known_mask=known,unknown_mask=not known,category=category,legal_refs=candidate_refs,selected_ref=selected,qualified_owner_GT_OFFLINE_ONLY=selected_owner,certified_correct_available=available,person_content_refs=content,wrong_owner=wrong,correct_owner=correct,cosine_margin=margin,Recall_at_K=recall,candidates=past,current_normalized_box=(x['detection_meta'][0,row,:4]).tolist(),available_stale_bank=prefix.get('old_reids') is not None and bool(prefix.get('old_reids')),source_state=source,prefix=entry['prefix'],valid_counterfactual_outcome={'scope':'four actually executed joint assignments only; no candidate-wise utility fabricated','results':{a:str(OUT/'factorial/prefix'/a/f'video{video:02d}'/name/'RESULT.json') for a in ['B0','B1','B2','B3']},'unexecuted_candidate_utility':None},strict=video!=14)
    ev['provenance_hash']=hashlib.sha256(json.dumps(ev,sort_keys=True,allow_nan=False).encode()).hexdigest();all_events.append(ev);c=summary['strict_TRAIN' if video!=14 else 'diagnostic_video14'];c[category]+=1;c['events']+=1;c['known_current']+=gt is not None;c['known_selected_owner']+=known;c['wrong_selected_owner']+=wrong;c['correct_selected_owner']+=correct;c['legal_correct_owner_available']+=bool(available);c['alias_events']+=len(aliases)>1;c['mixed_content_events']+=bool(mixed);c['candidate_count_gt16']+=len(candidate_refs)>16
    c['correction_opportunity']+=wrong and bool(available);c['false_correction_opportunity']+=correct and bool(wrong_competitors)
 # Independent event groups, no adjacent-frame multiplication. Uniform seeded group sample.
 grouped={}
 for ev in sorted(all_events,key=lambda e:e['GT_free_event_key']):
  if not ev['strict'] or ev['GT_OFFLINE_ONLY'] is None:continue
  if not (ev['wrong_owner'] or ev['category'].startswith(('C2','C3','C4','C6'))):continue
  key=(ev['source_video'],ev['GT_OFFLINE_ONLY'],ev['GT_free_event_key'][2],ev['time_block']);grouped.setdefault(key,ev)
 import random
 keys=sorted(grouped);random.Random(20261009).shuffle(keys);selected_keys=set(keys[:64]);probability=min(1,64/max(1,len(keys)));teacher=[]
 for key,ev in grouped.items():
  ev['sampling_probability']=probability;ev['sampled']=key in selected_keys
  if ev['sampled']:teacher.append(ev)
 allpath=OUT/'owner_audit/EVENTS.jsonl.gz';allpath.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(allpath,'wt') as stream:
  for ev in all_events:stream.write(json.dumps(ev,allow_nan=False)+'\n')
 save(OUT/'owner_audit/TEACHER_EVENTS.json',teacher)
 report=dict(status='COMPLETE_BOUNDED_PREFIX_CENSUS',binding=binding(),counts={k:dict(v) for k,v in summary.items()},events=ref(allpath),teacher_events=ref(OUT/'owner_audit/TEACHER_EVENTS.json'),independent_difficult_groups=len(grouped),teacher_group_count=len(teacher),maximum_teacher_groups=64,sampling='uniform seeded sample without replacement of independent groups; earliest actual first-payload event per group',candidate_sources=sources,category_scope='exclusive hierarchical offline diagnostic; C4 is limited numerical similarity proxy, not proof of fundamental information absence',GT_diagnostic_categories_are_runtime_features=False,counterfactual_scope='native executed joint arm outcomes only; independent candidate utilities null',identifiability_claim='certified past owner support is diagnostic, appearance consistency does not establish owner correctness')
 save(REPORTS/'OWNER_IDENTIFIABILITY.json',report);save(REPORTS/'PHASE17_OWNER_EVIDENCE_AUDIT.json',report)
 save(REPORTS/'PHASE17_CORRECTION_OPPORTUNITY.json',dict(status=report['status'],counts=report['counts'],scope=report['category_scope'],independent_groups=len(grouped),teacher_events=report['teacher_events']))
 print('OWNER_AUDIT_COMPLETE',report['counts'],len(grouped),len(teacher),flush=True)
if __name__=='__main__':main()
