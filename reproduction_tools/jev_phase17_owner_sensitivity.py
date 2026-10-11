"""Separate current pure-history support from immutable earliest certified owner."""
from jev_phase17_common import *
import gzip,collections
from jev_phase13_runtime import cache_inputs
from jev_phase16_common import duplicate_masked_labels
from jev_phase14_artifacts import load_dense
from jev_phase15_pilot_native import converted_prefix
from gtr.modeling.jev_phase16.evidence_ledger import EvidenceLedger
if __name__=='__main__':
 protect()
 with gzip.open(OUT/'owner_audit/EVENTS.jsonl.gz','rt') as f:events=[json.loads(line) for line in f]
 totals=collections.defaultdict(collections.Counter);details=[]
 for video in TRAIN:
  _,_,reader=cache_inputs(video);labels=duplicate_masked_labels(video,reader);cached={}
  for event in [e for e in events if e['source_video']==video]:
   path=event['prefix']['path']
   if path not in cached:
    prefix=converted_prefix(load_dense(path,map_location='cpu'));ledger=EvidenceLedger.from_prefix(prefix);ledger.verify_raw_galleries(prefix['galleries']);owners={}
    for identity,records in ledger.records.items():
     votes=collections.Counter();n=0
     for record in records:
      n+=1;gt=labels.current(*record['key'][:2])[record['key'][2]]
      if gt is not None:votes[gt]+=1
      if identity not in owners and len(votes)==1 and sum(votes.values())>=3 and sum(votes.values())/n>=.8:owners[identity]=next(iter(votes))
    cached[path]=owners
   owners=cached[path];gt=event['GT_OFFLINE_ONLY'];selected=event['selected_ref'];owner=owners.get(selected);known=owner is not None and gt is not None;wrong=known and owner!=gt;available=[r for r in event['legal_refs'] if owners.get(r)==gt] if gt is not None else []
   c=totals['strict' if event['strict'] else 'diagnostic14'];c['events']+=1;c['anchored_known_selected']+=known;c['anchored_wrong_selected']+=wrong;c['anchored_wrong_with_correct_candidate']+=wrong and bool(available);c['original_pure_wrong_with_correct_candidate']+=event['wrong_owner'] and bool(event['certified_correct_available']);c['known_owner_disagreement']=c.get('known_owner_disagreement',0)+int(known and event['known_mask'] and owner!=event['qualified_owner_GT_OFFLINE_ONLY'])
   if wrong and available:details.append(dict(key=event['GT_free_event_key'],strict=event['strict'],selected=selected,earliest_owner_OFFLINE=owner,current_GT_OFFLINE=gt,correct_refs=available,original_pure_category=event['category'],original_pure_qualification=event['wrong_owner'] and bool(event['certified_correct_available']),group=[video,gt,event['GT_free_event_key'][2],event['time_block']]))
 groups={tuple(d['group']) for d in details if d['strict']}
 save(REPORTS/'OWNER_CERTIFICATION_SENSITIVITY.json',dict(status='COMPLETE_DIAGNOSTIC_NOT_LABEL_UPGRADE',counts={k:dict(v) for k,v in totals.items()},anchored_correctable_groups=len(groups),details=details,source=ref(REPORTS/'OWNER_IDENTIFIABILITY.json'),interpretation='P4 uses currently single-GT history support. P3 propagation preserves earliest certified owner across later mixtures. These answer different questions; pure-support count is not a census of all recoverable anchored-owner errors. No new intervention, teacher sampling or learning qualification follows this sensitivity.',stale_bank_field='available_stale_bank in P4 means prefix bank payload exists, not a candidate-specific lawful REACT qualification',binding=binding()))
 print('OWNER_CERTIFICATION_SENSITIVITY',dict(totals),len(groups),flush=True)
