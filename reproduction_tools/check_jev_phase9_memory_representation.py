"""Check the birth/write representation contract on every frozen data prefix."""
import collections,json
import torch
from jev_phase9_common import *
def main():
 protect();source=binding();records=[];diff=collections.Counter();failures=[];seen=set()
 for v in TRAIN+VAL:
  for parent,name in [(PREVIOUS/'opportunity_scan_v1','SCAN_RESULT.json'),(OUT/'capture_v1','CAPTURE_RESULT.json')]:
   m=json.loads((parent/f'video{v:02d}'/name).read_text())
   for r in m['bounded_snapshots']:
    if r['sha256']in seen:continue
    seen.add(r['sha256']);assert sha(r['path'])==r['sha256'];s=torch.load(r['path'],map_location='cpu');state=s['state'];local=collections.Counter(int(n)-len(state.memory.get(t,[]))for t,n in state.track_hits.items());diff.update(local)
    bad={str(t):{'hits':int(n),'writes':len(state.memory.get(t,[]))}for t,n in state.track_hits.items()if int(n)-len(state.memory.get(t,[]))!=1}
    if bad:failures.append({'key':list(s['key']),'tracks':bad})
    records.append({'key':list(s['key']),'path':r['path'],'sha256':r['sha256'],'tracks':len(state.track_hits),'hits_minus_memory_writes':dict(local),'prior_reactivation_rows':state.counters.get('reactivated_rows',0)})
 report={'status':'PASS'if not failures else'FAIL_PREFIX_REPRESENTATION_CONTRACT','binding':source,'snapshots':len(records),'per_track_instances_checked':sum(diff.values()),'offset_histogram':dict(diff),'failures':failures,'records':records,'bank_compensation':'native gallery includes birth, becomes possible at len==bank_size+1; replay subsequent writes become possible at len>=bank_size. These thresholds compensate the one-entry offset. Mature bank averages use final bank_size raw write vectors, which exclude the initial birth observation.','uncovered_limit':'If additional skipped/reactivation writes break the offset, first-vector reconstruction alone is insufficient and formal causal learning remains blocked. A passing offset is necessary, not full deployment proof.','H32_vs_window':'production window40 >H32; a newly born/reactivated current identity cannot leave that window and need a second bank lookup within32 frames; no lifecycle redesign authorized','WHAT_DID_WE_LEARN':'Different container encodings must be translated, not falsely identified as identical native gallery evidence.'}
 save(REPORTS/'MEMORY_REPRESENTATION_AUDIT.json',report);print(report['status'],report['snapshots'],report['offset_histogram'],len(failures),flush=True)
if __name__=='__main__':main()
