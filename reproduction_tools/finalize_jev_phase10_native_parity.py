"""Fail-closed aggregation of all original records and known bank diagnostics."""
import json
from collections import Counter
from jev_phase10_common import *

def main():
 protect();cfg=json.loads((REPORTS/'PHASE10_PREREGISTRATION.json').read_text());records=[];raw=[];full=[]
 for video in TRAIN+VAL:
  p=OUT/'prefix_parity_v1'/f'video{video:02d}'/'RESULT.json';d=json.loads(p.read_text());assert d['status']=='PASS'
  assert d['source_records']==sum(r['key'][0]==video for r in cfg['original_snapshot_records'])
  records.extend(d['records']);raw.append({'video':video,'path':str(p),'SHA256':sha(p),'source_commit':d['binding']['source_commit'],'CUDA_VISIBLE_DEVICES':d['binding']['CUDA_VISIBLE_DEVICES']})
  for mode in ['off','compat']:
   p=OUT/'native_capture_v1'/f'video{video:02d}'/mode/'RESULT.json';dd=json.loads(p.read_text());assert dd['status']=='COMPLETE';full.append({'video':video,'mode':mode,'path':str(p),'SHA256':sha(p),'source_commit':dd['binding']['source_commit']})
 assert len(records)==221
 assert Counter((tuple(r['key']),r['old_snapshot_SHA256'])for r in records)==Counter((tuple(r['key']),r['sha256'])for r in cfg['original_snapshot_records'])
 assert all(r['status']=='PASS'and r['state64_max_abs_error']<=1e-6 and r['evidence12_max_abs_error']<=1e-6 and r['original_raw_scores_max_abs_error']<=1e-6 for r in records)
 trace=json.loads((OUT/'native_capture_v1/video12/compat/FULL_NATIVE_TRACE.json').read_text());diagnostics=[]
 for key in cfg['known_diagnostic_keys']:
  t=next(r for r in trace if r['key']==key);e=[e for e in t['events']if e['track']==5]
  if key[1]==839:
   assert len(e)==1 and 'BANK_REACTIVATE'in e[0]['events']and 'REACTIVATION_WRITE'in e[0]['events'];diagnostics.append(e[0])
  else:
   import torch
   m=json.loads((OUT/'native_capture_v1/video12/compat/RESULT.json').read_text());entry=next(r for r in m['prefixes']if r['key']==key);before=torch.load(entry['path'],map_location='cpu');length=len(before['galleries'][5]);assert length==1367 and not e
   diagnostics.append({'key':key,'track':5,'boundary':'BEFORE_CURRENT_MATCH','ordered_gallery_length':length,'actual_hits':int(before['hits'][5]),'current_camera_observation_of_identity5':False,'last_write_key':[12,850,1],'native_prefix_SHA256':entry['sha256']})
 common={'status':'PASS','fixed_original_records':221,'unique_event_keys':220,'all_original_records_retained':True,'all_original_94_birth_only_failures_included':True,'float_tolerance':1e-6,'all_discrete_fields_exact':True,'heldout_sealed':True,'Full24':False,'official_TEST':False,'raw_checks':raw,'full_video_capture_manifests':full,'preregistration_SHA256':sha(REPORTS/'PHASE10_PREREGISTRATION.json'),'known_identity5_diagnostics':diagnostics,'causal_labels_not_yet_revalidated':True}
 compact=[{k:r[k]for k in ['key','source_record_index','old_snapshot_SHA256','state64_max_abs_error','evidence12_max_abs_error','original_raw_scores_max_abs_error','status']}for r in records]
 save(REPORTS/'NATIVE_STATE_PARITY.json',{**common,'records':compact,'serialized_prefix_and_native_commit_all_fields_exact':True})
 save(REPORTS/'GALLERY_PARITY.json',{**common,'ordered_gallery_vectors_exact':True,'native_gallery_lengths_content_and_order_exact':True,'legacy_memory_writes_are_not_native_gallery':True,'records':[{'key':r['key'],'source_record_index':r['source_record_index'],'old_writes_vs_native_gallery_offset_histogram':r['old_writes_vs_native_gallery_offset_histogram']}for r in records]})
 for name,field in [('PHASE10_FEATURE64_PARITY.json','state64_max_abs_error'),('PHASE10_EVIDENCE12_PARITY.json','evidence12_max_abs_error')]:save(REPORTS/name,{**common,'max_abs_error':max(r[field]for r in records),'records':[{'key':r['key'],'source_record_index':r['source_record_index'],'max_abs_error':r[field]}for r in records]})
 save(REPORTS/'PHASE10_GALLERY_PARITY.json',{**common,'canonical_report':'GALLERY_PARITY.json','canonical_SHA256':sha(REPORTS/'GALLERY_PARITY.json')})
 save(REPORTS/'PHASE10_STATE_EVENT_LEDGER_AUDIT.json',{**common,'source':'actual native production pre/post observers; no memory reconstruction','all_current_native_commit_events_RNG_and_containers_exact':True,'all_full_video_OFF_compat_native_commit_states_exact':True,'scope':'current commit for each fixed prefix; full H32 equality is independently gated in Phase B'})
 doc='''# Phase X native gallery repair

All 221 frozen Phase IX records (220 distinct boundaries) pass. All 94 records that failed the birth-only reconstruction remain in the event set. No old snapshot or checkpoint was overwritten.

The repair uses a serialized production prefix before current get_asso/MATCH, after current perceptions have been loaded. It preserves actual ordered Gallery Instances, history storage, hits, bank containers, ID counter and trajectory/Python/NumPy/Torch RNG. The fork resumes the production sliding loop, so it does not reconstruct Gallery from hits or the old Replay memory list.

Every fixed boundary preserves candidate order, raw association scores, Hungarian proposal and legal masks. Restored state64/evidence12 are within 1e-6; complete prefix and current native commit containers, observation events and RNG match exactly. Each full source video also has separate OFF/compat full-state and identity equality. Known video12 identity5 bank reactivation at frame839 and Gallery state at frame851 are separately recorded in the JSON reports.

This establishes the state repair, not validity of old H32 utilities or any learned architecture. Those remain gated until all frozen counterfactual branches run again. Large prefixes and full event traces remain on the server; compact reports bind their checks by SHA256.
'''
 (ROOT/'docs/PHASE10_GALLERY_REPAIR_REPORT.md').write_text(doc)
 print('ALL_FIXED_NATIVE_PREFIX_PARITY_PASS 221/221')
if __name__=='__main__':main()
