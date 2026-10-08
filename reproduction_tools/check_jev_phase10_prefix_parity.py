"""All frozen records: actual native prefix serialization and real commit parity."""
import argparse,json,time
import numpy as np
import torch
from jev_phase10_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,fingerprint,prefix_state,pack

def error(a,b):
 a=torch.as_tensor(a).detach().cpu();b=torch.as_tensor(b).detach().cpu()
 assert a.shape==b.shape
 return float((a-b).abs().max()) if a.numel() else 0.

def main(video):
 protect();source=binding();assert not source['dirty']
 cfg=json.loads((REPORTS/'PHASE10_PREREGISTRATION.json').read_text());records=[r for r in cfg['original_snapshot_records']if r['key'][0]==video]
 cap=OUT/'native_capture_v1'/f'video{video:02d}'
 native=json.loads((cap/'compat/RESULT.json').read_text());off=json.loads((cap/'off/RESULT.json').read_text())
 assert native['full_raw_output_SHA256']==off['full_raw_output_SHA256'],'new full GMT OFF/compat behavior mismatch'
 trace=json.loads((cap/'compat/FULL_NATIVE_TRACE.json').read_text());baseline={tuple(r['key']):r for r in trace}
 offtrace=json.loads((cap/'off/FULL_NATIVE_TRACE.json').read_text());assert len(trace)==len(offtrace)
 assert all(a['full_native_commit_SHA256']==b['full_native_commit_SHA256']and a['ids']==b['ids']for a,b in zip(trace,offtrace)),'all full factual containers/RNG mismatch'
 registry={tuple(p['key']):p for p in native['prefixes']};out=OUT/'prefix_parity_v1'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'RESULT.json').exists():
  done=json.loads((out/'RESULT.json').read_text());assert done['binding']==source;print('PREFIX_PARITY_ALREADY_COMPLETE',video);return
 model=build_model(video);results=[];all_rows=inputs(video,native['frames'])
 for index,oldrecord in enumerate(records):
  key=tuple(oldrecord['key']);r=registry[key];assert sha(oldrecord['path'])==oldrecord['sha256']
  assert sha(r['path'])==r['sha256']and sha(r['packet_path'])==r['packet_sha256']
  original=torch.load(oldrecord['path'],map_location='cpu');expected=torch.load(r['packet_path'],map_location='cpu');before=torch.load(r['path'],map_location='cpu')
  packet=expected['packet'];batch=packet['batch'];p=original['proposal']
  assert tuple(p.track_ids)==tuple(batch.candidate_ids),'original frozen candidate references drifted'
  raw_error=error(p.scores,batch.scores);assert raw_error<=1e-6,('original raw-score drift',key,raw_error)
  oldmask=torch.isfinite(p.scores).clone()
  for row,col in p.banned_edges:oldmask[row,col]=False
  assert torch.equal(oldmask,batch.legal_mask),'original legal candidates differ'
  pairs={row:int(torch.where(batch.evidence12[row,:,2]>0)[0][0])for row in range(len(batch.scores))if bool((batch.evidence12[row,:,2]>0).any())}
  assert dict(p.pairs)==pairs,'original Hungarian proposal changed'
  fork=NativeProductionPrefixRecorder(model,[],out/f'check{index:03d}');pre=[False];got=[None]
  def observe_before(**d):
   current=prefix_state(model,**d)
   if tuple(current['key'])!=key:
    fork.before(**d);return
   assert fingerprint(current)==fingerprint(before),'native prefix full content/RNG serialization mismatch'
   assert set(current['galleries'])==set(before['galleries'])
   for t in current['galleries']:
    assert torch.equal(current['galleries'][t].reid_features.detach().cpu(),before['galleries'][t].reid_features),'ordered gallery vectors differ'
   pre[0]=True;fork.before(**d)
  def after(**d):
   candidate=d['candidate']
   if (d['frame'],d['view'])==key[1:]:got[0]=candidate['batch']
   fork.after(**d)
  model.jev_native_prefix_observer=observe_before;model.jev_candidate_commit_observer=after
  with torch.no_grad():NativeStateForkAdapter(model).run(r['path'],all_rows,stop_frame=key[1])
  assert pre[0]and got[0]is not None
  assert fork.trace[0]==baseline[key],'native commit containers/events/RNG drifted'
  state_error=error(got[0].state64,batch.state64);evidence_error=error(got[0].evidence12,batch.evidence12)
  assert state_error<=1e-6 and evidence_error<=1e-6
  assert tuple(got[0].candidate_ids)==tuple(batch.candidate_ids)and torch.equal(got[0].legal_mask.cpu(),batch.legal_mask)
  oldwrites={int(t):len(original['state'].memory.get(t,[]))for t in before['galleries']}
  expectedlengths={int(t):len(g)for t,g in before['galleries'].items()}
  result={'source_record_index':index,'key':list(key),'old_snapshot_SHA256':oldrecord['sha256'],'native_prefix':r,'status':'PASS','full_ordered_gallery_and_all_prefix_fields_exact':True,'full_commit_containers_events_RNG_exact':True,'state64_max_abs_error':state_error,'evidence12_max_abs_error':evidence_error,'original_raw_scores_max_abs_error':raw_error,'original_candidate_order_mask_proposal_exact':True,'legacy_memory_is_not_gallery':oldwrites!=expectedlengths,'old_writes_vs_native_gallery_offset_histogram':dict(__import__('collections').Counter(expectedlengths[t]-oldwrites[t]for t in expectedlengths))}
  results.append(result);save(out/'PROGRESS.json',{'status':'RUNNING','video':video,'checked':len(results),'total':len(records),'last_key':list(key)});save(out/'RESULT.partial.json',{'binding':source,'records':results})
 summary={'status':'PASS','binding':source,'video':video,'source_records':len(records),'records':results,'all_fixed_records_retained':True,'new_full_video_GMT_OFF_compat_all_state_parity':True,'legacy_raw_snapshots_unchanged':True,'factual_capture_SHA256':sha(cap/'compat/RESULT.json'),'no_utility_or_training_performed':True}
 save(out/'RESULT.json',summary);save(out/'PROGRESS.json',{'status':'COMPLETE','video':video,'checked':len(records),'total':len(records)});print('PREFIX_PARITY_COMPLETE',video,len(records),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)
