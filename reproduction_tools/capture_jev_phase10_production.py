"""Frozen whole native production capture and original factual ID verification."""
import argparse,gzip,random,time
import numpy as np
import torch
from jev_phase10_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,fingerprint

def main(video,mode):
 protect();assert video in TRAIN+VAL;source=binding();assert not source['dirty'],'canonical workers must run a frozen clean checkout'
 protocol=json.loads((REPORTS/'PHASE10_PREREGISTRATION.json').read_text());out=OUT/'native_capture_v1'/f'video{video:02d}'/mode
 out.mkdir(parents=True,exist_ok=True);complete=out/'RESULT.json'
 if complete.exists():
  d=json.loads(complete.read_text());assert d['binding']==source;print('CAPTURE_ALREADY_COMPLETE',video,mode);return
 start=out/'START_MANIFEST.json'
 if start.exists():assert json.loads(start.read_text())['binding']==source
 else:save(start,{'binding':source,'video':video,'mode':mode,'fixed_snapshot_keys':[r['key']for r in protocol['original_snapshot_records']if r['key'][0]==video]})
 scan=Path('/home/liuyeqiang/WWW_jev_phase8_runtime/20261008_v1/opportunity_scan_v1')/f'video{video:02d}'
 prior=json.loads((scan/'SCAN_RESULT.json').read_text());total=prior['max_frame']+1
 factual={}
 for d in map(json.loads,gzip.open(scan/'natural_event_index.jsonl.gz','rt')):factual[tuple(d['key'])+(d['row'],)]=int(d['actual_committed_id'])
 model=build_model(video)
 if mode=='off':model.jev_candidate_policy=None
 rows=inputs(video,total)
 random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 wanted=[r['key']for r in protocol['original_snapshot_records']if r['key'][0]==video] if mode=='compat' else []
 if video==12 and mode=='compat':wanted+=protocol['known_diagnostic_keys']
 stamp=[0.];observed=[0]
 def progress(row,saved):
  for r,t in enumerate(row['ids']):
   k=tuple(row['key'])+(r,)
   assert factual[k]==t,('original native factual commit mismatch',k,t,factual[k])
   observed[0]+=1
  if time.monotonic()-stamp[0]>=20:
   save(out/'PROGRESS.json',{'status':'RUNNING','video':video,'mode':mode,'last_key':row['key'],'commits':len(recorder.trace),'observed_rows':observed[0],'captured_prefixes':len(saved),'pid':os.getpid(),'binding_source_commit':source['source_commit']});stamp[0]=time.monotonic()
 recorder=NativeProductionPrefixRecorder(model,wanted,out/'prefixes',progress,resume=True)
 model.jev_native_prefix_observer=recorder.before;model.jev_candidate_commit_observer=recorder.after
 begin=time.monotonic()
 with torch.no_grad():actual,_=model.sliding_inference_GMT(rows,2,[None,None,list(range(2*total))],native_raw=True)
 assert set(recorder.saved)==set(map(tuple,wanted))
 for i,instance in enumerate(actual):
  frame,view=divmod(i,2)
  for row,track in enumerate(instance.track_ids.tolist()):
   # The frame-zero seed camera has no association policy event in the index.
   if (video,frame,view,row)in factual:assert factual[video,frame,view,row]==track
 save(out/'FULL_NATIVE_TRACE.json',recorder.trace)
 saved=[]
 for d in recorder.saved.values():
  saved.append(dict(d,sha256=sha(d['path']),packet_sha256=sha(d['packet_path'])))
 result={'status':'COMPLETE','binding':source,'video':video,'mode':mode,'frames':total,'commits':len(recorder.trace),'original_factual_rows_verified':observed[0],'all_original_factual_IDs_exact':True,'full_raw_output_SHA256':fingerprint(actual),'prefixes':saved,'full_native_trace':{'path':str(out/'FULL_NATIVE_TRACE.json'),'sha256':sha(out/'FULL_NATIVE_TRACE.json')},'elapsed_seconds':time.monotonic()-begin,'no_GT_or_future_inputs':True}
 save(complete,result);save(out/'PROGRESS.json',{'status':'COMPLETE','video':video,'mode':mode,'frames':total,'captured_prefixes':len(saved),'commits':len(recorder.trace)});print('NATIVE_CAPTURE_COMPLETE',video,mode,len(saved),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--mode',choices=['off','compat'],required=True);a=p.parse_args();main(a.video,a.mode)
