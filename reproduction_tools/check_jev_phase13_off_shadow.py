"""Re-run all 221 registered original native prefixes in OFF and new SHADOW.

The original GMT 1152D perception cache is retained in this engineering
regression only. SHADOW reads its first 1024 components as an explicitly
labelled Stage2 transfer probe; no main experiment uses these features.
"""
import argparse,time,traceback
import torch
from jev_phase13_common import *
from gtr.modeling.jev_stage2.model import GlobalIdentityJev
from gtr.modeling.jev_stage2.native import NativeDirectExecutor
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,prefix_state,fingerprint

def main(video):
    protect();source=binding();assert not source['dirty'];allowed(video);torch.set_num_threads(1)
    from jev_phase12_common import build_model,inputs,PREVIOUS
    old=json.loads((ROOT/'reports/JEV_PHASE10/PHASE10_PREREGISTRATION.json').read_text());records=[r for r in old['original_snapshot_records'] if r['key'][0]==video]
    capture=PREVIOUS/'native_capture_v1'/f'video{video:02d}'/'compat';manifest=json.loads((capture/'RESULT.json').read_text());registry={tuple(r['key']):r for r in manifest['prefixes']};trace={tuple(r['key']):r for r in json.loads((capture/'FULL_NATIVE_TRACE.json').read_text())}
    out=OUT/'off_shadow_native_v1'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists();model=build_model(video);torch.manual_seed(20261009);network=GlobalIdentityJev().cuda().eval();executor=NativeDirectExecutor(network,mode='SHADOW',react_learned=False);rows=inputs(video,manifest['frames']);results=[];begin=time.monotonic();score_calls=[0]
    def observe(**d):score_calls[0]+=1
    executor.observer=observe
    try:
        for index,record in enumerate(records):
            key=tuple(record['key']);registered=registry[key];assert sha(registered['path'])==registered['sha256'];assert sha(registered['packet_path'])==registered['packet_sha256'];expected=torch.load(registered['packet_path'],map_location='cpu')['packet']['batch'];before=torch.load(registered['path'],map_location='cpu');modes=[]
            for mode in ['OFF','SHADOW']:
                if mode=='SHADOW':model.jev_stage2_executor=executor
                else:model.__dict__.pop('jev_stage2_executor',None)
                observer=NativeProductionPrefixRecorder(model,[],out/f'record{index:03d}'/mode);pre=[False];errors=[]
                def prehook(**d):
                    if (d['frame'],d['view'])==key[1:]:assert fingerprint(prefix_state(model,**d))==fingerprint(before);pre[0]=True
                    observer.before(**d)
                def posthook(**d):
                    b=d['candidate']['batch']
                    if (d['frame'],d['view'])==key[1:]:
                        assert b.candidate_ids==expected.candidate_ids and torch.equal(b.legal_mask.cpu(),expected.legal_mask)
                        for name in ['scores','state64','evidence12']:
                            a=getattr(b,name).cpu();z=getattr(expected,name);error=float((a-z).abs().max()) if a.numel() else 0.;assert error<=1e-6;(errors.append([name,error]))
                    observer.after(**d)
                    if mode=='SHADOW':
                        current=d['instances'][-1]
                        for row,ref in enumerate(current.track_ids.tolist()):executor.memory.update(ref,current.reid_features[row,:1024],current.pred_boxes.tensor[row],current.image_size,d['frame'],d['view'])
                model.jev_native_prefix_observer=prehook;model.jev_candidate_commit_observer=posthook
                with torch.no_grad():NativeStateForkAdapter(model).run(registered['path'],rows,stop_frame=min(key[1]+1,manifest['frames']-1))
                assert pre[0] and observer.trace
                for actual in observer.trace:assert actual==trace[tuple(actual['key'])],(mode,key,actual['key'])
                modes.append({'mode':mode,'prefix_exact':True,'current_and_next_full_state_trace_exact':True,'payloads':len(observer.trace),'tensor_errors':errors})
            results.append({'key':list(key),'status':'PASS','modes':modes,'original_prefix_SHA256':registered['sha256']});save(out/'PROGRESS.json',{'status':'RUNNING','video':video,'checked':len(results),'total':len(records),'seconds':time.monotonic()-begin});print('PHASE13_OFF_SHADOW_PREFIX_PASS',video,len(results),len(records),flush=True)
        save(out/'RESULT.json',{'status':'PASS','binding':source,'records':results,'video':video,'original_record_count':len(records),'shadow_score_calls':score_calls[0],'seconds':time.monotonic()-begin,'scope':'full original GMT OFF parity plus read-only new Stage2 SHADOW on historical Stage2-transfer1024; not fresh Stage1 main evidence','heldout':'SEALED'});print('PHASE13_OFF_SHADOW_COMPLETE',video,flush=True)
    except Exception:save(out/'FAILED.json',{'status':'FAIL','binding':source,'completed':len(results),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)
