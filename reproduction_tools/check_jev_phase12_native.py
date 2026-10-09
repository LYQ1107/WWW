"""All original prefixes: actual resumed native OFF/SHADOW and fresh next frame."""
import argparse,time,traceback
import torch
from jev_phase12_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,prefix_state,fingerprint
from gtr.modeling.visual_jev_mcmot import VisualJev
from gtr.modeling.visual_jev_mcmot.lifecycle_controller import VisualLifecycleController

def main(video):
    protect();source=binding();assert not source['dirty'],'native workers need clean pinned source'
    assert json.loads((REPORTS/'STRUCTURAL_TESTS.json').read_text())['status']=='PASS'
    assert json.loads((REPORTS/'QUESTION_CONDITIONING_TESTS.json').read_text())['status']=='PASS'
    old=json.loads((ROOT/'reports/JEV_PHASE10/PHASE10_PREREGISTRATION.json').read_text())
    records=[r for r in old['original_snapshot_records'] if r['key'][0]==video]
    capture=PREVIOUS/'native_capture_v1'/f'video{video:02d}'/'compat'
    manifest=json.loads((capture/'RESULT.json').read_text());registry={tuple(r['key']):r for r in manifest['prefixes']}
    trace={tuple(r['key']):r for r in json.loads((capture/'FULL_NATIVE_TRACE.json').read_text())}
    out=OUT/os.environ.get('JEV_PHASE12_NATIVE_OUTPUT','native_parity_v1')/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():
        complete=json.loads((out/'RESULT.json').read_text());assert complete['binding']==source;print('NATIVE_ALREADY_PASS',video);return
    model=build_model(video);torch.manual_seed(20261009);network=VisualJev().cuda().eval()
    controller=VisualLifecycleController(network,'SHADOW');model.visual_jev_controller=controller
    rows=inputs(video,manifest['frames']);results=[];start=time.perf_counter()
    try:
        for index,record in enumerate(records):
            key=tuple(record['key']);registered=registry[key]
            assert sha(registered['path'])==registered['sha256']
            assert sha(registered['packet_path'])==registered['packet_sha256']
            expected=torch.load(registered['packet_path'],map_location='cpu')['packet']['batch']
            before=torch.load(registered['path'],map_location='cpu')
            modes=[]
            for mode in ['OFF','SHADOW']:
                model.visual_jev_enabled=mode=='SHADOW'
                observer=NativeProductionPrefixRecorder(model,[],out/f'record{index:03d}'/mode)
                pre=[False];errors=[]
                def observe_before(**values):
                    if (values['frame'],values['view'])==key[1:]:
                        actual=prefix_state(model,**values)
                        assert fingerprint(actual)==fingerprint(before),'full restored prefix/RNG mismatch'
                        pre[0]=True
                    observer.before(**values)
                def observe_after(**values):
                    batch=values['candidate']['batch']
                    if (values['frame'],values['view'])==key[1:]:
                        assert batch.candidate_ids==expected.candidate_ids
                        assert torch.equal(batch.legal_mask.cpu(),expected.legal_mask)
                        for name in ['scores','state64','evidence12']:
                            a=getattr(batch,name).cpu();b=getattr(expected,name);error=float((a-b).abs().max()) if a.numel() else 0.
                            assert error<=1e-6,(name,key,error);errors.append([name,error])
                    observer.after(**values)
                model.jev_native_prefix_observer=observe_before;model.jev_candidate_commit_observer=observe_after
                with torch.no_grad():NativeStateForkAdapter(model).run(registered['path'],rows,stop_frame=min(key[1]+1,manifest['frames']-1))
                assert pre[0] and observer.trace
                for actual in observer.trace:
                    assert actual==trace[tuple(actual['key'])],('full native commit/Gallery/bank/hits/events/RNG changed',mode,key,actual['key'])
                    assert len(actual['ids'])==len(set(actual['ids']))
                modes.append({'mode':mode,'prefix_full_exact':True,'current_and_next_frame_full_native_commits_exact':True,'payloads':len(observer.trace),'feature_errors':errors,'no_ID_duplicates':True})
            results.append({'original_record_index':index,'key':list(key),'status':'PASS','prefix_SHA256':registered['sha256'],'modes':modes})
            progress={'status':'RUNNING','video':video,'checked':len(results),'total':len(records),'elapsed_seconds':time.perf_counter()-start,'last_key':list(key),'results':results}
            save(out/'PROGRESS.json',progress);print('NATIVE_PREFIX_PASS',video,len(results),len(records),key,flush=True)
        summary={'status':'PASS','binding':source,'video':video,'source_records':len(records),'records':results,'seconds':time.perf_counter()-start,'shadow_typed_tasks':sorted({r['task'] for r in controller.records}),'shadow_predictions':len(controller.records),'old_840_conclusions_unchanged':True,'protected_historical_evidence':True,'no_real_model_training':True}
        save(out/'RESULT.json',summary);print('NATIVE_VIDEO_COMPLETE',video,len(records),flush=True)
    except Exception:
        save(out/'FAILED.json',{'status':'FAIL','binding':source,'video':video,'completed_records':results,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)
