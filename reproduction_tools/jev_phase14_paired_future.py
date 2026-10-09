"""Frozen inference factor deletions from identical serialized production prefixes."""
import argparse,gzip,time,torch
from jev_phase14_common import *
from jev_phase13_runtime import build_tracker,cache_inputs,run
from audit_jev_stage2_gta_free import phase13_prefix
from jev_phase14_artifacts import load_dense,save_dense
from jev_phase14_question_diagnostics import QuestionProbe,delete_factor
from jev_phase14_native_risk import NativeRisk
from gtr.modeling.jev_native_state import fingerprint
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory

def main(mode,seed):
    protect();source=binding();torch.set_num_threads(1);torch.manual_seed(20261009)
    p=read(REPORTS/'PAIRED_NATIVE_PROTOCOL.json');root=OUT/'paired_native_future_v1';root.mkdir(exist_ok=True)
    if mode=='capture':
        references=[]
        for video in TRAIN:
            values,frames,reader=cache_inputs(video);model=build_tracker(video);model.jev_stage2_executor.memory_factory=CachedIdentityMemory
            def before(**d):
                if d['frame'] not in p['prefix_frames'] or d['view']!=0:return
                path=root/f'PREFIX_{video}_{d["frame"]}_0.pth.xz'
                if not path.exists():save_dense(path,phase13_prefix(model,d))
                references.append(dict(video=video,frame=d['frame'],view=0,path=str(path),SHA256=sha(path)))
            model.jev_native_prefix_observer=before
            with torch.no_grad():run(model,values,frames,stop=max(p['prefix_frames']))
            del model;torch.cuda.empty_cache()
        assert len(references)==12;save(root/'PREFIX_MANIFEST.json',dict(status='COMPLETE',binding=source,prefixes=references,GT_actor_inputs=False,policy='actual frozen cosine B1',serialization_at_boundary=True));return
    manifest=read(root/'PREFIX_MANIFEST.json');m=read(OLD/'formal_training_v1/full'/f'seed{seed}/RESULT.json');ck=m['checkpoint'];assert sha(ck['path'])==ck['SHA256']
    policy=QuestionProbe().cuda().eval();policy.load_state_dict(torch.load(ck['path'],map_location='cpu')['model']);policy.probe='unchanged'
    folder=root/f'seed{seed}';folder.mkdir(exist_ok=True);results=[];begin=time.monotonic()
    for video in TRAIN:
        values,frames,reader=cache_inputs(video);model=build_tracker(video,policy=policy);ex=model.jev_stage2_executor;ex.memory_factory=CachedIdentityMemory;builder=ex.batch
        factor=['unchanged']
        def batch(*a,**k):return delete_factor(builder(*a,**k),factor[0])
        ex.batch=batch
        for ref in [r for r in manifest['prefixes'] if r['video']==video]:
            assert sha(ref['path'])==ref['SHA256'];baseline=None
            for condition in p['factors']:
                path=folder/f'video{video}_frame{ref["frame"]}_{condition}.json'
                if path.exists():r=read(path)
                else:
                    prefix=load_dense(ref['path'],map_location='cuda:0');initial=fingerprint(prefix)
                    factor[0]=condition;policy.variant='no_shared_state' if condition=='shared_state' else 'full'
                    risk=NativeRisk(video,reader,prefix);ex.observer=risk.before;ex.commit_observer=risk.after
                    with torch.no_grad():raw,_=run(model,values,frames,stop=ref['frame']+p['future_scene_frames']-1,prefix=prefix)
                    assert risk.counts['payloads']==2*p['future_scene_frames']
                    trace=folder/f'video{video}_frame{ref["frame"]}_{condition}.jsonl.gz'
                    with gzip.open(trace,'wt') as h:
                        for event in risk.trace:h.write(json.dumps(event)+'\n')
                    r=dict(status='COMPLETE',binding=source,seed=seed,checkpoint=ck,condition=condition,prefix=ref,starting_state_SHA256=initial,
                        future_scene_frames=p['future_scene_frames'],audit=risk.summary(),trace=dict(path=str(trace),SHA256=sha(trace)),
                        actual_mutated_future=True,GT_actor_inputs=False,legal_actions_solver_lifecycle_unchanged=True,dependency_OOD_only=True,
                        committed_instances_SHA256=fingerprint(raw[:2*(ref['frame']+p['future_scene_frames'])]))
                    save(path,r);del prefix,raw,risk
                if condition=='unchanged':baseline=r
                assert baseline is not None and r['starting_state_SHA256']==baseline['starting_state_SHA256']
                results.append(r);print('PHASE14_PAIRED_FUTURE',seed,video,ref['frame'],condition,r['audit']['counts'],flush=True)
                save(folder/'PROGRESS.json',dict(done=len(results),total=156,key=[video,ref['frame'],condition]))
        del model;torch.cuda.empty_cache()
    assert len(results)==156;save(folder/'RESULT.json',dict(status='COMPLETE',binding=source,seed=seed,checkpoint=ck,results=results,seconds=time.monotonic()-begin))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['capture','branches'],required=True);a.add_argument('--seed',type=int,default=20261009);v=a.parse_args();main(v.mode,v.seed)
