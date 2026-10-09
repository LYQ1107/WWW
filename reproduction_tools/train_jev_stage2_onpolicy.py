"""One bounded own-state TRAIN round, identical protocol for four main models."""
import argparse,time,random,math,traceback,gc
import numpy as np
import torch
from jev_phase13_learning import *
from jev_phase13_runtime import *
from evaluate_jev_stage2_online import load_policy
from run_jev_phase10_closed_loop import raw_predictions
from build_dense_jev_stage2_dataset import OfflineLabels

LARGE=Path('/data1/liuyeqiang/WWW_jev_phase13_runtime/20261009_v1')
MAIN_MODELS=['full','motip','camel','set_transformer']

def freeze():
    # Freeze before formal tracking outcomes; no phenotype-based clip choice.
    p=REPORTS/'ONPOLICY_PROTOCOL.json';assert not p.exists()
    save(p,{'status':'FROZEN_BEFORE_ONPOLICY','binding':binding(),'rounds':1,'models':MAIN_MODELS,'seeds':SEEDS,'videos':TRAIN,'frames_per_video':256,'bound_scope':'first256 actual scene frames of all four TRAIN videos, no selected corrective windows; finite round, not a full-TRAIN outcome claim','updates':4000,'total_updates':24000,'batch_payloads':4,'learning_rate':.00005,'weight_decay':.01,'gradient_clip':1.,'blend':'two original natural MATCH payloads and two own-state MATCH payloads per update','eligibility':'formal Full TRAIN fixed audit certified accuracy>=0.90; each actual own-state rollout needs>=5000 certified positive MATCH rows across all4 videos and remains UNKNOWN on contaminated histories','REACT':'NOT_RUN_INSUFFICIENT_DENSE_SUPPORT; no automatic task enablement','MEMORY':'NOT_RUN_NO_WRITE_KEEP_LABELS','fine_tune_primary_checkpoint':'LAST exactly4000 extra updates; same budget for all main models/seeds, separate total24k comparison','heldout':'SEALED','source_SHA256':sha(__file__)})

def rollout(variant,seed):
    assert variant in MAIN_MODELS and seed in SEEDS;protocol=json.loads((REPORTS/'ONPOLICY_PROTOCOL.json').read_text());assert sha(__file__)==protocol['source_SHA256'];source=binding();assert not source['dirty'];protect();torch.set_num_threads(1);torch.manual_seed(20261009)
    full=json.loads((OUT/'formal_training_v1/full'/f'seed{seed}'/'RESULT.json').read_text());assert full['TRAIN_FIXED_AUDIT_SUBSET']['certified_accuracy']>=.90,'normal association not qualified; do not run on-policy'
    policy,trained,temp=load_policy(variant,seed,'formal');results=[];total=collections.Counter();root=LARGE/'onpolicy_native_v1'/variant/f'seed{seed}'
    for v in TRAIN:
        out=root/f'video{v:02d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'RESULT.json').exists():
            r=json.loads((out/'RESULT.json').read_text());assert r['binding']==source and r['trained']==trained;results.append(r);total.update(r['counts']);continue
        model=build_tracker(v,policy,variant=variant,temperature=(temp,1.),react_learned=False);values,frames,reader=cache_inputs(v);labels=OfflineLabels(v,reader);model.jev_stage2_executor.observer=labels.before;model.jev_stage2_executor.commit_observer=labels.after;start=time.monotonic()
        with torch.no_grad():raw,_=run(model,values,frames,stop=protocol['frames_per_video']-1)
        data=out/'DATASET.pth';atomic_torch(data,{'records':labels.records,'video':v,'scope':'all actual decisions in first256 native self-state TRAIN frames; offline GT labels never inputs; no corrective filtering'});pred=raw_predictions(raw,labels.images);save(out/'RAW_PREDICTIONS.json',pred)
        from jev_phase7_offline import IdentityEvaluator
        evaluator=IdentityEvaluator(v);identity=evaluator.summarize(evaluator.align(pred));save(out/'IDENTITY_AUDIT.json',identity)
        result={'status':'COMPLETE','binding':source,'trained':trained,'variant':variant,'seed':seed,'video':v,'frames':protocol['frames_per_video'],'counts':dict(labels.counts),'dataset':{'path':str(data),'SHA256':sha(data),'bytes':data.stat().st_size},'raw_predictions_SHA256':sha(out/'RAW_PREDICTIONS.json'),'identity_summary':{k:val for k,val in identity.items() if k not in ['wrong_ID_episodes','identity_index_query']},'known_Gallery_IDs':sum(bool(s) for s in labels.past.values()),'contaminated_Gallery_IDs':sum(len(s)>1 for s in labels.past.values()),'GTA_throw_mocks':True,'actual_mutated_state':True,'seconds':time.monotonic()-start};save(out/'RESULT.json',result);results.append(result);total.update(labels.counts);del model,labels,raw;gc.collect();torch.cuda.empty_cache();print('PHASE13_OWN_STATE_ROLLOUT',variant,seed,v,result['counts'],flush=True)
    passed=total['normal_active_positive']>=5000
    manifest={'status':'PASS' if passed else 'FAIL','binding':source,'variant':variant,'seed':seed,'results':results,'counts':dict(total),'training_labels_unknown_not_negative':True,'history_source':'each current model real own-state commits, not B1/GMT future replay','REACT':'UNTRAINED_FALLBACK','MEMORY':'UNTRAINED_FALLBACK'};save(root/'MANIFEST.json',manifest);return manifest

def train(variant,seed):
    protect();source=binding();assert not source['dirty'];protocol=json.loads((REPORTS/'ONPOLICY_PROTOCOL.json').read_text());assert sha(__file__)==protocol['source_SHA256'];manifest=json.loads((LARGE/'onpolicy_native_v1'/variant/f'seed{seed}'/'MANIFEST.json').read_text());assert manifest['status']=='PASS'
    original,mm=load_data(TRAIN);original=[r for r in original if r['task']==0];own=[]
    for item in manifest['results']:
        p=item['dataset'];assert sha(p['path'])==p['SHA256'];own.extend(r for r in torch.load(p['path'],map_location='cpu')['records'] if r['task']==0)
    formal=json.loads((OUT/'formal_training_v1'/variant/f'seed{seed}'/'RESULT.json').read_text());assert sha(formal['checkpoint']['path'])==formal['checkpoint']['SHA256'];model=GlobalIdentityJev(variant).cuda();model.load_state_dict(torch.load(formal['checkpoint']['path'],map_location='cpu')['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay']);rng=random.Random(seed);torch.manual_seed(seed);torch.set_num_threads(1);out=OUT/'onpolicy_training_v1'/variant/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists();history=[];max_grad=0.;start=time.monotonic()
    original_audit=original[:64];own_audit=own[:64];step=0
    try:
        for step in range(1,protocol['updates']+1):
            part=rng.sample(original,2)+rng.sample(own,2);x,y=collate(part,variant=variant);model.train();optimizer.zero_grad(set_to_none=True);z=model(x);loss,parts=losses(z,x,y);assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip']));assert math.isfinite(grad);max_grad=max(max_grad,grad);optimizer.step()
            if step%1000==0:
                history.append({'update':step,'original_fixed_audit':evaluate(model,original_audit,variant),'own_state_fixed_audit':evaluate(model,own_audit,variant),'loss_parts':parts});save(out/'PROGRESS.json',{'status':'RUNNING','update':step,'target_updates':protocol['updates'],'seconds':time.monotonic()-start});print('PHASE13_ONPOLICY_UPDATE',variant,seed,step,flush=True)
        ck=out/'LAST_FROZEN.pth';atomic_torch(ck,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'variant':variant,'seed':seed,'updates':step,'total_updates':24000,'binding':source,'formal_checkpoint':formal['checkpoint'],'onpolicy_manifest_SHA256':sha(LARGE/'onpolicy_native_v1'/variant/f'seed{seed}'/'MANIFEST.json'),'history':history,'qualified_tasks':['MATCH']})
        # Same first256 TRAIN clips: actual mutated-state rollouts before/after.
        post=[];model.eval()
        for v in TRAIN:
            native=build_tracker(v,model,variant=variant,temperature=(formal['temperature'],1.),react_learned=False);values,frames,_=cache_inputs(v)
            with torch.no_grad():raw,_=run(native,values,frames,stop=255)
            ann=json.loads(ANNOTATIONS.read_text());ims={(i['frame_id']-1,i['view_id']-1):i for i in ann['images'] if i['video_id']==v};pred=raw_predictions(raw,ims);save(out/f'POST_RAW_video{v}.json',pred)
            from jev_phase7_offline import IdentityEvaluator
            evaluator=IdentityEvaluator(v);identity=evaluator.summarize(evaluator.align(pred));post.append({'video':v,'summary':{k:val for k,val in identity.items() if k not in ['wrong_ID_episodes','identity_index_query']},'predictions_SHA256':sha(out/f'POST_RAW_video{v}.json')});del native,raw;gc.collect()
        result={'status':'COMPLETE','binding':source,'variant':variant,'seed':seed,'phase':'onpolicy','actual_updates':step,'target_updates':protocol['updates'],'total_updates':24000,'checkpoint':{'path':str(ck),'SHA256':sha(ck)},'formal_checkpoint':formal['checkpoint'],'dataset_SHA256':{str(r['video']):r['dataset']['SHA256'] for r in manifest['results']},'original_dataset_SHA256':{str(r['video']):r['DATASET']['SHA256'] for r in mm},'temperature':formal['temperature'],'temperature_refit':False,'history':history,'before_online_TRAIN':[{'video':r['video'],'summary':r['identity_summary']} for r in manifest['results']],'after_online_TRAIN':post,'scope':'paired complete native mutated-state first256 TRAIN frames, censored error episodes; not full-TRAIN HOTA or independent test','gradient_health':{'finite':True,'max_gradient':max_grad},'REACT':'UNTRAINED_FALLBACK','MEMORY':'UNTRAINED_FALLBACK','seconds':time.monotonic()-start};save(out/'RESULT.json',result);print('PHASE13_ONPOLICY_COMPLETE',variant,seed,flush=True)
    except Exception:save(out/'FAILED.json',{'status':'FAIL','binding':source,'updates':step,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','rollout','train'],required=True);p.add_argument('--variant',choices=MAIN_MODELS);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage=='rollout':rollout(a.variant,a.seed)
    else:train(a.variant,a.seed)
