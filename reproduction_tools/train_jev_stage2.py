"""Frozen Tiny -> bounded Dense Pilot -> equal-update formal association training."""
import argparse,random,time,math,traceback
import numpy as np
import torch
from jev_phase13_learning import *

def train(variant,seed,phase):
    protect();source=binding();assert not source['dirty'],'train only immutable committed source'
    protocol=json.loads((REPORTS/'TRAINING_PROTOCOL.json').read_text());assert protocol['status']=='FROZEN_BEFORE_TRAINING'
    assert variant in protocol['models'] and seed in protocol['seeds']
    for path,h in protocol['training_source_SHA256'].items():assert sha(ROOT/path)==h,path
    for gate in ['GTA_FREE_CONTRACT','STRUCTURAL_TESTS','DENSE_DATASET_MANIFEST','NATIVE_PARITY']:
        assert json.loads((REPORTS/(gate+'.json')).read_text())['status']=='PASS',gate
    rows,manifests=load_data(TRAIN);validation,valman=load_data(VAL);trainrows=[r for r in rows if r['task']==0];valrows=[r for r in validation if r['task']==0]
    if phase=='tiny':
        selected={tuple(r) for r in protocol['tiny_record_keys']};trainrows=[r for r in rows if tuple(r['key'])+(r['task'],) in selected];assert len(trainrows)==len(selected);valrows=trainrows
    else:
        assert json.loads((REPORTS/'TINY_LEARNABILITY.json').read_text())['status']=='PASS'
        if phase=='formal':assert json.loads((REPORTS/'PILOT.json').read_text())['status']=='PASS'
    updates_target=protocol['updates'][phase];batch_size=protocol['batch_payloads'];out=OUT/f'{phase}_training_v1'/variant/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():assert json.loads((out/'RESULT.json').read_text())['binding']==source;return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=GlobalIdentityJev(variant).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay']);max_gradient=0.;zero=0;history=[];cursor=0;order=[];epoch=0;update=0;start=time.monotonic()
    manifest_hashes={str(m['video']):m['DATASET']['SHA256'] for m in manifests};last=out/'LAST.pth';best=out/'BEST.pth';best_score=float('inf')
    def state():return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'variant':variant,'seed':seed,'phase':phase,'updates':update,'binding':source,'dataset_SHA256':manifest_hashes,'training_protocol_SHA256':sha(REPORTS/'TRAINING_PROTOCOL.json'),'order':order,'cursor':cursor,'epoch':epoch,'history':history,'best_score':best_score,'max_gradient':max_gradient,'zero_gradients':zero,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'qualified_tasks':['MATCH'],'REACT':'UNTRAINED_FROZEN_COSINE_FALLBACK','MEMORY':'UNTRAINED_FROZEN_NATIVE_WRITE'}
    if last.exists():
        ck=torch.load(last,map_location='cpu');assert ck['binding']==source and ck['dataset_SHA256']==manifest_hashes;model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);update=ck['updates'];order=ck['order'];cursor=ck['cursor'];epoch=ck['epoch'];history=ck['history'];best_score=ck['best_score'];max_gradient=ck['max_gradient'];zero=ck['zero_gradients'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    example,_=collate(trainrows[:batch_size],variant=variant);capacity=profile(model,example);torch.cuda.reset_peak_memory_stats()
    # Lexicographic audit subset fixed by protocol, never tracking outcomes.
    assessment_keys={tuple(k) for k in protocol['assessment_record_keys']}
    val_subset=[r for r in valrows if phase=='tiny' or tuple(r['key'])+(r['task'],) in assessment_keys]
    train_subset=trainrows[:protocol['assessment_train_groups']]
    try:
        while update<updates_target:
            if cursor>=len(order):
                order=list(range(len(trainrows)));random.Random(seed+epoch).shuffle(order);cursor=0;epoch+=1
            part=[trainrows[i] for i in order[cursor:cursor+batch_size]];cursor+=len(part);model.train();x,y=collate(part,variant=variant);optimizer.zero_grad(set_to_none=True);z=model(x);loss,parts=losses(z,x,y);assert torch.isfinite(loss)
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip']));assert math.isfinite(norm);max_gradient=max(max_gradient,norm);zero+=norm==0;optimizer.step();update+=1
            if update%protocol['assessment_interval'][phase]==0 or update==updates_target:
                tr=evaluate(model,train_subset,variant);va=evaluate(model,val_subset,variant);criterion=va['known_NLL'];assert criterion is not None and math.isfinite(criterion)
                history.append({'update':update,'TRAIN':tr,'DEVELOPMENT':va,'loss_parts':parts,'seconds':time.monotonic()-start})
                if criterion<best_score:best_score=criterion;atomic_torch(best,state())
                atomic_torch(last,state());save(out/'PROGRESS.json',{'status':'RUNNING','variant':variant,'seed':seed,'update':update,'target_updates':updates_target,'last_loss':float(loss),'validation_known_NLL':criterion,'seconds':time.monotonic()-start});print('STAGE2_TRAIN_PROGRESS',phase,variant,seed,update,updates_target,round(criterion,6),flush=True)
        # LAST is the preregistered primary checkpoint: identical optimizer
        # budget. BEST is retained as a transparent development diagnostic.
        frozen=out/'LAST_FROZEN.pth';assert not frozen.exists();atomic_torch(frozen,state());model.load_state_dict(torch.load(frozen,map_location='cpu')['model']);validation_metrics=evaluate(model,valrows,variant);training_metrics=evaluate(model,train_subset,variant)
        temperatures=protocol['temperature_grid'];calibration=[{'temperature':t,'metrics':evaluate(model,val_subset,variant,t)} for t in temperatures];temperature=min(calibration,key=lambda a:(a['metrics']['known_NLL'],a['temperature']))['temperature'] if phase=='formal' else 1.
        result={'status':'COMPLETE','binding':source,'phase':phase,'variant':variant,'seed':seed,'actual_updates':update,'target_updates':updates_target,'checkpoint':{'path':str(frozen),'SHA256':sha(frozen),'selection':'LAST at common frozen update budget'},'dataset_SHA256':manifest_hashes,'validation_dataset_SHA256':{str(m['video']):m['DATASET']['SHA256'] for m in valman},'capacity':capacity,'TRAIN_FIXED_AUDIT_SUBSET':training_metrics,'DEVELOPMENT_DENSE':validation_metrics,'temperature':temperature,'calibration_diagnostics':calibration,'gradient_health':{'finite':True,'max_unclipped_norm':max_gradient,'zero_steps':zero,'nonfinite_steps':0},'peak_training_VRAM_MiB':torch.cuda.max_memory_allocated()/1048576,'elapsed_seconds':time.monotonic()-start,'qualified_tasks':['MATCH'],'tiny_other_tasks':'engineering learnability only' if phase=='tiny' else None,'REACT':'UNTRAINED_FROZEN_COSINE_FALLBACK','MEMORY':'UNTRAINED_FROZEN_NATIVE_WRITE','heldout':'SEALED'}
        save(out/'RESULT.json',result);save(out/'PROGRESS.json',{'status':'COMPLETE','update':update,'target_updates':updates_target});print('STAGE2_TRAIN_COMPLETE',phase,variant,seed,flush=True)
    except Exception:
        save(out/'FAILED.json',{'status':'FAIL','binding':source,'updates':update,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['tiny','pilot','formal'],required=True);a=p.parse_args();train(a.variant,a.seed,a.phase)
