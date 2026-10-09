"""Native/data gated MATCH learning, atomic resumable checkpoints and curves."""
import argparse,random,time,math,traceback,shutil
import numpy as np
import torch
from jev_phase12_learning import *

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)

def train(name,supervision,seed,phase,steps=1000,epochs=100):
    gate_hashes=gates();rows,manifest=load_data();source=binding();assert not source['dirty']
    protocol=json.loads((REPORTS/'MATCH_TRAINING_PROTOCOL.json').read_text())
    assert name in protocol['models'] and seed in protocol['seeds']
    for path,digest in protocol['script_SHA256'].items():assert sha(ROOT/path)==digest
    if phase=='tiny':assert name==protocol['tiny_model'] and steps in [protocol['tiny_updates_initial'],2000]
    else:assert epochs==protocol['formal_epochs']
    stats=normalization(rows);trainrows=[r for r in rows if r['partition']=='train'];valrows=[r for r in rows if r['partition']=='validation']
    if phase=='tiny':
        selection=json.loads((PREVIOUS/'tiny_v1/SELECTION.json').read_text())
        selected={tuple(r['key'])+(r['row'],) for r in selection['rows']}
        trainrows=[r for r in trainrows if tuple(r['key'])+(r['row'],) in selected]
        assert len(trainrows)==12 and len({r['group'] for r in trainrows})==8
        valrows=trainrows
    else:
        assert phase=='formal';assert (REPORTS/'MATCH_TINY_RESULTS.json').exists(),'first finish independent Tiny; no95% gate'
    out=OUT/f'{phase}_training_v1'/name/supervision/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():
        done=json.loads((out/'RESULT.json').read_text());assert done['binding']==source and done['requested_steps']==steps and done['requested_epochs']==epochs;print('MATCH_ALREADY_COMPLETE',name,supervision,seed,phase);return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1)
    model=build_network(name,stats).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    updates=0;history=[];best=float('inf');start_epoch=0;max_grad=0.;gradient_zeros=0;begin=time.perf_counter()
    lastpath=out/'LAST.pth';bestpath=out/'BEST.pth'
    if lastpath.exists():
        saved=torch.load(lastpath,map_location='cuda:0');assert saved['binding']==source and saved['visual_dataset_SHA256']==manifest['SHA256']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);updates=saved['updates'];history=saved['history'];best=saved['best'];start_epoch=saved['epoch'];max_grad=saved['max_grad'];gradient_zeros=saved['gradient_zeros']
    target_updates=steps if phase=='tiny' else math.ceil(len(trainrows)/16)*epochs
    example_args,_=collate(trainrows[:min(16,len(trainrows))],'cuda:0')
    capacity=profile_macs(model,example_args)
    groups=collections.defaultdict(list)
    for row in trainrows:groups[row['group']].append(row)
    def checkpoint(epoch):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'model_name':name,'supervision':supervision,'seed':seed,'phase':phase,'epoch':epoch,'updates':updates,'history':history,'best':best,'max_grad':max_grad,'gradient_zeros':gradient_zeros,'binding':source,'visual_dataset_SHA256':manifest['SHA256'],'normalization':stats,'gate_SHA256':gate_hashes}
    def assessment(epoch):
        nonlocal best
        tr=evaluate(model,trainrows,stats,supervision,'cuda:0');va=evaluate(model,valrows,stats,supervision,'cuda:0')
        active='CE' if name=='no_H32' else supervision
        criterion=va['known_NLL'] if active=='CE' else va['ranking_loss'] if active=='H32' else (va['known_NLL'] or 0)+(va['ranking_loss'] or 0)
        assert criterion is not None and math.isfinite(criterion)
        item={'epoch':epoch,'update':updates,'TRAIN':{k:v for k,v in tr.items() if k!='records'},'VALIDATION':{k:v for k,v in va.items() if k!='records'},'selection_criterion':criterion,'elapsed_seconds':time.perf_counter()-begin};history.append(item)
        if criterion<best:
            best=criterion;atomic_torch(bestpath,checkpoint(epoch))
        atomic_torch(lastpath,checkpoint(epoch))
        save(out/'PROGRESS.json',{'status':'RUNNING','updates':updates,'target_updates':target_updates,'epoch':epoch,'TRAIN_correct':tr['CertifiedCorrect'],'TRAIN_unknown':tr['UNKNOWNChoice'],'selection_criterion':criterion,'elapsed_seconds':time.perf_counter()-begin})
        print('MATCH_PROGRESS',phase,name,supervision,seed,updates,target_updates,round(criterion,6),flush=True)
    try:
        epoch=start_epoch
        while updates<target_updates:
            model.train()
            if phase=='tiny':ordered=trainrows;batchsize=len(trainrows)
            else:
                generator=random.Random(seed+100000+epoch);keys=sorted(groups);generator.shuffle(keys);ordered=[]
                for key in keys:
                    within=list(groups[key]);generator.shuffle(within);ordered.extend(within)
                batchsize=16
            for start in range(0,len(ordered),batchsize):
                if updates>=target_updates:break
                model.train();chunk=ordered[start:start+batchsize];args,labels=collate(chunk,'cuda:0');optimizer.zero_grad(set_to_none=True)
                output=model(*args);perrow,parts=losses(output,labels,stats,supervision,name)
                loss=(perrow*labels['weight']).sum()/labels['weight'].sum().clamp_min(1e-8)
                assert torch.isfinite(loss),'nonfinite MATCH loss'
                loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));assert math.isfinite(gradient)
                max_grad=max(max_grad,gradient);gradient_zeros+=gradient==0;optimizer.step();updates+=1
                if phase=='tiny' and (updates%50==0 or updates==target_updates):assessment(updates)
            epoch+=1
            if phase=='formal':assessment(epoch)
        last=torch.load(lastpath,map_location='cuda:0');selected=torch.load(bestpath,map_location='cuda:0');model.load_state_dict(selected['model'])
        temperatures=[.25,.5,1.,2.,4.];candidates=[(evaluate(model,valrows,stats,supervision,'cuda:0',t),t) for t in temperatures]
        eligible=[item for item in candidates if item[0]['known_NLL'] is not None]
        validation,temperature=min(eligible,key=lambda item:(item[0]['known_NLL'],item[1])) if phase=='formal' and eligible else (evaluate(model,valrows,stats,supervision,'cuda:0'),1.)
        training=evaluate(model,trainrows,stats,supervision,'cuda:0',temperature)
        for namepath in ['BEST','LAST']:
            src=out/(namepath+'.pth');dst=out/(namepath+'_FROZEN.pth');assert not dst.exists();shutil.copyfile(src,dst)
        result={'status':'COMPLETE_STABLE','binding':source,'phase':phase,'model':name,'supervision':supervision,'seed':seed,'requested_steps':steps,'requested_epochs':epochs,'actual_updates':updates,'best_update':selected['updates'],'best_epoch':selected['epoch'],'visual_dataset_SHA256':manifest['SHA256'],'gate_SHA256':gate_hashes,'capacity':capacity,'TRAIN':training,'VALIDATION':validation,'temperature':temperature,'temperature_scope':'known candidates only; no calibrated UNKNOWN or abstain claim','history':history,'gradient_health':{'finite':True,'max_unclipped_norm':max_grad,'zero_gradient_steps':gradient_zeros,'nonfinite_steps':0},'checkpoints':{p:{'path':str(out/(p+'_FROZEN.pth')),'SHA256':sha(out/(p+'_FROZEN.pth'))} for p in ['BEST','LAST']},'elapsed_seconds':time.perf_counter()-begin,'partial_labels_not_negative':True,'unexecuted_Q_remains_NaN':True,'old_PhaseX_Tiny_FAIL_preserved':True,'all_models95_Tiny_gate_required':False,'heldout_status':'SEALED'}
        save(out/'RESULT.json',result);save(out/'PROGRESS.json',{'status':'COMPLETE_STABLE','updates':updates,'target_updates':target_updates});print('MATCH_TRAIN_COMPLETE',phase,name,supervision,seed,flush=True)
    except Exception:
        save(out/'FAILED.json',{'status':'FAIL','binding':source,'updates':updates,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--supervision',choices=['CE','H32','joint'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['tiny','formal'],required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--epochs',type=int,default=100);a=p.parse_args();train(a.model,a.supervision,a.seed,a.phase,a.steps,a.epochs)
