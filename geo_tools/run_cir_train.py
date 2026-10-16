"""First-seed bounded CPU training on sealed FIT12 risk labels, not veto value.

No GMT/teacher training, DEV reading, scene selection or online-success claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import resource
import signal
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,sha256,save_json
from cir_risk_training import read_risk_training,calibrate_risk_threshold,dependence_metrics
from gtr.modeling.identity_revision.model import NativeRiskHead,INPUT_CONTRACT
from run_cir_guarded import save


def main():
    def stopped(signum,frame):raise TimeoutError('External owned CPU wall guard stopped this training attempt')
    signal.signal(signal.SIGTERM,stopped)
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'results/CIR_GMT/P4_RISK_LEARNING_PROTOCOL.json')
    p.add_argument('--ledger',type=Path,default=ROOT/'results/CIR_GMT/P4_CPU_RESOURCE_LEDGER.json');a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());train=protocol['shared_training']
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1') or torch.cuda.is_initialized():raise ValueError('Explicit CPU-only execution required')
    if train['first_seed']!=17 or train['epochs']!=2 or train['batch_size']!=512:raise ValueError('Frozen first-seed budget drift')
    if a.out.exists():raise FileExistsError(a.out)
    ledger=json.loads(a.ledger.read_text()) if a.ledger.exists() else dict(measured_CPU_wall_seconds=0.,GPU_wall_hours=0.,stages=[])
    name='P4_FIT12_RISK_SEED17_V1'
    if any(s['status']=='RUNNING' or s['stage']==name for s in ledger['stages']):raise ValueError('Preserve existing cost/attempt records')
    a.out.mkdir(parents=True);begun=time.monotonic();deadline=begun+600
    stage=dict(stage=name,status='RUNNING',maximum_CPU_wall_seconds=600,GPU_wall_hours=0.,pid=os.getpid(),out=str(a.out))
    ledger['stages'].append(stage);save(a.ledger,ledger)
    result=dict(status='RUNNING',scope='FIT12_RISK_ONLY_NOT_FULL24_OR_ONLINE',seed=17,GT_in_model_inputs=False,
        protocol=dict(path=str(a.protocol),sha256=sha256(a.protocol)),data=dict(path=str(a.data),sha256=protocol['training_file_sha256']),models={},
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),ROOT/'geo_tools/cir_risk_training.py',ROOT/'gtr/modeling/identity_revision/model.py')})
    def check_time():
        if time.monotonic()>deadline:raise TimeoutError('Registered CPU wall bound exceeded')
    def score(model,x):
        chunks=[]
        with torch.inference_mode():
            for start in range(0,len(x),8192):check_time();chunks.append(model(x[start:start+8192]).numpy())
        return np.concatenate(chunks)
    try:
        data=read_risk_training(a.data,protocol['training_file_sha256'],set(protocol['FIT_video_ids']))
        stats=data['stats'];admit=protocol['admission']
        if stats['wrong_queries']<admit['minimum_wrong_queries'] or stats['wrong_scene_person_groups']<admit['minimum_wrong_scene_person_groups'] or len(stats['wrong_scenes'])<admit['minimum_wrong_scenes']:
            raise ValueError('Insufficient genuine classification supervision')
        x,y,w=data['features'],data['labels'],data['weights'];mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.001)
        result['coverage']=stats
        for architecture in ('ordinary_mlp','grouped_risk'):
            check_time();torch.manual_seed(17);random.seed(17);np.random.seed(17)
            model=NativeRiskHead(architecture);model.set_normalization(mean,scale);model.train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            generator=torch.Generator().manual_seed(17+1001);order_hash=hashlib.sha256();seen=torch.zeros(len(x),dtype=torch.long)
            losses=[];steps=0;t=time.monotonic()
            for epoch in range(2):
                order=torch.randperm(len(x),generator=generator);order_hash.update(order.numpy().tobytes());total=0.
                for start in range(0,len(x),512):
                    check_time();idx=order[start:start+512];seen[idx]+=1;optimizer.zero_grad(set_to_none=True)
                    loss=(F.binary_cross_entropy_with_logits(model(x[idx]),y[idx],reduction='none')*w[idx]).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite risk training loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite risk training gradient')
                    optimizer.step();steps+=1;total+=float(loss)*len(idx)
                losses.append(total/len(x));print(json.dumps(dict(model=architecture,epoch=epoch+1,steps=steps,weighted_fit_loss=losses[-1])),flush=True)
            if not (seen==2).all():raise RuntimeError('Not every eligible record was trained twice')
            model.eval();known_scores=score(model,x);calibration=calibrate_risk_threshold(known_scores,y.numpy(),.001)
            all_scores=score(model,data['all_features']);trigger=all_scores>calibration['threshold']
            known_mask=np.zeros(len(all_scores),dtype=bool);known_mask[data['known_indices'].numpy()]=True
            checkpoint=a.out/(architecture+'_seed17.pt')
            torch.save(dict(input_contract=INPUT_CONTRACT,architecture=architecture,state_dict=model.state_dict(),
                threshold=calibration['threshold'],seed=17,epochs=2,protocol_sha256=sha256(a.protocol),training_data_sha256=protocol['training_file_sha256']),checkpoint)
            result['models'][architecture]=dict(checkpoint=dict(path=str(checkpoint),sha256=sha256(checkpoint)),parameters=961,
                steps=steps,epochs=2,all_eligible_records_seen_twice=True,sample_order_sha256=order_hash.hexdigest(),weighted_fit_losses=losses,
                fitting_calibration=calibration,group_metrics=dependence_metrics(known_scores,y.numpy(),data['people'],calibration['threshold']),
                all_FIT_queries_above_threshold=int(trigger.sum()),UNKNOWN_FIT_queries_above_threshold=int((trigger&~known_mask).sum()),
                elapsed_CPU_wall_seconds=time.monotonic()-t,tracking_metrics=None,online_improvement=None)
            save_json(a.out/'TRAINING_RESULTS.json',result)
        models=list(result['models'].values())
        if models[0]['sample_order_sha256']!=models[1]['sample_order_sha256'] or models[0]['steps']!=models[1]['steps']:
            raise RuntimeError('Matched training budget/order failed')
        result.update(status='COMPLETE_FIRST_SEED_FIT12_RISK_ONLY_NATIVE_ONLINE_PENDING',scientific_success=False,full24_fit_complete=False,
            timing_scope='CPU data hash/read/mask/train/calibrate; no GPU/GMT/teacher; Python/Torch import excluded',peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        stage['status']='COMPLETE'
    except BaseException as error:
        result.update(status='FAILED_NO_ADMISSION',error=repr(error));stage.update(status='FAILED_EVIDENCE_RETAINED',error=repr(error));raise
    finally:
        elapsed=time.monotonic()-begun;result['CPU_wall_seconds']=elapsed;stage['CPU_wall_seconds']=elapsed
        ledger['measured_CPU_wall_seconds']+=elapsed;save(a.ledger,ledger);save_json(a.out/'TRAINING_RESULTS.json',result)


if __name__=='__main__':main()
