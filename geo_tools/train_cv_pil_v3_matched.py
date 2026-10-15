"""One preregistered CPU300-step C2/C3/C4 fit, after real DEV admission.

No DEV examples or labels are loaded by this training program.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from cv_pil_dev12_assets import require
from cv_pil_native_handoff import sha256, inference_inputs
from cv_pil_v3_fit import load_fit
from cv_pil_v2_model import (CandidatePredictor, tensorize, training_eligible,
                              supervised_losses, parameter_digest)
from run_cv_pil_v2 import summarize, one_row


def execute(protocol_path, protocol_sha, out):
    require(sha256(protocol_path)==protocol_sha, 'Preregistration SHA differs')
    p=json.loads(Path(protocol_path).read_text());require(not out.exists(), 'Existing TRAIN result cannot be overwritten')
    require(p['status']=='PREREGISTERED_BEFORE_V3_TRAINING' and p['steps']==300 and p['seed']==20261015,
            'Unregistered training protocol')
    d=p['development_admission']; require(sha256(d['path'])==d['sha256'], 'DEV admission changed')
    admission=json.loads(Path(d['path']).read_text())
    require(admission['status']=='PASS_FULL_NATIVE_DEV12_INPUTS', 'Complete real DEV admission required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);started=time.monotonic()
    events,audit=load_fit(**p['fit_source'])
    batches=[tensorize(inference_inputs(e)) for e in events]
    eligible=[i for i,(e,b) in enumerate(zip(events,batches)) if training_eligible(b,e['OFFLINE_ONLY'])]
    require(eligible==p['eligible_source_indices'], 'Frozen common TRAIN eligibility differs')
    schedule=p['schedule_source_indices']; require(len(schedule)==300 and set(schedule)<=set(eligible), 'Invalid shared schedule')
    require([list(events[i]['INPUT_ONLY']['event_key']) for i in schedule]==p['schedule_query_keys'], 'Shared physical sample keys differ')
    out.mkdir(parents=True); training={};initials=set()
    for arm in ('C2','C3','C4'):
        torch.manual_seed(p['seed']);model=CandidatePredictor(use_camera=arm!='C4')
        initial=parameter_digest(model);initials.add(initial)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);logs=[]
        for step,index in enumerate(schedule,1):
            b=batches[index];s,h=model(b);rank,predict,count=supervised_losses(s,h,b,events[index]['OFFLINE_ONLY'])
            loss=rank+(0. if arm=='C2' else .5)*predict
            require(torch.isfinite(loss).item(), 'Nonfinite TRAIN objective')
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step in (1,100,200,300):
                log=dict(arm=arm,step=step,rank_loss=float(rank.detach()),prediction_loss=float(predict.detach()),
                         total_loss=float(loss.detach()),gradient_norm_before_clip=float(grad),cross_view_sources=count)
                logs.append(log);print(json.dumps(log),flush=True)
        file=out/(arm+'_step300.pth')
        torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=p['seed'],steps=300,
                        cohort_sha256=p['fit_source']['expected_sha'],protocol_sha256=protocol_sha),file)
        model.eval()
        training[arm]=dict(steps=300,parameters=sum(x.numel() for x in model.parameters()),
            local_checkpoint=str(file.resolve()),checkpoint_sha256=sha256(file),
            initial_parameter_sha256=initial,final_parameter_sha256=parameter_digest(model),logs=logs)
        # Store only a tiny head; reconstruct the three models for common
        # original-distribution TRAIN evaluation after all fits are completed.
    require(len(initials)==1, 'Arms have different initial parameters')
    from cv_pil_native_handoff import safe_torch_load
    from cv_pil_v2_model import fused_scores
    models={}
    for arm,t in training.items():
        model=CandidatePredictor(use_camera=arm!='C4');model.load_state_dict(safe_torch_load(t['local_checkpoint'])['state_dict'])
        models[arm]=model.eval()
    all_rows={arm:[] for arm in ('C0','C1','C2','C3','C4')}
    with torch.no_grad():
        for e,b in zip(events,batches):
            scores=dict(C0=b.native_scores,C1=fused_scores(b,b.histories))
            scores.update({arm:m(b)[0] for arm,m in models.items()})
            for arm,s in scores.items():all_rows[arm].append(one_row(e,b,s))
    result=dict(status='MATCHED_V3_FIT_COMPLETE_NOT_DEVELOPMENT_SUCCESS',training=training,
        protocol=dict(seed=p['seed'],cohort_sha256=p['fit_source']['expected_sha']),
        protocol_file_sha256=protocol_sha,source_audit=audit,original_distribution_train=summarize(all_rows),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1]).decode().strip(),
        source_sha256={f:sha256(Path(__file__).with_name(f)) for f in ['train_cv_pil_v3_matched.py','cv_pil_v3_fit.py','cv_pil_v2_model.py']},
        wall_seconds=time.monotonic()-started,gpu_wall_hours=0,DEV_loaded=False,scientific_success=False)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();execute(a.protocol,a.protocol_sha256,a.out)


if __name__=='__main__':main()
