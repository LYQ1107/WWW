"""Reload immutable V3 heads and verify all3000 TRAIN outputs; no fitting."""
import argparse
import csv
import json
from pathlib import Path
import time

import torch

from cv_pil_dev12_assets import require
from cv_pil_native_handoff import sha256,safe_torch_load,inference_inputs
from cv_pil_v3_fit import load_fit,fixed_schedule
from cv_pil_v2_model import CandidatePredictor,tensorize,parameter_digest,training_eligible,fused_scores
from run_cv_pil_v2 import one_row,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'Existing saved-head verification cannot be overwritten')
    started=time.monotonic();torch.set_num_threads(1)
    original=json.loads((a.run/'RESULTS.json').read_text());proto=json.loads(a.protocol.read_text())
    require(sha256(a.protocol)==original['protocol_file_sha256'],'Training protocol SHA differs')
    for name,digest in original['source_sha256'].items():
        require(sha256(Path(__file__).with_name(name))==digest,'Training source changed: '+name)
    events,audit=load_fit(**proto['fit_source']);require(audit==original['source_audit'],'Training inputs changed')
    batches=[tensorize(inference_inputs(e)) for e in events]
    eligible=[i for i,(e,b) in enumerate(zip(events,batches)) if training_eligible(b,e['OFFLINE_ONLY'])]
    hard=[i for i in eligible if events[i]['OFFLINE_ONLY']['actual_primary_state'] in ('wrong_link','missed_known_positive')]
    require(eligible==proto['eligible_source_indices'] and hard==proto['hard_source_indices'] and
            fixed_schedule(eligible,hard)==proto['schedule_source_indices'],'Fixed shared TRAIN schedule differs')
    models={};head_records={}
    for arm,info in original['training'].items():
        require(sha256(info['local_checkpoint'])==info['checkpoint_sha256'],'Saved head SHA differs')
        state=safe_torch_load(info['local_checkpoint'])
        require(state['seed']==20261015 and state['steps']==300 and state['arm']==arm and
                state['cohort_sha256']==proto['fit_source']['expected_sha'] and
                state['protocol_sha256']==original['protocol_file_sha256'],'Saved head training contract differs')
        model=CandidatePredictor(use_camera=arm!='C4');model.load_state_dict(state['state_dict'],strict=True)
        require(parameter_digest(model)==info['final_parameter_sha256'] and
                sum(x.numel() for x in model.parameters())==38080 and
                all(torch.isfinite(x).all().item() and x.device.type=='cpu' for x in model.parameters()),'Saved model parameters differ')
        models[arm]=model.eval();head_records[arm]=dict(sha256=info['checkpoint_sha256'],parameters=38080,steps=300,seed=20261015)
    rows={arm:[] for arm in ('C0','C1','C2','C3','C4')};flat=[]
    with torch.no_grad():
        for e,b in zip(events,batches):
            scores=dict(C0=b.native_scores,C1=fused_scores(b,b.histories));scores.update({arm:model(b)[0] for arm,model in models.items()})
            r=dict(key=list(e['INPUT_ONLY']['event_key']),person=e['OFFLINE_ONLY']['GT_person'])
            for arm,s in scores.items():
                record=one_row(e,b,s);rows[arm].append(record)
                r.update({arm+'_native_id':record['prediction'],arm+'_known_pair_correct':record['outcome'],arm+'_RR':record['rr']})
            flat.append(r)
    require(summarize(rows)==original['original_distribution_train'],'Saved-head full TRAIN outputs differ from frozen training endpoint')
    a.out.mkdir(parents=True)
    with (a.out/'TRAIN_QUERY_RESULTS_OFFLINE_ONLY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    report=dict(status='PASS_ALL3000_SAVED_HEAD_OUTPUTS',queries=3000,methods=5,heads=head_records,
                exact_original_aggregate_scene_person_results=True,shared_schedule_verified=True,
                protocol_sha256=sha256(a.protocol),training_result_sha256=sha256(a.run/'RESULTS.json'),
                query_csv_sha256=sha256(a.out/'TRAIN_QUERY_RESULTS_OFFLINE_ONLY.csv'),
                wall_seconds=time.monotonic()-started,retraining=False,DEV_loaded=False,gpu_wall_hours=0)
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
