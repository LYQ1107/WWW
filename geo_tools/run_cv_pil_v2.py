"""Fixed CPU 1000/300 native-history controls; no collection or online tracker."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
import torch

from cv_pil_native_handoff import load_frozen_cohort, inference_inputs, sha256
from cv_pil_v2_model import (CandidatePredictor,tensorize,fused_scores,
                              supervised_losses,training_eligible,parameter_digest)

ROOT=Path(__file__).resolve().parents[1]
ARMS=('C2','C3','C4')


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def hash_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def one_row(event,batch,scores):
    info=event['OFFLINE_ONLY'];actor=event['INPUT_ONLY']
    order=torch.argsort(scores,descending=True,stable=True).tolist();first=order[0]
    targets=info['certified_target_indices'];known=info['candidate_known_mask']
    outcome=None if info['GT_person'] is None or not known[first] else first in targets
    rr=1./(min(order.index(t) for t in targets)+1) if targets else None
    return dict(key=list(actor['event_key']),scene=actor['video'],person=info['GT_person'],
                primary_state=info['actual_primary_state'],positive_covered=bool(targets),
                cross_view_positive=info['has_strict_past_cross_view_positive'],
                known_query=info['GT_person'] is not None,any_unknown_candidate=not all(known),
                alias_positive_count=len(targets),native_opportunity=info['correctable_primary_error'],
                final_commit_certified=info['final_native_commit_certified'],
                prediction=int(actor['native_candidate_ids'][first]),outcome=outcome,
                strict_retrieval_correct=bool(first in targets) if targets else None,rr=rr)


def metrics(rows):
    scoreable=[r for r in rows if r['positive_covered']]
    return dict(queries=len(rows),known_queries=sum(r['known_query'] for r in rows),
        positive_covered_queries=len(scoreable),
        candidate_coverage_all_queries=len(scoreable)/len(rows) if rows else None,
        candidate_coverage_known_queries=len(scoreable)/sum(r['known_query'] for r in rows) if any(r['known_query'] for r in rows) else None,
        strict_past_cross_view_positive_queries=sum(r['cross_view_positive'] for r in rows),
        top1=sum(r['strict_retrieval_correct'] for r in scoreable)/len(scoreable) if scoreable else None,
        mrr=sum(r['rr'] for r in scoreable)/len(scoreable) if scoreable else None,
        certified_known_correct=sum(r['outcome'] is True for r in rows),
        certified_known_wrong=sum(r['outcome'] is False for r in rows),
        prediction_unknown=sum(r['outcome'] is None for r in rows),
        prediction_unknown_ratio=sum(r['outcome'] is None for r in rows)/len(rows) if rows else None,
        query_unknown=sum(not r['known_query'] for r in rows),
        query_unknown_ratio=sum(not r['known_query'] for r in rows)/len(rows) if rows else None,
        query_with_unknown_candidate=sum(r['any_unknown_candidate'] for r in rows),
        alias_positive_queries=sum(r['alias_positive_count']>1 for r in rows),
        actual_primary_states=dict(Counter(r['primary_state'] for r in rows)),
        certified_native_committed_fixes=None,certified_native_committed_harms=None)


def paired(before,after):
    assert [r['key'] for r in before]==[r['key'] for r in after]
    fixes=[b for b,a in zip(before,after) if b['outcome'] is False and a['outcome'] is True]
    harms=[b for b,a in zip(before,after) if b['outcome'] is True and a['outcome'] is False]
    return dict(certified_pair_rank_fixes=len(fixes),certified_pair_rank_harms=len(harms),
        correct_to_unknown=sum(b['outcome'] is True and a['outcome'] is None for b,a in zip(before,after)),
        unknown_to_correct=sum(b['outcome'] is None and a['outcome'] is True for b,a in zip(before,after)),
        fix_person_groups=len({(r['scene'],r['person']) for r in fixes}),
        harm_person_groups=len({(r['scene'],r['person']) for r in harms}),
        changed_choices=sum(b['prediction']!=a['prediction'] for b,a in zip(before,after)),
        fixed_query_keys=[r['key'] for r in fixes],harmed_query_keys=[r['key'] for r in harms],
        native_committed_corrections=None,native_committed_harms=None)


def summarize(all_rows):
    base=all_rows['C0'];result=dict(methods={arm:metrics(rows) for arm,rows in all_rows.items()})
    result['paired_vs_C0']={arm:paired(base,rows) for arm,rows in all_rows.items() if arm!='C0'}
    result['C3_vs_C2']=paired(all_rows['C2'],all_rows['C3'])
    result['C3_vs_C4']=paired(all_rows['C4'],all_rows['C3'])
    for grouping,key in [('by_scene',lambda r:str(r['scene'])),
                         ('by_person',lambda r:f"{r['scene']}:{r['person']}")]:
        groups=defaultdict(list)
        for i,r in enumerate(base):
            if grouping=='by_person' and r['person'] is None:continue
            groups[key(r)].append(i)
        result[grouping]={g:dict(methods={arm:metrics([rs[i] for i in ii]) for arm,rs in all_rows.items()},
            C3_vs_C2=paired([all_rows['C2'][i] for i in ii],[all_rows['C3'][i] for i in ii]),
            C3_vs_C4=paired([all_rows['C4'][i] for i in ii],[all_rows['C3'][i] for i in ii])) for g,ii in groups.items()}
    result['unique_scene_person_groups']=len(result['by_person'])
    result['person_macro_top1']={arm:float(np.mean([v['methods'][arm]['top1'] for v in result['by_person'].values() if v['methods'][arm]['top1'] is not None])) for arm in all_rows}
    return result


def execute(protocol,out):
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert protocol['seed']==20261015 and protocol['steps_per_arm']==300
    assert protocol['feature_dim']==1152 and protocol['rank']==16
    out.mkdir(parents=True,exist_ok=False)
    write(out/'STATE.json',dict(status='VERIFYING_FROZEN_INPUTS',gpu_wall_hours=0))
    fit,dev,audit=load_frozen_cohort(Path(protocol['runtime']),Path(protocol['cohort']),protocol['cohort_sha256'])
    data={'fit':fit,'development':dev}
    batches={split:[tensorize(inference_inputs(e),protocol['age_denominator']) for e in es] for split,es in data.items()}
    eligible=[i for i,(e,b) in enumerate(zip(fit,batches['fit'])) if training_eligible(b,e['OFFLINE_ONLY'])]
    if not eligible:raise ValueError('No certified training ranking targets')
    rng=np.random.default_rng(protocol['seed']);schedule=[eligible[int(rng.integers(len(eligible)))] for _ in range(300)]
    write(out/'SAMPLING.json',dict(eligible_query_keys=[fit[i]['INPUT_ONLY']['event_key'] for i in eligible],
        eligible_count=len(eligible),schedule_source_indices=schedule,schedule_sha256=hash_json(schedule),
        eligibility='Common frozen supervised subset; no DEV, no candidate removal from actor inputs.'))
    training={};models={};initials=set()
    for arm in ARMS:
        torch.manual_seed(protocol['seed']);model=CandidatePredictor(use_camera=protocol['explicit_camera'][arm])
        initial=parameter_digest(model);initials.add(initial)
        optim=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        arm_start=time.monotonic();logs=[];with_prediction=0
        for step,index in enumerate(schedule,1):
            b=batches['fit'][index];info=fit[index]['OFFLINE_ONLY']
            scores,pred=model(b);rank,predict,count=supervised_losses(scores,pred,b,info)
            loss=rank+protocol['prediction_loss_weight'][arm]*predict
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['grad_clip_norm']);optim.step()
            with_prediction+=count>0
            if step==1 or step%100==0:
                row=dict(arm=arm,step=step,ranking_loss=float(rank.detach()),prediction_loss=float(predict.detach()),
                         total_loss=float(loss.detach()),cross_view_positive_history_rows=count)
                logs.append(row);print(json.dumps(row),flush=True)
                write(out/'STATE.json',dict(status='TRAINING',arm=arm,step=step,wall_seconds=time.monotonic()-started,gpu_wall_hours=0))
        path=out/(arm+'_step300.pth');model.eval()
        torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=protocol['seed'],steps=300,
                        cohort_sha256=protocol['cohort_sha256'],protocol_sha256=hash_json(protocol)),path)
        training[arm]=dict(steps=300,parameters=sum(p.numel() for p in model.parameters()),
            initial_parameter_sha256=initial,final_parameter_sha256=parameter_digest(model),
            schedule_sha256=hash_json(schedule),steps_with_cross_view_prediction_target=with_prediction,
            training_wall_seconds=time.monotonic()-arm_start,logs=logs,
            local_checkpoint=str(path),checkpoint_sha256=sha256(path))
        models[arm]=model
    assert len(initials)==1 and len({v['parameters'] for v in training.values()})==1
    splits={};query_rows=[]
    with torch.no_grad():
        for split,events in data.items():
            rows={arm:[] for arm in ('C0','C1',*ARMS)}
            for event,b in zip(events,batches[split]):
                score={'C0':b.native_scores,'C1':fused_scores(b,b.histories)}
                score.update({arm:model(b)[0] for arm,model in models.items()})
                for arm,s in score.items():rows[arm].append(one_row(event,b,s))
                r=rows['C0'][-1]
                flat=dict(split=split,scene=r['scene'],image_id=r['key'][1],row_id=r['key'][2],
                    GT_person_OFFLINE_ONLY=r['person'],primary_state=r['primary_state'],positive_covered=r['positive_covered'],
                    cross_view_positive=r['cross_view_positive'],alias_positive_count=r['alias_positive_count'])
                for arm in rows:
                    z=rows[arm][-1];flat[arm+'_native_id']=z['prediction'];flat[arm+'_known_pair_correct']=z['outcome'];flat[arm+'_RR']=z['rr']
                query_rows.append(flat)
            splits[split]=summarize(rows)
    with (out/'QUERY_RESULTS_OFFLINE_ONLY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(query_rows[0]));writer.writeheader();writer.writerows(query_rows)
    dev_summary=splits['development'];opportunity=audit['counts'].get('validation_correctable_primary_error',0)
    unique_groups=sum(v['C3_vs_C2']['certified_pair_rank_fixes']>v['C3_vs_C2']['certified_pair_rank_harms'] for v in dev_summary['by_person'].values())
    compare=dev_summary['C3_vs_C2'];camera=dev_summary['C3_vs_C4']
    predictive_unique=(compare['certified_pair_rank_fixes']>compare['certified_pair_rank_harms'] and unique_groups>=2 and
                       camera['certified_pair_rank_fixes']>camera['certified_pair_rank_harms'])
    status=('INSUFFICIENT_NATIVE_OPPORTUNITY' if opportunity==0 else
            'NO_UNIQUE_PREDICTIVE_EFFECT' if not predictive_unique else 'DEVELOPMENT_POSITIVE_PENDING_FULL_NATIVE')
    # Even a positive old cohort does not certify final commits. No deployment
    # job is launched by this program under any outcome.
    result=dict(status=status,scientific_success=False,primary_opportunity_queries=opportunity,
        final_commit_opportunity_certified=False,unique_predictive_rank_effect_qualified=predictive_unique,
        secondary_status=None if predictive_unique else 'NO_UNIQUE_PREDICTIVE_EFFECT',
        source_audit=audit,protocol=protocol,protocol_canonical_sha256=hash_json(protocol),
        training=training,splits=splits,wall_seconds=time.monotonic()-started,
        peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_wall_hours=0,full_online=dict(HOTA=None,AssA=None,CVIDF1=None,IDSW=None,Frag=None,FPS=None),
        native_integration='NOT_STARTED_GATE_CLOSED',miner_handoff='WAITING_EXTERNAL_FROZEN_MINER_RELEASE',
        limitations=['Development scene12 previously exposed; one seed, no independent validation.',
            'Top1/MRR refer to positive-covered native Top8 records, not final committed IDs.',
            'UNKNOWN candidates remain legal at inference; masked only in supervised ranking loss.',
            'Alias tracks are multi-positive; success grouped by scene and true person, not candidate edges.',
            'Full1152 includes native appearance+spatial/camera cues; no claim of camera-free representations.',
            'No encoder/video collection, no TrackVGGT weights/scripts imported, no legacy ledger edits.'])
    for source,expected in audit['verified_sources'].items():
        assert sha256(source)==expected['events_sha256']
        assert sha256(Path(source).parent/'manifest.json')==expected['manifest_sha256']
    write(out/'RESULTS.json',result)
    write(out/'SOURCE_BINDING.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('cv_pil_v2_model.py'),Path(__file__).with_name('cv_pil_native_handoff.py')]},
        python=__import__('platform').python_version(),torch=torch.__version__,numpy=np.__version__,
        written_utc=datetime.now(timezone.utc).isoformat(),query_results_sha256=sha256(out/'QUERY_RESULTS_OFFLINE_ONLY.csv')))
    write(out/'STATE.json',dict(status='COMPLETE',research_status=status,gpu_wall_hours=0,wall_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=status,development=dev_summary['methods'],C3_vs_C2=compare,C3_vs_C4=camera)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=ROOT/'manifests/cv_pil_v2_fixed_protocol.json')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out.parent).free<30*2**30:raise RuntimeError('Free disk below30GiB')
    protocol=json.loads(a.protocol.read_text())
    try:execute(protocol,a.out)
    except Exception as exc:
        if a.out.exists():write(a.out/'FAILURE.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':main()
