"""Load saved heads and verify every frozen query prediction, without training."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from cv_pil_native_handoff import load_frozen_cohort,inference_inputs,sha256
from cv_pil_v2_model import CandidatePredictor,tensorize,fused_scores,training_eligible,parameter_digest
from run_cv_pil_v2 import one_row,hash_json,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    dest=a.run/'VERIFY_SAVED_HEADS.json'
    if dest.exists():raise FileExistsError(dest)
    started=time.monotonic();torch.set_num_threads(1)
    result=json.loads((a.run/'RESULTS.json').read_text());proto=result['protocol']
    binding=json.loads((a.run/'SOURCE_BINDING.json').read_text())
    for name,expected in binding['source_sha256'].items():
        assert sha256(Path(__file__).with_name(name))==expected
    assert sha256(a.run/'QUERY_RESULTS_OFFLINE_ONLY.csv')==binding['query_results_sha256']
    frozen_path=Path(__file__).resolve().parents[1]/'results/cv_pil_v2_20261015/V1_FROZEN_EVIDENCE.json'
    frozen=json.loads(frozen_path.read_text());root=Path(__file__).resolve().parents[1]
    for name,expected in frozen['repository_files'].items():assert sha256(root/name)==expected
    for w in frozen['local_heads'].values():assert sha256(w['path'])==w['sha256']
    w=frozen['frozen_gmt_plus_mlp_reference'];assert sha256(w['path'])==w['sha256']
    fit,dev,audit=load_frozen_cohort(Path(proto['runtime']),Path(proto['cohort']),proto['cohort_sha256'])
    assert audit==result['source_audit']
    models={}
    for arm,info in result['training'].items():
        assert info['steps']==300 and info['parameters']==38080
        path=Path(info['local_checkpoint']);assert path.resolve().is_relative_to(a.run.resolve())
        assert sha256(path)==info['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['arm']==arm and state['steps']==300 and state['seed']==proto['seed']
        assert state['cohort_sha256']==proto['cohort_sha256'] and state['protocol_sha256']==hash_json(proto)
        model=CandidatePredictor(use_camera=proto['explicit_camera'][arm]);model.load_state_dict(state['state_dict'],strict=True)
        assert parameter_digest(model)==info['final_parameter_sha256'];models[arm]=model.eval()
    rows=list(csv.DictReader((a.run/'QUERY_RESULTS_OFFLINE_ONLY.csv').open()))
    assert len(rows)==1300
    eligible=[];checked=0
    with torch.no_grad():
        for split,events in [('fit',fit),('development',dev)]:
            source_rows=[r for r in rows if r['split']==split]
            assert len(source_rows)==len(events)
            for index,(event,old) in enumerate(zip(events,source_rows)):
                b=tensorize(inference_inputs(event),proto['age_denominator'])
                if split=='fit' and training_eligible(b,event['OFFLINE_ONLY']):eligible.append(index)
                scores={'C0':b.native_scores,'C1':fused_scores(b,b.histories)}
                scores.update({arm:model(b)[0] for arm,model in models.items()})
                assert [int(old['scene']),int(old['image_id']),int(old['row_id'])]==list(event['INPUT_ONLY']['event_key'])
                for arm,score in scores.items():
                    r=one_row(event,b,score)
                    assert int(old[arm+'_native_id'])==r['prediction']
                    known={'True':True,'False':False,'':None}[old[arm+'_known_pair_correct']]
                    assert known==r['outcome']
                    rr=float(old[arm+'_RR']) if old[arm+'_RR'] else None
                    assert rr==r['rr']
                checked+=1
    sample=json.loads((a.run/'SAMPLING.json').read_text());rng=np.random.default_rng(proto['seed'])
    expected=[eligible[int(rng.integers(len(eligible)))] for _ in range(300)]
    assert expected==sample['schedule_source_indices'] and sample['eligible_count']==len(eligible)
    assert sample['eligible_query_keys']==[list(fit[i]['INPUT_ONLY']['event_key']) for i in eligible]
    assert {t['schedule_sha256'] for t in result['training'].values()}=={hash_json(expected)}
    write(dest,dict(status='PASS',checked_unique_physical_queries=checked,
        checked_methods=5,loaded_saved_heads=3,head_prediction_certification_and_MRR_parity=True,
        frozen_V1_and_MLP_SHA_preserved=True,source_SHA_rechecked=True,
        common_training_schedule_verified=True,no_retraining=True,gpu_wall_hours=0,
        wall_seconds=time.monotonic()-started,query_results_sha256=binding['query_results_sha256']))
    print(dest.read_text())


if __name__=='__main__':main()
