"""Complete closed-loop diagnostic with canonical typed features and frozen native B2 MATCH."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from jev_phase6_common import OUT,B2,sha,save,load_controller,protect_anchor
from jev_phase6_rollouts import NativeReplayLab
from run_jev_phase6_lifecycle_probe import canonical_row
from gtr.modeling.jev_lifecycle import pack_typed_state

def main(video,kind,device):
    checkpoint=OUT/'standalone_heads_native'/kind/'model_calibrated.pth'
    binding=json.loads((checkpoint.parent/'result.json').read_text())
    if binding['status']!='COMPLETE':raise RuntimeError('head has no eligible canonical fit')
    model=load_controller(checkpoint,'cpu');lab=NativeReplayLab(video,device,native_match_validation=True,fast_match=True)
    question='MEMORY_DECISION' if kind=='MEMORY' else 'REACTIVATION_DECISION'
    decisions=[];counts=Counter()
    def intervene(feature,q,legal,action,context):
        if q!=question:return action
        row=canonical_row(lab,{'question':q,'context':context})
        x=pack_typed_state(row['feature_vector'])
        with torch.no_grad():probs=model(x,[q],[legal])['probs'][0].cpu()
        selected=legal[int(probs.argmax())];counts[selected]+=1
        decisions.append({**row,'action':selected,'probs':probs.tolist(),'baseline_action':action})
        return selected
    output=OUT/f'standalone_closed_loop_native/{kind}/video{video:02d}'
    result=lab.run(output,intervention=intervene)
    with (output/'typed_decisions.jsonl').open('w') as f:
        for row in decisions:f.write(json.dumps(row,sort_keys=True)+'\n')
    dataset=lab.pilot.prepare_eval_dataset(lab.subset)
    prepared,evaluated=lab.pilot.run_eval('jev',Path(result['predictions']),dataset)
    metrics=lab.pilot.extract_metrics(evaluated)
    value={'status':'COMPLETE','video':video,'kind':kind,'metrics':metrics,'result':result,
        'typed_checkpoint_sha256':sha(checkpoint),'MATCH_checkpoint_sha256':sha(B2),
        'prediction_sha256':sha(result['predictions']),'actions':dict(counts),
        'MATCH':'frozen B2 with native second-round learned validation',
        'other_question':'instantaneous GMT','typed_parameters':39712,'total_controller_runtime_parameters':73792,
        'canonical_relative_feature_contract':True,'GT_or_future_runtime_inputs':False,
        'unified_shared_model':False,'video01_model_selection':False,'official_test_read':False,'full24_authorized':False}
    save(output/'result.json',value);protect_anchor();print(json.dumps({k:v for k,v in value.items() if k!='result'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[23,24],required=True)
    p.add_argument('--kind',choices=['MEMORY','REACT'],required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();main(a.video,a.kind,a.device)
