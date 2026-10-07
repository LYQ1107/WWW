"""Fixed-checkpoint action removal on all existing heldouts, native transitions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from jev_phase6_common import OUT,B2,sha,save,protect_anchor
from jev_phase6_rollouts import NativeReplayLab

def main(video,condition,device):
    lab=NativeReplayLab(video,device,native_match_validation=True,fast_match=True,compact_context=True)
    counts=Counter()
    def intervene(feature,question,legal,action,context):
        if question=='MATCH_DECISION' and action=='REASSOCIATE':
            counts['removed_REASSOCIATE']+=1
            return 'START_NEW' if condition=='G4a' else 'ACCEPT_CURRENT'
        return action
    output=OUT/f'native_removal/video{video:02d}/{condition}'
    result=lab.run(output,intervention=intervene)
    dataset=lab.pilot.prepare_eval_dataset(lab.subset)
    _,evaluated=lab.pilot.run_eval('jev',Path(result['predictions']),dataset)
    value={'status':'COMPLETE','video':video,'condition':condition,'metrics':lab.pilot.extract_metrics(evaluated),
        'result':result,'checkpoint_sha256':sha(B2),'prediction_sha256':sha(result['predictions']),
        'removed_REASSOCIATE':counts['removed_REASSOCIATE'],'future':'live frozen native B2 with forced current action removal',
        'official_test_read':False,'full24_authorized':False,'original_development_gate_revised':False}
    save(output/'result.json',value);protect_anchor();print(json.dumps({k:v for k,v in value.items() if k!='result'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[2,3,5],required=True)
    p.add_argument('--condition',choices=['G4a','G4b'],required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();main(a.video,a.condition,a.device)
