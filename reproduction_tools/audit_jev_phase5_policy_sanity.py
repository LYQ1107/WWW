"""Verify the catastrophic MLP run before treating it as experimental evidence."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
from audit_jev_phase5 import OUT,BASE,save,records
from gtr.modeling.jev_runtime import build_controller_from_checkpoint
from gtr.modeling.jev_decision import ACTION_NAMES,action_index,question_index


def main():
    raw=OUT/'minimal_training/C2/model.pth';cal=OUT/'minimal_training/C2/calibration/model_calibrated.pth'
    r=torch.load(raw,map_location='cpu');c=torch.load(cal,map_location='cpu');weights_equal=all(torch.equal(v,c['model'][k]) for k,v in r['model'].items());assert weights_equal
    model=build_controller_from_checkpoint(cal,device='cpu');online=[json.loads(l) for l in (OUT/'minimal_tracking/C2/online_decisions.jsonl').open()];off=[json.loads(l) for l in (OUT/'ablations/A0/online_decisions.jsonl').open() if json.loads(l)['question']=='MATCH_DECISION']
    online=[d for d in online if d['question']=='MATCH_DECISION'];mismatch=0;counter=Counter()
    with torch.no_grad():
        for start in range(0,len(online),256):
            group=online[start:start+256];features=torch.tensor([d['feature_vector'] for d in group]);legal=torch.full((len(group),len(ACTION_NAMES)),-1,dtype=torch.long)
            for i,d in enumerate(group):legal[i,:len(d['legal_actions'])]=torch.tensor([action_index(a) for a in d['legal_actions']])
            output=model(features,torch.full((len(group),),question_index('MATCH_DECISION')),legal);columns=output['probs'].argmax(1);chosen=[ACTION_NAMES[int(legal[i,col])] for i,col in enumerate(columns)]
            mismatch+=sum(a!=d['action'] for a,d in zip(chosen,group));counter.update(chosen)
        # The list/string runtime API must select the same semantic action as
        # the training tensor/padded-mask API on identical state values.
        first=online[0];x=torch.tensor(first['feature_vector']);o=model(x,'MATCH_DECISION',first['legal_actions']);first_choice=first['legal_actions'][int(o['probs'].argmax())]
        assert first_choice==first['action']
        off_counts=Counter()
        for start in range(0,len(off),256):
            group=off[start:start+256];features=torch.tensor([d['feature_vector'] for d in group]);legal=torch.full((len(group),len(ACTION_NAMES)),-1,dtype=torch.long)
            for i,d in enumerate(group):legal[i,:len(d['legal_actions'])]=torch.tensor([action_index(a) for a in d['legal_actions']])
            output=model(features,torch.full((len(group),),question_index('MATCH_DECISION')),legal);off_counts.update(ACTION_NAMES[int(legal[i,col])] for i,col in enumerate(output['probs'].argmax(1)))
    assert mismatch==0
    first_state_equal=np.array_equal(np.asarray(first['feature_vector'],dtype=np.float32),np.asarray(off[0]['feature_vector'],dtype=np.float32))
    report={'status':'PASS','checkpoint_model_name':c['model_name'],'hidden_dim':c['hidden_dim'],'calibration_did_not_change_weights':weights_equal,'temperature':c['temperature'],'all_online_decisions_recomputed_on_CPU':len(online),'action_mismatches':mismatch,'online_CPU_recomputed_actions':dict(counter),'same_model_on_OFF_state_features':dict(off_counts),'first_state_exactly_equals_OFF_state':first_state_equal,'first_probabilities':o['probs'].tolist(),'first_legal_actions':first['legal_actions'],'first_training_tensor_and_runtime_list_action_identical':True,'interpretation':'Recorded all-START_NEW collapse is produced by this trained model, not an action-index/loading mismatch. Its predictions on stable OFF trajectories can differ substantially from its self-induced states. This is one small-sequence/seed failure, not a universal MLP claim.','official_test_read':False}
    save('MLP_POLICY_SANITY.json',report);print(json.dumps(report))
if __name__=='__main__':main()
