"""Check deployment interfaces against the frozen trained forward, not new fits."""
import torch
from jev_phase12_learning import *
from gtr.modeling.visual_jev_mcmot.baselines import LegacyNumericAdapter as RuntimeNumeric, NumericalOnlyVisualAdapter as RuntimeNoVisual

def main():
    torch.set_num_threads(1);rows,manifest=load_data();stats=normalization(rows);checks=[]
    for device in ['cpu','cuda:0']:
        args,_=collate(rows[:3],device)
        for name in ['CandidateMLP','CandidateDeepSets','CandidateJEV','numerical_only','full']:
            result=json.loads((OUT/'formal_training_v1'/name/'joint/seed20261008/RESULT.json').read_text())
            ck=result['checkpoints']['BEST'];assert sha(ck['path'])==ck['SHA256']
            checkpoint=torch.load(ck['path'],map_location=device)
            frozen=build_network(name,stats).to(device).eval();frozen.load_state_dict(checkpoint['model'])
            runtime=(RuntimeNumeric(name) if name.startswith('Candidate') else RuntimeNoVisual() if name=='numerical_only' else VisualJev(name)).to(device).eval()
            runtime.load_state_dict(checkpoint['model'])
            with torch.no_grad():
                a=frozen(*args);b=runtime(*args)
                errors={field:float((a[field]-b[field]).abs().max()) for field in ['choice_logits','consequences']}
                assert max(errors.values())<=1e-6,(name,errors)
                if name.startswith('Candidate'):
                    state,q,o=args;c=runtime.forward_numeric(q.context[:,0],o.evidence[:,0],o.mask[:,0])
                    assert torch.equal(c['choice_logits'],a['choice_logits'][:,0])
                    assert torch.equal(c['consequences'],a['consequences'][:,0])
                assert torch.equal(preference(a,'joint',name)/result['temperature'],preference(b,'joint',name)/result['temperature'])
            checks.append({'model':name,'device':device,'checkpoint_SHA256':ck['SHA256'],'max_errors':errors,'final_preference_temperature_matches':True})
    save(OUT/'deployment_contract_v1/RESULT.json',{'status':'PASS','binding':binding(),'checks':checks,'visual_dataset_SHA256':manifest['SHA256'],'scope':'same actual checkpoint, identical normalization/preference/temperature; training architecture and weights untouched'})
    print('FROZEN_CHECKPOINT_DEPLOYMENT_PASS',flush=True)
if __name__=='__main__':main()
