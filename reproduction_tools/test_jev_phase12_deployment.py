"""Check deployment interfaces against the frozen trained forward, not new fits."""
import torch
from jev_phase12_learning import *
from gtr.modeling.visual_jev_mcmot.baselines import LegacyNumericAdapter as RuntimeNumeric, NumericalOnlyVisualAdapter as RuntimeNoVisual
from gtr.modeling.visual_jev_mcmot.identity_history_tokens import build_match_inputs,native_time_metadata

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
    # Compare the actual all-question production tensor builder against each
    # stored single-question native training example, including competition
    # pooling. This catches an extra untrained terminal token at deployment.
    r=next(r for r in rows if r['partition']=='validation');key=tuple(r['key']);capture=json.loads((PREVIOUS/'native_capture_v1'/f'video{key[0]:02d}/compat/RESULT.json').read_text());entry=next(e for e in capture['prefixes'] if tuple(e['key'])==key)
    pkt=torch.load(entry['packet_path'],map_location='cuda:0')['packet']['batch'];prefix=torch.load(entry['path'],map_location='cuda:0');obs=prefix['instances'][key[1]*2+key[2]].reid_features
    args=build_match_inputs(pkt,obs,prefix['galleries'],frame=key[1],view=key[2],metadata=native_time_metadata(prefix['instances'],key[1],key[2],first=prefix['first']),include_terminal=False)
    individual=[x for x in rows if tuple(x['key'])==key];native=[]
    for name in ['full','set_transformer','visual_deepsets','question_plain','numerical_only']:
        result=json.loads((OUT/'formal_training_v1'/name/'joint/seed20261008/RESULT.json').read_text());saved=torch.load(result['checkpoints']['BEST']['path'],map_location='cuda:0');model=build_network(name,stats).cuda().eval();model.load_state_dict(saved['model'])
        with torch.no_grad():
            many=model(*args)
            for sample in individual:
                one_args,_=collate([sample],'cuda:0');one=model(*one_args);row=sample['row'];errors={f:float((many[f][0,row]-one[f][0,0]).abs().max()) for f in ['choice_logits','consequences']};assert max(errors.values())<2e-5,(name,row,errors)
                native.append({'model':name,'key':key,'row':row,'errors':errors})
    save(OUT/'deployment_contract_v2/RESULT.json',{'status':'PASS','binding':binding(),'checks':checks,'actual_native_multiquestion_vs_training_rows':native,'visual_dataset_SHA256':manifest['SHA256'],'scope':'same actual checkpoint, same real production builder and normalization/preference/temperature; no untrained DEFER terminal competition token; DEFER remains frozen lawful solver dummy; architecture/weights untouched'})
    print('FROZEN_CHECKPOINT_DEPLOYMENT_PASS',flush=True)
if __name__=='__main__':main()
