"""Extend frozen v3 with real pre-association visual evidence and executed Q."""
from collections import defaultdict
import torch
from jev_phase12_common import *
from gtr.modeling.visual_jev_mcmot.identity_history_tokens import build_match_inputs,native_time_metadata
from gtr.modeling.visual_jev_mcmot.schemas import OnlineVisualState,QuestionDescriptor,OptionTensors

def main():
    protect();source=binding();assert not source['dirty']
    old=torch.load(PREVIOUS/'dataset_v3/DATASET.pth',map_location='cpu');rows=old['rows']
    out=OUT/'visual_dataset_v1';out.mkdir(exist_ok=True)
    if (out/'MANIFEST.json').exists():
        manifest=json.loads((out/'MANIFEST.json').read_text());assert manifest['binding']==source and sha(manifest['path'])==manifest['SHA256'];print('VISUAL_DATA_ALREADY_DONE');return
    registries={}
    for video in TRAIN+VAL:
        d=json.loads((PREVIOUS/'native_capture_v1'/f'video{video:02d}'/'compat/RESULT.json').read_text());registries[video]={tuple(r['key']):r for r in d['prefixes']}
    causal={}
    for e in json.loads((ROOT/'reports/JEV_PHASE10/H32_CAUSAL_PARITY.json').read_text())['raw_event_manifests']:
        assert sha(e['path'])==e['SHA256'];data=json.loads(Path(e['path']).read_text());causal[tuple(data['key'])+(data['row'],)]=data
    groups=defaultdict(list)
    for index,row in enumerate(rows):groups[tuple(row['key'])].append((index,row))
    extended=[None]*len(rows);provenance=[]
    for counter,(key,selected) in enumerate(sorted(groups.items())):
        r=registries[key[0]][key];assert sha(r['path'])==r['sha256'] and sha(r['packet_path'])==r['packet_sha256']
        prefix=torch.load(r['path'],map_location='cpu');packet=torch.load(r['packet_path'],map_location='cpu')['packet'];batch=packet['batch']
        obs=prefix['instances'][key[1]*2+key[2]].reid_features
        assert obs.shape[0]==len(batch.scores)
        meta=native_time_metadata(prefix['instances'],key[1],key[2],first=prefix['first'])
        state,questions,options=build_match_inputs(batch,obs,prefix['galleries'],frame=key[1],view=key[2],metadata=meta,include_terminal=False)
        assert torch.equal(options.mask[0],batch.legal_mask),'missing visual candidate support cannot be hidden'
        # Clone row views so saved tensors do not retain all questions' storage.
        shared=OnlineVisualState(state.visual.clone(),state.metadata.clone(),state.mask.clone())
        for index,row in selected:
            rr=row['row'];assert tuple(row['candidate_ids'])==batch.candidate_ids
            assert torch.equal(row['state64'],batch.state64[rr]) and torch.equal(row['evidence12'],batch.evidence12[rr])
            query=QuestionDescriptor(questions.visual[:,rr:rr+1].clone(),questions.context[:,rr:rr+1].clone(),questions.types[:,rr:rr+1].clone(),questions.mask[:,rr:rr+1].clone())
            opts=OptionTensors(options.visual[:,rr:rr+1].clone(),options.visual_mask[:,rr:rr+1].clone(),options.evidence[:,rr:rr+1].clone(),options.kinds[:,rr:rr+1].clone(),options.mask[:,rr:rr+1].clone())
            q=torch.full((len(row['candidate_ids']),3),float('nan'));qm=torch.zeros_like(q,dtype=torch.bool)
            event=causal.get(key+(rr,))
            if event:
                offers=defaultdict(list)
                for executed in row['executed_candidate_utilities']:
                    if not executed['single_candidate_ranking_eligible']:continue
                    tag=executed['tag'];branch=event['branches'][tag]
                    assert branch['native_trace_SHA256']==executed['native_trace_SHA256']
                    ref=executed['existing_candidate_reference'];col=row['candidate_ids'].index(ref)
                    offers[col].append([branch['horizons'][str(h)]['delta_utility'] for h in [8,16,32]])
                for col,values in offers.items():
                    x=torch.tensor(values)
                    for h in range(3):
                        if float(x[:,h].max()-x[:,h].min())<=1e-6:q[col,h]=x[0,h];qm[col,h]=True
            assert torch.equal(qm[:,2],row['utility_mask'])
            assert torch.equal(q[qm[:,2],2],row['utility'][row['utility_mask']])
            assert not torch.isfinite(q[~qm]).any()
            extended[index]=dict(row,visual_state=shared,visual_question=query,visual_options=opts,consequence_targets=q,consequence_mask=qm)
        provenance.append({'key':list(key),'prefix_SHA256':r['sha256'],'packet_SHA256':r['packet_sha256'],'rows':len(selected),'visual_tokens':int(state.mask.sum()),'legal_options':len(batch.candidate_ids)})
        save(out/'PROGRESS.json',{'status':'RUNNING','prefixes':counter+1,'total_prefixes':len(groups),'rows':sum(p['rows'] for p in provenance),'last_key':list(key)})
        print('VISUAL_PREFIX_EXTRACTED',counter+1,len(groups),key,flush=True)
    path=out/'DATASET.pth';assert not path.exists();temp=path.with_suffix('.pth.tmp')
    torch.save({'version':'PhaseXII_visual_v1','rows':extended,'prior_dataset_SHA256':sha(PREVIOUS/'dataset_v3/DATASET.pth'),'provenance':provenance},temp);temp.replace(path)
    manifest={'status':'PASS','binding':source,'path':str(path),'SHA256':sha(path),'bytes':path.stat().st_size,'rows':len(extended),'partitions':{p:sum(r['partition']==p for r in extended) for p in ['train','validation']},'visual_dimension':1152,'all_candidates_and_labels_unchanged':True,'no_future_runtime_features':True,'consequence_utility':'executed native branches only; independent horizon ambiguity masked; H32 exact v3','provenance':provenance,'not_uploaded_to_git':True}
    save(out/'MANIFEST.json',manifest);print('VISUAL_DATA_COMPLETE',len(extended),path.stat().st_size,flush=True)

if __name__=='__main__':main()
