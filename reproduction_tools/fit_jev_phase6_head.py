"""Small standalone typed-head diagnostic; never a unified lifecycle model."""
import argparse
from collections import Counter
import json
import random

import numpy as np
import torch

from jev_phase6_common import OUT,REPORTS,B2,sha,save,new_output,protect_anchor
from gtr.modeling.jev_lifecycle import TypedJEVController,pack_typed_state
from jev_phase6_label_contract import censor_unknown_target


def main(kind,device,native_contract=False):
    protect_anchor()
    if not native_contract:raise RuntimeError('legacy prefixes cannot train a canonical head; --native-contract required')
    question='MEMORY_DECISION' if kind=='MEMORY' else 'REACTIVATION_DECISION'
    source={v:sorted((OUT/f'lifecycle_native/video{v:02d}/labels').glob(f'{kind}_*.json')) for v in (24,23)}
    for v in (24,23):
        manifest=json.loads((OUT/f'lifecycle_native/video{v:02d}/CANONICAL_MANIFEST.json').read_text())
        if not manifest['native_MATCH_transition_contract']:raise AssertionError('canonical native prefix required')
    rows={}
    for v,paths in source.items():
        rows[v]=[]
        for path in paths:
            row=json.loads(path.read_text())
            if censor_unknown_target(row):save(path,row)
            rows[v].append(row)
    if len(rows[24])!=(10 if kind=='MEMORY' else 32) or len(rows[23])!=(10 if kind=='MEMORY' else 16):
        raise RuntimeError('canonical bounded dataset incomplete; no fit on partial outcomes')
    counts={v:sum(float(r['sample_weight'])>0 for r in records) for v,records in rows.items()}
    output=new_output(OUT/'standalone_heads_native'/kind)
    eligibility={'status':'ELIGIBLE' if counts[24]>=4 and counts[23]>=2 else 'INSUFFICIENT_INFORMATIVE_DATA',
                 'kind':kind,'informative':counts,'minimum_train':4,'minimum_validation':2,
                 'role':'bounded standalone diagnostic, not unified JEV training or generalization evidence',
                 'source_sha256':{str(p):sha(p) for paths in source.values() for p in paths}}
    if eligibility['status']!='ELIGIBLE':
        save(output/'result.json',eligibility);print(json.dumps(eligibility));return
    random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003)
    model=TypedJEVController().to(device)
    anchor=torch.load(B2,map_location='cpu')
    model.core.load_state_dict(anchor['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def tensors(records):
        if any(r['question']!=question or r['GT_or_future_inputs'] for r in records):raise AssertionError('canonical online feature contract failed')
        x=torch.stack([pack_typed_state(r['feature_vector']) for r in records]).to(device)
        y=torch.tensor([r['target_probs'] for r in records],device=device)
        w=torch.tensor([r['sample_weight'] for r in records],device=device)
        return x,y,w,[r['legal_actions'] for r in records]
    tx,ty,tw,ta=tensors(rows[24]);vx,vy,vw,va=tensors(rows[23])
    def evaluate():
        model.eval()
        with torch.no_grad():
            probs=model(vx,[question]*len(vx),va)['probs'];chosen=probs.argmax(-1)
            correct=torch.tensor([va[i][int(c)] in rows[23][i]['best_actions'] for i,c in enumerate(chosen)],device=device)
            nll=(-(vy*probs.clamp_min(1e-8).log()).sum(-1)*vw).sum()/vw.sum()
            return {'weighted_nll':float(nll),'positive_weight_accuracy':float((correct*vw).sum()/vw.sum())}
    history=[];rng=np.random.default_rng(20261003)
    for epoch in range(20):
        model.train();order=rng.permutation(len(tx));total=0
        for start in range(0,len(order),128):
            ids=order[start:start+128];legal=[ta[i] for i in ids]
            probs=model(tx[ids],[question]*len(ids),legal)['probs'];loss=(-(ty[ids]*probs.clamp_min(1e-8).log()).sum(-1)*tw[ids]).sum()/tw[ids].sum().clamp_min(1e-8)
            optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss)
        history.append({'epoch':epoch+1,'loss':total,'validation':evaluate()})
    # Fixed last epoch. Calibration never changes the selected semantic action.
    model.eval()
    with torch.no_grad():raw=model(vx,[question]*len(vx),va)['probs'].detach().cpu().double()
    target=vy.detach().cpu().double();weights=vw.detach().cpu().double();logits=raw.clamp_min(1e-8).log()
    temperature_log=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    calibration=torch.optim.LBFGS([temperature_log],lr=.25,max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        calibration.zero_grad();temp=temperature_log.exp().clamp(.05,20)
        loss=(-(target*torch.log_softmax(logits/temp,-1)).sum(-1)*weights).sum()/weights.sum()
        loss.backward();return loss
    calibration.step(closure);temperature=float(temperature_log.detach().exp().clamp(.05,20))
    payload={'model_name':'phase6_typed_head','model':model.cpu().state_dict(),'kind':kind,'seed':20261003,
             'epochs':20,'calibration_temperature':temperature,'permanent_B2_sha256':sha(B2),'binding':eligibility}
    torch.save(payload,output/'model_calibrated.pth')
    report={'status':'COMPLETE','kind':kind,'checkpoint_sha256':sha(output/'model_calibrated.pth'),
            'history':history,'temperature_validation_only':temperature,'eligibility':eligibility,
            'best_action_counts':{str(v):dict(Counter(a for r in records if r['sample_weight']>0 for a in r['best_actions'])) for v,records in rows.items()},
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'runtime_MATCH':'permanent frozen B2, separate unchanged controller; this typed core handles the selected diagnostic question only',
            'checkpoint_selection':'fixed last20epoch, train24/val23 only; no video01 or heldout tuning',
            'unified_training':False,'official_test_read':False}
    save(output/'result.json',report);protect_anchor();print(json.dumps({'kind':kind,'status':'COMPLETE','informative':counts}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['MEMORY','REACT'],required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--native-contract',action='store_true')
    a=p.parse_args();main(a.kind,a.device,a.native_contract)
