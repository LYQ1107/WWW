"""Fixed-budget historical-label controls; not Native Causal MiniSet training."""
import argparse,json,random
import numpy as np
import torch
from jev_phase7_common import *
from jev_phase7_policies import StrongMLP, DynamicThreshold
from jev_compact_dataset import CompactJEVData
from train_jev import load_policy_split
from train_jev_compact import batch_forward,evaluate

def train(name,device):
    protect();dest=OUT/'training'/name
    if dest.exists():raise RuntimeError('refusing reused training output')
    dest.mkdir(parents=True)
    random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003)
    source=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/match_training')
    data=CompactJEVData(source/'compact');splits=load_policy_split(source/'policy_split.json')
    train_ids=data.indices_for_sequences(splits['train']);val_ids=data.indices_for_sequences(splits['val'])
    assert len(train_ids)==1703 and len(val_ids)==2600
    model=(StrongMLP() if name=='MLP' else DynamicThreshold()).to(device)
    params=sum(p.numel()for p in model.parameters());assert abs(params/34080-1)<.01
    bind=dict(binding(),condition=name,parameters=params,train_records=len(train_ids),val_records=len(val_ids),
              data_manifest_sha256=sha(source/'compact/manifest.json'),split_sha256=sha(source/'policy_split.json'),
              epochs=20,batch_size=128,optimizer='AdamW',lr=.001,
              label_scope='HISTORICAL_B2_TARGETS_NOT_NATIVE_CAUSAL_TRUTH')
    save(dest/'binding.json',bind);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=np.random.default_rng(20261003);history=[]
    for epoch in range(20):
        model.train();order=train_ids.copy();rng.shuffle(order);losses=[]
        for start in range(0,len(order),128):
            loss,*_=batch_forward(model,data,order[start:start+128],torch.device(device))
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
        history.append({'epoch':epoch+1,'train_loss_mean_batch':float(np.mean(losses)),
                        'val':evaluate(model,data,val_ids,torch.device(device))})
    model.cpu().eval();payload={'model':model.state_dict(),'model_name':name,'binding':bind,'temperature':1.}
    torch.save(payload,dest/'model.pth')
    with torch.no_grad():
        _,output,_,weights=batch_forward(model,data,val_ids,torch.device('cpu'))
        logits=output['logits'].double();target=torch.tensor(np.asarray(data.target_probs[val_ids]),dtype=torch.float64)
        valid=output['legal_mask'];weights=weights.double()
    t=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64));optim=torch.optim.LBFGS([t],lr=.25,max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optim.zero_grad();scaled=(logits/t.exp().clamp(.05,20)).masked_fill(~valid,torch.finfo(torch.float64).min)
        loss=(-(target*torch.log_softmax(scaled,-1).masked_fill(~valid,0)).sum(-1)*weights).sum()/weights.sum()
        loss.backward();return loss
    optim.step(closure);payload['temperature']=float(t.detach().exp().clamp(.05,20))
    torch.save(payload,dest/'model_calibrated.pth')
    save(dest/'metrics.json',{'status':'COMPLETE','binding':bind,'history':history,'temperature':payload['temperature'],
                            'checkpoint_sha256':sha(dest/'model_calibrated.pth'),'calibration_split':'video06_only'})
    protect();print(json.dumps({'condition':name,'status':'COMPLETE','parameters':params}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('condition',choices=['MLP','DYNAMIC']);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();train(a.condition,a.device)
