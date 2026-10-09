"""Dense payload batches; offline labels never enter model.forward."""
import random,math,collections
import numpy as np
import torch
from jev_phase13_common import *
from gtr.modeling.jev_stage2.model import GlobalIdentityJev
from gtr.modeling.jev_stage2.memory import transform_inputs
from gtr.modeling.jev_stage2.assignment import lawful_choice,structured_assignment_loss

VARIANTS=['full','motip','camel','set_transformer','no_question_reader','fixed_question','no_option_reader','no_gating','no_cross_camera','no_long_term','no_competition','no_typed_head']
SEEDS=[20261008,20261009,20261010]

def load_data(videos,version='dense_native_v3',verify=True):
    records=[];manifests=[]
    for v in videos:
        allowed(v);p=OUT/version/f'video{v:02d}';m=json.loads((p/'RESULT.json').read_text());assert m['status']=='COMPLETE'
        if verify:assert sha(m['DATASET']['path'])==m['DATASET']['SHA256']
        records.extend(torch.load(m['DATASET']['path'],map_location='cpu')['records']);manifests.append(m)
    return records,manifests

def collate(records,device='cuda:0',variant='full'):
    b=len(records);q=max(len(r['rows']) for r in records);k=max(len(r['refs']) for r in records)
    prototype=records[0]['inputs'];shapes={'detection_visual':(b,q,1024),'detection_meta':(b,q,8),'history_visual':(b,k,4,1024),'history_mask':(b,k,4),'identity_meta':(b,k,12),'pair_evidence':(b,q,k,16),'legal':(b,q,k),'question_mask':(b,q),'identity_mask':(b,k),'question_type':(b,q)}
    x={key:torch.zeros(shape,dtype=prototype[key].dtype) for key,shape in shapes.items()}
    positive=torch.zeros(b,q,k+1,dtype=torch.bool);known=positive.clone();targets=torch.full((b,q),k,dtype=torch.long);supervised=torch.zeros(b,q,dtype=torch.bool)
    for i,r in enumerate(records):
        qq=len(r['rows']);kk=len(r['refs']);a=r['inputs']
        for key in ['detection_visual','detection_meta','question_mask','question_type']:x[key][i,:qq]=a[key][0]
        for key in ['history_visual','history_mask','identity_meta','identity_mask']:x[key][i,:kk]=a[key][0]
        for key in ['pair_evidence','legal']:x[key][i,:qq,:kk]=a[key][0]
        positive[i,:qq,:kk]=r['positive'][:,:kk];positive[i,:qq,k]=r['positive'][:,kk]
        known[i,:qq,:kk]=r['known_options'][:,:kk];known[i,:qq,k]=r['known_options'][:,kk]
        targets[i,:qq]=torch.where(r['targets']==kk,k,r['targets']);supervised[i,:qq]=r['supervised']
    x=transform_inputs({key:value.to(device) for key,value in x.items()},variant)
    y={'positive':positive.to(device),'known_options':known.to(device),'targets':targets.to(device),'supervised':supervised.to(device)}
    return x,y

def losses(logits,x,y):
    valid=y['supervised'] & x['question_mask'];known=y['known_options'];positive=y['positive']
    masked=logits.masked_fill(~known,-1e4);lp=torch.log_softmax(masked,-1);mass=torch.logsumexp(lp.masked_fill(~positive,-1e4),-1)
    ce=(-mass[valid]).mean() if valid.any() else logits.sum()*0
    target=positive.float()/positive.sum(-1,keepdim=True).clamp_min(1);p=lp.exp();brier=(((p-target)**2*known).sum(-1)[valid]).mean() if valid.any() else logits.sum()*0
    structured=structured_assignment_loss(logits,x['legal'] & known[:,:,:-1],y['targets'],valid)
    return ce+.25*structured+.1*brier,{'choice':float(ce.detach()),'assignment':float(structured.detach()),'brier':float(brier.detach()),'known_rows':int(valid.sum())}

def cosine_logits(x):
    values=x['pair_evidence'][:,:,:,[0,1,2]].max(-1).values
    return torch.cat([values,values.new_full((*values.shape[:2],1),.75)],-1)

@torch.no_grad()
def evaluate(model,records,variant='full',temperature=1.,batch_size=4,device='cuda:0'):
    if model is not None:model.eval()
    count=collections.Counter();nll=[];brier=[];confidence=[];correct=[];risks=[]
    for start in range(0,len(records),batch_size):
        part=records[start:start+batch_size];x,y=collate(part,device,variant);z=(model(x) if model is not None else cosine_logits(x))/temperature
        known=y['known_options'];valid=y['supervised'] & x['question_mask'];lp=torch.log_softmax(z.masked_fill(~known,-1e4),-1);p=lp.exp();pos=y['positive']
        nll.extend((-torch.logsumexp(lp.masked_fill(~pos,-1e4),-1)[valid]).cpu().tolist());target=pos.float()/pos.sum(-1,keepdim=True).clamp_min(1)
        brier.extend((((p-target)**2*known).sum(-1)[valid]).cpu().tolist())
        # Reliability is conditional on certified options; full action accuracy
        # below additionally counts selections of UNKNOWN options explicitly.
        cond_conf,cond_choice=p.max(-1);cond_correct=pos.gather(-1,cond_choice[...,None]).squeeze(-1)
        confidence.extend(cond_conf[valid].cpu().tolist());correct.extend(cond_correct[valid].cpu().tolist())
        for i,r in enumerate(part):
            qq=len(r['rows']);kk=len(r['refs']);local=torch.cat([z[i,:qq,:kk],z[i,:qq,-1:]],-1);legal=x['legal'][i,:qq,:kk]
            choices=lawful_choice(local,legal)
            for row,col in enumerate(choices):
                col=kk if col<0 else col;count['rows']+=1
                if not bool(r['supervised'][row]):count['unlabelled_rows']+=1;continue
                count['certified_rows']+=1
                if not bool(r['known_options'][row,col]):count['UNKNOWN_selected']+=1
                elif bool(r['positive'][row,col]):count['certified_correct']+=1
                else:count['certified_wrong']+=1
    ece=0.
    if confidence:
        a=np.asarray(confidence);c=np.asarray(correct);bins=[]
        for lo in np.linspace(0,1,11)[:-1]:
            m=(a>=lo)&(a<lo+.100000001)
            if m.any():ece+=float(m.mean()*abs(a[m].mean()-c[m].mean()));bins.append({'lo':float(lo),'count':int(m.sum()),'confidence':float(a[m].mean()),'accuracy':float(c[m].mean())})
        order=np.argsort(-a,kind='stable');risks=[{'coverage':f,'risk':float(1-c[order[:max(1,math.ceil(f*len(c)))]].mean())} for f in [.1,.25,.5,.75,1.]]
    else:bins=[];ece=None
    return {'counts':dict(count),'known_NLL':float(np.mean(nll)) if nll else None,'known_Brier':float(np.mean(brier)) if brier else None,'conditional_ECE':ece,'reliability_bins':bins,'conditional_risk_coverage':risks,'certified_accuracy':count['certified_correct']/count['certified_rows'] if count['certified_rows'] else None,'scope':'NLL/Brier/ECE conditional on certified options only; ambiguous/unanchored options are UNKNOWN, never false negatives; action accuracy includes UNKNOWN selections in denominator'}

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def profile(model,x):
    """Executed Linear/MHA matmuls only; never count dormant heads as MACs."""
    totals=collections.Counter();handles=[]
    def linear(m,args,out):totals['linear_MAC']+=int(out.numel()*m.in_features)
    def attention(m,args,out):
        q,k=args[:2];b,n,d=q.shape;s=k.shape[1];totals['attention_MAC']+=int(b*((n+2*s)*d*d+n*d*d+2*n*s*d))
    for m in model.modules():
        if isinstance(m,torch.nn.MultiheadAttention):handles.append(m.register_forward_hook(attention))
        elif isinstance(m,torch.nn.Linear) and not isinstance(m,torch.nn.modules.linear.NonDynamicallyQuantizableLinear):handles.append(m.register_forward_hook(linear))
    with torch.no_grad():model(x)
    for h in handles:h.remove()
    mac=sum(totals.values());return {'registered_parameters':sum(p.numel() for p in model.parameters()),'executed_MAC':mac,'matmul_FLOPs':2*mac,'breakdown':dict(totals),'input_shapes':{k:list(v.shape) for k,v in x.items()},'scope':'real executed matrix multiply count; nonlinear/normalization/softmax/Hungarian excluded; dormant MEMORY and REACT heads disclosed separately'}
