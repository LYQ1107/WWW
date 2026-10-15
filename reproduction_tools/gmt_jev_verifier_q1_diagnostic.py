"""Bounded, diagnostic-only fallback. No Q2 pseudo-labels or native decisions."""
import collections,torch,numpy as np
from gmt_jev_verifier_common import *
CLASSES=['RELIABLE','CONFLICTED','UNKNOWN']
def run_diagnostic(items,buffer_bytes):
    from gtr.modeling.gmt_jev.typed_verifier import HistoryReliabilityHead
    train=[s for s in items if s['video'] in [12,13] and not s['heldout']]
    valid=[s for s in items if s['video']==14]
    support={label:len({(s['video'],s['identity']) for s in train if s['label']==label}) for label in CLASSES}
    if min(support['RELIABLE'],support['CONFLICTED'])<4 or not valid:
        result=dict(status='Q1_DIAGNOSTIC_NOT_TRAINED_INSUFFICIENT_ID_SUPPORT',support=support,
                    Q2_updates=0,native_decisions_changed=0,scientific_success=False)
        save(REPORTS/'Q1_RELIABILITY_DIAGNOSTIC.json',result);return result
    torch.manual_seed(20261015);head=HistoryReliabilityHead().cpu()
    x=torch.stack([s['x'] for s in train]);y=torch.tensor([CLASSES.index(s['label']) for s in train])
    counts=torch.bincount(y,minlength=3);weight=(1/counts.clamp_min(1).float());weight/=weight.mean()
    optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.RandomState(20261015);logs=[]
    for step in range(1,301):
        indices=torch.tensor(rng.randint(len(train),size=min(64,len(train))))
        optim.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(head(x[indices]),y[indices],weight=weight)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        assert torch.isfinite(norm) and torch.isfinite(loss)
        optim.step()
        if step==1 or step%25==0:logs.append(dict(step=step,Q1_loss=float(loss.detach()),Q2_loss=None,Q2_labels=0,gradient_norm=float(norm)))
    with torch.no_grad():
        probabilities=head(torch.stack([s['x'] for s in valid])).softmax(-1);pred=probabilities.argmax(-1)
    confusion=[[0]*3 for _ in CLASSES];protect=[];proxy=[]
    for s,p,pr in zip(valid,pred.tolist(),probabilities.tolist()):
        truth=CLASSES.index(s['label']);confusion[truth][p]+=1
        if p==1 and pr[1]>=.9:protect.append(s)
        if float(s['x'][15])<.5 and float(s['x'][18])>0:proxy.append(s)
    precision=sum(s['label']=='CONFLICTED' for s in protect)/len(protect) if protect else None
    folder=OUT/'q1_diagnostic';folder.mkdir(exist_ok=False);path=folder/'q1_step300.pth'
    torch.save(dict(state_dict=head.state_dict(),classes=CLASSES,steps=300,seed=20261015),path)
    result=dict(status='COMPLETE_Q1_DIAGNOSTIC_ONLY',checkpoint=ref(path),steps=300,Q2_updates=0,
       support=support,training_rows=len(train),diagnostic_video14_rows=len(valid),confusion_true_by_pred=confusion,
       classes=CLASSES,high_confidence_conflicted_predictions=len(protect),
       high_confidence_conflicted_native_ID_groups=len({s['identity'] for s in protect}),
       high_confidence_conflicted_precision=precision,
       appearance_consistency_proxy=dict(predictions=len(proxy),conflicted_precision=sum(s['label']=='CONFLICTED' for s in proxy)/len(proxy) if proxy else None),
       parameters=sum(p.numel() for p in head.parameters()),logs=logs,buffer_accounted_bytes=buffer_bytes,
       native_decisions_changed=0,causal_identity_protection='NOT_ESTABLISHED',scientific_success=False,
       split='Q1 trains only TRAIN12/13 non-heldout matrices; video14 is Q1 diagnostic only, previously exposed to GMT/MLP research and not qualified independent validation.',
       unknown_semantics='UNKNOWN is a past-certificate abstention/insufficiency label, never a KEEP/APPLY pseudo-label; current-person and future labels are not model inputs.',
       label_dependency='Past certificates carry across original time buckets; earlier temporal-heldout labels may affect later training certificates. Not strict label-isolated temporal validation.',
       source=binding())
    save(REPORTS/'Q1_RELIABILITY_DIAGNOSTIC.json',result)
    print('Q1_DIAGNOSTIC_COMPLETE',len(train),len(valid),precision,flush=True);return result
