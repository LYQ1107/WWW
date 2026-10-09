"""Bounded real native visual observations; no synthetic Gallery or stale IDs."""
import math
import torch
from .schemas import OnlineVisualState, QuestionDescriptor, OptionTensors

def history_tokens(galleries, ids, reference):
    visual=reference.new_zeros((len(ids),3,1152))
    mask=torch.zeros((len(ids),3),device=reference.device,dtype=torch.bool)
    lengths=[]
    for j,track in enumerate(ids):
        gallery=galleries.get(int(track))
        length=0 if gallery is None else len(gallery)
        lengths.append(length)
        if length:
            values=gallery.reid_features.detach().to(reference.device)
            visual[j]=torch.stack((values[-1],values[0],values.mean(0)))
            mask[j]=True
    return visual,mask,lengths

def native_time_metadata(instances,frame,view,view_num=2,first=False,history_limit=80):
    """Use actual ordered native history; missing old metadata stays unknown."""
    last={}
    if first:
        return last
    # At prefix instances includes both current cameras; rows without committed
    # IDs are excluded. History index is actual native frame*view_num+camera.
    start=0 if history_limit is None else max(0,len(instances)-history_limit)
    for index,inst in enumerate(instances[start:],start):
        time,camera=divmod(index,view_num)
        if time>frame or (time==frame and camera>=view) or not inst.has('track_ids'):
            continue
        for track in inst.track_ids.detach().cpu().tolist():
            last[int(track)]=(time,camera)
    return last

def build_match_inputs(batch,observations,galleries,*,frame=0,view=0,metadata=None,question_type=0,include_terminal=True):
    """Same builder used by real native actor and offline prefix extraction."""
    if observations.shape != (len(batch.scores),1152):
        raise ValueError('current actual ReID dimension/count mismatch')
    m,n=batch.scores.shape
    hist,hmask,lengths=history_tokens(galleries,batch.candidate_ids,observations)
    raw=torch.cat((observations.detach(),hist.reshape(n*3,1152)),0)
    token_meta=raw.new_zeros((m+3*n,8))
    token_meta[:m,0]=1;token_meta[:m,6]=float(view);token_meta[:m,7]=1
    for j,track in enumerate(batch.candidate_ids):
        sl=slice(m+3*j,m+3*j+3)
        token_meta[sl,1:4]=torch.eye(3,device=raw.device)
        token_meta[sl,4]=math.log1p(lengths[j])
        token_meta[sl,6]=-1
        if metadata and int(track) in metadata:
            time,camera=metadata[int(track)]
            token_meta[m+3*j,5]=math.log1p(max(0,frame-time))
            token_meta[m+3*j,6]=float(camera);token_meta[m+3*j,7]=1
    smask=torch.cat((torch.ones(m,device=raw.device,dtype=torch.bool),hmask.reshape(-1)))
    if not len(raw) or not smask.any():
        raw=observations.new_zeros((1,1152));token_meta=observations.new_zeros((1,8));smask=torch.ones(1,dtype=torch.bool,device=raw.device)
    k=n+int(include_terminal)
    ov=observations.new_zeros((1,m,k,3,1152))
    om=torch.zeros((1,m,k,3),device=raw.device,dtype=torch.bool)
    ev=observations.new_zeros((1,m,k,12))
    kinds=torch.full((1,m,k),0 if question_type==0 else 2,device=raw.device,dtype=torch.long)
    legal=torch.zeros((1,m,k),device=raw.device,dtype=torch.bool)
    if n:
        ov[0,:,:n]=hist[None].expand(m,-1,-1,-1)
        om[0,:,:n]=hmask[None].expand(m,-1,-1)
        ev[0,:,:n]=batch.evidence12
        # Native identity without gallery is not replaced by fabricated visual.
        legal[0,:,:n]=batch.legal_mask & hmask.any(-1)[None]
    if include_terminal:
        ov[0,:,n,0]=observations.detach();om[0,:,n,0]=True
        kinds[0,:,n]=1 if question_type==0 else 3;legal[0,:,n]=True
    state=OnlineVisualState(raw[None],token_meta[None],smask[None])
    questions=QuestionDescriptor(observations.detach()[None],batch.state64[None],torch.full((1,m),question_type,device=raw.device,dtype=torch.long),torch.ones((1,m),device=raw.device,dtype=torch.bool))
    return state,questions,OptionTensors(ov,om,ev,kinds,legal)

def build_memory_inputs(observation,gallery,context,view):
    from types import SimpleNamespace
    # Ref stays Python bookkeeping only; correct committed gallery supplied.
    batch=SimpleNamespace(scores=observation.new_zeros((1,1)),candidate_ids=(0,),state64=context.reshape(1,64),evidence12=observation.new_zeros((1,1,12)),legal_mask=torch.ones((1,1),device=observation.device,dtype=torch.bool))
    state,q,_=build_match_inputs(batch,observation.reshape(1,1152),{0:gallery},view=view,include_terminal=False)
    hist,mask,_=history_tokens({0:gallery},(0,),observation)
    evidence=observation.new_zeros((1,1,2,12))
    if len(gallery):
        consistency=torch.nn.functional.cosine_similarity(observation.reshape(1,-1),hist[0])
        evidence[0,0,:,0:3]=consistency
        evidence[0,0,:,5]=len(gallery)
    evidence[0,0,0,9]=1 # actual write action descriptor, not supervision
    visual=hist[None,None].expand(1,1,2,3,1152).clone()
    # Include actual new observation for both actions, never omit current data.
    visual[0,0,:,0]=observation
    masks=mask[None,None].expand(1,1,2,3).clone();masks[...,0]=True
    q=QuestionDescriptor(q.visual,q.context,torch.full_like(q.types,2),q.mask)
    options=OptionTensors(visual,masks,evidence,torch.tensor([[[4,5]]],device=observation.device),torch.ones((1,1,2),device=observation.device,dtype=torch.bool))
    return state,q,options
