"""Exact token batching optimization; no new inputs, weights or policy rules."""
import torch

def native_time_metadata(instances,frame,view,view_num=2,first=False,history_limit=80):
    if first:return {}
    vectors=[];times=[];start=0 if history_limit is None else max(0,len(instances)-history_limit)
    for index,inst in enumerate(instances[start:],start):
        time,camera=divmod(index,view_num)
        if time>frame or (time==frame and camera>=view) or not inst.has('track_ids'):continue
        vectors.append(inst.track_ids.detach());times.extend([(time,camera)]*len(inst))
    ids=torch.cat(vectors).cpu().tolist() if vectors else []
    return {int(track):stamp for track,stamp in zip(ids,times)}

def history_tokens(galleries,ids,reference):
    values=[];lengths=[];valid=[];zero=None
    for track in ids:
        gallery=galleries.get(int(track));length=0 if gallery is None else len(gallery);lengths.append(length);valid.append([bool(length)]*3)
        if length:
            gallery_values=gallery.reid_features.detach().to(reference.device)
            values.extend((gallery_values[-1],gallery_values[0],gallery_values.mean(0)))
        else:
            if zero is None:zero=reference.new_zeros(1152)
            values.extend((zero,zero,zero))
    visual=torch.stack(values).reshape(len(ids),3,1152) if values else reference.new_zeros((0,3,1152))
    return visual,torch.tensor(valid,dtype=torch.bool,device=reference.device).reshape(len(ids),3),lengths

def build_match_inputs(batch,observations,galleries,*,frame=0,view=0,metadata=None,question_type=0,include_terminal=True):
    # The mathematical construction below is the original frozen builder.
    # Only history stacking and CPU->CUDA metadata/ID transfers are batched.
    if observations.shape!=(len(batch.scores),1152):raise ValueError('current actual ReID dimension/count mismatch')
    from .schemas import OnlineVisualState,QuestionDescriptor,OptionTensors
    import math
    m,n=batch.scores.shape;hist,hmask,lengths=history_tokens(galleries,batch.candidate_ids,observations)
    raw=torch.cat((observations.detach(),hist.reshape(n*3,1152)),0)
    meta=[[1.,0.,0.,0.,0.,0.,float(view),1.] for _ in range(m)]
    for j,track in enumerate(batch.candidate_ids):
        for slot in range(3):
            item=[0.,float(slot==0),float(slot==1),float(slot==2),math.log1p(lengths[j]),0.,-1.,0.]
            if slot==0 and metadata and int(track) in metadata:
                time,camera=metadata[int(track)];item[5]=math.log1p(max(0,frame-time));item[6]=float(camera);item[7]=1.
            meta.append(item)
    token_meta=raw.new_tensor(meta).reshape(m+3*n,8)
    smask=torch.cat((torch.ones(m,device=raw.device,dtype=torch.bool),hmask.reshape(-1)))
    if not len(raw) or not smask.any():
        raw=observations.new_zeros((1,1152));token_meta=observations.new_zeros((1,8));smask=torch.ones(1,dtype=torch.bool,device=raw.device)
    k=n+int(include_terminal);ov=observations.new_zeros((1,m,k,3,1152));om=torch.zeros((1,m,k,3),device=raw.device,dtype=torch.bool);ev=observations.new_zeros((1,m,k,12));kinds=torch.full((1,m,k),0 if question_type==0 else 2,device=raw.device,dtype=torch.long);legal=torch.zeros((1,m,k),device=raw.device,dtype=torch.bool)
    if n:
        ov[0,:,:n]=hist[None].expand(m,-1,-1,-1);om[0,:,:n]=hmask[None].expand(m,-1,-1);ev[0,:,:n]=batch.evidence12;legal[0,:,:n]=batch.legal_mask & hmask.any(-1)[None]
    if include_terminal:
        ov[0,:,n,0]=observations.detach();om[0,:,n,0]=True;kinds[0,:,n]=1 if question_type==0 else 3;legal[0,:,n]=True
    state=OnlineVisualState(raw[None],token_meta[None],smask[None]);questions=QuestionDescriptor(observations.detach()[None],batch.state64[None],torch.full((1,m),question_type,device=raw.device,dtype=torch.long),torch.ones((1,m),device=raw.device,dtype=torch.bool))
    return state,questions,OptionTensors(ov,om,ev,kinds,legal)
