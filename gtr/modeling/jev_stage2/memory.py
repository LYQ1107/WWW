"""Only committed past observations enter global memory. IDs are opaque references."""
import copy
import torch
import torch.nn.functional as F

class IdentityMemory:
    def __init__(self):self.meta={}
    def update(self,ref,feature,box,image_size,frame,view):
        r=self.meta.setdefault(int(ref),{'views':{},'last_frame':frame,'hits':0})
        v=r['views'].setdefault(int(view),{'sum':torch.zeros_like(feature),'count':0,'box':None,'previous_box':None,'frame':frame,'previous_frame':frame})
        v['previous_box']=v['box'];v['previous_frame']=v['frame'];v['box']=box.detach().clone()/box.new_tensor([image_size[1],image_size[0],image_size[1],image_size[0]]);v['frame']=frame;v['sum']=v['sum']+feature.detach();v['count']+=1;r['last_frame']=frame;r['hits']+=1
    def state_dict(self):return copy.deepcopy(self.meta)
    def load_state_dict(self,state):self.meta=copy.deepcopy(state)
    def build(self,current,refs,galleries,frame,view,active,*,no_cross=False,no_long=False):
        device=current.reid_features.device;d=len(current);k=len(refs);hist=torch.zeros(k,4,1024,device=device);hm=torch.zeros(k,4,dtype=torch.bool,device=device);bm=torch.zeros(k,12,device=device);pair=torch.zeros(d,k,16,device=device)
        boxes=current.pred_boxes.tensor/current.pred_boxes.tensor.new_tensor([current.image_size[1],current.image_size[0],current.image_size[1],current.image_size[0]])
        dv=F.normalize(current.reid_features[:,:1024],dim=-1);dm=torch.cat([boxes,current.scores[:,None],boxes.new_full((d,1),view),boxes.new_full((d,1),min(frame/2000,10)),boxes.new_full((d,1),float(not active))],dim=1)
        for col,ref in enumerate(refs):
            g=galleries[ref];visual=g.reid_features[:,:1024];meta=self.meta.get(ref,{'views':{},'last_frame':frame,'hits':len(g)});own=meta['views'].get(view);other=meta['views'].get(1-view)
            vals=[visual[-1],visual.mean(0),own['sum']/own['count'] if own else None,other['sum']/other['count'] if other else None]
            for token,val in enumerate(vals):
                if val is not None:hist[col,token]=F.normalize(val,dim=-1);hm[col,token]=True
            age=max(0,frame-meta['last_frame']);bm[col]=bm.new_tensor([age/40,min(meta['hits'],2000)/2000,min(len(g),2000)/2000,float(ref in active),float(own is not None),float(other is not None),min(max(0,frame-own['frame']),2000)/2000 if own else 0,min(max(0,frame-other['frame']),2000)/2000 if other else 0,view,float(age>39),float(len(g)>=10),1])
            pair[:,col,:4]=dv@hist[col].T;pair[:,col,4]=age/40;pair[:,col,5]=float(own is not None);pair[:,col,6]=float(other is not None);pair[:,col,7]=bm[col,2];pair[:,col,8]=float(ref in active);pair[:,col,9]=current.scores
            if own:
                prior=own['box'];pred=prior
                if own['previous_box'] is not None and own['frame']>own['previous_frame']:
                    pred=prior+(prior-own['previous_box'])/max(1,own['frame']-own['previous_frame'])*min(max(0,frame-own['frame']),40)
                pair[:,col,10:14]=boxes-pred;pair[:,col,14]=max(0,frame-own['frame'])/40
            pair[:,col,15]=bm[col,1]
        x={'detection_visual':dv[None],'detection_meta':dm[None],'history_visual':hist[None],'history_mask':hm[None],'identity_meta':bm[None],'pair_evidence':pair[None],'legal':torch.ones(1,d,k,dtype=torch.bool,device=device),'question_mask':torch.ones(1,d,dtype=torch.bool,device=device),'identity_mask':torch.ones(1,k,dtype=torch.bool,device=device),'question_type':torch.zeros(1,d,dtype=torch.long,device=device)}
        return transform_inputs(x,'no_cross_camera' if no_cross else 'no_long_term' if no_long else 'full')

def transform_inputs(x,variant):
    """Exactly the same causal tensor transform in training and native inference.

    No-cross keeps the opaque shared namespace/legal candidates, but removes all
    other-view visual evidence and all global counts/timestamps. It uses only
    the own-view appearance summary (slot 2) and own-view geometry. Eligibility
    remains the frozen executor's rule; this ablation is not a new tracker.
    """
    if variant not in ['no_cross_camera','no_long_term']:return x
    x={k:v.clone() for k,v in x.items()}
    h=x['history_visual'];hm=x['history_mask'];bm=x['identity_meta'];p=x['pair_evidence']
    if variant=='no_long_term':h[:,:,1:]=0;hm[:,:,1:]=False
    else:
        own=h[:,:,2].clone();present=hm[:,:,2].clone()
        h[:]=0;hm[:]=False;h[:,:,:3]=own[:,:,None];hm[:,:,:3]=present[:,:,None]
        bm[:,:,0]=bm[:,:,6]*50;bm[:,:,[1,2,5,7,10]]=0;bm[:,:,9]=(bm[:,:,0]>39/40).float()
        p[:,:,:,4]=bm[:,None,:,0];p[:,:,:,[6,7,15]]=0
    p[:,:,:,:4]=torch.einsum('bqd,bkhd->bqkh',x['detection_visual'],h)
    return x
