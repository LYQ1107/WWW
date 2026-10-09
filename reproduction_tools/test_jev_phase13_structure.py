"""Actual dynamic candidate/ID invariants and gradients, including empty pools."""
import torch
from jev_phase13_learning import *

def main():
    torch.set_num_threads(1);torch.manual_seed(20261009);rows,man=load_data([12]);r=next(r for r in rows if r['task']==0 and len(r['refs'])>=3 and r['supervised'].any());x,y=collate([r]);checks=[]
    for variant in VARIANTS:
        model=GlobalIdentityJev(variant).cuda();xx=transform_inputs(x,variant);z=model(xx);assert torch.isfinite(z).all()
        k=xx['history_visual'].shape[1];perm=torch.arange(k-1,-1,-1,device='cuda:0');xp={a:b.clone() for a,b in xx.items()}
        for a in ['history_visual','history_mask','identity_meta','identity_mask']:xp[a]=xp[a][:,perm]
        for a in ['pair_evidence','legal']:xp[a]=xp[a][:,:,perm]
        zz=model(xp);expected=torch.cat([z[:,:,perm],z[:,:,-1:]],-1);error=float((zz-expected).abs().max());assert error<3e-5,(variant,error)
        renamed=dict(r,refs=[10000+83*i for i in range(k)]);rx,ry=collate([renamed],variant=variant);assert all(torch.equal(xx[key],rx[key]) for key in xx);assert torch.equal(model(rx),z)
        q=xx['detection_visual'].shape[1];empty={a:b.clone() for a,b in xx.items()}
        for a in ['history_visual','history_mask','identity_meta','identity_mask']:empty[a]=empty[a][:,:0]
        for a in ['pair_evidence','legal']:empty[a]=empty[a][:,:,:0]
        ez=model(empty);assert ez.shape==(1,q,1) and torch.isfinite(ez).all();assert lawful_choice(ez[0],empty['legal'][0])==[-1]*q
        no_q={a:b.clone() for a,b in xx.items()}
        for a in ['detection_visual','detection_meta','question_mask','question_type']:no_q[a]=no_q[a][:,:0]
        for a in ['pair_evidence','legal']:no_q[a]=no_q[a][:,:0]
        assert model(no_q).shape==(1,0,k+1)
        loss,_=losses(z,xx,y);loss.backward();grad={a:float(p.grad.abs().sum()) if p.grad is not None else None for a,p in model.named_parameters()};assert all(g is None or np.isfinite(g) for g in grad.values())
        if variant=='full':
            for block in ['visual.','state_read.','question_evidence.','question_read.','option_question.','option_state.','gate.']:
                assert sum(v or 0 for key,v in grad.items() if key.startswith(block))>0,block
        bad=dict(xx,GT=torch.ones(1,device='cuda'))
        try:model(bad);raise AssertionError('GT key accepted')
        except ValueError:pass
        # Padding a completely masked identity cannot affect existing options.
        pad={a:b.clone() for a,b in xx.items()}
        for a in ['history_visual','history_mask','identity_meta','identity_mask']:pad[a]=torch.cat([pad[a],torch.zeros_like(pad[a][:,:1])],1)
        for a in ['pair_evidence','legal']:pad[a]=torch.cat([pad[a],torch.zeros_like(pad[a][:,:,:1])],2)
        zp=model(pad);paderr=float((torch.cat([zp[:,:,:k],zp[:,:,-1:]],-1)-z).abs().max());assert paderr<3e-5,(variant,paderr)
        checks.append({'variant':variant,'permutation_max_error':error,'masked_padding_max_error':paderr,'ID_integer_rename_exact':True,'empty_pool_private_terminal':True,'zero_questions':True,'GT_key_rejected':True,'capacity':profile(model,xx),'gradient_by_parameter':grad})
    # Tensor ablations preserve exact train/runtime transformations and remove
    # the original similarities as well as the visual vectors.
    for v in ['no_cross_camera','no_long_term']:
        a=transform_inputs(r['inputs'],v);b,_=collate([r],device='cpu',variant=v);assert all(torch.equal(a[t],b[t]) for t in a)
    result={'status':'PASS','binding':binding(),'dataset_SHA256':man[0]['DATASET']['SHA256'],'actual_record_key':r['key'],'checks':checks,'no_cross_and_no_long_train_runtime_tensor_transform_exact':True,'model_GT_inputs':False,'history_features_causal_by_native_builder':True}
    save(OUT/'structural_v1/RESULT.json',result);print('PHASE13_STRUCTURE_PASS',flush=True)
if __name__=='__main__':main()
