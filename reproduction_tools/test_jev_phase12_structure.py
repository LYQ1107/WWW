"""Actual forward/gradient/behavior gates, independently of tracking labels."""
import dataclasses,inspect,ast,traceback
import torch
from jev_phase12_common import *
from gtr.modeling.visual_jev_mcmot import *
from gtr.modeling.visual_jev_mcmot.schemas import ActionType
from gtr.modeling.visual_jev_mcmot.native_action_adapter import lawful_match,transition_request
from gtr.modeling.jev_candidate_features import CandidateBatch

def fixture(device='cpu',k=7,q=3):
    state=OnlineVisualState(torch.randn(2,11,1152,device=device),torch.randn(2,11,8,device=device),torch.ones(2,11,device=device,dtype=torch.bool))
    types=torch.tensor([[i%3 for i in range(q)]]*2,device=device,dtype=torch.long)
    qmask=torch.ones(2,q,device=device,dtype=torch.bool)
    question=QuestionDescriptor(torch.randn(2,q,1152,device=device),torch.randn(2,q,64,device=device),types,qmask)
    kinds=torch.zeros((2,q,k),device=device,dtype=torch.long)
    for j in range(q):
        kinds[:,j]=0 if j%3==0 else 2 if j%3==1 else 4
        if k:kinds[:,j,-1]=1 if j%3==0 else 3 if j%3==1 else 5
    legal=torch.ones(2,q,k,device=device,dtype=torch.bool)
    for j in range(q):
        if j%3==2:
            legal[:,j]=False
            if k>=2:legal[:,j,0]=True;legal[:,j,-1]=True
            else:qmask[:,j]=False
    opt=OptionTensors(torch.randn(2,q,k,3,1152,device=device),torch.ones(2,q,k,3,device=device,dtype=torch.bool),torch.randn(2,q,k,12,device=device),kinds,legal)
    return state,question,opt

def main():
    protect();torch.set_num_threads(1);torch.manual_seed(20261009)
    report={'status':'RUNNING','binding':binding(),'tests':[]};out=OUT/os.environ.get('JEV_PHASE12_TEST_RUN','structure_v2');out.mkdir(exist_ok=True)
    def check(name,fn):
        try:details=fn();report['tests'].append({'name':name,'status':'PASS','details':details})
        except Exception:
            report['tests'].append({'name':name,'status':'FAIL','traceback':traceback.format_exc()});report['status']='FAIL';save(out/'FAILED.json',report);raise
        save(out/'PROGRESS.json',report);print('STRUCTURE_PASS',name,flush=True)
    model=VisualJev().eval()
    def shapes():
        records=[]
        for device in ['cpu','cuda:0']:
            net=model.to(device)
            for k in [0,1,2,7,41]:
                for q in [0,1,3,5]:
                    args=fixture(device,k,q);result=net(*args)
                    assert result['state_memory'].shape==(2,8,128)
                    assert result['question_state'].shape==(2,q,128)
                    assert result['action_representations'].shape==(2,q,k,128)
                    assert result['choice_logits'].shape==(2,q,k)
                    assert result['consequences'].shape==(2,q,k,3)
                    assert all(torch.isfinite(x).all() for x in result.values() if torch.is_tensor(x))
                    if k and q:assert torch.allclose(result['probabilities'].sum(-1),args[1].mask.float())
                    records.append([device,k,q])
        return {'cases':records,'params':parameters(model)}
    check('T1_CPU_CUDA_dynamic_shapes_empty_and_finite',shapes)
    model.cpu();args=fixture()
    def conditioning():
        state,q,opt=args
        q.context.requires_grad_();q.visual.requires_grad_();state.visual.requires_grad_();opt.visual.requires_grad_();opt.evidence.requires_grad_()
        invoked={};handles=[]
        for name in ['state_encoder','question_reader','option_encoder','option_reader']:
            def hook(module,inputs,result,n=name):invoked[n]=invoked.get(n,0)+1
            handles.append(getattr(model,name).register_forward_hook(hook))
        result=model(state,q,opt)
        loss=result['choice_logits'].square().sum()+result['consequences'].square().sum()
        loss.backward()
        gradient={name:float(x.grad.norm()) for name,x in [('dynamic_context',q.context),('current_visual',q.visual),('state_visual',state.visual),('option_visual',opt.visual),('option_evidence',opt.evidence)]}
        assert all(value>1e-8 for value in gradient.values())
        assert all(invoked.get(n,0)>0 for n in ['state_encoder','question_reader','option_encoder','option_reader'])
        changed=dataclasses.replace(q,context=q.context.detach()+torch.randn_like(q.context)*2)
        altered=model(state,changed,opt)
        delta=float((altered['question_state']-result['question_state']).abs().max());assert delta>1e-4
        for handle in handles:handle.remove()
        return {'invoked':invoked,'gradient_norms':gradient,'same_state_dynamic_context_max_difference':delta,'not_merely_different_static_types':True}
    check('T2_dynamic_question_and_real_gradient_paths',conditioning)
    def sensitivity():
        with torch.no_grad():
            original=model(*args)['choice_logits']
            changedq=dataclasses.replace(args[1],visual=-args[1].visual)
            changedo=dataclasses.replace(args[2],visual=-args[2].visual)
            a=float((model(args[0],changedq,args[2])['choice_logits']-original).abs().max())
            b=float((model(args[0],args[1],changedo)['choice_logits']-original).abs().max())
        assert a>1e-6 and b>1e-6
        return {'current_visual_logit_max_difference':a,'history_visual_logit_max_difference':b}
    check('T3_current_and_history_visual_used',sensitivity)
    def permutation():
        state,q,opt=args;p=torch.tensor([4,2,0,5,1,3,6])
        # Preserve terminal semantic reference under the same permutation.
        shuffled=OptionTensors(opt.visual[:,:,p],opt.visual_mask[:,:,p],opt.evidence[:,:,p],opt.kinds[:,:,p],opt.mask[:,:,p])
        with torch.no_grad():a=model(state,q,opt);b=model(state,q,shuffled)
        errors={name:float((a[name][:,:,p]-b[name]).abs().max()) for name in ['choice_logits','consequences','probabilities','action_representations']}
        assert all(value<2e-5 for value in errors.values())
        assert torch.equal(p[b['choice_logits'].argmax(-1)],a['choice_logits'].argmax(-1))
        return {'max_errors':errors,'semantic_selection_equivariant':True}
    check('T4_option_permutation_and_reference_selection',permutation)
    def semantics():
        results={action.name:transition_request(action) for action in [ActionType.ABSTAIN_OR_FALLBACK,ActionType.DEFER_TO_REACTIVATION,ActionType.START_NEW]}
        assert len({x['request'] for x in results.values()})==3
        assert results['ABSTAIN_OR_FALLBACK']['birth_increment']==0 and results['DEFER_TO_REACTIVATION']['birth_increment']==0
        return results
    check('T5_ABSTAIN_DEFER_NEW_distinct_requests',semantics)
    def capacity():
        s=torch.tensor([[4.,2.],[3.,1.],[4.,2.]]);legal=torch.ones(3,2,dtype=torch.bool)
        batch=CandidateBatch((17,31),s,torch.zeros(3,64),torch.zeros(3,2,12),legal,torch.ones(2),0)
        assigned,fallback=lawful_match(s,batch,torch.tensor([True,False,False]),views=torch.tensor([0,0,1]))
        assert len(set(assigned.existing_ids[:2]))==2 and assigned.existing_ids[2] in assigned.existing_ids[:2]
        assert fallback==(0,1)
        return {'assigned_refs':assigned.existing_ids,'whole_conflict_component_fallback':fallback,'cross_camera_reuse_legal':True}
    check('T6_lawful_assignment_and_group_fallback',capacity)
    def leakage():
        fields={cls.__name__:[f.name for f in dataclasses.fields(cls)] for cls in [OnlineVisualState,QuestionDescriptor,OptionTensors]}
        forbidden={'gt','labels','targets','anchors','evaluator','future','oracle','utility','track_id','identity_id'}
        assert not any(name in forbidden for names in fields.values() for name in names)
        rejected=False
        try:OnlineVisualState(**dict(dataclasses.asdict(args[0]),future=torch.ones(1)))
        except TypeError:rejected=True
        assert rejected
        runtime=ROOT/'gtr/modeling/visual_jev_mcmot'
        imports=[]
        for path in runtime.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
                elif isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        assert not any(any(word in name for word in ['offline','evaluator','anchors','phase10_learning','phase8_utility','phase7_offline']) for name in imports)
        return {'schemas':fields,'unknown_future_field_rejected':True,'runtime_imports':sorted(set(imports))}
    check('T7_runtime_schema_and_import_leakage_barrier',leakage)
    def masking():
        s,q,o=fixture();mask=o.mask.clone();mask[0,0]=False;mask[1,1,2:]=False;o=dataclasses.replace(o,mask=mask)
        result=model(s,q,o);assert torch.equal(result['probabilities'][~mask],torch.zeros_like(result['probabilities'][~mask]))
        assert result['probabilities'][0,0].sum()==0
        bad=dataclasses.replace(q,context=q.context.clone());bad.context[0,0,0]=float('nan')
        rejected=False
        try:model(s,bad,o)
        except ValueError:rejected=True
        assert rejected
        illegal=dataclasses.replace(o,kinds=o.kinds.clone());illegal.kinds[1,0,0]=3
        rejected=False
        try:model(s,q,illegal)
        except ValueError:rejected=True
        assert rejected
        return {'all_illegal_probabilities_zero':True,'all_masked_row_zero':True,'NaN_and_wrong_typed_action_rejected':True}
    check('T1_masks_nonfinite_wrong_action_rejection',masking)
    report['capacity']={variant:profile_macs(VisualJev(variant).eval(),fixture(k=16,q=3)) for variant in ['full','set_transformer','visual_deepsets','question_plain','no_question_reader','fixed_question','no_option_reader','no_gating','no_shared_state','no_history','no_H32','no_consequence','similarity']}
    report['status']='PASS';report['real_tracking_training_started']=False
    save(out/'RESULT.json',report);save(REPORTS/'STRUCTURAL_TESTS.json',report)
    save(REPORTS/'QUESTION_CONDITIONING_TESTS.json',{'status':'PASS_FORWARD_GRADIENT','tests':[r for r in report['tests'] if r['name'].startswith('T2')],'synthetic_learnability':'PENDING_AFTER_STRUCTURAL_PASS'})
    save(REPORTS/'OPTION_READER_TESTS.json',{'status':'PASS','tests':[r for r in report['tests'] if r['name'].startswith(('T2','T3','T4'))]})
    print('ALL_STRUCTURE_PASS',flush=True)

if __name__=='__main__':main()
