"""Bounded R2 diagnostic fuser learning, separate IDs/weights/native namespace."""
import argparse, collections, random, torch
from jev_recovery_common import *
from jev_recovery_learning import load_records, load_model, batch, loss, conflict_groups, question_conflict_groups
from gtr.modeling.jev_recovery.diagnostic_fusion import DiagnosticFusion
from jev_recovery_complete_input_assessment import assess

KINDS = ['explicit_scores','independent_readers','mlp','attention']


def build(kind, phase=None):
    backbone, source, _ = load_model('jev',20261009,'pilot',2,'XIV')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261009)
        model = DiagnosticFusion(backbone,kind).cuda()
    if phase:
        descriptor = read(OUT/'diagnostic_fusion_v2'/kind/phase/'RESULT.json')['checkpoint']
        assert sha(descriptor['path']) == descriptor['SHA256']
        saved = torch.load(descriptor['path'],map_location='cuda:0')
        assert saved['backbone_checkpoint'] == source
        missing,unexpected = model.load_state_dict(saved['new_fuser'],strict=False)
        assert not unexpected and all(n.startswith('backbone.') for n in missing)
        return model.eval(),descriptor
    return model,source


def train(kind, phase):
    protect(); torch.set_num_threads(1)
    protocol = read(REPORTS/'DIAGNOSTIC_FUSION_PREREGISTRATION_V2.json')
    target = protocol['updates'][phase]
    folder = OUT/'diagnostic_fusion_v2'/kind/phase; folder.mkdir(parents=True,exist_ok=False)
    records,validation,sources = load_records(2)
    groups = conflict_groups(records,2); semantic = question_conflict_groups(records)
    keys = sorted(groups); semantic_keys = sorted(semantic)
    ordinary = collections.defaultdict(list)
    for r in records: ordinary[r['key'][0],r['key'][1]//64].append(r)
    blocks = sorted(ordinary); rng = random.Random(20261009)
    fixtures = [groups[next(k for k in keys if k[0]==v)][0] for v in [12,13,14]]
    fixtures.append(groups[keys[0]][0])
    weights = {}
    for n in ['cross','stability']:
        pos = sum(int((r['oneweek_labels'][n]==1).sum()) for r in records)
        neg = sum(int((r['oneweek_labels'][n]==0).sum()) for r in records)
        weights[n] = neg/max(1,pos)
    pos = sum(int(r['positive'][:,:-1].sum()) for r in records)
    neg = sum(int((r['known_options'][:,:-1]&~r['positive'][:,:-1]).sum()) for r in records)
    weights['compatibility'] = neg/max(1,pos)
    model,source = build(kind); parameters = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in parameters); assert 81000 <= count <= 83000
    optimizer = torch.optim.AdamW(parameters,lr=.0001,weight_decay=.01)
    x,y = batch(fixtures)
    with torch.no_grad():
        old = model.backbone.details(x); initial = model.details(x)
        assert torch.equal(old['edge_logits'],initial['edge_logits'])
        assert old['plans']==initial['plans']
        assert all(torch.equal(a,b) for a,b in zip(old['plan_scores'],initial['plan_scores']))
        fixture_before = float(loss(initial,x,y,fixtures,weights)[0])
    before = assess(model,validation,128)
    save(folder/'START.json',dict(status='FROZEN_MODEL_AND_BATCHES',protocol=ref(REPORTS/'DIAGNOSTIC_FUSION_PREREGISTRATION_V2.json'),
        backbone_checkpoint=source,new_parameters=count,frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        exact_initial_plan_scores_and_edges=True,TRAIN16_gradient=False,DEV_gradient=False,sources=sources))
    digest = hashlib.sha256(); rank_batches=0; history=[]; began=time.monotonic()
    for step in range(1,target+1):
        if phase=='tiny': part=fixtures
        else:
            part=[]
            for slot in range(4):
                if slot==1 and semantic_keys:
                    key=rng.choice(semantic_keys); part.append(rng.choice(semantic[key]))
                elif slot<2:
                    key=rng.choice(keys); part.append(rng.choice(groups[key]))
                else:
                    video=rng.choice([12,13,14]); key=rng.choice([k for k in blocks if k[0]==video])
                    part.append(rng.choice(ordinary[key]))
        assert all(r['key'][0] in [12,13,14] and not r['audit_block'] for r in part)
        digest.update(json.dumps([r['key'] for r in part]).encode()); x,y=batch(part)
        optimizer.zero_grad(set_to_none=True); model.train()
        d=model.details(x); objective,components,pairs=loss(d,x,y,part,weights)
        assert torch.isfinite(objective); objective.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.); assert torch.isfinite(norm)
        optimizer.step(); rank_batches+=bool(pairs)
        if step%32==0 or step==1 or step==target:
            row=dict(step=step,total=target,loss=float(objective.detach()),components=components,rank_pairs=len(pairs),seconds=time.monotonic()-began)
            history.append(row); save(folder/'PROGRESS.json',dict(status='TRAINING',**row));print('FUSER_TRAIN',kind,phase,row,flush=True)
        if step%128==0: protect()
    model.eval(); x,y=batch(fixtures)
    with torch.no_grad(): fixture_after=float(loss(model.details(x),x,y,fixtures,weights)[0])
    final=assess(model,validation,128); checkpoint=folder/'FUSER.pth'
    torch.save(dict(new_fuser={k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('backbone.')},
        backbone_checkpoint=source,updates=target,kind=kind),checkpoint)
    assert sha(source['path'])==source['SHA256']
    result=dict(status='COMPLETE',kind=kind,phase=phase,new_parameters=count,updates=target,backbone_checkpoint=source,
        checkpoint=ref(checkpoint),batch_key_digest=digest.hexdigest(),before=before,final=final,
        fixture_before=fixture_before,fixture_after=fixture_after,rank_batches=rank_batches,history=history,
        engineering_pass=rank_batches>0 and (phase!='tiny' or fixture_after<fixture_before),
        scope='Final fusion learning on common frozen trained JEV frontend; not pure end-to-end ordinary controller',
        auxiliary_heads_frozen=True,independent_validation=False,binding=binding('R2 supplementary final-fuser diagnostic'))
    save(folder/'RESULT.json',result);save(REPORTS/'diagnostic_fusion'/f'{kind}_{phase}.json',result)
    print('FUSER_TRAIN_COMPLETE',kind,phase,flush=True)


def online(kind, phase, identity=False):
    protect()
    if identity: import jev_oneweek_identity_analysis as old
    else: import jev_oneweek_online as old
    from jev_recovery_online import configure
    configure(old,2,'XIV')
    old.OUT=OUT/'diagnostic_fusion_native_v2';old.REPORTS=REPORTS/'diagnostic_fusion_native'
    old.policy=lambda arm,seed,ph:build(arm,ph)
    if identity:
        # Compare with the actual common frozen frontend trajectory, rather
        # than an unexecuted parent in the supplementary namespace.
        analyse=old.analyse; original_save=old.save
        old.REPORTS=REPORTS/'diagnostic_fusion_native/identity_v3'
        def preserved_save(path,value):
            path=pathlib.Path(path)
            if path.name=='STRICT_CAUSAL_IDENTITY_ANALYSIS.json':
                path=path.with_name('STRICT_CAUSAL_IDENTITY_ANALYSIS_FUSION_V3.json')
                assert not path.exists(),'Preserve completed identity audit'
            return original_save(path,value)
        old.save=preserved_save
        def routed(arm,seed,ph,video):
            if arm!='parent':return analyse(arm,seed,ph,video)
            current=old.OUT;old.OUT=OUT/'R2_XIV'
            try:return analyse('jev',seed,ph,video)
            finally:old.OUT=current
        old.analyse=routed
    old.main(kind,20261009,phase,16)


def probe(kind):
    protect(); torch.set_num_threads(1)
    _,records,sources=load_records(2); model,descriptor=build(kind,'pilot')
    count=collections.Counter()
    selected=[records[i] for i in torch.linspace(0,len(records)-1,min(128,len(records))).long().tolist()]
    with torch.no_grad():
        for r in selected:
            x,_=batch([r]); before=model.backbone.details(x); after=model.details(x)
            assert before['plans']==after['plans']
            pp=before['plans'][0]
            u=pp[int(before['plan_scores'][0].argmax())];v=pp[int(after['plan_scores'][0].argmax())]
            k=len(r['refs'])
            for row,(a,b) in enumerate(zip(u,v)):
                ca,cb=k if a<0 else a,k if b<0 else b
                count['rows']+=1; count['changed_rows']+=a!=b
                if bool(r['known_options'][row,ca]) and bool(r['known_options'][row,cb]):
                    count['N%d%d'%(int(r['positive'][row,ca]),int(r['positive'][row,cb]))]+=1
                else:
                    count['UNKNOWN_comparison']+=1;count['UNKNOWN_changed_rows']+=a!=b
    save(REPORTS/'diagnostic_fusion'/f'{kind}_CERTIFIED.json',dict(status='COMPLETE_PAIRED_FROZEN_STATE',
        checkpoint=descriptor,counts=dict(count),net=count['N01']-count['N10'],sources=sources,
        before='Identical frozen R2 JEV Pilot1000 seed09',native_mutation=False,independent_validation=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=KINDS,required=True)
    p.add_argument('--phase',choices=['tiny','pilot'],default='pilot')
    p.add_argument('--mode',choices=['train','probe','online','identity'],default='train');a=p.parse_args()
    try:
        if a.mode=='train': train(a.kind,a.phase)
        elif a.mode=='probe': probe(a.kind)
        else: online(a.kind,a.phase,a.mode=='identity')
    except Exception:
        import traceback
        failure('diagnostic_fusion_'+a.kind+'_'+a.mode,traceback.format_exc());raise
