"""Limited TRAIN-only low-rank prototypes and matched-parameter controls.

Frozen detector/foundation vectors are offline TRAIN inputs, never a speed test.
Epoch selection uses held-out association-TRAIN video12, not development/TEST.
"""
import argparse,time,csv,numpy as np,torch
from common import *
from gtr.geometry.lightweight_head import LowRankIdentityHead,matched_hidden_width

def groups(samples,kind,feature_dir):
    values=[];hashes={}
    for sample in samples:
        path=Path(sample['path']);arr=np.load(path);hashes[str(path)]=sha256(path)
        foundations=[None,None]
        if kind!='gmt':
            feature_path=feature_dir/path.name;feat=np.load(feature_path);hashes[str(feature_path)]=sha256(feature_path)
            for view in range(2):
                if kind=='vggt':
                    key=f'layer_17/roi_align/{view}'
                    vector=np.concatenate([feat[key],feat['reverse_'+key]],axis=1).astype(np.float32)/2**.5
                else:vector=feat[f'roi_align/{view}'].astype(np.float32)
                foundations[view]=torch.from_numpy(vector)
        value=dict(video_id=sample['video_id'],frame_id=sample['frame_id'],
            gmt=[torch.tensor(arr[f'gmt_{view}'],dtype=torch.float32) for view in range(2)],foundation=foundations,
            ids=[torch.tensor(arr[f'labels_{view}'],dtype=torch.long) for view in range(2)],base=torch.tensor(arr['gmt_logits'],dtype=torch.float32))
        if (value['ids'][1]>0).any():values.append(value)
    return values,hashes

def scores(model,group,weight):
    residual=model(group['gmt'][1],group['gmt'][0],group['foundation'][1],group['foundation'][0])
    rows=group['ids'][1]>0;cols=group['ids'][0]>0
    candidate=(group['base']+weight*residual)[rows][:,cols]
    candidate=torch.cat([candidate,candidate.new_zeros((len(candidate),1))],dim=1)
    positive=group['ids'][1][rows,None]==group['ids'][0][None,cols]
    target=torch.cat([positive,~positive.any(1,keepdim=True)],dim=1)
    return candidate,target

def objective(model,group,weight):
    candidate,target=scores(model,group,weight)
    return (torch.logsumexp(candidate,dim=1)-torch.logsumexp(candidate.masked_fill(~target,-float('inf')),dim=1)).mean()

@torch.no_grad()
def evaluate(model,data,weight):
    losses=[];queries=correct=shared=rank_correct=0
    for group in data:
        candidate,target=scores(model,group,weight)
        losses.append(float(objective(model,group,weight)))
        queries+=len(candidate);correct+=int(target.gather(1,candidate.argmax(1,keepdim=True)).sum())
        known=target[:,:-1].any(1);shared+=int(known.sum())
        if known.any():rank_correct+=int(target[known,:-1].gather(1,candidate[known,:-1].argmax(1,keepdim=True)).sum())
    return dict(loss=float(np.mean(losses)),queries=queries,association_accuracy=correct/queries,
                shared_queries=shared,retrieval_recall1=rank_correct/shared if shared else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--samples',type=Path,default=RUNTIME/'samples/screen_train_v1/samples.json')
    parser.add_argument('--vggt-features',type=Path,default=RUNTIME/'features/feature_train_native_v1')
    parser.add_argument('--dino-features',type=Path,default=RUNTIME/'features/dinov2_train_v1')
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--seeds',type=int,nargs='+',default=[20261014,20261015,20261016]);args=parser.parse_args()
    if args.steps not in [500,1000,2000]:raise ValueError('predefined limited training budget only')
    gate=json.loads((ROOT/'results/train_complementarity_native_v1.json').read_text())
    if gate['status']!='LIMITED_TRAIN_PAIR_POTENTIAL' or not gate['corrections']:raise ValueError('TRAIN potential gate not met')
    source=json.loads(args.samples.read_text());split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if source['provenance'].get('split')!='train':raise ValueError('TRAIN only')
    vids={s['video_id'] for s in source['samples']}
    if vids!={5,8,12} or not vids.issubset(set(split['association_train_video_ids'])):raise ValueError('frozen scene split mismatch')
    out=RUNTIME/'heads'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);results=[];curves=[];artifact_hashes={};start=time.perf_counter()
    vggt_data,vggt_hashes=groups(source['samples'],'vggt',args.vggt_features)
    gmt_dim=vggt_data[0]['gmt'][0].shape[1];native_dim=vggt_data[0]['foundation'][0].shape[1]//2
    target=LowRankIdentityHead(gmt_dim,native_dim,hidden_dim=96,symmetric=True)
    target_parameters=sum(p.numel() for p in target.parameters())
    del target
    for kind in ['gmt','dinov2','vggt']:
        data,hashes=(vggt_data,vggt_hashes) if kind=='vggt' else groups(source['samples'],kind,args.dino_features)
        artifact_hashes.update(hashes)
        fit=[g for g in data if g['video_id'] in [5,8]];validation=[g for g in data if g['video_id']==12]
        extra=(data[0]['foundation'][0].shape[1]//(2 if kind=='vggt' else 1)) if kind!='gmt' else 0
        hidden=96 if kind=='vggt' else matched_hidden_width(gmt_dim,extra,target_parameters)
        settings=dict(gmt_dim=gmt_dim,foundation_dim=extra,hidden_dim=hidden,output_dim=32,symmetric=kind=='vggt')
        for seed in args.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=LowRankIdentityHead(**settings)
            with torch.no_grad():model.scale.fill_(1.);model.bias.fill_(-.5)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            best_loss=float('inf');best_step=None;checkpoint=out/f'{kind}_seed{seed}.pt'
            for step in range(1,args.steps+1):
                model.train();optimizer.zero_grad(set_to_none=True)
                group=fit[int(rng.integers(len(fit)))];loss=objective(model,group,2.)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite prototype loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                with torch.no_grad():model.scale.clamp_(.01,10.)
                if step%500==0:
                    model.eval();training=evaluate(model,fit,2.);heldout=evaluate(model,validation,2.)
                    row=dict(backbone=kind,seed=seed,step=step,train_loss=training['loss'],train_recall1=training['retrieval_recall1'],validation_loss=heldout['loss'],validation_recall1=heldout['retrieval_recall1'])
                    curves.append(row);print(json.dumps(row),flush=True)
                    if heldout['loss']<best_loss:
                        best_loss=heldout['loss'];best_step=step
                        torch.save(dict(state_dict=model.state_dict(),settings=settings,backbone=kind,seed=seed,step=step,source_split='association_train',fit_video_ids=[5,8],validation_video_ids=[12],
                            feature_weight=2.,max_residual=2.,layers=([17] if kind=='vggt' else [-1]),projection_dim=(0 if kind in ['gmt','vggt'] else 256),pooling=('reid' if kind=='gmt' else 'roi_align'),sample_manifest_sha256=sha256(args.samples)),checkpoint)
            selected=torch.load(checkpoint,map_location='cpu');model.load_state_dict(selected['state_dict']);model.eval()
            results.append(dict(backbone=kind,seed=seed,parameters=sum(p.numel() for p in model.parameters()),target_parameters=target_parameters,selected_step=best_step,
                settings=settings,train=evaluate(model,fit,2.),validation=evaluate(model,validation,2.),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint)))
    with (out/'curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=curves[0]);writer.writeheader();writer.writerows(curves)
    report=dict(status='LIMITED_PROTOTYPES_TRAINED_NOT_CLOSED_LOOP_VALIDATED',provenance=provenance(),used_cache=True,cache_scope='only frozen association-TRAIN target vectors; no inference speed claim',
        fit_video_ids=[5,8],heldout_video_ids=[12],heldout_scope='association TRAIN, not development; backbone historically trained all original TRAIN',
        steps=args.steps,seeds=args.seeds,seconds=time.perf_counter()-start,sample_manifest_sha256=sha256(args.samples),input_artifact_sha256=artifact_hashes,
        weight_rule='fixed2 to permit correction of TRAIN proposal gap2.1596; bounded residual magnitude2; no DEV/TEST tuning',
        head_hyperparameters='AdamW lr0.001 wd0.0001, gradient norm cap10, train initialization scale1/bias-0.5, positive scale constrained0.01..10; fixed before any DEV head evaluation',
        selection_rule='minimum held-out association-TRAIN video12 loss at every500 updates (500/1000/1500/2000 up to the fixed budget), all3 seeds retained; no DEV selection',
        supervised_rows='GT-eligible queries; GT-known gallery for loss; fixed unmatched logit0 when no known positive; labels never encoder inputs',
        controls='same2-layer architecture, output32, trainable scalar calibration; hidden widths give <=0.5% parameter count difference; VGGT shares projection across2 order slots',
        scope_limitations=['tiny15-pair TRAIN sample','not full historical-frame GTA training','independent DINO also corrects the single TRAIN GMT error','no track accuracy or scientific gain established'],results=results)
    save_json(out/'manifest.json',report);save_json(ROOT/f'results/{args.tag}_manifest.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
